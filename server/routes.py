import asyncio
import json
import os
from pathlib import Path

from aiohttp import web

from server.session_manager import session_manager
from utils.logger import logger


VALID_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AVATAR_ROOT = PROJECT_ROOT / "data" / "avatars"
DEFAULT_ASSET_AVATAR_ROOT = PROJECT_ROOT / "assets" / "avatars"

ASSET_AVATAR_DISPLAY = {
    "avatar1": {
        "name": "角色 1",
        "description": "董卿音色数字人",
    },
    "avatar2": {
        "name": "角色 2",
        "description": "女性音色数字人",
    },
    "avatar3": {
        "name": "角色 3",
        "description": "女性音色数字人",
    },
    "avatar4": {
        "name": "角色 4",
        "description": "撒贝宁音色数字人",
    },
    "avatar5": {
        "name": "角色 5",
        "description": "撒贝宁音色数字人",
    },
    "avatar6": {
        "name": "角色 6",
        "description": "撒贝宁增强音色数字人",
    },
    "avatar7": {
        "name": "角色 7",
        "description": "静音微呼吸数字人",
    },
}


def json_ok(data=None):
    body = {"code": 0, "msg": "ok"}
    if data is not None:
        body["data"] = data
    return web.Response(content_type="application/json", text=json.dumps(body))


def json_error(msg: str, code: int = -1):
    return web.Response(
        content_type="application/json",
        text=json.dumps({"code": code, "msg": str(msg)}),
    )


def get_session(request, sessionid: str):
    return session_manager.get_session(sessionid)


def resolve_avatar_root() -> Path:
    env_root = os.environ.get("LIVETALKING_AVATAR_ROOT")
    candidates = []
    if env_root:
        candidates.append(Path(env_root))
    candidates.append(DEFAULT_AVATAR_ROOT)
    candidates.append(Path("data/avatars"))

    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return DEFAULT_AVATAR_ROOT


def natural_sort_key(value: str):
    parts = []
    current = ""
    current_is_digit = None
    for ch in value:
        is_digit = ch.isdigit()
        if current_is_digit is None or current_is_digit == is_digit:
            current += ch
        else:
            parts.append(int(current) if current_is_digit else current.lower())
            current = ch
        current_is_digit = is_digit
    if current:
        parts.append(int(current) if current_is_digit else current.lower())
    return parts


def build_preview_url(image_path: Path, avatar_root: Path) -> str:
    relative_path = image_path.relative_to(avatar_root).as_posix()
    return f"/avatar-data/{relative_path}"


def build_asset_preview_url(image_path: Path, avatar_root: Path) -> str:
    relative_path = image_path.relative_to(avatar_root).as_posix()
    return f"/avatar-assets/{relative_path}"


def pick_avatar_preview(avatar_dir: Path, avatar_root: Path):
    search_dirs = [
        avatar_dir / "preview",
        avatar_dir / "full_imgs",
        avatar_dir / "face_imgs",
        avatar_dir,
    ]
    for search_dir in search_dirs:
        if not search_dir.is_dir():
            continue
        image_files = sorted(
            [
                item
                for item in search_dir.iterdir()
                if item.is_file() and item.suffix.lower() in VALID_IMAGE_EXTENSIONS
            ],
            key=lambda item: natural_sort_key(item.name),
        )
        if image_files:
            return build_preview_url(image_files[0], avatar_root)
    return None


async def list_avatars(request):
    try:
        asset_root = DEFAULT_ASSET_AVATAR_ROOT
        if asset_root.exists():
            asset_images = sorted(
                [
                    item
                    for item in asset_root.iterdir()
                    if item.is_file() and item.suffix.lower() in VALID_IMAGE_EXTENSIONS
                ],
                key=lambda item: natural_sort_key(item.stem),
            )
            if asset_images:
                avatars = []
                for index, image_path in enumerate(asset_images, start=1):
                    display = ASSET_AVATAR_DISPLAY.get(image_path.stem, {})
                    avatars.append(
                        {
                            "id": image_path.stem,
                            "name": display.get("name", image_path.stem),
                            "description": display.get("description", f"EchoMimicV3 数字人角色 {index}"),
                            "image": build_asset_preview_url(image_path, asset_root),
                        }
                    )
                return json_ok(data={"avatars": avatars, "avatar_root": str(asset_root.resolve())})

        avatar_root = resolve_avatar_root()
        if not avatar_root.exists():
            return json_ok(data={"avatars": [], "avatar_root": str(avatar_root)})

        avatar_dirs = sorted(
            [item for item in avatar_root.iterdir() if item.is_dir()],
            key=lambda item: natural_sort_key(item.name),
        )

        avatars = []
        for index, avatar_dir in enumerate(avatar_dirs, start=1):
            avatars.append(
                {
                    "id": avatar_dir.name,
                    "name": avatar_dir.name,
                    "description": f"数字人角色 {index}",
                    "image": pick_avatar_preview(avatar_dir, avatar_root),
                }
            )

        return json_ok(data={"avatars": avatars, "avatar_root": str(avatar_root)})
    except Exception as exc:
        logger.exception("list_avatars exception:")
        return json_error(str(exc))


async def list_sessions(request):
    try:
        sessions = session_manager.list_sessions()
        return json_ok(
            data={
                "sessions": sessions,
                "active_sessionid": session_manager.latest_ready_sessionid(),
            }
        )
    except Exception as exc:
        logger.exception("list_sessions exception:")
        return json_error(str(exc))


async def human(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        logger.info(
            "human request type=%s sessionid=%s text_len=%d interrupt=%s",
            params.get("type"),
            sessionid,
            len(params.get("text", "")),
            bool(params.get("interrupt")),
        )
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.warning("human request rejected: session not found, sessionid=%s", sessionid)
            return json_error("session not found")

        if params.get("interrupt"):
            avatar_session.flush_talk()

        datainfo = {}
        if params.get("tts"):
            datainfo["tts"] = params.get("tts")

        if params["type"] == "echo":
            avatar_session.put_msg_txt(params["text"], datainfo)
        elif params["type"] == "chat":
            llm_response = request.app.get("llm_response")
            if llm_response:
                asyncio.get_event_loop().run_in_executor(
                    None, llm_response, params["text"], avatar_session, datainfo
                )

        return json_ok()
    except Exception as exc:
        logger.exception("human route exception:")
        return json_error(str(exc))


async def interrupt_talk(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.flush_talk()
        return json_ok()
    except Exception as exc:
        logger.exception("interrupt_talk exception:")
        return json_error(str(exc))


async def humanaudio(request):
    try:
        form = await request.post()
        sessionid = str(form.get("sessionid", ""))
        fileobj = form["file"]
        filebytes = fileobj.file.read()

        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.put_audio_file(filebytes, {})
        return json_ok()
    except Exception as exc:
        logger.exception("humanaudio exception:")
        return json_error(str(exc))


async def set_audiotype(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        avatar_session.set_custom_state(params["audiotype"])
        return json_ok()
    except Exception as exc:
        logger.exception("set_audiotype exception:")
        return json_error(str(exc))


async def record(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        if params["type"] == "start_record":
            avatar_session.start_recording()
        elif params["type"] == "end_record":
            avatar_session.stop_recording()
        return json_ok()
    except Exception as exc:
        logger.exception("record exception:")
        return json_error(str(exc))


async def is_speaking(request):
    params = await request.json()
    sessionid = params.get("sessionid", "")
    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")
    return json_ok(data=avatar_session.is_speaking())


async def choice_init(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        tree_id = params.get("tree_id") or getattr(avatar_session.opt, "choice_tree_id", "default_choice_tree")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        orchestrator = request.app.get("choice_orchestrator")
        if orchestrator is None:
            return json_error("choice orchestrator not configured")
        payload = orchestrator.init_session(avatar_session, tree_id)
        return json_ok(data=payload)
    except Exception as exc:
        logger.exception("choice_init exception:")
        return json_error(str(exc))


async def choice_select(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        choice_id = params.get("choice_id", "")
        logger.info("choice select request sessionid=%s choice_id=%s", sessionid, choice_id)
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            logger.warning("choice select rejected: session not found, sessionid=%s", sessionid)
            return json_error("session not found")
        orchestrator = request.app.get("choice_orchestrator")
        if orchestrator is None:
            return json_error("choice orchestrator not configured")
        payload = orchestrator.select_choice(
            avatar_session,
            choice_id=choice_id,
            interrupt=bool(params.get("interrupt", True)),
        )
        return json_ok(data=payload)
    except Exception as exc:
        logger.exception("choice_select exception:")
        return json_error(str(exc))


async def choice_state(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        orchestrator = request.app.get("choice_orchestrator")
        if orchestrator is None:
            return json_error("choice orchestrator not configured")
        payload = orchestrator.get_state(avatar_session)
        return json_ok(data=payload)
    except Exception as exc:
        logger.exception("choice_state exception:")
        return json_error(str(exc))


async def choice_reset(request):
    try:
        params = await request.json()
        sessionid = params.get("sessionid", "")
        avatar_session = get_session(request, sessionid)
        if avatar_session is None:
            return json_error("session not found")
        orchestrator = request.app.get("choice_orchestrator")
        if orchestrator is None:
            return json_error("choice orchestrator not configured")
        avatar_session.flush_talk()
        payload = orchestrator.reset_session(avatar_session)
        return json_ok(data=payload)
    except Exception as exc:
        logger.exception("choice_reset exception:")
        return json_error(str(exc))


def setup_routes(app):
    avatar_root = resolve_avatar_root()
    asset_avatar_root = DEFAULT_ASSET_AVATAR_ROOT

    app.router.add_post("/human", human)
    app.router.add_post("/humanaudio", humanaudio)
    app.router.add_post("/set_audiotype", set_audiotype)
    app.router.add_post("/record", record)
    app.router.add_post("/interrupt_talk", interrupt_talk)
    app.router.add_post("/is_speaking", is_speaking)
    app.router.add_post("/choice/init", choice_init)
    app.router.add_post("/choice/select", choice_select)
    app.router.add_post("/choice/state", choice_state)
    app.router.add_post("/choice/reset", choice_reset)
    app.router.add_get("/api/avatars", list_avatars)
    app.router.add_get("/api/sessions", list_sessions)

    if avatar_root.exists():
        app.router.add_static("/avatar-data/", path=str(avatar_root))
    if asset_avatar_root.exists():
        app.router.add_static("/avatar-assets/", path=str(asset_avatar_root))

    app.router.add_static("/", path="web")
