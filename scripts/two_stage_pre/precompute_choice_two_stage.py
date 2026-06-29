#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import soundfile as sf
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).resolve().with_name("config.yaml")
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from choice.echomimicv3_cache import ChoiceEchoMimicV3CacheStore, hash_choice_tree
from choice.orchestrator import ChoiceOrchestrator, StaticChoiceTreeProvider


def status(message: str) -> None:
    print(f"[choice-two-stage] {message}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Precompute EchoMimicV3 -> LatentSync videos for choice-mode male/female WAV caches."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output_root", type=Path, default=None, help="Override output_root from config.yaml.")
    parser.add_argument("--voices", nargs="*", default=None, help="Voice keys from config.yaml, e.g. male female.")
    parser.add_argument("--nodes", nargs="*", default=None, help="Only process these node ids, e.g. root cache api.")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--skip_stage1", action="store_true")
    parser.add_argument("--skip_stage2", action="store_true")
    parser.add_argument("--skip_import", action="store_true", help="Only generate two-stage mp4; do not import into runtime cache.")
    parser.add_argument("--force", action="store_true", help="Regenerate existing stage outputs.")
    parser.add_argument("--python", default=sys.executable)
    return parser.parse_args()


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML must be a mapping: {path}")
    return data


def write_yaml(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, allow_unicode=True, sort_keys=False)


def require_path(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def sha1_file(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def iter_audio_files(audio_dir: Path, nodes: set[str] | None):
    for audio_path in sorted(audio_dir.glob("*.wav")):
        if nodes and audio_path.stem not in nodes:
            continue
        yield audio_path


def build_item_config(batch_config: dict, voice: str, image_path: Path, audio_path: Path, output_root: Path) -> dict:
    item_root = output_root / voice / audio_path.stem
    return {
        "image_path": str(image_path),
        "audio_path": str(audio_path),
        "video_path": str(item_root / "final" / f"{audio_path.stem}.mp4"),
        "output_root": str(item_root),
        "stage1_output_dir": str(item_root / "stage1_echomimicv3"),
        "stage2_output_dir": str(item_root / "stage2_latentsync"),
        "echomimicv3": dict(batch_config.get("echomimicv3", {}) or {}),
        "latentsync": dict(batch_config.get("latentsync", {}) or {}),
    }


def ensure_silence_audio(audio_path: Path, duration_seconds: float, sample_rate: int = 16000) -> None:
    if audio_path.exists():
        return
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    samples = max(1, int(round(float(duration_seconds) * sample_rate)))
    silence = np.zeros(samples, dtype=np.float32)
    sf.write(str(audio_path), silence, sample_rate)


def stage1_ready(item_config: dict) -> bool:
    manifest_path = Path(item_config["stage1_output_dir"]) / "manifest.json"
    stage1_video = Path(item_config["stage1_output_dir"]) / f"{Path(item_config['audio_path']).stem}.mp4"
    return manifest_path.exists() and stage1_video.exists()


def stage2_ready(item_config: dict) -> bool:
    return Path(item_config["video_path"]).exists()


def run_command(cmd: list[str], cwd: Path, dry_run: bool) -> int:
    status("$ " + " ".join(cmd))
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=cwd).returncode


def run_pipeline_stage(args, runner_path: Path, config_path: Path, stage: str) -> int:
    cmd = [args.python, str(runner_path), "--config", str(config_path)]
    if args.dry_run:
        cmd.append("--dry-run")
    if stage == "stage1":
        cmd.append("--skip-stage2")
    elif stage == "stage2":
        cmd.append("--skip-stage1")
    else:
        raise ValueError(f"unknown stage: {stage}")
    return run_command(cmd, PROJECT_ROOT, args.dry_run)


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"cannot open video: {video_path}")
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"video has no frames: {video_path}")
    return frames


def _match_lab_stats(source: np.ndarray, target: np.ndarray, strength: float) -> np.ndarray:
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    output_lab = source_lab.copy()
    strength = float(np.clip(strength, 0.0, 1.0))
    for channel in range(3):
        src = source_lab[:, :, channel]
        tgt = target_lab[:, :, channel]
        src_mean, src_std = cv2.meanStdDev(src)
        tgt_mean, tgt_std = cv2.meanStdDev(tgt)
        src_mean = float(src_mean[0][0])
        src_std = max(float(src_std[0][0]), 1e-6)
        tgt_mean = float(tgt_mean[0][0])
        tgt_std = max(float(tgt_std[0][0]), 1e-6)
        matched = (src - src_mean) * (tgt_std / src_std) + tgt_mean
        output_lab[:, :, channel] = src * (1.0 - strength) + matched * strength
    output_lab = np.clip(output_lab, 0, 255).astype(np.uint8)
    return cv2.cvtColor(output_lab, cv2.COLOR_LAB2BGR)


def _sharpen_image(image: np.ndarray, amount: float) -> np.ndarray:
    amount = float(max(0.0, amount))
    if amount <= 0:
        return image
    blurred = cv2.GaussianBlur(image, (0, 0), 1.0)
    return cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)


def export_reference_image_from_video(
    video_path: Path,
    avatar_image_path: Path,
    image_path: Path,
    *,
    mode: str = "color_match_avatar",
    frame_start: int = 3,
    frame_count: int = 6,
    color_strength: float = 0.75,
    sharpen_amount: float = 0.25,
    force: bool = False,
) -> Path:
    if image_path.exists() and not force:
        return image_path

    frames = read_video_frames(video_path)
    start = max(0, int(frame_start))
    count = max(1, int(frame_count))
    selected = frames[start:start + count]
    if not selected:
        selected = frames[:count]
    if not selected:
        raise RuntimeError(f"cannot export reference image from empty video: {video_path}")

    min_h = min(frame.shape[0] for frame in selected)
    min_w = min(frame.shape[1] for frame in selected)
    aligned = []
    for frame in selected:
        if frame.shape[:2] != (min_h, min_w):
            frame = cv2.resize(frame, (min_w, min_h), interpolation=cv2.INTER_AREA)
        aligned.append(frame.astype(np.float32))
    video_reference = np.median(np.stack(aligned, axis=0), axis=0).clip(0, 255).astype(np.uint8)

    mode = (mode or "color_match_avatar").strip().lower()
    if mode == "video_frame":
        reference = video_reference
    else:
        avatar_reference = cv2.imread(str(avatar_image_path))
        if avatar_reference is None:
            raise FileNotFoundError(f"cannot read avatar image: {avatar_image_path}")
        if avatar_reference.shape[:2] != video_reference.shape[:2]:
            avatar_reference = cv2.resize(
                avatar_reference,
                (video_reference.shape[1], video_reference.shape[0]),
                interpolation=cv2.INTER_LANCZOS4,
            )
        reference = _match_lab_stats(avatar_reference, video_reference, color_strength)
        reference = _sharpen_image(reference, sharpen_amount)

    image_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(image_path), reference):
        raise RuntimeError(f"failed to write reference image: {image_path}")
    return image_path


def export_idle_loop_frames(
    video_path: Path,
    frames_dir: Path,
    *,
    max_frames: int = 125,
    force: bool = False,
) -> int:
    if frames_dir.exists() and any(frames_dir.glob("*.png")) and not force:
        return len(list(frames_dir.glob("*.png")))

    frames = read_video_frames(video_path)
    if max_frames and max_frames > 0:
        frames = frames[:max_frames]
    frames_dir.mkdir(parents=True, exist_ok=True)

    if force:
        for old_frame in frames_dir.glob("*.png"):
            old_frame.unlink()

    for index, frame in enumerate(frames):
        output_path = frames_dir / f"{index:06d}.png"
        if output_path.exists() and not force:
            continue
        cv2.imwrite(str(output_path), frame)
    return len(frames)


def read_audio_16k(audio_path: Path) -> np.ndarray:
    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]
    if sample_rate != 16000 and audio.shape[0] > 0:
        import resampy

        audio = resampy.resample(audio, sample_rate, 16000)
    return np.asarray(audio, dtype=np.float32)


def make_runtime_session(
    batch_config: dict,
    preset: dict,
    voice: str,
    tree_id: str,
    image_path: Path,
) -> SimpleNamespace:
    runtime = batch_config.get("runtime", {}) or {}
    tts = batch_config.get("tts", {}) or {}
    echo = batch_config.get("echomimicv3", {}) or {}
    paths = batch_config.get("paths", {}) or {}
    opt = SimpleNamespace(
        model=runtime.get("model", "echomimicv3"),
        avatar_id=preset.get("avatar_id", voice),
        tts=tts.get("name", runtime.get("tts", "gpt-sovits")),
        REF_FILE=str(resolve_path(preset["ref_file"])),
        REF_TEXT=str(preset.get("ref_text", "")),
        fps=int(echo.get("fps", 25)),
        echomimicv3_sample_size=list(echo.get("sample_size", [512, 512])),
        echomimicv3_num_steps=echo.get("num_inference_steps"),
        echomimicv3_guidance_scale=echo.get("guidance_scale"),
        echomimicv3_audio_guidance_scale=echo.get("audio_guidance_scale"),
        echomimicv3_transformer_path=str(resolve_path(paths.get("echomimicv3_transformer_path", ""))) if paths.get("echomimicv3_transformer_path") else "",
        echomimicv3_weight_dtype=echo.get("weight_dtype", "bfloat16"),
    )
    avatar = SimpleNamespace(
        avatar_id=opt.avatar_id,
        ref_image_path=str(image_path),
        prompt=echo.get("prompt", "A person is speaking."),
        negative_prompt=echo.get(
            "negative_prompt",
            "blur, low quality, distorted face, bad hands, extra fingers, deformed body, strange movement, jitter, flicker",
        ),
    )
    return SimpleNamespace(opt=opt, avatar=avatar, _choice_state={"tree_id": tree_id})


def import_runtime_cache(
    batch_config: dict,
    tree_payload: dict,
    tree_hash: str,
    cache_store: ChoiceEchoMimicV3CacheStore,
    orchestrator: ChoiceOrchestrator,
    voice: str,
    preset: dict,
    image_path: Path,
    audio_path: Path,
    video_path: Path,
):
    tree_id = tree_payload.get("tree_id") or "default_choice_tree"
    node_id = audio_path.stem
    node = tree_payload["node_map"].get(node_id)
    if node is None:
        raise KeyError(f"choice node not found for audio stem: {node_id}")
    tts_text = orchestrator._node_tts_text(node)
    runtime_session = make_runtime_session(batch_config, preset, voice, tree_id, image_path)
    params = cache_store.build_params(runtime_session, tree_id, node, tts_text)
    frames = read_video_frames(video_path)
    audio = read_audio_16k(audio_path)
    cache_dir = cache_store.set(tree_id, params["cache_key"], params, audio, frames, source_tree_hash=tree_hash)
    return {
        "node_id": node_id,
        "cache_key": params["cache_key"],
        "cache_dir": str(cache_dir),
        "duration_seconds": float(audio.shape[0] / 16000.0),
        "frames": len(frames),
        "audio_samples": int(audio.shape[0]),
        "avatar_id": runtime_session.opt.avatar_id,
        "ref_file": runtime_session.opt.REF_FILE,
        "ref_text": runtime_session.opt.REF_TEXT,
    }


def write_runtime_manifest(cache_store: ChoiceEchoMimicV3CacheStore, tree_id: str, tree_hash: str, imports: list[dict]):
    manifest_path = cache_store.manifest_path(tree_id)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            payload = {}
    else:
        payload = {}
    items = payload.get("items", {})
    for item in imports:
        entry = {
            "cache_key": item["cache_key"],
            "status": "ready",
            "duration_seconds": item["duration_seconds"],
            "frames": item["frames"],
            "audio_samples": item["audio_samples"],
            "message": "imported from two-stage precompute",
            "avatar_id": item["avatar_id"],
            "ref_file": item["ref_file"],
            "ref_text": item["ref_text"],
        }
        existing = items.get(item["node_id"], [])
        entries = existing if isinstance(existing, list) else [existing]
        entries = [old for old in entries if old.get("cache_key") != entry["cache_key"]]
        entries.append(entry)
        items[item["node_id"]] = entries
    payload.update(
        {
            "schema_version": "choice_echomimicv3_v1",
            "tree_id": tree_id,
            "tree_hash": tree_hash,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "items": items,
        }
    )
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def main() -> int:
    args = parse_args()
    config_path = args.config.resolve()
    require_path(config_path, "two-stage precompute config")
    batch_config = load_yaml(config_path)
    runner_path = resolve_path(batch_config.get("runner", "two_stage/run_two_stage_yaml_pipeline.py"))
    output_root = resolve_path(args.output_root or batch_config.get("output_root", "cache/choice_two_stage"))
    require_path(runner_path, "two-stage runner")
    voice_presets = batch_config.get("voices", {}) or {}
    if not isinstance(voice_presets, dict) or not voice_presets:
        raise ValueError(f"config must define voices: {config_path}")
    voices = args.voices or list(voice_presets.keys())
    nodes = set(args.nodes) if args.nodes else None
    tree_id = (batch_config.get("choice", {}) or {}).get("tree_id", "default_choice_tree")
    provider = StaticChoiceTreeProvider(PROJECT_ROOT / "data" / "choice_trees")
    tree_payload = provider.load_tree(tree_id)
    tree_hash = hash_choice_tree(tree_payload)
    cache_store = ChoiceEchoMimicV3CacheStore(PROJECT_ROOT)
    orchestrator = ChoiceOrchestrator(str(PROJECT_ROOT))
    imported_items = []
    idle_loop = (batch_config.get("choice", {}) or {}).get("idle_loop", {}) or {}
    idle_loop_enabled = bool(idle_loop.get("enabled", True))
    idle_loop_audio_name = str(idle_loop.get("audio_name") or "idle_loop_silence")
    idle_loop_duration_seconds = float(idle_loop.get("duration_seconds") or 5.0)
    idle_loop_reference_source_node_id = str(idle_loop.get("reference_source_node_id") or tree_payload.get("root_node_id") or "root")
    idle_loop_reference_mode = str(idle_loop.get("reference_mode") or "color_match_avatar")
    idle_loop_reference_frame_start = int(idle_loop.get("reference_frame_start") or 3)
    idle_loop_reference_frame_count = int(idle_loop.get("reference_frame_count") or 6)
    idle_loop_reference_color_strength = float(idle_loop.get("reference_color_strength") or 0.75)
    idle_loop_reference_sharpen_amount = float(idle_loop.get("reference_sharpen_amount") or 0.25)
    idle_loop_dir_name = str(idle_loop.get("frames_dir_name") or "idle_frames_two_stage")
    idle_loop_max_frames = int(idle_loop.get("max_frames") or 125)

    manifest = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "config": str(config_path),
        "runner": str(runner_path),
        "output_root": str(output_root),
        "tree_id": tree_id,
        "items": [],
    }
    failed = 0
    ready = 0
    tasks = []

    for voice in voices:
        if voice not in voice_presets:
            raise ValueError(f"voice {voice!r} not defined in {config_path}")
        preset = voice_presets[voice] or {}
        image_path = resolve_path(preset["image_path"])
        audio_dir = resolve_path(preset["audio_dir"])
        require_path(image_path, f"{voice} avatar image")
        require_path(audio_dir, f"{voice} audio dir")

        audio_files = list(iter_audio_files(audio_dir, nodes))
        if not audio_files:
            status(f"no audio files matched: voice={voice} audio_dir={audio_dir}")

        for audio_path in audio_files:
            item_config = build_item_config(batch_config, voice, image_path, audio_path.resolve(), output_root)
            item_root = Path(item_config["output_root"])
            config_path = item_root / "config.yaml"
            write_yaml(config_path, item_config)
            tasks.append({
                "voice": voice,
                "preset": preset,
                "node_id": audio_path.stem,
                "image_path_obj": image_path,
                "audio_path_obj": audio_path.resolve(),
                "config_path_obj": config_path,
                "item_config": item_config,
                "image_path": str(image_path),
                "audio_path": str(audio_path),
                "config_path": str(config_path),
                "output_root": item_config["output_root"],
                "video_path": item_config["video_path"],
            })

        if idle_loop_enabled and (nodes is None or idle_loop_audio_name in nodes):
            idle_audio_path = output_root / "_idle_audio" / f"{voice}_{idle_loop_audio_name}.wav"
            if not args.dry_run:
                ensure_silence_audio(idle_audio_path, idle_loop_duration_seconds)
            idle_image_path = image_path
            reference_video_path = output_root / voice / idle_loop_reference_source_node_id / "final" / f"{idle_loop_reference_source_node_id}.mp4"
            reference_image_path = output_root / voice / idle_loop_audio_name / "reference" / f"{voice}_{idle_loop_reference_source_node_id}_stage2_ref.png"
            if not args.dry_run and reference_video_path.exists():
                idle_image_path = export_reference_image_from_video(
                    reference_video_path,
                    image_path,
                    reference_image_path,
                    mode=idle_loop_reference_mode,
                    frame_start=idle_loop_reference_frame_start,
                    frame_count=idle_loop_reference_frame_count,
                    color_strength=idle_loop_reference_color_strength,
                    sharpen_amount=idle_loop_reference_sharpen_amount,
                    force=args.force,
                )
                status(
                    "idle loop reference image ready: voice=%s source=%s image=%s"
                    % (voice, reference_video_path, idle_image_path)
                )
            elif not args.dry_run:
                status(
                    "idle loop reference video missing, fallback to avatar image: voice=%s source=%s"
                    % (voice, reference_video_path)
                )
            item_config = build_item_config(batch_config, voice, idle_image_path, idle_audio_path.resolve(), output_root)
            item_root = Path(item_config["output_root"])
            config_path = item_root / "config.yaml"
            write_yaml(config_path, item_config)
            tasks.append({
                "voice": voice,
                "preset": preset,
                "node_id": idle_loop_audio_name,
                "image_path_obj": idle_image_path,
                "audio_path_obj": idle_audio_path.resolve(),
                "config_path_obj": config_path,
                "item_config": item_config,
                "image_path": str(idle_image_path),
                "audio_path": str(idle_audio_path),
                "config_path": str(config_path),
                "output_root": item_config["output_root"],
                "video_path": item_config["video_path"],
                "is_idle_loop": True,
                "idle_reference_video": str(reference_video_path),
            })

    if not args.skip_stage1:
        status(f"stage1 batch start: tasks={len(tasks)}")
        for task in tasks:
            item = {key: task[key] for key in ("voice", "node_id", "image_path", "audio_path", "config_path", "output_root", "video_path")}
            if stage1_ready(task["item_config"]) and not args.force:
                item["stage1_status"] = "skipped_ready"
                item["stage1_returncode"] = 0
                status(f"stage1 skip ready: voice={task['voice']} node={task['node_id']}")
            else:
                returncode = run_pipeline_stage(args, runner_path, task["config_path_obj"], "stage1")
                item["stage1_returncode"] = returncode
                item["stage1_status"] = "ready" if returncode == 0 else "failed"
                if returncode != 0:
                    failed += 1
            manifest["items"].append(item)
        status("stage1 batch done")
    else:
        status("stage1 skipped by --skip_stage1")

    if not args.skip_stage2:
        status(f"stage2 batch start: tasks={len(tasks)}")
        for task in tasks:
            item = {key: task[key] for key in ("voice", "node_id", "image_path", "audio_path", "config_path", "output_root", "video_path")}
            if stage2_ready(task["item_config"]) and not args.force:
                returncode = 0
                item["stage2_status"] = "skipped_ready"
                item["stage2_returncode"] = 0
                status(f"stage2 skip ready: voice={task['voice']} node={task['node_id']}")
            else:
                if not args.dry_run and not stage1_ready(task["item_config"]):
                    item["stage2_status"] = "failed"
                    item["stage2_returncode"] = 1
                    item["message"] = "stage1 output missing"
                    manifest["items"].append(item)
                    failed += 1
                    status(f"stage2 missing stage1: voice={task['voice']} node={task['node_id']}")
                    continue
                returncode = run_pipeline_stage(args, runner_path, task["config_path_obj"], "stage2")
                item["stage2_returncode"] = returncode
                item["stage2_status"] = "ready" if returncode == 0 else "failed"

            if returncode == 0 and not args.dry_run and not args.skip_import:
                if task.get("is_idle_loop"):
                    item["import_status"] = "skipped_idle_loop"
                else:
                    video_path = Path(task["item_config"]["video_path"])
                    require_path(video_path, "two-stage final video")
                    imported = import_runtime_cache(
                        batch_config,
                        tree_payload,
                        tree_hash,
                        cache_store,
                        orchestrator,
                        task["voice"],
                        task["preset"],
                        task["image_path_obj"],
                        task["audio_path_obj"],
                        video_path,
                    )
                    imported_items.append(imported)
                    item["choice_cache_key"] = imported["cache_key"]
                    item["choice_cache_dir"] = imported["cache_dir"]
                    item["import_status"] = "ready"

            if (
                returncode == 0
                and not args.dry_run
                and idle_loop_enabled
                and task.get("is_idle_loop")
            ):
                video_path = Path(task["item_config"]["video_path"])
                frames_dir = (
                    PROJECT_ROOT
                    / "data"
                    / "avatars"
                    / str(task["preset"].get("avatar_id", task["voice"]))
                    / "echomimicv3"
                    / idle_loop_dir_name
                )
                exported_frames = export_idle_loop_frames(
                    video_path,
                    frames_dir,
                    max_frames=idle_loop_max_frames,
                    force=args.force,
                )
                item["idle_loop_frames_dir"] = str(frames_dir)
                item["idle_loop_frames"] = exported_frames
                status(
                    "silent idle loop frames ready: voice=%s avatar=%s frames=%d dir=%s"
                    % (
                        task["voice"],
                        task["preset"].get("avatar_id", task["voice"]),
                        exported_frames,
                        frames_dir,
                    )
                )

            manifest["items"].append(item)
            if returncode == 0:
                ready += 1
            else:
                failed += 1
        status("stage2 batch done")
    else:
        status("stage2 skipped by --skip_stage2")

    manifest_path = output_root / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if imported_items and not args.dry_run:
        runtime_manifest_path = write_runtime_manifest(cache_store, tree_id, tree_hash, imported_items)
        status(f"runtime choice cache manifest updated: {runtime_manifest_path}")
    status(f"finished: ready={ready} failed={failed} manifest={manifest_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
