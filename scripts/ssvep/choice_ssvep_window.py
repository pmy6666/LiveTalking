#!/usr/bin/env python3
import argparse
import json
import math
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request

import yaml

# Keep the OpenGL setup consistent with ../../ssvep_flicker.py.
os.environ.setdefault("__NV_PRIME_RENDER_OFFLOAD", "1")
os.environ.setdefault("__GLX_VENDOR_LIBRARY_NAME", "nvidia")
os.environ.setdefault("PYGLET_SHADOW_WINDOW", "0")

DEFAULT_FREQS = [12.8, 11.2, 8.8]
DEFAULT_PHASES = [0.0, 0.0, 0.0]


def setup_psychopy():
    import pyglet

    pyglet.options["shadow_window"] = False
    try:
        temp_win = pyglet.window.Window(width=1, height=1, visible=False)
    except Exception as exc:
        display = os.environ.get("DISPLAY")
        raise RuntimeError(
            "无法连接到图形显示环境 DISPLAY=%r。请在有桌面显示的终端中运行，"
            "或在 scripts/ssvep/config.yaml 中设置 display: \":0\" 后重试。"
            % display
        ) from exc
    temp_win.switch_to()

    from psychopy import core, event, prefs, visual

    prefs.general["winType"] = "pyglet"
    prefs.general["autoLog"] = False
    return core, event, visual


def post_json(base_url, path, payload, timeout=1.5):
    url = base_url.rstrip("/") + path
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body or "{}")
    if parsed.get("code") != 0:
        raise RuntimeError(parsed.get("msg") or ("request failed: " + path))
    return parsed.get("data") or {}


def get_json(base_url, path, timeout=1.5):
    url = base_url.rstrip("/") + path
    with urllib.request.urlopen(url, timeout=timeout) as response:
        body = response.read().decode("utf-8")
    parsed = json.loads(body or "{}")
    if parsed.get("code") != 0:
        raise RuntimeError(parsed.get("msg") or ("request failed: " + path))
    return parsed.get("data") or {}


def resolve_sessionid(base_url, configured_sessionid):
    sessionid = str(configured_sessionid or "").strip()
    if sessionid and sessionid not in {"auto", "latest", "你的sessionid"}:
        return sessionid

    try:
        data = get_json(base_url, "/api/sessions", timeout=2.0)
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise RuntimeError(
                "server does not provide /api/sessions yet. "
                "Please restart LiveTalking after pulling this change, "
                "or set sessionid manually in scripts/ssvep/config.yaml."
            ) from exc
        raise
    active = data.get("active_sessionid")
    if active:
        return str(active)

    sessions = data.get("sessions") or []
    for item in sessions:
        if item.get("ready") and item.get("sessionid"):
            return str(item["sessionid"])
    raise RuntimeError("no active LiveTalking session found; please start the web connection first")


def load_choice_tree(path):
    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)
    nodes = {node["node_id"]: node for node in payload.get("nodes", [])}
    root_node_id = payload.get("root_node_id") or "root"
    if root_node_id not in nodes:
        raise ValueError("root node not found in choice tree: %s" % root_node_id)
    return {
        "tree_id": payload.get("tree_id") or "default_choice_tree",
        "root_node_id": root_node_id,
        "nodes": nodes,
    }


def load_yaml_config(path):
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as file:
        payload = yaml.safe_load(file) or {}
    if not isinstance(payload, dict):
        raise ValueError("YAML config must be a mapping: %s" % path)
    return payload


def add_configurable_argument(parser, name, config, key, default=None, **kwargs):
    parser.add_argument(name, default=config.get(key, default), **kwargs)


def configure_display(display):
    display = str(display or "").strip()
    if display:
        os.environ["DISPLAY"] = display
    elif not os.environ.get("DISPLAY"):
        # Most local Ubuntu desktop sessions expose the first X display as :0.
        # This keeps the YAML command usable from terminals that did not inherit DISPLAY.
        default_x11_socket = Path("/tmp/.X11-unix/X0")
        if default_x11_socket.exists():
            os.environ["DISPLAY"] = ":0"


def get_node_choices(tree, node_id):
    node = tree["nodes"].get(node_id) or {}
    return (node.get("choices") or [])[:3]


def build_lut(frequency, phase, actual_fps, lut_len):
    fps = actual_fps or 60
    return [
        (math.sin(2 * math.pi * frequency * i / fps + phase * math.pi) + 1) / 2 * 0.9 + 0.1
        for i in range(lut_len)
    ]


def get_choice_frequency(choice, index):
    ssvep = choice.get("ssvep") or {}
    configured = ssvep.get("frequency") or choice.get("ssvep_frequency")
    try:
        frequency = float(configured)
    except (TypeError, ValueError):
        frequency = 0.0
    return frequency if frequency > 0 else DEFAULT_FREQS[index % len(DEFAULT_FREQS)]


def get_choice_phase(choice, index):
    ssvep = choice.get("ssvep") or {}
    configured = ssvep.get("phase") or choice.get("ssvep_phase")
    try:
        return float(configured)
    except (TypeError, ValueError):
        return DEFAULT_PHASES[index % len(DEFAULT_PHASES)]


def shorten_text(text, max_chars=18):
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def make_layout(width, height, block_size, spacing):
    usable_width = width - block_size
    max_spacing = max(0, usable_width / 2)
    actual_spacing = min(spacing, max_spacing)
    y = -height * 0.05
    return [(-actual_spacing, y), (0, y), (actual_spacing, y)]


def make_target(win, index, pos, block_size):
    colors = [
        ([0.0, 0.35, 1.0], "12.8Hz"),
        ([0.0, 1.0, 0.45], "11.2Hz"),
        ([0.65, 0.2, 1.0], "8.8Hz"),
    ]
    edge_color, _ = colors[index % len(colors)]
    block = visual.Rect(
        win,
        width=block_size,
        height=block_size * 0.58,
        pos=pos,
        fillColor=[-0.85, -0.85, -0.85],
        lineColor=edge_color,
        lineWidth=6,
    )
    number = visual.TextStim(
        win,
        text=str(index + 1),
        pos=(pos[0], pos[1] + block_size * 0.15),
        height=max(32, block_size * 0.18),
        color=[1, 1, 1],
        bold=True,
    )
    label = visual.TextStim(
        win,
        text="等待选项",
        pos=(pos[0], pos[1] - block_size * 0.02),
        height=max(26, block_size * 0.11),
        color=[1, 1, 1],
        bold=True,
        wrapWidth=block_size * 0.88,
    )
    freq_label = visual.TextStim(
        win,
        text="",
        pos=(pos[0], pos[1] - block_size * 0.2),
        height=max(18, block_size * 0.075),
        color=[0.85, 0.96, 1.0],
        wrapWidth=block_size * 0.8,
    )
    return {
        "block": block,
        "number": number,
        "label": label,
        "freq_label": freq_label,
    }


def update_target_positions(targets, width, height, block_size, spacing):
    for target, pos in zip(targets, make_layout(width, height, block_size, spacing)):
        target["block"].pos = pos
        target["block"].width = block_size
        target["block"].height = block_size * 0.58
        target["number"].pos = (pos[0], pos[1] + block_size * 0.15)
        target["number"].height = max(32, block_size * 0.18)
        target["label"].pos = (pos[0], pos[1] - block_size * 0.02)
        target["label"].height = max(26, block_size * 0.11)
        target["label"].wrapWidth = block_size * 0.88
        target["freq_label"].pos = (pos[0], pos[1] - block_size * 0.2)
        target["freq_label"].height = max(18, block_size * 0.075)


def draw_targets(targets, lut_list, frame_cnt, active_count):
    for index, target in enumerate(targets):
        if index >= active_count:
            target["block"].fillColor = [-0.92, -0.92, -0.92]
        else:
            intensity = lut_list[index][frame_cnt % len(lut_list[index])]
            color = intensity * 2 - 1
            target["block"].fillColor = [color, color, color]
        target["block"].draw()
        target["number"].draw()
        target["label"].draw()
        target["freq_label"].draw()


def main():
    project_root = Path(__file__).resolve().parents[2]
    default_tree_path = project_root / "data" / "choice_trees" / "default_choice_tree.json"
    default_config_path = project_root / "scripts" / "ssvep" / "config.yaml"

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", default=str(default_config_path), help="YAML config path")
    config_args, remaining_argv = config_parser.parse_known_args()
    config = load_yaml_config(config_args.config)

    parser = argparse.ArgumentParser(
        description="LiveTalking 3-option SSVEP choice window",
        parents=[config_parser],
    )
    add_configurable_argument(parser, "--server", config, "server", default="http://127.0.0.1:8010", help="LiveTalking server URL")
    add_configurable_argument(parser, "--sessionid", config, "sessionid", default="", help="LiveTalking session id")
    add_configurable_argument(parser, "--display", config, "display", default="", help="X11 display, for example :0")
    add_configurable_argument(parser, "--choice-tree", config, "choice_tree", default=str(default_tree_path), help="fixed choice tree json path")
    add_configurable_argument(parser, "--width", config, "width", type=int, default=1280, help="window width in pixels")
    add_configurable_argument(parser, "--height", config, "height", type=int, default=520, help="window height in pixels")
    parser.add_argument("--fullscreen", action="store_true", default=bool(config.get("fullscreen", False)), help="use fullscreen window")
    add_configurable_argument(parser, "--block-size", config, "block_size", type=int, default=300, help="option block width in pixels")
    add_configurable_argument(parser, "--spacing", config, "spacing", type=int, default=460, help="center-to-center spacing between option blocks")
    parser.add_argument("--no-server-init", action="store_true", default=bool(config.get("no_server_init", False)), help="do not call /choice/init on startup")
    add_configurable_argument(parser, "--lut-len", config, "lut_len", type=int, default=1000, help="SSVEP LUT length")
    args = parser.parse_args(remaining_argv)
    configure_display(args.display)
    if not args.no_server_init:
        try:
            args.sessionid = resolve_sessionid(args.server, args.sessionid)
        except Exception as exc:
            parser.error(str(exc))
    core, event, visual = setup_psychopy()
    choice_tree = load_choice_tree(args.choice_tree)
    current_node_id = choice_tree["root_node_id"]

    win = visual.Window(
        size=[args.width, args.height],
        fullscr=args.fullscreen,
        color=[-1, -1, -1],
        units="pix",
        allowGUI=True,
    )
    actual_fps = win.getActualFrameRate() or 60

    current_width, current_height = win.size
    targets = [
        make_target(win, index, pos, args.block_size)
        for index, pos in enumerate(make_layout(current_width, current_height, args.block_size, args.spacing))
    ]

    title_text = visual.TextStim(
        win,
        text="LiveTalking SSVEP 选项窗口",
        pos=(0, current_height * 0.4),
        height=30,
        color=[1, 1, 1],
        bold=True,
    )
    status_text = visual.TextStim(
        win,
        text="Q/ESC退出 | R刷新 | 1/2/3模拟选择",
        pos=(0, -current_height * 0.42),
        height=22,
        color=[0.75, 0.85, 1.0],
        wrapWidth=current_width * 0.9,
    )

    frame_cnt = 0
    choices = get_node_choices(choice_tree, current_node_id)
    lut_list = [build_lut(freq, phase, actual_fps, args.lut_len) for freq, phase in zip(DEFAULT_FREQS, DEFAULT_PHASES)]
    last_message = "已加载固定选项树: %s" % choice_tree["tree_id"]

    print("==== LiveTalking SSVEP 选项窗口 ====")
    print("Q/ESC：退出")
    print("R：重置到根节点")
    print("1/2/3：模拟 SSVEP 识别并提交选择")
    print("+/-：调整窗口内目标间距")
    print("[/]：调整目标块大小")
    print("====================================")

    def render_local_choices():
        nonlocal lut_list, last_message
        freqs = []
        phases = []
        for index in range(3):
            choice = choices[index] if index < len(choices) else {}
            freqs.append(get_choice_frequency(choice, index))
            phases.append(get_choice_phase(choice, index))
            targets[index]["label"].text = shorten_text(choice.get("choice_text") if choice else "等待选项")
            targets[index]["freq_label"].text = ("%.1fHz" % freqs[index]).replace(".0Hz", "Hz")
        lut_list = [build_lut(freq, phase, actual_fps, args.lut_len) for freq, phase in zip(freqs, phases)]
        last_message = "节点: %s | 选项数: %d | FPS: %.1f" % (current_node_id or "root", len(choices), actual_fps)

    def reset_to_root(sync_server=True):
        nonlocal choices, current_node_id, last_message
        current_node_id = choice_tree["root_node_id"]
        choices = get_node_choices(choice_tree, current_node_id)
        render_local_choices()
        if sync_server:
            try:
                post_json(
                    args.server,
                    "/choice/reset",
                    {"sessionid": args.sessionid},
                    timeout=3.0,
                )
                last_message = "已重置到根节点并同步后端"
            except Exception as exc:
                last_message = "已本地重置，后端同步失败: %s" % exc

    def select_choice(index):
        nonlocal choices, current_node_id, last_message
        if index >= len(choices):
            last_message = "选项 %d 不存在" % (index + 1)
            return
        choice = choices[index]
        child_node_id = choice.get("child_node_id")
        if child_node_id not in choice_tree["nodes"]:
            last_message = "子节点不存在: %s" % child_node_id
            return
        post_json(
            args.server,
            "/choice/select",
            {
                "sessionid": args.sessionid,
                "choice_id": choice["choice_id"],
                "interrupt": True,
            },
            timeout=3.0,
        )
        current_node_id = child_node_id
        choices = get_node_choices(choice_tree, current_node_id)
        render_local_choices()
        last_message = "已提交选择 %d: %s | 下一个节点: %s" % (
            index + 1,
            choice.get("choice_text", ""),
            current_node_id,
        )

    if not args.no_server_init:
        try:
            post_json(
                args.server,
                "/choice/init",
                {"sessionid": args.sessionid, "tree_id": choice_tree["tree_id"]},
                timeout=3.0,
            )
            last_message = "已初始化后端选项树: %s" % choice_tree["tree_id"]
        except Exception as exc:
            last_message = "本地选项已加载，后端初始化失败: %s" % exc
    render_local_choices()

    while True:
        keys = event.getKeys()
        if "q" in keys or "escape" in keys:
            win.close()
            core.quit()
        if "r" in keys:
            reset_to_root(sync_server=True)
            frame_cnt = 0
        for key, index in (("1", 0), ("num_1", 0), ("2", 1), ("num_2", 1), ("3", 2), ("num_3", 2)):
            if key in keys:
                try:
                    select_choice(index)
                except Exception as exc:
                    last_message = "选择提交失败: %s" % exc
        if "equal" in keys or "plus" in keys:
            args.spacing += 20
        if "minus" in keys:
            args.spacing = max(args.block_size, args.spacing - 20)
        if "bracketright" in keys:
            args.block_size += 20
        if "bracketleft" in keys:
            args.block_size = max(160, args.block_size - 20)

        width, height = win.size
        if width != current_width or height != current_height:
            current_width, current_height = width, height
            title_text.pos = (0, current_height * 0.4)
            status_text.pos = (0, -current_height * 0.42)
            status_text.wrapWidth = current_width * 0.9

        update_target_positions(targets, current_width, current_height, args.block_size, args.spacing)
        title_text.draw()
        draw_targets(targets, lut_list, frame_cnt, len(choices))
        status_text.text = last_message + " | Q/ESC退出 | R重置 | 1/2/3选择 | +/-间距 | [/]大小"
        status_text.draw()
        win.flip()
        frame_cnt += 1


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
