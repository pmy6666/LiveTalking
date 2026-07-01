#!/usr/bin/env python3
# cd C:\Users\1234\Desktop\PMY\LiveTalking & "F:\miniforge3\envs\LiveTalking\python.exe" scripts\ssvep\choice_ssvep_window.py --server http://127.0.0.1:8010 --sessionid auto
import argparse
import importlib.util
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
AUTO_MODES = {"off", "sim", "udp"}


def setup_psychopy():
    import pyglet

    pyglet.options["shadow_window"] = False
    try:
        temp_win = pyglet.window.Window(width=1, height=1, visible=False)
    except Exception as exc:
        display = os.environ.get("DISPLAY")
        raise RuntimeError(
            "cannot connect to graphical display DISPLAY=%r. "
            "Run from a desktop session, or set display: \":0\" in scripts/ssvep/config.yaml."
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


def wait_for_sessionid(base_url, configured_sessionid, timeout_seconds=120.0, poll_interval=1.0):
    sessionid = str(configured_sessionid or "").strip()
    if sessionid and sessionid not in {"auto", "latest", "你的sessionid"}:
        return sessionid

    timeout_seconds = max(1.0, float(timeout_seconds or 120.0))
    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        try:
            return resolve_sessionid(base_url, configured_sessionid)
        except Exception as exc:
            last_error = exc
            print(
                "waiting for active LiveTalking session; open dashboard and click start connection...",
                flush=True,
            )
            time.sleep(max(0.2, float(poll_interval or 1.0)))

    raise RuntimeError(
        "no active LiveTalking session found within %.0fs; please start the web connection first. last error: %s"
        % (timeout_seconds, last_error)
    )


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


def choices_signature(node_id, choices):
    return json.dumps(
        {
            "node_id": node_id,
            "choices": [
                {
                    "choice_id": choice.get("choice_id"),
                    "choice_text": choice.get("choice_text"),
                    "frequency": (choice.get("ssvep") or {}).get("frequency") or choice.get("ssvep_frequency"),
                    "phase": (choice.get("ssvep") or {}).get("phase") or choice.get("ssvep_phase"),
                }
                for choice in choices
            ],
        },
        ensure_ascii=False,
        sort_keys=True,
    )


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


def print_choice_log(index, choice, node_id, source="choice"):
    choice_text = str(choice.get("choice_text") or "").strip() or "(empty)"
    choice_id = str(choice.get("choice_id") or "").strip() or "(no choice_id)"
    child_node_id = str(choice.get("child_node_id") or "").strip()
    child_text = " child_node=%s" % child_node_id if child_node_id else ""
    print(
        "[SSVEP choice] source=%s selected=%d text=%s choice_id=%s node=%s%s"
        % (source, index + 1, choice_text, choice_id, node_id or "root", child_text),
        flush=True,
    )


class AutoSSVEPController:
    def __init__(self, args):
        mode = str(getattr(args, "eeg_mode", "off") or "off").strip().lower()
        if mode not in AUTO_MODES:
            raise ValueError("--eeg-mode must be one of: %s" % ", ".join(sorted(AUTO_MODES)))
        self.args = args
        self.mode = mode
        self.receiver = None
        self.classifier = None
        self.smoother = None
        self.last_sim_at = 0.0
        self.next_sim_target = 1
        self.ignore_until = 0.0
        self.last_status = "auto SSVEP: off"

        if self.mode == "udp":
            from eeg_udp_receiver import EEGUDPReceiver
            from ssvep_classifier import DecisionSmoother, FFTSSVEPClassifier

            self.receiver = EEGUDPReceiver(
                host=args.eeg_udp_host,
                port=args.eeg_udp_port,
                expected_channels=args.eeg_channels,
                expected_samples=args.eeg_samples,
            )
            self.classifier = FFTSSVEPClassifier(sample_rate=args.eeg_sample_rate)
            self.smoother = DecisionSmoother(
                decision_windows=args.decision_windows,
                min_votes=args.min_votes,
                confidence_threshold=args.confidence_threshold,
                submit_cooldown_sec=args.submit_cooldown_sec,
            )
            self.last_status = "auto SSVEP: udp waiting on %s:%d" % (args.eeg_udp_host, args.eeg_udp_port)
        elif self.mode == "sim":
            self.last_status = "auto SSVEP: simulator ready"

    def start(self):
        if self.receiver:
            self.receiver.start()

    def stop(self):
        if self.receiver:
            self.receiver.stop()

    def notify_state_change(self):
        self.ignore_until = time.monotonic() + max(0.0, float(self.args.ignore_after_state_change_sec or 0.0))
        if self.smoother:
            self.smoother.reset()

    def poll(self, choices):
        if self.mode == "off":
            return None
        if time.monotonic() < self.ignore_until:
            return None
        if not choices:
            self.last_status = "auto SSVEP: no available choices"
            return None
        if self.mode == "sim":
            return self._poll_sim(choices)
        return self._poll_udp(choices)

    def _poll_sim(self, choices):
        now = time.monotonic()
        interval = max(0.1, float(self.args.sim_target_interval_sec or 3.0))
        if now - self.last_sim_at < interval:
            return None
        self.last_sim_at = now
        target = min(self.next_sim_target, len(choices))
        self.next_sim_target = 1 if self.next_sim_target >= 3 else self.next_sim_target + 1
        self.last_status = "auto SSVEP sim target=%d" % target
        return target

    def _poll_udp(self, choices):
        packet = self.receiver.get_latest() if self.receiver else None
        if not packet:
            if self.receiver and self.receiver.last_error:
                self.last_status = "auto SSVEP UDP warning: %s" % self.receiver.last_error
                self.receiver.last_error = ""
            return None

        freqs = [get_choice_frequency(choices[index], index) for index in range(min(3, len(choices)))]
        result = self.classifier.predict(packet.eeg, freqs)
        scores_text = ",".join("%.2g" % score for score in result.scores)
        self.last_status = (
            "auto SSVEP packet=%d target=%s conf=%.3f scores=[%s]"
            % (packet.packet_id, result.target_index or "-", result.confidence, scores_text)
        )
        target = self.smoother.update(result) if self.smoother else None
        if target is None or target > len(choices):
            return None
        return target


class SelectionPauseGate:
    def __init__(self, args):
        self.args = args
        self.pause_sec = max(0.0, float(getattr(args, "selection_pause_sec", 10.0) or 10.0))
        self.locked_until = 0.0

    def locked(self):
        return time.monotonic() < self.locked_until

    def pause(self):
        if self.pause_sec > 0:
            self.locked_until = time.monotonic() + self.pause_sec

    def remaining(self):
        return max(0.0, self.locked_until - time.monotonic())

    def status_text(self):
        if self.locked():
            return "selection paused %.1fs for avatar speech" % self.remaining()
        return ""


def shorten_text(text, max_chars=18):
    text = str(text or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1] + "..."


def actual_target_spacing(width, block_size, spacing):
    usable_width = width - block_size
    max_spacing = max(0, usable_width / 2)
    return min(spacing, max_spacing)


def target_text_width(width, block_size, spacing):
    actual_spacing = actual_target_spacing(width, block_size, spacing)
    by_spacing = actual_spacing - 48 if actual_spacing > 0 else block_size
    by_window = (width - 48) / 3
    return int(max(80, min(block_size * 0.95, by_spacing, by_window)))


def shorten_choice_text(text, text_width, font_size):
    chars_per_line = max(4, int(text_width / max(8, font_size * 0.7)))
    return shorten_text(text, max_chars=max(8, chars_per_line * 2))


def display_choice_text(choice, max_chars=48):
    return shorten_text(choice.get("choice_text") or "", max_chars=max_chars) or "已选择当前选项"


def make_layout(width, height, block_size, spacing):
    actual_spacing = actual_target_spacing(width, block_size, spacing)
    rect_h = block_size * 0.58
    top_margin = max(12, height * 0.06)
    bottom_text_space = max(78, block_size * 0.31)
    y = min(height / 2 - top_margin - rect_h / 2, bottom_text_space)
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
        pos=(pos[0], pos[1] - block_size * 0.42),
        height=max(18, block_size * 0.065),
        color=[1, 1, 1],
        bold=True,
        wrapWidth=block_size,
    )
    freq_label = visual.TextStim(
        win,
        text="",
        pos=(pos[0], pos[1] - block_size * 0.56),
        height=max(14, block_size * 0.045),
        color=[0.85, 0.96, 1.0],
        wrapWidth=block_size,
    )
    return {
        "block": block,
        "number": number,
        "label": label,
        "freq_label": freq_label,
    }


def update_target_positions(targets, width, height, block_size, spacing):
    text_width = target_text_width(width, block_size, spacing)
    for target, pos in zip(targets, make_layout(width, height, block_size, spacing)):
        target["block"].pos = pos
        target["block"].width = block_size
        target["block"].height = block_size * 0.58
        target["number"].pos = (pos[0], pos[1] + block_size * 0.15)
        target["number"].height = max(32, block_size * 0.18)
        target["label"].pos = (pos[0], pos[1] - block_size * 0.42)
        target["label"].height = max(18, block_size * 0.065)
        target["label"].wrapWidth = text_width
        target["freq_label"].pos = (pos[0], pos[1] - block_size * 0.56)
        target["freq_label"].height = max(14, block_size * 0.045)
        target["freq_label"].wrapWidth = text_width


def parse_args():
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
    add_configurable_argument(parser, "--poll-interval", config, "poll_interval", type=float, default=0.5, help="seconds between /choice/state sync checks")
    add_configurable_argument(parser, "--wait-session-timeout", config, "wait_session_timeout", type=float, default=120.0, help="seconds to wait for --sessionid auto")
    add_configurable_argument(parser, "--wait-session-interval", config, "wait_session_interval", type=float, default=1.0, help="seconds between active session discovery checks")
    add_configurable_argument(parser, "--eeg-mode", config, "eeg_mode", default="off", choices=sorted(AUTO_MODES), help="auto selection mode: off, sim, or udp")
    add_configurable_argument(parser, "--sim-target-interval-sec", config, "sim_target_interval_sec", type=float, default=3.0, help="seconds between simulated target outputs")
    add_configurable_argument(parser, "--eeg-udp-host", config, "eeg_udp_host", default="0.0.0.0", help="EEG UDP bind host")
    add_configurable_argument(parser, "--eeg-udp-port", config, "eeg_udp_port", type=int, default=5555, help="EEG UDP bind port")
    add_configurable_argument(parser, "--eeg-channels", config, "eeg_channels", type=int, default=21, help="expected EEG channel count")
    add_configurable_argument(parser, "--eeg-samples", config, "eeg_samples", type=int, default=750, help="expected EEG sample count per packet")
    add_configurable_argument(parser, "--eeg-sample-rate", config, "eeg_sample_rate", type=float, default=300.0, help="EEG sample rate")
    add_configurable_argument(parser, "--decision-windows", config, "decision_windows", type=int, default=3, help="number of recent classifier windows to vote over")
    add_configurable_argument(parser, "--min-votes", config, "min_votes", type=int, default=2, help="minimum votes required to submit")
    add_configurable_argument(parser, "--confidence-threshold", config, "confidence_threshold", type=float, default=0.2, help="minimum classifier confidence")
    add_configurable_argument(parser, "--submit-cooldown-sec", config, "submit_cooldown_sec", type=float, default=2.0, help="seconds between automatic submissions")
    add_configurable_argument(parser, "--ignore-after-state-change-sec", config, "ignore_after_state_change_sec", type=float, default=0.5, help="seconds to ignore classifier output after choices change")
    add_configurable_argument(parser, "--selection-pause-sec", config, "selection_pause_sec", type=float, default=10.0, help="seconds to pause SSVEP selection after a successful choice submit")
    args = parser.parse_args(remaining_argv)
    return parser, args, project_root


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


def run_pyglet_main(args):
    import pyglet
    from pyglet import shapes

    auto_ssvep = AutoSSVEPController(args)
    selection_pause = SelectionPauseGate(args)
    choice_tree = load_choice_tree(args.choice_tree)
    current_node_id = choice_tree["root_node_id"]
    choices = get_node_choices(choice_tree, current_node_id)
    actual_fps = 60.0
    frame_cnt = 0
    last_state_signature = choices_signature(current_node_id, choices)
    last_message = "node: %s | choices: %d" % (current_node_id, len(choices))

    config = pyglet.gl.Config(double_buffer=True)
    window = pyglet.window.Window(
        width=args.width,
        height=args.height,
        caption="LiveTalking SSVEP Choice Window",
        fullscreen=bool(args.fullscreen),
        config=config,
    )

    colors = [
        (0, 90, 255),
        (0, 255, 115),
        (166, 51, 255),
    ]
    batch = pyglet.graphics.Batch()
    status_label = pyglet.text.Label(
        "",
        font_size=12,
        color=(190, 215, 255, 255),
        anchor_x="center",
        anchor_y="center",
        multiline=True,
        width=max(200, int(args.width * 0.9)),
        batch=batch,
    )
    waiting_title = pyglet.text.Label(
        "已选择",
        font_size=24,
        color=(210, 235, 255, 255),
        anchor_x="center",
        anchor_y="center",
    )
    waiting_choice = pyglet.text.Label(
        "",
        font_size=28,
        color=(255, 255, 255, 255),
        anchor_x="center",
        anchor_y="center",
        multiline=True,
        width=max(260, int(args.width * 0.72)),
    )
    waiting_status = pyglet.text.Label(
        "数字人回答中...",
        font_size=16,
        color=(180, 210, 230, 255),
        anchor_x="center",
        anchor_y="center",
    )
    waiting_view = {"active": False, "choice_text": ""}

    target_items = []
    for index in range(3):
        rect = shapes.Rectangle(0, 0, args.block_size, int(args.block_size * 0.58), color=(20, 20, 20), batch=batch)
        border = shapes.BorderedRectangle(
            0,
            0,
            args.block_size,
            int(args.block_size * 0.58),
            border=6,
            color=(20, 20, 20),
            border_color=colors[index],
            batch=batch,
        )
        number = pyglet.text.Label(
            str(index + 1),
            font_size=max(20, int(args.block_size * 0.11)),
            color=(255, 255, 255, 255),
            anchor_x="center",
            anchor_y="center",
            batch=batch,
        )
        label = pyglet.text.Label(
            "等待选项",
            font_size=max(16, int(args.block_size * 0.065)),
            color=(255, 255, 255, 255),
            anchor_x="center",
            anchor_y="top",
            multiline=True,
            width=args.block_size,
            batch=batch,
        )
        freq_label = pyglet.text.Label(
            "",
            font_size=max(12, int(args.block_size * 0.045)),
            color=(215, 245, 255, 255),
            anchor_x="center",
            anchor_y="top",
            batch=batch,
        )
        target_items.append(
            {
                "rect": rect,
                "border": border,
                "number": number,
                "label": label,
                "freq_label": freq_label,
            }
        )

    def format_freq(value):
        return ("%.1fHz" % value).replace(".0Hz", "Hz")

    def build_current_luts():
        freqs = []
        phases = []
        text_width = target_text_width(window.width, args.block_size, args.spacing)
        label_font_size = max(16, int(args.block_size * 0.065))
        for index in range(3):
            choice = choices[index] if index < len(choices) else {}
            freqs.append(get_choice_frequency(choice, index))
            phases.append(get_choice_phase(choice, index))
            target_items[index]["label"].text = shorten_choice_text(
                choice.get("choice_text") if choice else "等待选项",
                text_width,
                label_font_size,
            )
            target_items[index]["freq_label"].text = format_freq(freqs[index])
        return [build_lut(freq, phase, actual_fps, args.lut_len) for freq, phase in zip(freqs, phases)]

    lut_list = build_current_luts()

    def apply_server_state(payload, status_prefix="server sync"):
        nonlocal choices, current_node_id, lut_list, last_state_signature, last_message
        current = payload.get("current") or {}
        server_choices = (current.get("choices") or [])[:3]
        server_node_id = current.get("node_id") or current_node_id
        signature = choices_signature(server_node_id, server_choices)
        if signature == last_state_signature:
            return False
        current_node_id = server_node_id
        choices = server_choices
        lut_list = build_current_luts()
        last_state_signature = signature
        last_message = "node: %s | choices: %d" % (current_node_id or "root", len(choices))
        auto_ssvep.notify_state_change()
        return True

    def refresh_from_server(status_prefix="server sync"):
        payload = post_json(args.server, "/choice/state", {"sessionid": args.sessionid}, timeout=1.5)
        if not payload.get("initialized"):
            return False
        return apply_server_state(payload, status_prefix=status_prefix)

    def reset_to_root():
        nonlocal choices, current_node_id, lut_list, last_message
        waiting_view["active"] = False
        waiting_view["choice_text"] = ""
        current_node_id = choice_tree["root_node_id"]
        choices = get_node_choices(choice_tree, current_node_id)
        lut_list = build_current_luts()
        auto_ssvep.notify_state_change()
        if args.no_server_init:
            last_message = "node: %s | choices: %d" % (current_node_id, len(choices))
            return
        try:
            payload = post_json(args.server, "/choice/reset", {"sessionid": args.sessionid}, timeout=3.0)
            apply_server_state(payload, status_prefix="reset")
        except Exception as exc:
            last_message = "reset failed: %s" % exc

    def select_choice(index):
        nonlocal last_message
        if selection_pause.locked():
            last_message = "selection ignored: waiting %.1fs" % selection_pause.remaining()
            auto_ssvep.notify_state_change()
            return
        if index >= len(choices):
            last_message = "choice %d is not available" % (index + 1)
            return
        choice = choices[index]
        print_choice_log(index, choice, current_node_id, source="submit")
        if args.no_server_init:
            current_child = choice.get("child_node_id")
            if current_child in choice_tree["nodes"]:
                nonlocal_choices_update(current_child)
                selection_pause.pause()
                waiting_view["active"] = True
                waiting_view["choice_text"] = display_choice_text(choice)
            return
        try:
            payload = post_json(
                args.server,
                "/choice/select",
                {
                    "sessionid": args.sessionid,
                    "choice_id": choice["choice_id"],
                    "interrupt": True,
                },
                timeout=3.0,
            )
            apply_server_state(payload, status_prefix="selected %d" % (index + 1))
            selection_pause.pause()
            waiting_view["active"] = True
            waiting_view["choice_text"] = display_choice_text(choice)
            last_message = "selected %d | node: %s" % (index + 1, current_node_id)
        except Exception as exc:
            last_message = "select failed: %s" % exc

    def nonlocal_choices_update(node_id):
        nonlocal choices, current_node_id, lut_list, last_message, last_state_signature
        current_node_id = node_id
        choices = get_node_choices(choice_tree, current_node_id)
        lut_list = build_current_luts()
        last_state_signature = choices_signature(current_node_id, choices)
        auto_ssvep.notify_state_change()
        last_message = "node: %s | choices: %d" % (current_node_id, len(choices))

    if not args.no_server_init:
        try:
            payload = post_json(
                args.server,
                "/choice/init",
                {"sessionid": args.sessionid, "tree_id": choice_tree["tree_id"]},
                timeout=3.0,
            )
            apply_server_state(payload, status_prefix="initialized")
            last_message = "node: %s | choices: %d" % (current_node_id, len(choices))
        except Exception as exc:
            last_message = "server init failed, using local tree: %s" % exc

    def update_layout():
        width, height = window.get_size()
        block_size = args.block_size
        positions = make_layout(width, height, block_size, args.spacing)
        text_width = target_text_width(width, block_size, args.spacing)
        status_label.x = width // 2
        status_label.y = int(height * 0.035)
        status_label.width = max(200, int(width * 0.9))
        waiting_title.x = width // 2
        waiting_title.y = int(height * 0.70)
        waiting_choice.x = width // 2
        waiting_choice.y = int(height * 0.50)
        waiting_choice.width = max(260, int(width * 0.72))
        waiting_status.x = width // 2
        waiting_status.y = int(height * 0.28)
        for item, pos in zip(target_items, positions):
            center_x = width / 2 + pos[0]
            center_y = height / 2 + pos[1]
            rect_w = block_size
            rect_h = int(block_size * 0.58)
            left = center_x - rect_w / 2
            bottom = center_y - rect_h / 2
            item["rect"].position = (left, bottom)
            item["rect"].width = rect_w
            item["rect"].height = rect_h
            item["border"].position = (left, bottom)
            item["border"].width = rect_w
            item["border"].height = rect_h
            item["number"].x = center_x
            item["number"].y = center_y + block_size * 0.15
            item["label"].x = center_x
            item["label"].y = center_y - block_size * 0.34
            item["label"].width = text_width
            item["freq_label"].x = center_x
            item["freq_label"].y = center_y - block_size * 0.54

    @window.event
    def on_key_press(symbol, modifiers):
        nonlocal last_message
        if symbol in {pyglet.window.key.Q, pyglet.window.key.ESCAPE}:
            window.close()
        elif symbol == pyglet.window.key.R:
            reset_to_root()
        elif symbol in {pyglet.window.key._1, pyglet.window.key.NUM_1}:
            select_choice(0)
        elif symbol in {pyglet.window.key._2, pyglet.window.key.NUM_2}:
            select_choice(1)
        elif symbol in {pyglet.window.key._3, pyglet.window.key.NUM_3}:
            select_choice(2)
        elif symbol in {pyglet.window.key.PLUS, pyglet.window.key.EQUAL}:
            args.spacing += 20
            update_layout()
        elif symbol == pyglet.window.key.MINUS:
            args.spacing = max(args.block_size, args.spacing - 20)
            update_layout()
        elif symbol == pyglet.window.key.BRACKETRIGHT:
            args.block_size += 20
            update_layout()
        elif symbol == pyglet.window.key.BRACKETLEFT:
            args.block_size = max(160, args.block_size - 20)
            update_layout()
        else:
            last_message = "Q/Esc exit | R reset | 1/2/3 select"

    @window.event
    def on_draw():
        window.clear()
        update_layout()
        if waiting_view["active"] and selection_pause.locked():
            waiting_title.draw()
            waiting_choice.draw()
            waiting_status.draw()
        else:
            batch.draw()

    @window.event
    def on_close():
        auto_ssvep.stop()

    def tick(dt):
        nonlocal frame_cnt, last_message
        if waiting_view["active"] and not selection_pause.locked():
            waiting_view["active"] = False
        if not args.no_server_init and frame_cnt % max(1, int(actual_fps * args.poll_interval)) == 0:
            try:
                refresh_from_server(status_prefix="server sync")
            except Exception as exc:
                last_message = "state sync failed: %s" % exc
        for index, item in enumerate(target_items):
            if index >= len(choices):
                value = 20
            else:
                intensity = lut_list[index][frame_cnt % len(lut_list[index])]
                value = int(max(0, min(255, round(intensity * 255))))
            item["rect"].color = (value, value, value)
            item["border"].color = (value, value, value)
        gate_status = selection_pause.status_text()
        if not selection_pause.locked():
            target_index = auto_ssvep.poll(choices)
            if target_index is not None:
                select_choice(target_index - 1)
            elif args.eeg_mode != "off" and auto_ssvep.last_status:
                last_message = "%s | %s" % (last_message.split(" | auto SSVEP", 1)[0], auto_ssvep.last_status)
        if gate_status:
            last_message = "%s | %s" % (last_message.split(" | selection paused", 1)[0], gate_status)
        waiting_choice.text = waiting_view["choice_text"]
        waiting_status.text = "数字人回答中，请稍候 %.1fs" % selection_pause.remaining()
        status_label.text = last_message
        frame_cnt += 1

    print("==== LiveTalking SSVEP pyglet window ====")
    print("sessionid:", args.sessionid)
    print("auto SSVEP mode:", args.eeg_mode)
    print("Q/ESC: exit | R: reset | 1/2/3: select")
    print("=========================================")
    update_layout()
    auto_ssvep.start()
    pyglet.clock.schedule_interval(tick, 1 / actual_fps)
    pyglet.app.run()


def main():
    parser, args, _project_root = parse_args()
    configure_display(args.display)
    if not args.no_server_init:
        try:
            args.sessionid = wait_for_sessionid(
                args.server,
                args.sessionid,
                timeout_seconds=args.wait_session_timeout,
                poll_interval=args.wait_session_interval,
            )
        except Exception as exc:
            parser.error(str(exc))
    if importlib.util.find_spec("psychopy") is None:
        print("PsychoPy is not installed; using pyglet fallback renderer.", flush=True)
        return run_pyglet_main(args)
    core, event, visual = setup_psychopy()
    auto_ssvep = AutoSSVEPController(args)
    selection_pause = SelectionPauseGate(args)
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

    status_text = visual.TextStim(
        win,
        text="Q/ESC 退出 | R 刷新 | 1/2/3 模拟选择",
        pos=(0, -current_height * 0.48),
        height=22,
        color=[0.75, 0.85, 1.0],
        wrapWidth=current_width * 0.9,
    )
    waiting_title = visual.TextStim(
        win,
        text="已选择",
        pos=(0, current_height * 0.22),
        height=30,
        color=[0.72, 0.88, 1.0],
        bold=True,
    )
    waiting_choice = visual.TextStim(
        win,
        text="",
        pos=(0, current_height * 0.02),
        height=34,
        color=[1, 1, 1],
        bold=True,
        wrapWidth=current_width * 0.76,
    )
    waiting_status = visual.TextStim(
        win,
        text="数字人回答中...",
        pos=(0, -current_height * 0.24),
        height=20,
        color=[0.72, 0.82, 0.9],
        wrapWidth=current_width * 0.86,
    )
    waiting_view = {"active": False, "choice_text": ""}

    frame_cnt = 0
    choices = get_node_choices(choice_tree, current_node_id)
    lut_list = [build_lut(freq, phase, actual_fps, args.lut_len) for freq, phase in zip(DEFAULT_FREQS, DEFAULT_PHASES)]
    last_message = "节点:%s 选项:%d" % (current_node_id, len(choices))
    last_state_signature = choices_signature(current_node_id, choices)
    next_poll_at = 0.0

    print("==== LiveTalking SSVEP choice window ====")
    print("auto SSVEP mode:", args.eeg_mode)
    print("Q/ESC: exit")
    print("R: reset to root")
    print("1/2/3: submit manual choice")
    print("+/-: adjust target spacing")
    print("[/]: adjust target block size")
    print("====================================")

    def render_current_choices(status_prefix=None):
        nonlocal lut_list, last_message, last_state_signature
        freqs = []
        phases = []
        text_width = target_text_width(current_width, args.block_size, args.spacing)
        label_font_size = max(18, args.block_size * 0.065)
        for index in range(3):
            choice = choices[index] if index < len(choices) else {}
            freqs.append(get_choice_frequency(choice, index))
            phases.append(get_choice_phase(choice, index))
            targets[index]["label"].text = shorten_choice_text(
                choice.get("choice_text") if choice else "等待选项",
                text_width,
                label_font_size,
            )
            targets[index]["freq_label"].text = ("%.1fHz" % freqs[index]).replace(".0Hz", "Hz")
        lut_list = [build_lut(freq, phase, actual_fps, args.lut_len) for freq, phase in zip(freqs, phases)]
        last_state_signature = choices_signature(current_node_id, choices)
        auto_ssvep.notify_state_change()
        last_message = "节点:%s 选项:%d" % (current_node_id or "root", len(choices))

    def apply_server_state(payload, status_prefix="后端同步"):
        nonlocal choices, current_node_id
        current = payload.get("current") or {}
        server_choices = (current.get("choices") or [])[:3]
        server_node_id = current.get("node_id") or current_node_id
        signature = choices_signature(server_node_id, server_choices)
        if signature == last_state_signature:
            return False
        current_node_id = server_node_id
        choices = server_choices
        render_current_choices(status_prefix=status_prefix)
        return True

    def refresh_from_server(status_prefix="后端同步"):
        payload = post_json(
            args.server,
            "/choice/state",
            {"sessionid": args.sessionid},
            timeout=1.5,
        )
        if not payload.get("initialized"):
            return False
        return apply_server_state(payload, status_prefix=status_prefix)

    def reset_to_root(sync_server=True):
        nonlocal choices, current_node_id, last_message
        waiting_view["active"] = False
        waiting_view["choice_text"] = ""
        current_node_id = choice_tree["root_node_id"]
        choices = get_node_choices(choice_tree, current_node_id)
        render_current_choices(status_prefix="local reset")
        if sync_server and not args.no_server_init:
            try:
                payload = post_json(
                    args.server,
                    "/choice/reset",
                    {"sessionid": args.sessionid},
                    timeout=3.0,
                )
                apply_server_state(payload, status_prefix="reset")
                last_message = "节点:%s 选项:%d" % (current_node_id, len(choices))
            except Exception as exc:
                last_message = "reset failed: %s" % exc

    def select_choice(index):
        nonlocal choices, current_node_id, last_message
        if selection_pause.locked():
            last_message = "selection ignored: waiting %.1fs" % selection_pause.remaining()
            auto_ssvep.notify_state_change()
            return
        if index >= len(choices):
            last_message = "choice %d is not available" % (index + 1)
            return
        choice = choices[index]
        print_choice_log(index, choice, current_node_id, source="submit")
        if args.no_server_init:
            current_child = choice.get("child_node_id")
            if current_child in choice_tree["nodes"]:
                current_node_id = current_child
                choices = get_node_choices(choice_tree, current_node_id)
                render_current_choices(status_prefix="local selected")
                selection_pause.pause()
                waiting_view["active"] = True
                waiting_view["choice_text"] = display_choice_text(choice)
                last_message = "selected:%d node:%s" % (index + 1, current_node_id)
            return
        payload = post_json(
            args.server,
            "/choice/select",
            {
                "sessionid": args.sessionid,
                "choice_id": choice["choice_id"],
                "interrupt": True,
            },
            timeout=3.0,
        )
        apply_server_state(payload, status_prefix="selected")
        selection_pause.pause()
        waiting_view["active"] = True
        waiting_view["choice_text"] = display_choice_text(choice)
        last_message = "selected:%d node:%s" % (index + 1, current_node_id)

    if not args.no_server_init:
        try:
            payload = post_json(
                args.server,
                "/choice/init",
                {"sessionid": args.sessionid, "tree_id": choice_tree["tree_id"]},
                timeout=3.0,
            )
            apply_server_state(payload, status_prefix="initialized")
            last_message = "节点:%s 选项:%d" % (current_node_id, len(choices))
        except Exception as exc:
            last_message = "server init failed, using local tree: %s" % exc
    render_current_choices(status_prefix="current")
    auto_ssvep.start()

    try:
        while True:
            now = time.monotonic()
            if waiting_view["active"] and not selection_pause.locked():
                waiting_view["active"] = False
            if not args.no_server_init and now >= next_poll_at:
                next_poll_at = now + max(0.1, float(args.poll_interval or 0.5))
                try:
                    refresh_from_server(status_prefix="server sync")
                except Exception as exc:
                    last_message = "state sync failed: %s" % exc
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
                        last_message = "select failed: %s" % exc
            gate_status = selection_pause.status_text()
            if not selection_pause.locked():
                auto_target_index = auto_ssvep.poll(choices)
                if auto_target_index is not None:
                    try:
                        select_choice(auto_target_index - 1)
                    except Exception as exc:
                        last_message = "auto SSVEP select failed: %s" % exc
                elif args.eeg_mode != "off" and auto_ssvep.last_status:
                    last_message = "%s | %s" % (last_message.split(" | auto SSVEP", 1)[0], auto_ssvep.last_status)
            if gate_status:
                last_message = "%s | %s" % (last_message.split(" | selection paused", 1)[0], gate_status)
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
                status_text.pos = (0, -current_height * 0.48)
                status_text.wrapWidth = current_width * 0.9
                waiting_title.pos = (0, current_height * 0.22)
                waiting_choice.pos = (0, current_height * 0.02)
                waiting_choice.wrapWidth = current_width * 0.76
                waiting_status.pos = (0, -current_height * 0.24)
                waiting_status.wrapWidth = current_width * 0.86
    
            update_target_positions(targets, current_width, current_height, args.block_size, args.spacing)
            if waiting_view["active"] and selection_pause.locked():
                waiting_choice.text = waiting_view["choice_text"]
                waiting_status.text = "数字人回答中，请稍候 %.1fs" % selection_pause.remaining()
                waiting_title.draw()
                waiting_choice.draw()
                waiting_status.draw()
            else:
                draw_targets(targets, lut_list, frame_cnt, len(choices))
                status_text.text = last_message
                status_text.draw()
            win.flip()
            frame_cnt += 1
    finally:
        auto_ssvep.stop()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(0)
