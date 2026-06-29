import math
import os
import sys
import time
import tkinter as tk
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tkinter import colorchooser, ttk


class SSVEPVisualizer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("SSVEP 闪烁可视器")
        self.geometry("1040x680")
        self.minsize(820, 560)

        self.running = False
        self.start_time = 0.0
        self.frequency = tk.DoubleVar(value=10.0)
        self.shape_size = tk.IntVar(value=180)
        self.shape_name = tk.StringVar(value="circle")
        self.original_color = tk.BooleanVar(value=False)
        self.on_color = "#ffffff"
        self.off_color = "#151f2a"
        self.stage_color = "#101820"
        self.after_id = None
        self.is_on = True
        self.selected_count = 0
        self.last_selected_at = ""
        self.selection_flash_until = 0.0

        self._build_ui()
        self._draw_shape(True)

    def _build_ui(self):
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        controls = ttk.Frame(self, padding=18)
        controls.grid(row=0, column=0, sticky="ns")
        controls.columnconfigure(0, weight=1)

        title = ttk.Label(controls, text="SSVEP 闪烁可视器", font=("Microsoft YaHei", 16, "bold"))
        title.grid(row=0, column=0, sticky="w")

        intro = ttk.Label(
            controls,
            text="中心区域显示指定形状，可调节闪烁频率、颜色和大小。",
            wraplength=260,
            foreground="#607086",
        )
        intro.grid(row=1, column=0, sticky="w", pady=(8, 20))

        self.frequency_label = ttk.Label(controls, text="")
        self.frequency_label.grid(row=2, column=0, sticky="w")

        frequency_frame = ttk.Frame(controls)
        frequency_frame.grid(row=3, column=0, sticky="ew", pady=(6, 18))
        frequency_frame.columnconfigure(0, weight=1)

        self.frequency_scale = ttk.Scale(
            frequency_frame,
            from_=1.0,
            to=30.0,
            variable=self.frequency,
            command=self._on_frequency_scale,
        )
        self.frequency_scale.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        frequency_spin = ttk.Spinbox(
            frequency_frame,
            from_=1.0,
            to=30.0,
            increment=0.1,
            textvariable=self.frequency,
            width=7,
            command=self._sync_labels,
        )
        frequency_spin.grid(row=0, column=1)
        frequency_spin.bind("<KeyRelease>", lambda _event: self._sync_labels())

        ttk.Label(controls, text="中心形状").grid(row=4, column=0, sticky="w")
        shape_select = ttk.Combobox(
            controls,
            textvariable=self.shape_name,
            values=("circle", "square", "diamond", "triangle"),
            state="readonly",
        )
        shape_select.grid(row=5, column=0, sticky="ew", pady=(6, 18))
        shape_select.bind("<<ComboboxSelected>>", lambda _event: self._draw_shape(self.is_on))

        self.size_label = ttk.Label(controls, text="")
        self.size_label.grid(row=6, column=0, sticky="w")

        size_frame = ttk.Frame(controls)
        size_frame.grid(row=7, column=0, sticky="ew", pady=(6, 18))
        size_frame.columnconfigure(0, weight=1)

        size_scale = ttk.Scale(
            size_frame,
            from_=80,
            to=300,
            variable=self.shape_size,
            command=self._on_size_scale,
        )
        size_scale.grid(row=0, column=0, sticky="ew", padx=(0, 10))

        size_spin = ttk.Spinbox(
            size_frame,
            from_=80,
            to=300,
            increment=4,
            textvariable=self.shape_size,
            width=7,
            command=self._on_size_change,
        )
        size_spin.grid(row=0, column=1)
        size_spin.bind("<KeyRelease>", lambda _event: self._on_size_change())

        ttk.Button(controls, text="选择亮色", command=self._choose_on_color).grid(
            row=8, column=0, sticky="ew", pady=(0, 10)
        )
        ttk.Button(controls, text="选择暗色", command=self._choose_off_color).grid(
            row=9, column=0, sticky="ew", pady=(0, 10)
        )
        ttk.Button(controls, text="选择背景色", command=self._choose_stage_color).grid(
            row=10, column=0, sticky="ew", pady=(0, 18)
        )

        original_toggle = ttk.Checkbutton(
            controls,
            text="原色低刺激显示",
            variable=self.original_color,
            command=lambda: self._draw_shape(self.is_on),
        )
        original_toggle.grid(row=11, column=0, sticky="w", pady=(0, 6))

        original_hint = ttk.Label(
            controls,
            text="开启后保持亮色原色，只用轻微明暗变化闪烁，肉眼刺激更小。",
            wraplength=260,
            foreground="#607086",
        )
        original_hint.grid(row=12, column=0, sticky="w", pady=(0, 18))

        button_frame = ttk.Frame(controls)
        button_frame.grid(row=13, column=0, sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.toggle_button = ttk.Button(button_frame, text="开始闪烁", command=self._toggle_running)
        self.toggle_button.grid(row=0, column=0, sticky="ew", padx=(0, 8))

        ttk.Button(button_frame, text="恢复默认", command=self._reset_defaults).grid(row=0, column=1, sticky="ew")

        self.status_label = ttk.Label(controls, text="", foreground="#607086")
        self.status_label.grid(row=14, column=0, sticky="w", pady=(18, 0))

        self.selection_label = ttk.Label(controls, text="", foreground="#607086")
        self.selection_label.grid(row=15, column=0, sticky="w", pady=(8, 0))

        self.canvas = tk.Canvas(self, background=self.stage_color, highlightthickness=0)
        self.canvas.grid(row=0, column=1, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._draw_shape(self.is_on))
        self.canvas.bind("<Button-1>", lambda _event: self._select_current_target())
        self.canvas.bind("<space>", lambda _event: self._select_current_target())
        self.canvas.bind("<Return>", lambda _event: self._select_current_target())
        self.canvas.configure(takefocus=True)
        self.canvas.focus_set()

        self._sync_labels()

    def _safe_frequency(self):
        try:
            value = float(self.frequency.get())
        except (TypeError, tk.TclError, ValueError):
            value = 10.0
        return max(1.0, min(30.0, value))

    def _safe_size(self):
        try:
            value = int(float(self.shape_size.get()))
        except (TypeError, tk.TclError, ValueError):
            value = 180
        return max(80, min(300, value))

    def _sync_labels(self):
        freq = self._safe_frequency()
        size = self._safe_size()
        self.frequency.set(round(freq, 1))
        self.shape_size.set(size)
        self.frequency_label.configure(text=f"闪烁频率：{freq:.1f} Hz")
        self.size_label.configure(text=f"形状大小：{size} px")
        mode = "原色低刺激" if self.original_color.get() else "亮暗切换"
        self.status_label.configure(text=f"状态：{'闪烁中' if self.running else '已暂停'}，{freq:.1f} Hz，{mode}")
        selection_text = (
            f"选择：已选择 {self.selected_count} 次，最近 {self.last_selected_at}"
            if self.last_selected_at
            else "选择：尚未选择"
        )
        self.selection_label.configure(text=selection_text)

    def _on_frequency_scale(self, _value):
        self._sync_labels()

    def _on_size_scale(self, _value):
        self._on_size_change()

    def _on_size_change(self):
        self._sync_labels()
        self._draw_shape(self.is_on)

    def _choose_color(self, initial):
        _rgb, value = colorchooser.askcolor(color=initial, parent=self)
        return value

    def _choose_on_color(self):
        value = self._choose_color(self.on_color)
        if value:
            self.on_color = value
            self._draw_shape(self.is_on)

    def _choose_off_color(self):
        value = self._choose_color(self.off_color)
        if value:
            self.off_color = value
            self._draw_shape(self.is_on)

    def _choose_stage_color(self):
        value = self._choose_color(self.stage_color)
        if value:
            self.stage_color = value
            self.canvas.configure(background=value)

    def _toggle_running(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        self.running = True
        self.start_time = time.perf_counter()
        self.toggle_button.configure(text="暂停闪烁")
        self._tick()
        self._sync_labels()

    def _stop(self):
        self.running = False
        if self.after_id is not None:
            self.after_cancel(self.after_id)
            self.after_id = None
        self.is_on = True
        self.toggle_button.configure(text="开始闪烁")
        self._draw_shape(True)
        self._sync_labels()

    def _reset_defaults(self):
        self._stop()
        self.frequency.set(10.0)
        self.shape_size.set(180)
        self.shape_name.set("circle")
        self.original_color.set(False)
        self.on_color = "#ffffff"
        self.off_color = "#151f2a"
        self.stage_color = "#101820"
        self.selected_count = 0
        self.last_selected_at = ""
        self.selection_flash_until = 0.0
        self.canvas.configure(background=self.stage_color)
        self._sync_labels()
        self._draw_shape(True)

    def _select_current_target(self):
        self.selected_count += 1
        self.last_selected_at = time.strftime("%H:%M:%S")
        self.selection_flash_until = time.perf_counter() + 0.42
        self._sync_labels()
        self._draw_shape(self.is_on)
        self.after(430, lambda: self._draw_shape(self.is_on))

    def _tick(self):
        if not self.running:
            return

        elapsed = time.perf_counter() - self.start_time
        wave = math.sin(2.0 * math.pi * self._safe_frequency() * elapsed)
        next_is_on = wave >= 0
        if next_is_on != self.is_on:
            self.is_on = next_is_on
            self._draw_shape(self.is_on)

        self.after_id = self.after(8, self._tick)

    def _draw_shape(self, is_on):
        self.canvas.delete("all")

        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        cx = width / 2
        cy = height / 2
        size = self._safe_size()
        original_mode = self.original_color.get()
        color = self.on_color if (is_on or original_mode) else self.off_color
        stipple = "gray50" if (original_mode and not is_on) else ""
        zone_outline = "#f59e0b" if time.perf_counter() < self.selection_flash_until else "#4b5d70"
        zone_width = 4 if time.perf_counter() < self.selection_flash_until else 2

        zone_size = min(width, height) * 0.55
        self.canvas.create_rectangle(
            cx - zone_size / 2,
            cy - zone_size / 2,
            cx + zone_size / 2,
            cy + zone_size / 2,
            outline=zone_outline,
            dash=(6, 5),
            width=zone_width,
        )
        self.canvas.create_line(cx, 0, cx, height, fill="#2b3a48")
        self.canvas.create_line(0, cy, width, cy, fill="#2b3a48")

        shape = self.shape_name.get()
        half = size / 2
        if shape == "square":
            self.canvas.create_rectangle(cx - half, cy - half, cx + half, cy + half, fill=color, outline="", stipple=stipple)
        elif shape == "diamond":
            self.canvas.create_polygon(cx, cy - half, cx + half, cy, cx, cy + half, cx - half, cy, fill=color, outline="", stipple=stipple)
        elif shape == "triangle":
            self.canvas.create_polygon(cx, cy - half, cx + half, cy + half, cx - half, cy + half, fill=color, outline="", stipple=stipple)
        else:
            self.canvas.create_oval(cx - half, cy - half, cx + half, cy + half, fill=color, outline="", stipple=stipple)

        self.canvas.create_text(
            20,
            height - 24,
            anchor="w",
            text=f"{self._safe_frequency():.1f} Hz",
            fill="#d7deea",
            font=("Segoe UI", 12, "bold"),
        )
        self.canvas.create_text(
            width - 20,
            height - 24,
            anchor="e",
            text="中心刺激区域",
            fill="#d7deea",
            font=("Microsoft YaHei", 12),
        )


WEB_PAGE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>SSVEP 闪烁可视器</title>
    <style>
        :root {
            --shape-color: #ffffff;
            --shape-off: #151f2a;
            --stage-bg: #101820;
            --shape-size: 180px;
        }
        * { box-sizing: border-box; }
        body {
            margin: 0;
            min-height: 100vh;
            color: #142033;
            font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", Arial, sans-serif;
            background: #f4f7fb;
        }
        .app {
            min-height: 100vh;
            display: grid;
            grid-template-columns: 320px minmax(0, 1fr);
        }
        .controls {
            background: #fff;
            border-right: 1px solid #d7deea;
            padding: 22px;
            display: flex;
            flex-direction: column;
            gap: 18px;
        }
        h1 { margin: 0 0 8px; font-size: 1.35rem; }
        p, small, .readout { color: #607086; line-height: 1.5; }
        .field { display: grid; gap: 8px; }
        .field label { font-weight: 700; }
        .row { display: grid; grid-template-columns: 1fr 88px; gap: 10px; align-items: center; }
        input, select, button { font: inherit; }
        input[type="range"] { width: 100%; }
        input[type="number"], input[type="color"], select {
            width: 100%;
            border: 1px solid #d7deea;
            border-radius: 8px;
            background: #fff;
            color: #142033;
            min-height: 42px;
            padding: 8px 10px;
        }
        input[type="color"] { padding: 4px; }
        .button-row { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
        button {
            border: 0;
            border-radius: 8px;
            min-height: 44px;
            padding: 10px 12px;
            cursor: pointer;
            font-weight: 700;
        }
        .primary { background: #0f6cbd; color: #fff; }
        .danger { background: #bf1d1d; color: #fff; }
        .secondary { background: #edf2f8; color: #142033; border: 1px solid #d7deea; }
        .stage-wrap { min-width: 0; padding: 22px; display: grid; place-items: center; }
        .stage {
            width: min(100%, 980px);
            height: min(72vh, 680px);
            min-height: 420px;
            border-radius: 8px;
            background: var(--stage-bg);
            border: 1px solid #263544;
            position: relative;
            overflow: hidden;
            display: grid;
            place-items: center;
        }
        .stage::before, .stage::after { content: ""; position: absolute; pointer-events: none; }
        .stage::before {
            left: 50%; top: 0; width: 1px; height: 100%;
            background: rgba(255, 255, 255, 0.11);
        }
        .stage::after {
            top: 50%; left: 0; width: 100%; height: 1px;
            background: rgba(255, 255, 255, 0.11);
        }
        .center-zone {
            width: clamp(220px, 46vmin, 420px);
            aspect-ratio: 1;
            border: 1px dashed rgba(255, 255, 255, 0.28);
            border-radius: 8px;
            display: grid;
            place-items: center;
        }
        .shape {
            width: var(--shape-size);
            height: var(--shape-size);
            background: var(--shape-color);
            transform-origin: center;
            will-change: background-color;
        }
        .shape.off { background: var(--shape-off); }
        .shape.circle { border-radius: 50%; }
        .shape.square { border-radius: 8px; }
        .shape.diamond { border-radius: 8px; transform: rotate(45deg); }
        .shape.triangle {
            width: 0; height: 0; background: transparent;
            border-left: calc(var(--shape-size) / 2) solid transparent;
            border-right: calc(var(--shape-size) / 2) solid transparent;
            border-bottom: var(--shape-size) solid var(--shape-color);
        }
        .shape.triangle.off { background: transparent; border-bottom-color: var(--shape-off); }
        .status-bar {
            position: absolute;
            left: 16px;
            right: 16px;
            bottom: 14px;
            display: flex;
            justify-content: space-between;
            color: rgba(255, 255, 255, 0.78);
            font-size: 0.86rem;
            pointer-events: none;
        }
        @media (max-width: 820px) {
            .app { grid-template-columns: 1fr; }
            .controls { border-right: 0; border-bottom: 1px solid #d7deea; }
            .stage { height: 60vh; min-height: 360px; }
        }
    </style>
</head>
<body>
    <main class="app">
        <aside class="controls">
            <div>
                <h1>SSVEP 闪烁可视器</h1>
                <p>当前环境没有桌面 DISPLAY，因此自动使用浏览器可视化模式。</p>
            </div>
            <div class="field">
                <label for="frequency">闪烁频率</label>
                <div class="row">
                    <input id="frequency" type="range" min="1" max="30" step="0.1" value="10">
                    <input id="frequency-number" type="number" min="1" max="30" step="0.1" value="10">
                </div>
                <small id="frequency-label">当前频率：10.0 Hz</small>
            </div>
            <div class="field">
                <label for="shape-select">中心形状</label>
                <select id="shape-select">
                    <option value="circle">圆形</option>
                    <option value="square">方形</option>
                    <option value="diamond">菱形</option>
                    <option value="triangle">三角形</option>
                </select>
            </div>
            <div class="field">
                <label for="shape-size">形状大小</label>
                <div class="row">
                    <input id="shape-size" type="range" min="80" max="300" step="4" value="180">
                    <input id="shape-size-number" type="number" min="80" max="300" step="4" value="180">
                </div>
                <small id="size-label">当前大小：180 px</small>
            </div>
            <div class="field"><label for="on-color">亮色</label><input id="on-color" type="color" value="#ffffff"></div>
            <div class="field"><label for="off-color">暗色</label><input id="off-color" type="color" value="#151f2a"></div>
            <div class="field"><label for="stage-color">背景色</label><input id="stage-color" type="color" value="#101820"></div>
            <div class="button-row">
                <button id="toggle" class="primary" type="button">开始闪烁</button>
                <button id="fullscreen" class="secondary" type="button">全屏</button>
            </div>
            <button id="reset" class="secondary" type="button">恢复默认</button>
            <div class="readout" id="readout">状态：已暂停</div>
        </aside>
        <section class="stage-wrap">
            <div class="stage" id="stage">
                <div class="center-zone"><div class="shape circle" id="shape"></div></div>
                <div class="status-bar"><span id="stage-frequency">10.0 Hz</span><span>中心刺激区域</span></div>
            </div>
        </section>
    </main>
    <script>
        var state = { running: false, rafId: null, startTime: 0, frequency: 10 };
        var root = document.documentElement;
        var stage = document.getElementById("stage");
        var shape = document.getElementById("shape");
        var frequency = document.getElementById("frequency");
        var frequencyNumber = document.getElementById("frequency-number");
        var frequencyLabel = document.getElementById("frequency-label");
        var stageFrequency = document.getElementById("stage-frequency");
        var shapeSelect = document.getElementById("shape-select");
        var shapeSize = document.getElementById("shape-size");
        var shapeSizeNumber = document.getElementById("shape-size-number");
        var sizeLabel = document.getElementById("size-label");
        var onColor = document.getElementById("on-color");
        var offColor = document.getElementById("off-color");
        var stageColor = document.getElementById("stage-color");
        var toggle = document.getElementById("toggle");
        var fullscreen = document.getElementById("fullscreen");
        var reset = document.getElementById("reset");
        var readout = document.getElementById("readout");
        function clamp(value, min, max) { return Math.min(max, Math.max(min, value)); }
        function now() { return window.performance && window.performance.now ? window.performance.now() : Date.now(); }
        function formatHz(value) { return Number(value).toFixed(1) + " Hz"; }
        function setFrequency(value) {
            var next = clamp(Number(value) || 1, 1, 30);
            state.frequency = next;
            frequency.value = next;
            frequencyNumber.value = next.toFixed(1);
            frequencyLabel.textContent = "当前频率：" + formatHz(next);
            stageFrequency.textContent = formatHz(next);
            updateReadout();
        }
        function setShapeSize(value) {
            var next = clamp(Number(value) || 180, 80, 300);
            root.style.setProperty("--shape-size", next + "px");
            shapeSize.value = next;
            shapeSizeNumber.value = Math.round(next);
            sizeLabel.textContent = "当前大小：" + Math.round(next) + " px";
        }
        function setShape(name) { shape.className = "shape " + name + (shape.classList.contains("off") ? " off" : ""); }
        function updateColors() {
            root.style.setProperty("--shape-color", onColor.value);
            root.style.setProperty("--shape-off", offColor.value);
            root.style.setProperty("--stage-bg", stageColor.value);
        }
        function updateReadout() { readout.textContent = "状态：" + (state.running ? "闪烁中，" : "已暂停，") + formatHz(state.frequency); }
        function renderFrame(timestamp) {
            var elapsedSeconds = (timestamp - state.startTime) / 1000;
            var wave = Math.sin(2 * Math.PI * state.frequency * elapsedSeconds);
            shape.classList.toggle("off", wave < 0);
            state.rafId = window.requestAnimationFrame(renderFrame);
        }
        function start() {
            if (state.running) return;
            state.running = true;
            state.startTime = now();
            toggle.textContent = "暂停闪烁";
            toggle.className = "danger";
            state.rafId = window.requestAnimationFrame(renderFrame);
            updateReadout();
        }
        function stop() {
            state.running = false;
            if (state.rafId !== null && window.cancelAnimationFrame) window.cancelAnimationFrame(state.rafId);
            state.rafId = null;
            shape.classList.remove("off");
            toggle.textContent = "开始闪烁";
            toggle.className = "primary";
            updateReadout();
        }
        function resetDefaults() {
            stop();
            setFrequency(10);
            shapeSelect.value = "circle";
            setShape("circle");
            setShapeSize(180);
            onColor.value = "#ffffff";
            offColor.value = "#151f2a";
            stageColor.value = "#101820";
            updateColors();
        }
        frequency.addEventListener("input", function() { setFrequency(this.value); });
        frequencyNumber.addEventListener("input", function() { setFrequency(this.value); });
        shapeSelect.addEventListener("change", function() { setShape(this.value); });
        shapeSize.addEventListener("input", function() { setShapeSize(this.value); });
        shapeSizeNumber.addEventListener("input", function() { setShapeSize(this.value); });
        onColor.addEventListener("input", updateColors);
        offColor.addEventListener("input", updateColors);
        stageColor.addEventListener("input", updateColors);
        toggle.addEventListener("click", function() { state.running ? stop() : start(); });
        fullscreen.addEventListener("click", function() { if (stage.requestFullscreen) stage.requestFullscreen(); });
        reset.addEventListener("click", resetDefaults);
        setFrequency(10);
        setShapeSize(180);
        updateColors();
    </script>
</body>
</html>
"""


class VisualizerPageHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path not in ("/", "/index.html"):
            self.send_error(404)
            return
        html_path = Path(__file__).with_name("index.html")
        if html_path.exists():
            payload = html_path.read_bytes()
        else:
            payload = WEB_PAGE.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        sys.stderr.write("[ssvep-web] " + fmt % args + "\n")


def run_web_visualizer(host="127.0.0.1", port=8765):
    server = ThreadingHTTPServer((host, port), VisualizerPageHandler)
    display_host = "127.0.0.1" if host == "0.0.0.0" else host
    print("当前环境没有 DISPLAY，已启动浏览器可视化模式。", flush=True)
    print(f"请在浏览器打开：http://{display_host}:{port}/", flush=True)
    print("按 Ctrl+C 停止服务。", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n已停止 SSVEP 可视化服务。")
    finally:
        server.server_close()


def main():
    parser = argparse.ArgumentParser(description="SSVEP 闪烁可视器")
    parser.add_argument("--web", action="store_true", help="强制使用浏览器可视化模式")
    parser.add_argument("--host", default="127.0.0.1", help="Web 模式监听地址")
    parser.add_argument("--port", default=8765, type=int, help="Web 模式监听端口")
    args = parser.parse_args()

    if args.web or not os.environ.get("DISPLAY"):
        run_web_visualizer(args.host, args.port)
        return

    try:
        app = SSVEPVisualizer()
    except tk.TclError as exc:
        if "display" in str(exc).lower():
            run_web_visualizer(args.host, args.port)
            return
        raise
    app.mainloop()


if __name__ == "__main__":
    main()
