# 前端 SSVEP 闪烁方案调整

## 结论

前端布局继续沿用当前“每个对话选项后面带一个闪烁目标”的方式，不再改成独立 3x3 九宫格刺激屏。

需要替换的是闪烁方法：当前前端使用 `requestAnimationFrame + sin > 0 + class 切换`，属于二值开关闪烁；建议改成参考 `ssvep_flicker.py` 的方式：

- 每个选项后面放固定尺寸灰度刺激块
- 根据显示刷新率生成 LUT
- 每一帧按 LUT 更新刺激块亮度
- 不使用 CSS transition，不使用 class 二值切换
- 保留当前普通按钮布局和点击选择逻辑

这样既保留现有选项对话 UI，又让闪烁波形更接近 `ssvep_flicker.py`。

## 当前问题

当前 `web/client.js` 中的核心逻辑是：

```js
var wave = Math.sin(2 * Math.PI * target.frequency * elapsedSeconds + target.phase);
var isOn = wave >= 0;
$(target.element)
  .toggleClass("ssvep-on", isOn)
  .toggleClass("ssvep-off", !isOn);
```

这个方案的问题：

1. 闪烁是二值亮/灭，不是 `ssvep_flicker.py` 中的连续正弦亮度。
2. `elapsedSeconds` 基于时间戳，视觉呈现仍受浏览器调度、CSS 样式和 transition 影响。
3. 当前 `.choice-btn` 有 `background-color 0.06s linear` transition，会平滑掉闪烁边界。
4. 整个按钮变色会影响文字阅读，且刺激区域大小会随选项文本布局变化。

## 保留的布局

保留现在的选项列表：

```text
选项 1 文本        [闪烁块 1]
选项 2 文本        [闪烁块 2]
选项 3 文本        [闪烁块 3]
```

建议不要让整个 `.choice-btn` 闪烁，而是在每个按钮右侧固定一个 `.ssvep-stimulus`。

按钮仍然负责：

- 显示选项文本
- 鼠标点击选择
- 显示频率标签
- 选中/禁用状态

刺激块负责：

- 灰度亮度正弦变化
- SSVEP 视觉刺激
- 固定尺寸、固定位置、固定频率

## 推荐频率

如果当前每轮固定 3 个选项，建议先使用三个间隔明显的频率：

```js
const SSVEP_OPTION_FREQS = [12.8, 11.2, 8.8];
const SSVEP_OPTION_PHASES = [0, 0, 0];
```

这三个频率来自 `ssvep_flicker.py` 的九宫格对角线位置：

```text
左上 12.8Hz
中心 11.2Hz
右下 8.8Hz
```

如果希望继续沿用旧的三频，也可以保留 `[10, 7.5, 6]`，但从 `ssvep_flicker.py` 一致性看，更推荐 `[12.8, 11.2, 8.8]`。

## DOM 结构建议

当前 `renderChoiceState()` 生成的按钮可以从：

```html
<button class="choice-btn ssvep-choice">
  <span class="choice-label">
    <span class="choice-text">选项文本</span>
    <span class="ssvep-frequency-badge">12.8Hz</span>
  </span>
</button>
```

调整为：

```html
<button class="choice-btn ssvep-choice">
  <span class="choice-label">
    <span class="choice-text">选项文本</span>
    <span class="choice-ssvep-side">
      <span class="ssvep-frequency-badge">12.8Hz</span>
      <span class="ssvep-stimulus" aria-hidden="true"></span>
    </span>
  </span>
</button>
```

关键点：

- `.ssvep-stimulus` 是真正闪烁的元素。
- `.choice-btn` 不再整体闪烁。
- `.ssvep-frequency-badge` 不闪烁，只显示频率。

## CSS 建议

需要去掉 SSVEP 状态下按钮背景色 transition 对刺激的影响。刺激块本身不能有 transition。

```css
.choice-btn {
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease;
}

.choice-ssvep-side {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}

.ssvep-stimulus {
  width: 72px;
  height: 72px;
  flex: 0 0 72px;
  background: rgb(25, 25, 25);
  border: 1px solid rgba(0, 0, 0, 0.18);
  border-radius: 4px;
  transition: none;
}

.ssvep-enabled .choice-btn {
  background: #fff;
  color: var(--text);
}

.ssvep-enabled .choice-btn.ssvep-locked {
  outline: 4px solid rgba(245, 158, 11, 0.36);
  border-color: var(--warning);
}
```

不建议继续使用：

```css
.ssvep-enabled .choice-btn.ssvep-on
.ssvep-enabled .choice-btn.ssvep-off
```

这些 class 可以暂时保留兼容，但新的主逻辑不再依赖它们。

## JavaScript 实现

### 状态结构

在 `web/client.js` 中扩展 `ssvepState`：

```js
var ssvepState = {
  enabled: false,
  originalColor: false,
  rafId: null,
  frameCnt: 0,
  actualFps: 60,
  lutLen: 1000,
  targets: [],
  defaultFrequencies: [12.8, 11.2, 8.8],
  defaultPhases: [0, 0, 0]
};
```

### LUT 生成

沿用 `ssvep_flicker.py` 的公式：

```python
intensity = (sin(2*pi*f*i/actual_fps + phase*pi)+1)/2*0.9+0.1
```

对应前端：

```js
function buildSsvepLut(frequency, phase, actualFps, lutLen) {
  var lut = [];
  for (var i = 0; i < lutLen; i += 1) {
    var intensity = (
      (Math.sin(2 * Math.PI * frequency * i / actualFps + phase * Math.PI) + 1) / 2
    ) * 0.9 + 0.1;
    lut.push(intensity);
  }
  return lut;
}
```

### 刷新率处理

PsychoPy 可以用：

```python
actual_fps = win.getActualFrameRate() or 60
```

浏览器无法完全等价，但可以做一个轻量估计：

```js
function estimateRefreshRate(sampleFrames) {
  sampleFrames = sampleFrames || 90;
  return new Promise(function(resolve) {
    var times = [];
    function step(now) {
      times.push(now);
      if (times.length < sampleFrames) {
        requestAnimationFrame(step);
        return;
      }
      var intervals = [];
      for (var i = 1; i < times.length; i += 1) {
        intervals.push(times[i] - times[i - 1]);
      }
      intervals.sort(function(a, b) { return a - b; });
      var median = intervals[Math.floor(intervals.length / 2)] || 16.6667;
      resolve(Math.round(1000 / median));
    }
    requestAnimationFrame(step);
  });
}
```

第一版可以默认 `60`，但正式实验建议开启估计，并把估计值显示在 `#ssvep-status`。

### 启动闪烁

替换当前 `startSsvepFlicker()`：

```js
function startSsvepFlicker() {
  stopSsvepFlicker();
  if (!ssvepState.enabled || !ssvepState.targets.length || !window.requestAnimationFrame) {
    updateSsvepStatus();
    return;
  }

  $("#choice-options").addClass("ssvep-enabled");
  ssvepState.frameCnt = 0;
  ssvepState.actualFps = ssvepState.actualFps || 60;

  ssvepState.targets.forEach(function(target) {
    target.lut = buildSsvepLut(
      target.frequency,
      target.phase || 0,
      ssvepState.actualFps,
      ssvepState.lutLen
    );
  });

  function tick() {
    ssvepState.targets.forEach(function(target) {
      if (!target.stimulus || !target.lut) {
        return;
      }
      var intensity = target.lut[ssvepState.frameCnt % target.lut.length];
      var value = Math.round(intensity * 255);
      target.stimulus.style.backgroundColor = "rgb(" + value + "," + value + "," + value + ")";
    });
    ssvepState.frameCnt += 1;
    ssvepState.rafId = window.requestAnimationFrame(tick);
  }

  ssvepState.rafId = window.requestAnimationFrame(tick);
  updateSsvepStatus();
}
```

### 停止闪烁

停止时只恢复刺激块，不要清理按钮文本布局：

```js
function resetSsvepButtonVisuals() {
  $("#choice-options .choice-btn")
    .removeClass("ssvep-locked")
    .css({ backgroundColor: "", borderColor: "", boxShadow: "", color: "" });

  $("#choice-options .ssvep-stimulus").css({
    backgroundColor: "",
    borderColor: ""
  });
}
```

## 渲染选项时如何绑定 target

在 `renderChoiceState(payload)` 中生成每个选项时，找到对应刺激块：

```js
var button = $(html);
var stimulus = button.find(".ssvep-stimulus")[0];

ssvepState.targets.push({
  element: button[0],
  stimulus: stimulus,
  choiceId: choice.choice_id,
  frequency: frequency,
  phase: phase
});
```

这样 `selectChoiceBySsvepTarget(0)` 仍然可以选择第一个选项，现有外部调用方式不用大改。

## 与 `ssvep_flicker.py` 的一致性

保持一致：

- 使用 LUT
- LUT 公式一致
- 亮度范围为 `0.1 ~ 1.0`
- 每帧按 `frameCnt` 查表
- phase 使用 `phase * Math.PI`

不完全一致：

- PsychoPy 可更准确控制 OpenGL 全屏刺激；浏览器受刷新率、后台降频、渲染队列影响。
- 当前布局不是 3x3 大方块，而是选项后的局部刺激块。
- 浏览器 `requestAnimationFrame` 只能跟随屏幕刷新，不能保证严格实验级时序。

因此，这个方案适合当前 Web 对话系统集成。如果要做严格实验采集，仍建议用 PsychoPy 或专门刺激呈现程序。

## 推荐落地步骤

1. 保留当前 `#choice-options` 布局。
2. 在每个选项按钮右侧新增 `.ssvep-stimulus`。
3. 删除或停用 `.ssvep-on / .ssvep-off` 对按钮背景的控制。
4. 把 `startSsvepFlicker()` 改成 LUT 按帧更新刺激块亮度。
5. 默认频率改为 `[12.8, 11.2, 8.8]`。
6. 增加刷新率估计，至少在状态栏显示当前使用的 FPS。
7. 用浏览器开发工具录制一小段，确认刺激块没有 CSS transition。
8. 再接入 BCI 分类结果调用 `selectChoiceBySsvepTarget(index)`。

## 最小改动清单

需要改：

- `web/dashboard.html`
  - 增加 `.choice-ssvep-side`
  - 增加 `.ssvep-stimulus`
  - 停用 SSVEP 对整个按钮背景的 on/off 样式

- `web/client.js`
  - `ssvepState` 增加 `frameCnt / actualFps / lutLen`
  - `defaultFrequencies` 改为 `[12.8, 11.2, 8.8]`
  - 新增 `buildSsvepLut()`
  - `startSsvepFlicker()` 改为 LUT 驱动
  - `renderChoiceState()` 给每个 target 保存 `stimulus`

不需要改：

- `/choice/init`
- `/choice/select`
- choice tree JSON
- two-stage 视频缓存
- 数字人播放逻辑
