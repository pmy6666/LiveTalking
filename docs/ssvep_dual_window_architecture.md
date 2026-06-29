# Web 数字人 + Python SSVEP 双窗口方案评估与实施

## 结论

这个方案可行，而且比“直接在浏览器里渲染 SSVEP 闪烁”更合理。

推荐架构是：

```text
网页窗口
  - 数字人视频
  - 对话文本
  - 选项 UI
  - 普通鼠标点击
        |
        | HTTP / WebSocket
        v
LiveTalking 后端
  - /choice/init
  - /choice/state
  - /choice/select
  - 数字人播放与缓存命中
        ^
        | HTTP / WebSocket
        |
Python SSVEP 窗口
  - PsychoPy/pyglet 稳定闪烁
  - 接收当前选项
  - 显示 SSVEP 刺激
  - 接收 BCI 分类结果或键盘模拟
  - 把选择结果传给后端/网页
```

也就是：网页继续负责数字人形象和 UI，Python 程序负责 SSVEP 刺激与选择输入。两个窗口同时存在，但通过 LiveTalking 后端共享同一个对话状态。

## 为什么可行

当前 LiveTalking 已经有完整的选项接口：

- `POST /choice/init`
- `POST /choice/state`
- `POST /choice/select`
- `POST /choice/reset`

网页端已经可以：

- 初始化选项对话
- 渲染当前节点
- 调用 `/choice/select`
- 更新下一轮选项

因此 Python SSVEP 程序不需要直接操作网页 DOM。它只需要知道：

1. 当前有哪些选项。
2. 每个 target 对应哪个 `choice_id`。
3. 用户/分类器选择了哪个 target。
4. 把对应 `choice_id` 发给后端。

后端处理完成后，网页可以通过现有逻辑刷新状态。

## 推荐实施模式

推荐采用“后端作为唯一状态源”的模式。

Python SSVEP 程序不要直接把选项传给网页端，而是：

1. Python 从后端读取当前选项。
2. Python 显示 SSVEP 闪烁。
3. Python 得到选择结果后调用后端 `/choice/select`。
4. 网页从后端刷新当前状态。

这样不会出现“网页状态”和“Python 状态”不一致的问题。

## 数据流

### 1. 网页建立数字人会话

网页保持现有流程：

```text
开始连接 -> 获得 sessionid -> /choice/init -> 展示当前对话和选项
```

### 2. Python SSVEP 程序连接同一个 sessionid

Python 程序启动时输入或读取当前 `sessionid`：

```bash
python ssvep_choice_window.py --server http://127.0.0.1:8010 --sessionid 123456
```

### 3. Python 获取当前选项

调用：

```http
POST /choice/state
```

请求：

```json
{
  "sessionid": "123456"
}
```

返回中的关键字段：

```json
{
  "current": {
    "node_id": "root",
    "display_text": "...",
    "choices": [
      {
        "choice_id": "root.c1",
        "choice_text": "你吃过饭了吗",
        "child_node_id": "capability"
      }
    ]
  }
}
```

### 4. Python 显示 SSVEP 刺激

Python 程序根据 `choices` 生成 target：

```text
target 1 -> choice root.c1
target 2 -> choice root.c2
target 3 -> choice root.c3
```

闪烁仍使用 `ssvep_flicker.py` 的方法：

- PsychoPy 窗口
- `win.getActualFrameRate()`
- LUT 查表
- 每帧更新 `fillColor`
- `win.flip()`

### 5. Python 得到选择结果

第一阶段可以先用键盘模拟：

```text
1 -> target 1
2 -> target 2
3 -> target 3
```

后续接入 BCI 分类器：

```json
{
  "target_index": 2,
  "confidence": 0.86
}
```

### 6. Python 调用后端选择接口

将 `target_index` 映射成 `choice_id`，调用：

```http
POST /choice/select
```

请求：

```json
{
  "sessionid": "123456",
  "choice_id": "root.c2",
  "interrupt": true
}
```

后端会：

- 更新当前节点
- 播放对应数字人视频
- 命中 two-stage 缓存
- 返回下一轮选项

### 7. 网页刷新

网页刷新有两种方式：

#### 方案 A：网页轮询 `/choice/state`

实现简单，推荐第一版使用。

网页每 500ms 或 1000ms 请求一次：

```http
POST /choice/state
```

如果发现当前 `node_id` 变化，就调用现有 `renderChoiceState()` 更新 UI。

优点：

- 改动小
- 稳定
- 不需要 WebSocket

缺点：

- 有轻微延迟

#### 方案 B：后端增加 WebSocket/SSE 推送

后端在 `/choice/select` 后主动推送新状态给网页。

优点：

- 实时性更好

缺点：

- 改动更大
- 需要维护 session 与浏览器连接关系

第一阶段建议使用方案 A。

## Python SSVEP 程序设计

建议新增脚本：

```text
scripts/ssvep/ssvep_choice_window.py
```

职责：

1. 从 LiveTalking 后端获取当前选项。
2. 使用 PsychoPy 创建 SSVEP 窗口。
3. 根据当前选项显示 3 个刺激块。
4. 用 LUT 方法稳定闪烁。
5. 接收键盘/BCI 分类结果。
6. 调用 `/choice/select`。
7. 选择后重新拉取下一轮状态并更新刺激内容。

## SSVEP 刺激布局

你现在希望保留“每个对话选项后面有闪烁”的语义。Python 窗口中可以做成三行布局：

```text
┌───────────────────────────────┐
│ 1. 你吃过饭了吗        [闪烁块] │
│ 2. 你最近忙不忙        [闪烁块] │
│ 3. 你周末一般做什么    [闪烁块] │
└───────────────────────────────┘
```

如果 BCI 识别效果不稳定，再切换成更标准的三块空间分离布局：

```text
[选项1]      [选项2]      [选项3]
```

建议第一版使用三个固定频率：

```python
FREQUENCIES = [12.8, 11.2, 8.8]
PHASES = [0.0, 0.0, 0.0]
```

这些频率来自 `ssvep_flicker.py` 的九宫格频率，间隔比较明显。

## LiveTalking 后端需要改什么

第一版可以不改后端，直接复用现有接口。

可选增强：

### 增加状态版本号

在 `/choice/state` 返回中增加：

```json
{
  "state_version": 12
}
```

网页和 Python 都可以用它判断状态是否变化。

### 增加 Python 专用状态接口

例如：

```http
POST /choice/ssvep_state
```

返回更轻量的数据：

```json
{
  "sessionid": "123456",
  "node_id": "root",
  "answer_text": "...",
  "targets": [
    {
      "target_index": 1,
      "choice_id": "root.c1",
      "text": "你吃过饭了吗",
      "frequency": 12.8,
      "phase": 0.0
    }
  ]
}
```

这不是第一版必须项。

## 网页端需要改什么

第一版最小改动：

1. 关闭网页中的 SSVEP 闪烁渲染。
2. 保留普通选项按钮。
3. 增加“外部 SSVEP 控制中”的状态提示。
4. 增加 `/choice/state` 轮询，让 Python 选择后网页自动更新。

当前网页已有 `renderChoiceState(payload)` 和 `/choice/select` 流程，所以只需要加轮询：

```js
function startChoiceStatePolling() {
  setInterval(function() {
    if (!ensureSessionReady() || !choiceState.initialized) {
      return;
    }
    fetch("/choice/state", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({sessionid: String(document.getElementById("sessionid").value)})
    })
      .then(function(response) { return response.json(); })
      .then(function(payload) {
        if (payload.code !== 0 || !payload.data || !payload.data.current) {
          return;
        }
        var nextNodeId = payload.data.current.node_id;
        var currentNodeId = choiceState.current && choiceState.current.node_id;
        if (nextNodeId && nextNodeId !== currentNodeId) {
          renderChoiceState(payload.data);
        }
      });
  }, 800);
}
```

## 可行性评估

### 优点

- 保留现有网页数字人界面，不推翻当前工作流。
- SSVEP 刺激从浏览器剥离，时序更可靠。
- Python 可以直接复用 `ssvep_flicker.py` 的 PsychoPy 逻辑。
- 后端接口已经基本满足需求。
- 后续接 BCI 分类器更自然。
- 出问题时容易定位：网页负责 UI，Python 负责刺激，后端负责状态。

### 风险

- 需要用户同时管理两个窗口。
- 两个窗口可能在不同屏幕上，需要明确实验屏幕位置。
- Python 和网页必须使用同一个 `sessionid`。
- 网页如果不轮询状态，Python 选择后网页不会自动更新。
- 如果 Python 刺激窗口没有焦点，键盘模拟可能收不到。

### 解决方法

- Python 启动时显示当前 sessionid 和连接状态。
- 网页显示“SSVEP 外部控制已连接/未连接”。
- 第一版用轮询同步状态。
- 后续用 WebSocket/SSE 做双向同步。
- 实验时固定显示器刷新率，并让 PsychoPy 窗口全屏显示在刺激屏幕上。

## 推荐实施步骤

### 第一步：Python SSVEP 独立窗口

新增：

```text
scripts/ssvep/ssvep_choice_window.py
```

实现：

- `--server`
- `--sessionid`
- `/choice/state`
- 三个刺激块
- 键盘 `1/2/3` 模拟选择
- `/choice/select`
- 选择后刷新下一轮选项

### 第二步：网页关闭内部 SSVEP 闪烁

保留按钮和普通点击选择。

将当前网页 SSVEP 开关改成：

```text
外部 SSVEP 控制：请启动 Python SSVEP 窗口
```

### 第三步：网页增加状态轮询

Python 完成选择后，网页自动从 `/choice/state` 拉取最新节点。

### 第四步：接入 BCI 分类器

Python 窗口提供统一函数：

```python
def handle_target_selected(target_index: int):
    ...
```

键盘模拟和真实分类器都调用它。

### 第五步：增加连接状态

可选新增接口：

```text
POST /choice/ssvep_register
POST /choice/ssvep_heartbeat
```

网页显示外部 SSVEP 是否在线。

## 最小可运行版本

最小版本只需要：

1. Python 程序调用 `/choice/state` 获取选项。
2. PsychoPy 显示三个闪烁块。
3. 按 `1/2/3` 调 `/choice/select`。
4. 网页每 800ms 轮询 `/choice/state` 更新 UI。

这已经能验证完整闭环：

```text
网页显示数字人 -> Python 闪烁 -> 键盘/BCI 选择 -> 后端播放数字人 -> 网页刷新下一轮
```

## 建议命令

未来脚本可以这样启动：

```bash
cd .
../envs/livetalking/bin/python scripts/ssvep/ssvep_choice_window.py \
  --server http://127.0.0.1:8010 \
  --sessionid <网页当前 sessionid>
```

网页仍然按原方式启动：

```bash
cd .
scripts/two_stage_pre/start_choice_talking_head.sh
```

## 最终建议

建议采用这个双窗口方案。

不要继续在浏览器里追求 SSVEP 刺激精度。网页负责数字人和交互展示，Python/PsychoPy 负责闪烁和选择输入，这是目前最稳妥、改动最可控的路线。
