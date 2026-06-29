# LiveTalking SSVEP 模型模拟端口开发文档

## 1. 目标

当前 LiveTalking 已具备三选一“选项对话”链路：

- 前端页面：`web/dashboard.html`
- 前端逻辑：`web/client.js`
- 后端接口：`server/routes.py`
- 选项状态机：`choice/orchestrator.py`
- 选项树数据：`data/choice_trees/default_choice_tree.json`

本阶段目标不是改动业务代码，而是先对齐 SSVEP 模型接入方式：新增一个“SSVEP 模型模拟端口”，接收或产生 SSVEP 识别结果，并把识别出的选项传给现有选项对话选择接口 `/choice/select`。

## 2. 当前代码阅读结论

### 2.1 服务启动与路由注册

`app.py` 使用 `aiohttp.web.Application` 启动 HTTP 服务，默认监听端口来自 `--listenport`，默认值为 `8010`。

关键流程：

1. `config.py` 解析 `--listenport`。
2. `app.py` 创建 `appasync = web.Application(...)`。
3. `app.py` 写入 `appasync["choice_orchestrator"] = ChoiceOrchestrator(...)`。
4. `app.py` 调用 `setup_routes(appasync)`。
5. `server/routes.py` 注册业务路由。

因此 SSVEP 模拟端口建议也放在 `server/routes.py` 中统一注册，或者拆成 `server/ssvep_routes.py` 后在 `setup_routes()` 中挂载。

### 2.2 现有选项对话接口

现有接口都使用 JSON 请求和 JSON 响应，成功响应格式统一为：

```json
{
  "code": 0,
  "msg": "ok",
  "data": {}
}
```

失败响应格式：

```json
{
  "code": -1,
  "msg": "错误信息"
}
```

已存在的选项对话接口：

| 接口 | 方法 | 作用 |
| --- | --- | --- |
| `/choice/init` | POST | 初始化选项树 |
| `/choice/select` | POST | 选择当前节点下的某个选项 |
| `/choice/state` | POST | 获取当前选项状态 |
| `/choice/reset` | POST | 重置选项树 |

其中 `/choice/select` 是 SSVEP 最终需要触发的核心接口。

请求格式：

```json
{
  "sessionid": "123456",
  "choice_id": "root.c1",
  "interrupt": true
}
```

### 2.3 前端已有 SSVEP 雏形

`web/client.js` 中已经存在 SSVEP 前端状态和函数：

- `ssvepState`
- `startSsvepFlicker()`
- `stopSsvepFlicker()`
- `selectChoiceBySsvepTarget(targetIndex)`
- `window.LiveTalkingSSVEP.selectTarget(targetIndex)`

当前前端选项渲染时，会把后端返回的 `payload.current.choices` 转成按钮，并维护：

```js
ssvepState.targets.push({
    index: index,
    choiceId: choice.choice_id,
    frequency: frequency,
    phase: phase,
    element: button[0]
});
```

也就是说，真实或模拟 SSVEP 模型只要输出目标编号，例如 `1`、`2`、`3`，前端已经可以通过：

```js
window.LiveTalkingSSVEP.selectTarget(1)
```

转换为当前选项的 `choice_id`，再调用已有 `/choice/select`。

### 2.4 后端选项状态机

`choice/orchestrator.py` 中：

- `_build_choices()` 当前返回 `choice_id`、`choice_text`、`child_node_id`。
- `init_session()` 初始化根节点。
- `select_choice()` 校验 `choice_id` 是否属于当前节点，然后跳转到子节点。
- `_serialize_node()` 返回当前节点文本和 choices。

因此 SSVEP 端口不应该绕过 `ChoiceOrchestrator` 直接修改状态，而应该复用 `/choice/select` 或调用同一层 `orchestrator.select_choice()`。

## 3. 推荐接入架构

### 3.1 总体链路

推荐链路如下：

```text
前端 SSVEP 闪烁选项
  -> SSVEP 模型或模拟器识别目标编号 target_index
  -> 新增 SSVEP 模拟端口
  -> 根据当前 choice state 把 target_index 映射成 choice_id
  -> 调用 ChoiceOrchestrator.select_choice()
  -> 返回与 /choice/select 相同的选项状态 payload
  -> 前端刷新选项和数字人播报
```

### 3.2 为什么建议后端端口做映射

前端已经能映射 `target_index -> choice_id`，但新增后端 SSVEP 端口仍有价值：

1. 后续真实 SSVEP 模型通常运行在服务端或独立进程。
2. 模型只需要输出稳定的目标编号，不需要理解业务 `choice_id`。
3. 后端可以记录置信度、延迟、原始频率等实验数据。
4. 后端可以统一校验当前会话和当前节点，避免选项切换时误触发旧目标。

## 4. 新增端口设计

### 4.1 端口命名

建议新增：

```text
POST /ssvep/select
```

该接口表示“接收 SSVEP 模型识别出的目标，并触发选项对话选择”。

可选新增调试接口：

```text
GET /ssvep/health
POST /ssvep/simulate
```

如果只做第一阶段，`/ssvep/select` 一个接口即可。

### 4.2 `/ssvep/select` 请求格式

最小可用格式：

```json
{
  "sessionid": "123456",
  "target_index": 1,
  "interrupt": true
}
```

推荐完整格式：

```json
{
  "sessionid": "123456",
  "target_index": 1,
  "target_id": "target-1",
  "frequency": 10.0,
  "confidence": 0.92,
  "timestamp": 1710000000.123,
  "source": "ssvep_simulator",
  "interrupt": true
}
```

字段说明：

| 字段 | 类型 | 必填 | 说明 |
| --- | --- | --- | --- |
| `sessionid` | string | 是 | LiveTalking WebRTC 会话 ID |
| `target_index` | number | 是 | SSVEP 识别出的目标编号，建议使用 1-based，即 1/2/3 |
| `target_id` | string | 否 | 模型侧目标 ID，便于日志追踪 |
| `frequency` | number | 否 | 识别出的刺激频率 |
| `confidence` | number | 否 | 模型置信度，范围建议 0 到 1 |
| `timestamp` | number | 否 | 模型识别时间戳 |
| `source` | string | 否 | `ssvep_simulator` 或真实模型名 |
| `interrupt` | boolean | 否 | 是否打断当前播报，默认 `true` |

### 4.3 `/ssvep/select` 响应格式

成功时建议直接复用 `/choice/select` 的 `data` 结构，并额外附带 SSVEP 元信息：

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "ssvep": {
      "target_index": 1,
      "choice_id": "root.c1",
      "confidence": 0.92,
      "source": "ssvep_simulator"
    },
    "choice": {
      "selected_choice_id": "root.c1",
      "path": ["root", "capability"],
      "current": {
        "node_id": "capability",
        "answer_text": "我吃过啦，你呢？再忙也记得按时吃饭。",
        "display_text": "我吃过啦，你呢？再忙也记得按时吃饭。",
        "tts_text": "我吃过啦，你呢？再忙也记得按时吃饭。",
        "choices": [],
        "audio_cache_hit": false
      },
      "audio_cache_hit": false,
      "video_cache_hit": false,
      "cache_mode": "realtime_fallback"
    }
  }
}
```

也可以为了前端改动更少，让 `data` 直接等于 `/choice/select` payload，同时增加顶层字段：

```json
{
  "code": 0,
  "msg": "ok",
  "data": {
    "selected_choice_id": "root.c1",
    "path": ["root", "capability"],
    "current": {},
    "audio_cache_hit": false,
    "video_cache_hit": false,
    "cache_mode": "realtime_fallback",
    "ssvep": {
      "target_index": 1,
      "confidence": 0.92,
      "source": "ssvep_simulator"
    }
  }
}
```

第一阶段推荐第二种格式，因为前端可以继续把整个 `payload.data` 交给 `renderChoiceState(payload.data)`。

## 5. target_index 到 choice_id 的映射规则

### 5.1 编号规则

建议 SSVEP 模型输出 `target_index` 使用 1-based 编号：

| SSVEP 输出 | 当前选项数组下标 | 含义 |
| --- | --- | --- |
| `1` | `choices[0]` | 第一个选项 |
| `2` | `choices[1]` | 第二个选项 |
| `3` | `choices[2]` | 第三个选项 |

原因：实验人员和前端 UI 都更容易理解“选项 1/2/3”。

### 5.2 后端映射流程

后端收到 `/ssvep/select` 后：

1. 根据 `sessionid` 获取 `avatar_session`。
2. 从 `ChoiceOrchestrator.get_state(avatar_session)` 获取当前节点。
3. 读取 `state["current"]["choices"]`。
4. 将 `target_index` 转为数组下标。
5. 取出 `choice_id`。
6. 调用 `ChoiceOrchestrator.select_choice(avatar_session, choice_id, interrupt=True)`。
7. 返回选择后的状态。

伪代码：

```python
async def ssvep_select(request):
    params = await request.json()
    sessionid = params.get("sessionid", "")
    target_index = int(params.get("target_index", 0))

    avatar_session = get_session(request, sessionid)
    if avatar_session is None:
        return json_error("session not found")

    orchestrator = request.app.get("choice_orchestrator")
    state = orchestrator.get_state(avatar_session)
    if not state.get("initialized"):
        return json_error("choice mode not initialized")

    choices = state["current"].get("choices", [])
    index = target_index - 1
    if index < 0 or index >= len(choices):
        return json_error("invalid ssvep target_index")

    choice_id = choices[index]["choice_id"]
    payload = orchestrator.select_choice(
        avatar_session,
        choice_id=choice_id,
        interrupt=bool(params.get("interrupt", True)),
    )
    payload["ssvep"] = {
        "target_index": target_index,
        "choice_id": choice_id,
        "confidence": params.get("confidence"),
        "source": params.get("source", "ssvep_simulator"),
    }
    return json_ok(data=payload)
```

## 6. 模拟器设计

### 6.1 第一阶段：HTTP 手动模拟

先不引入真实模型，只通过 HTTP 请求模拟 SSVEP 输出。

示例：

```bash
curl -X POST http://127.0.0.1:8010/ssvep/select \
  -H "Content-Type: application/json" \
  -d '{
    "sessionid": "实际页面里的 sessionid",
    "target_index": 1,
    "confidence": 0.95,
    "source": "manual_curl",
    "interrupt": true
  }'
```

这可以验证：

- SSVEP 端口能找到当前会话。
- `target_index` 能映射到当前选项。
- `/choice/select` 逻辑能被复用。
- 数字人会按选项结果播报。

### 6.2 第二阶段：定时自动模拟

可以新增一个独立脚本，例如：

```text
scripts/ssvep_simulator_client.py
```

脚本功能：

1. 接收 `--server http://127.0.0.1:8010`。
2. 接收 `--sessionid xxx`。
3. 每隔 N 秒随机选择 `target_index`。
4. POST 到 `/ssvep/select`。

这类脚本不需要嵌入主服务，便于独立模拟真实 SSVEP 模型进程。

### 6.3 第三阶段：真实 SSVEP 模型接入

真实模型可以继续沿用同一 HTTP 协议，只需要把预测结果组织成：

```json
{
  "sessionid": "123456",
  "target_index": 2,
  "frequency": 7.5,
  "confidence": 0.88,
  "source": "real_ssvep_model",
  "timestamp": 1710000000.123
}
```

如果真实模型只能输出频率，也可以在后端增加 `frequency -> target_index` 映射：

| 频率 | target_index |
| --- | --- |
| 10Hz | 1 |
| 7.5Hz | 2 |
| 6Hz | 3 |

不过第一阶段建议模型直接输出 `target_index`，避免屏幕刷新率、频率误差和浮点匹配带来的歧义。

## 7. 前端对齐方式

### 7.1 当前可用方式

前端已经暴露：

```js
window.LiveTalkingSSVEP.selectTarget(1)
```

浏览器控制台执行后，会选择当前第一个选项。

这适合前端本地调试，但不能模拟“模型从后端推送选择”的完整链路。

### 7.2 接入 `/ssvep/select` 后的前端策略

第一阶段可以不改前端：用 curl 或模拟器脚本直接请求 `/ssvep/select`。

如果后续希望页面主动连接 SSVEP 模拟器，有两个方向：

1. 前端轮询 `/ssvep/latest?sessionid=xxx`，拿到最新目标后调用 `window.LiveTalkingSSVEP.selectTarget()`。
2. 后端提供 WebSocket 或 Server-Sent Events，把 SSVEP 结果推给前端。

但考虑当前系统已经有 `/choice/select`，最小闭环建议先走服务端 HTTP 触发选择。

## 8. 后端文件改动建议

第一阶段最小改动：

| 文件 | 改动 |
| --- | --- |
| `server/routes.py` | 新增 `ssvep_select()`，在 `setup_routes()` 中注册 `/ssvep/select` |
| `web/client.js` | 暂不必改 |
| `web/dashboard.html` | 暂不必改 |
| `config.py` | 暂不必改 |

如果想拆分得更清晰：

| 文件 | 改动 |
| --- | --- |
| `server/ssvep_routes.py` | 放 SSVEP 专属接口 |
| `server/routes.py` | 导入并注册 SSVEP 路由 |

## 9. 错误处理约定

建议覆盖以下错误：

| 场景 | 返回 msg |
| --- | --- |
| session 不存在 | `session not found` |
| choice orchestrator 未配置 | `choice orchestrator not configured` |
| 选项模式未初始化 | `choice mode not initialized` |
| target_index 不是数字 | `invalid ssvep target_index` |
| target_index 超出当前选项数量 | `invalid ssvep target_index` |
| 当前节点没有选项 | `no choices available for current node` |

## 10. 验收步骤

1. 启动 LiveTalking：

```bash
cd .
../envs/livetalking/bin/python app.py --transport webrtc --model echomimicv3 --avatar_id avatar6 --listenport 8010
```

实际启动参数以 `run.txt` 为准。

2. 浏览器打开：

```text
http://127.0.0.1:8010/dashboard.html
```

3. 选择角色并点击“开始连接”。

4. 确认页面已初始化选项对话，拿到隐藏字段 `sessionid`。

5. 请求模拟 SSVEP 选择：

```bash
curl -X POST http://127.0.0.1:8010/ssvep/select \
  -H "Content-Type: application/json" \
  -d '{"sessionid":"页面中的 sessionid","target_index":1,"confidence":0.95,"source":"manual_curl","interrupt":true}'
```

6. 预期结果：

- HTTP 返回 `code=0`。
- 返回数据里包含 `selected_choice_id`。
- 页面当前选项状态发生变化。
- 数字人播放被选中节点对应文本。

## 11. 后续扩展

### 11.1 choice tree 配置 SSVEP 元数据

后续可以在 `data/choice_trees/default_choice_tree.json` 的每个 choice 中增加：

```json
{
  "choice_id": "root.c1",
  "choice_text": "你吃过饭了吗",
  "child_node_id": "capability",
  "ssvep": {
    "target_index": 1,
    "frequency": 10.0,
    "phase": 0
  }
}
```

同时在 `ChoiceOrchestrator._build_choices()` 中透传 `ssvep` 字段。

### 11.2 防误触发

真实实验中建议增加：

- `confidence_threshold`：低于阈值不触发。
- `cooldown_ms`：一次选择后短时间内忽略重复结果。
- `node_id` 校验：模型结果带上识别时的节点 ID，后端确认仍是同一节点再选择。
- `target_version`：每次渲染选项时递增版本，避免旧刺激结果命中新选项。

### 11.3 日志记录

建议每次 SSVEP 触发记录：

- `sessionid`
- `tree_id`
- 当前 `node_id`
- `target_index`
- 映射后的 `choice_id`
- `frequency`
- `confidence`
- `source`
- 处理耗时

这对后续实验复盘很重要。

## 12. 第一阶段实施清单

- [ ] 在 `server/routes.py` 新增 `/ssvep/select`。
- [ ] 复用 `session_manager.get_session()` 获取会话。
- [ ] 复用 `ChoiceOrchestrator.get_state()` 读取当前 choices。
- [ ] 将 `target_index` 映射为当前 `choice_id`。
- [ ] 复用 `ChoiceOrchestrator.select_choice()` 执行选择。
- [ ] 响应中附加 `ssvep` 调试信息。
- [ ] 使用 curl 完成端到端验证。
- [ ] 视需要再补一个独立模拟脚本。

