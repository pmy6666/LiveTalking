# EchoMimicV3 静态图与 DeepSeek 提示词任务方案

## 目标

将当前 LiveTalking 中 EchoMimicV3 的数字人播放流程改造成：

1. 使用一张静态参考图作为默认展示画面，不再循环播放一段照片帧。
2. 用户发起播报或任务动作时，GPT-SoVITS 生成驱动音频。
3. DeepSeek 根据当前数字人、播报内容和任务上下文生成简短直白的 `prompt` 与 `negative prompt`。
4. EchoMimicV3 使用参考图、驱动音频、prompt、negative prompt 生成任务动作视频。
5. 动作视频播放完成后，自动回到该数字人的静态参考图。

## 素材来源

### 参考图像

使用现有目录：

```text
assets/avatars/avatar1.jpg
assets/avatars/avatar2.jpg
assets/avatars/avatar3.jpg
assets/avatars/avatar4.jpg
assets/avatars/avatar5.jpg
```

后续可以建立一个映射表，将前端选择的数字人 ID 映射到对应参考图：

```text
avatar1 -> assets/avatars/avatar1.jpg
avatar2 -> assets/avatars/avatar2.jpg
avatar3 -> assets/avatars/avatar3.jpg
avatar4 -> assets/avatars/avatar4.jpg
avatar5 -> assets/avatars/avatar5.jpg
```

### 驱动音频

驱动音频使用 GPT-SoVITS TTS 的输出，不需要用户额外上传 `speech.wav` 或 `speech.mp3`。

流程为：

```text
文本输入 -> GPT-SoVITS -> wav/pcm 音频流 -> 缓存为临时 speech.wav -> EchoMimicV3 生成视频
```

为了兼容 EchoMimicV3，建议在适配层统一转换为：

```text
sample_rate = 16000
channels = mono
dtype = float32
container = wav
```

## DeepSeek Prompt 生成文件方案

新增一个独立文件：

```text
llm_prompt_deepseek.py
```

职责只做一件事：调用 DeepSeek API，返回 EchoMimicV3 所需的 `prompt` 和 `negative_prompt`。

建议暴露函数：

```text
generate_echomimicv3_prompts(
    avatar_name: str,
    avatar_description: str,
    speech_text: str,
    scene: str = "",
    action: str = ""
) -> dict
```

返回格式：

```json
{
  "prompt": "A front-facing female presenter speaks calmly, upper body visible, natural posture, realistic studio lighting.",
  "negative_prompt": "blur, low quality, distorted face, bad hands, extra fingers, strange body movement, jitter, motion blur"
}
```

### DeepSeek API 配置

通过环境变量配置，不把密钥写进代码：

```text
DEEPSEEK_API_KEY=你的 key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat
```

如果 API 调用失败，必须走本地兜底模板，避免影响主流程：

```text
prompt:
A front-facing digital human is speaking naturally, upper body visible, stable camera, realistic lighting.

negative_prompt:
blur, low quality, distorted face, bad hands, extra fingers, unnatural motion, jitter, flicker
```

## DeepSeek 提示词设计

### System Prompt

```text
You generate concise English prompts for EchoMimicV3 talking-head video generation.
Return only valid JSON.
The prompt must be short, direct, visual, and suitable for a realistic talking portrait.
Do not describe complex actions, large body movement, camera movement, or scene changes.
The negative_prompt must list visual defects to avoid.
```

### User Prompt 模板

```text
Avatar:
{avatar_description}

Speech text:
{speech_text}

Scene:
{scene}

Action:
{action}

Generate JSON with two fields:
prompt: one short English sentence, no more than 28 words.
negative_prompt: comma-separated English keywords, no more than 24 keywords.
```

### 生成约束

prompt 应该强调：

```text
front-facing
talking naturally
upper body or portrait
stable camera
realistic lighting
natural posture
```

negative prompt 应该覆盖：

```text
blur
low quality
distorted face
bad hands
extra fingers
deformed body
strange movement
jitter
flicker
motion blur
out of frame
```

不建议让 LLM 生成复杂动作，例如转身、走路、挥手、大幅身体动作。EchoMimicV3 当前更适合稳定说话头像，复杂动作容易带来画面跳变和肢体错误。

## 播放状态机方案

当前问题是默认画面循环播放照片帧。新的播放逻辑建议改为明确状态机：

```text
IDLE_STATIC
GENERATING
PLAYING_GENERATED
RETURNING_STATIC
```

### IDLE_STATIC

默认状态。

行为：

```text
持续推送同一张 avatar 静态图
持续推送静音音频帧
```

这里不再使用 `full_imgs` 的循环帧。

### GENERATING

收到文本任务后进入该状态。

行为：

```text
1. GPT-SoVITS 生成驱动音频
2. DeepSeek 生成 prompt 和 negative_prompt
3. EchoMimicV3 使用参考图和音频生成视频帧
4. 生成期间前端仍看到静态图
```

注意：不要先播放 TTS 音频，否则会破坏口型同步。应等待 EchoMimicV3 生成帧后，把生成帧和对应音频一起播放。

### PLAYING_GENERATED

EchoMimicV3 返回视频帧后进入该状态。

行为：

```text
按 fps 播放生成视频帧
同步推送对应音频帧
播放 start/end 事件
```

如果播放期间用户发起新任务：

```text
interrupt=true  -> 清空当前播放队列，重新进入 GENERATING
interrupt=false -> 新任务排队
```

### RETURNING_STATIC

动作视频播放结束后进入该状态。

第一版可以直接切回静态图。

后续如果想更自然，可以加 3 到 5 帧淡出过渡：

```text
last_generated_frame -> avatar_static_frame
```

## EchoMimicV3 适配器修改方案

主要修改文件：

```text
avatars/echomimicv3_avatar.py
```

建议调整点：

1. `load_avatar()` 支持从 `assets/avatars/avatar*.jpg` 读取参考图。
2. `EchoMimicV3AvatarData` 增加 `negative_prompt` 或 prompt provider 配置。
3. idle 逻辑从 `_next_idle_frame()` 改为固定返回静态图。
4. 音频任务结构 `EchoMimicV3AudioJob` 增加：

```text
speech_text
prompt
negative_prompt
ref_image_path
```

5. `_generation_loop()` 中调用 `llm_prompt_deepseek.generate_echomimicv3_prompts()`。
6. `EchoMimicV3FlashEngine._generate_frames_from_audio_path()` 将 `negative_prompt` 传给 pipeline。

当前代码里 pipeline 的 `negative_prompt` 是空字符串：

```text
negative_prompt=""
```

后续应替换成 DeepSeek 返回值或兜底值。

## 前端与会话选择方案

前端继续使用现有五个数字人卡片。

需要保证 `/offer` 请求中传入的 avatar 能映射到静态参考图，例如：

```json
{
  "avatar": "avatar1"
}
```

如果为了兼容现有 `wav2lip256_avatar1`，也可以先做映射：

```text
wav2lip256_avatar1 -> assets/avatars/avatar1.jpg
wav2lip_avatar_2   -> assets/avatars/avatar2.jpg
wav2lip_avatar_3   -> assets/avatars/avatar3.jpg
wav2lip_avatar_4   -> assets/avatars/avatar4.jpg
wav2lip_avatar_5   -> assets/avatars/avatar5.jpg
```

## 推荐目录结构

```text
LiveTalking/
  assets/
    avatars/
      avatar1.jpg
      avatar2.jpg
      avatar3.jpg
      avatar4.jpg
      avatar5.jpg
  cache/
    echomimicv3/
      audio/
      video/
      prompt/
  llm_prompt_deepseek.py
```

缓存建议：

```text
cache/echomimicv3/audio/   保存 GPT-SoVITS 生成的临时 wav
cache/echomimicv3/video/   可选，保存生成帧或 mp4 便于调试
cache/echomimicv3/prompt/  可选，保存 prompt/negative_prompt 便于复现
```

## 风险与限制

1. EchoMimicV3 不是实时模型，生成期间会有等待；静态图兜底可以让用户感知更稳定。
2. 不建议播放 TTS 后再等待视频，因为音频和口型会不同步。
3. LLM 生成的 prompt 必须短，不要让它写剧情或复杂动作。
4. negative prompt 只能降低坏结果概率，不能完全避免坏手、抖动、模糊。
5. 如果 `DeepSeek API` 不可用，必须使用本地默认 prompt，不能让 `/offer` 或 `/human` 失败。

## 分阶段实施

### 第一阶段：可连通

1. EchoMimicV3 idle 改为固定静态图。
2. 参考图读取 `assets/avatars/avatar*.jpg`。
3. GPT-SoVITS 音频继续作为驱动音频。
4. 视频生成完成后播放，播放结束回静态图。

### 第二阶段：DeepSeek Prompt

1. 新增 `llm_prompt_deepseek.py`。
2. 接入环境变量 API key。
3. 在 EchoMimicV3 生成前调用 DeepSeek。
4. 增加失败兜底 prompt。

### 第三阶段：体验优化

1. 加 prompt 缓存，减少重复调用。
2. 加视频/帧缓存，方便调试。
3. 加静态图和生成视频之间的淡入淡出。
4. 根据不同数字人维护固定 avatar description。

## 最小可行版本定义

最小可行版本只需要满足：

```text
打开页面 -> 选择数字人 -> 连接成功 -> 默认显示一张静态图
输入文本 -> GPT-SoVITS 生成音频 -> DeepSeek 生成 prompt
EchoMimicV3 生成视频 -> 播放人物动作和语音
播放结束 -> 回到同一张静态图
```

