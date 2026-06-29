# EchoMimicV3 整段音频与分段平滑生成任务方案

## 背景问题

当前 LiveTalking 接入 EchoMimicV3 后，音频会被切成多个 chunk 再分别送入 EchoMimicV3 生成视频片段。这样虽然能控制单次推理长度，但会带来两个明显问题：

1. 每个 chunk 都从同一张参考图重新开始生成，片段开头姿态趋近静态参考图。
2. 每个 chunk 的尾帧和下一个 chunk 的首帧没有时序记忆，容易出现跳变、抖动、表情断裂。

最终表现为：

```text
第 1 段动作视频 -> 第 2 段动作视频 -> 第 3 段动作视频
```

看似连续播放，但人物头部、嘴型、肩部、光照细节可能在片段边界发生突变。

## 目标

优化 EchoMimicV3 生成策略，使数字人动作更连续：

1. 优先尝试整段音频一次性生成完整动作视频。
2. 如果整段生成显存或时长不可接受，再采用分段生成加平滑拼接。
3. 保证播放阶段只播放连续视频，不在片段之间回到静态图。
4. 尽量避免破坏音频和口型同步。

## 方案一：整段音频一次性生成

### 思路

将 GPT-SoVITS 生成的完整音频保存为一个完整 wav，然后一次性传给 EchoMimicV3：

```text
完整文本 -> GPT-SoVITS -> full_speech.wav -> EchoMimicV3 -> full_video_frames -> WebRTC 连续播放
```

也就是说，不再按 1 秒音频切片，而是让 EchoMimicV3 看到完整上下文。

### 优点

1. 动作连续性最好。
2. 首尾帧断裂问题最少。
3. prompt、negative prompt、参考图只作用一次，风格更稳定。
4. 播放逻辑最简单：生成完成后按帧连续播放即可。

### 缺点

1. 显存压力更大。
2. 推理时间更长，用户首帧等待明显增加。
3. EchoMimicV3/Wan pipeline 可能有最大帧数限制。
4. 如果一句话很长，可能直接 OOM 或生成失败。

### 风险点

当前启动参数中有：

```text
--echomimicv3_video_length 25
```

这意味着当前代码会限制单次最多生成 25 帧，约 1 秒。如果直接传完整音频，但仍保留这个限制，实际仍只会生成前 1 秒。

若要整段音频一次生成，需要调整：

```text
echomimicv3_video_length = ceil(audio_duration * fps)
```

例如：

```text
5 秒音频 -> 125 帧
10 秒音频 -> 250 帧
```

这可能显著增加显存占用。

### 建议限制

第一版不要无限制整段生成。建议设置最大整段时长：

```text
max_full_audio_seconds = 5 到 8 秒
```

如果音频时长不超过这个阈值，则整段生成。

如果超过阈值，则回退到分段生成方案。

### 适用场景

适合：

```text
短句播报
选项对话里的单个回答
1 到 8 秒的数字人回答
```

不适合：

```text
长篇讲解
几十秒连续播报
多轮长文本 LLM 输出
```

## 方案二：分段生成，但增加上下文重叠

### 思路

仍然分段生成，但每段之间加入 overlap 音频上下文。

例如每段主体 4 秒，前后各带 0.5 秒上下文：

```text
segment 1: [0.0s - 4.5s]       输出保留 [0.0s - 4.0s]
segment 2: [3.5s - 8.5s]       输出保留 [4.0s - 8.0s]
segment 3: [7.5s - 12.5s]      输出保留 [8.0s - 12.0s]
```

这样每段生成时能看到前后一点音频上下文，边界处动作会更自然。

### 优点

1. 比纯切片更连续。
2. 显存压力比整段生成小。
3. 可处理更长音频。

### 缺点

1. 需要丢弃 overlap 区域对应的视频帧。
2. 代码复杂度更高。
3. EchoMimicV3 仍然是每段独立从参考图生成，无法完全继承上段尾帧状态。
4. 如果模型不支持上一段尾帧作为条件，仍会有一定跳变。

### 建议参数

```text
segment_seconds = 4
overlap_seconds = 0.5
fps = 25
overlap_frames = 12
```

每段生成完成后：

```text
第一段保留全部有效帧
中间段丢弃前 overlap_frames
最后一段丢弃前 overlap_frames，尾部按音频长度裁剪
```

## 方案三：分段生成后做视频帧过渡

### 思路

在第 N 段尾帧和第 N+1 段首帧之间做短过渡。

最简单方式是帧级 crossfade：

```text
transition_frame = frame_a * (1 - alpha) + frame_b * alpha
```

例如用 5 帧过渡：

```text
第 N 段最后 5 帧
第 N+1 段前 5 帧
做 alpha blend
```

### 优点

1. 实现简单。
2. 能缓解明显闪跳。
3. 不增加 EchoMimicV3 推理成本。

### 缺点

1. 只能视觉上淡化跳变，不能真正保证动作连续。
2. 嘴型区域可能变糊。
3. 如果两段头部位置差异很大，crossfade 会产生重影。

### 适用场景

适合作为补充方案，而不是主方案。

建议和方案二组合使用：

```text
overlap 分段生成 + 边界 crossfade
```

## 方案四：尾帧作为下一段参考图

### 思路

第 1 段生成完成后，将第 1 段最后一帧作为第 2 段的参考图：

```text
segment 1 ref = avatar.jpg
segment 2 ref = segment1_last_frame
segment 3 ref = segment2_last_frame
```

### 优点

1. 理论上可以改善姿态连续性。
2. 每段开头不再回到原始静态图。

### 缺点

1. 误差会累积，人物身份和画质可能逐段漂移。
2. EchoMimicV3 对参考图身份稳定性要求较高，尾帧可能带有表情、模糊、口型，作为参考图不一定稳定。
3. 长段视频可能越来越不像原角色。

### 建议

不建议作为第一优先方案。可以作为实验开关：

```text
--echomimicv3_use_last_frame_ref
```

默认关闭。

## 推荐方案

建议采用混合策略：

```text
短音频：整段音频一次生成
长音频：overlap 分段生成 + 边界 crossfade
```

### 推荐规则

```text
if audio_duration <= 6 秒:
    使用整段音频一次生成
else:
    使用 4 秒主体 + 0.5 秒 overlap 分段生成
    拼接时丢弃 overlap 帧
    边界处做 3 到 5 帧 crossfade
```

### 为什么这样选

选项对话通常回答不长，很多回答在 3 到 6 秒内。整段生成可以最大程度保证动作连续。

对于更长回答，强行整段生成可能等待过久或显存不稳，因此用 overlap 分段保底。

## 代码修改范围

主要修改文件：

```text
avatars/echomimicv3_avatar.py
```

建议新增或调整以下参数：

```text
--echomimicv3_full_audio_threshold 6.0
--echomimicv3_segment_seconds 4.0
--echomimicv3_overlap_seconds 0.5
--echomimicv3_transition_frames 5
```

也可以先不加 CLI 参数，直接在 EchoMimicV3 适配器中使用常量。

### 需要调整的函数

当前主要逻辑：

```text
_enqueue_audio_jobs()
_generate_utterance_frames()
_enqueue_playback()
```

建议改为：

```text
_enqueue_audio_jobs()
    不再创建多个播放 job，只创建一个 utterance job

_generate_utterance_frames()
    根据音频长度选择 full 或 overlap segment

_generate_full_audio_frames()
    一次性生成完整音频对应帧

_generate_overlap_segment_frames()
    分段生成并裁剪 overlap

_crossfade_frames()
    对拼接边界做 3 到 5 帧融合
```

## 整段生成伪代码

```text
duration = len(audio) / sample_rate
target_frames = ceil(duration * fps)

if duration <= full_audio_threshold:
    old_video_length = opt.echomimicv3_video_length
    opt.echomimicv3_video_length = target_frames
    frames = engine.generate_frames(ref_image, audio, prompt, negative_prompt)
    opt.echomimicv3_video_length = old_video_length
```

注意：更推荐改 `engine.generate_frames()` 支持传入 `max_video_length`，不要临时修改全局 opt。

## overlap 分段伪代码

```text
segment_samples = segment_seconds * sample_rate
overlap_samples = overlap_seconds * sample_rate
step_samples = segment_samples

for segment_start in range(0, total_samples, step_samples):
    context_start = max(0, segment_start - overlap_samples)
    context_end = min(total_samples, segment_start + segment_samples + overlap_samples)

    segment_audio = audio[context_start:context_end]
    segment_frames = generate(segment_audio)

    left_trim_frames = 0 if segment_start == 0 else overlap_seconds * fps
    right_keep_frames = segment_seconds * fps

    valid_frames = segment_frames[left_trim_frames:left_trim_frames + right_keep_frames]
    append valid_frames
```

## crossfade 伪代码

```text
def crossfade(prev_frames, next_frames, n=5):
    if len(prev_frames) < n or len(next_frames) < n:
        return prev_frames + next_frames

    output = prev_frames[:-n]
    for i in range(n):
        alpha = (i + 1) / (n + 1)
        blended = prev_frames[-n + i] * (1 - alpha) + next_frames[i] * alpha
        output.append(blended)
    output.extend(next_frames[n:])
    return output
```

## 对播放体验的影响

### 整段生成

用户体验：

```text
等待更久 -> 一次性连续播放完整动作
```

适合追求连续性。

### overlap 分段

用户体验：

```text
等待比整段短一些 -> 连续性比纯切片好
```

适合中长文本。

### 流式边生成边播放

当前不推荐。因为 EchoMimicV3 生成速度慢于播放速度：

```text
生成 1 秒视频约 5.8 秒
播放 1 秒视频只需 1 秒
```

如果继续边生成边播放，除非提前缓存很多段，否则必然再次断裂。

## 推荐实施顺序

### 第一阶段

实现短音频整段生成：

```text
audio_duration <= 6 秒 -> 整段生成
```

先解决选项对话短回答的连续性。

### 第二阶段

实现长音频 overlap 分段：

```text
segment_seconds = 4
overlap_seconds = 0.5
transition_frames = 5
```

### 第三阶段

增加可配置参数与日志：

```text
EchoMimicV3 generation mode: full_audio
EchoMimicV3 generation mode: overlap_segments
segment index / total
generated frames count
audio duration
```

## 验收标准

1. 3 到 6 秒短回答播放时，不再出现片段边界跳静态图。
2. 片段之间头部和嘴部动作没有明显瞬间跳变。
3. 播放期间音频和视频同步。
4. 长回答即使分段，也不会出现静态图插入。
5. 日志能明确看到本次使用的是 `full_audio` 还是 `overlap_segments`。

## 最终建议

优先实现：

```text
短句整段音频一次生成
```

这是对当前选项对话场景收益最大的方案。

然后再实现：

```text
长句 overlap 分段 + crossfade
```

这样能在连续性、显存、等待时间之间取得比较稳的平衡。

