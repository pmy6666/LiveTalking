# TTS + Talking Head 模型内部 batch_size=2 推理时间测试方案

更新时间：2026-06-01

本文档按新的最终要求重新定义测试：要测的是 **模型内部真正 `batch_size=2`** 的推理时间，而不是两条样本顺序请求、顺序生成后再相加。

## 1. 最终目标

本次最终要回答的问题是：

```text
当 TTS 模型内部 batch_size=2，Talking Head 模型内部 batch_size=2 时，
两条文本生成两条 talking-head 视频的真实模型批推理时间是多少？
```

核心要求：

| 阶段 | 必须满足 |
|---|---|
| TTS | 同一次 TTS 模型推理中，batch 维度真实为 2 |
| Talking Head | 同一次 Talking Head / LatentSync 模型推理中，batch 维度真实为 2 |
| 结果 | 输出两条音频、两条视频，并记录批推理 wall time |

## 2. 旧结果状态

之前已跑出的结果：

```text
TTS: 9.246s
Talking Head: 103.430s
TTS + Talking Head: 112.676s
```

该结果不是本次目标结果，原因：

1. GPT-SoVITs `/tts` API 一次请求只返回一个 wav；之前实际是两次顺序请求，每次请求参数里设置了 `batch_size=2`。
2. `LatentSync_test/run_latentsync_dongqing_batch.py` 是遍历音频文件逐条调用 LatentSync，不是模型内部 batch。
3. LatentSync 当前 pipeline 代码中存在 `assume batch size = 1`，现状不满足 Talking Head 内部 batch=2。

因此旧结果只能保留为：

```text
顺序执行 baseline，不作为模型内部 batch_size=2 结果。
```

## 3. 严格 batch 定义

### 3.1 错误口径

以下都不能算模型内部 batch_size=2：

| 错误方式 | 原因 |
|---|---|
| 两次 `/tts` 请求耗时相加 | 请求级顺序执行，不是同一次模型 batch |
| 一个脚本循环两条音频 | 外层任务 batch，不是模型 forward batch |
| 两个子进程各生成一个视频 | 进程级并发或顺序，不是同一模型 batch |
| API 参数写 `batch_size=2` 但实际只输入一条文本 | 不能证明模型处理了两个样本 |

### 3.2 正确口径

本次必须满足：

```text
TTS:
  input_texts = [text_1, text_2]
  模型内部 tensor batch 维度 = 2
  output_audios = [audio_1, audio_2]

Talking Head:
  input_videos = [video_1, video_2]
  input_audios = [audio_1, audio_2]
  LatentSync UNet / VAE / audio condition 相关 tensor batch 维度 = 2
  output_videos = [video_out_1, video_out_2]
```

如果使用 classifier-free guidance，LatentSync 某些 UNet 输入 batch 维度可能会变成：

```text
2 * batch_size = 4
```

这种情况下也可以接受，但需要在日志中说明：

```text
effective_sample_batch = 2
unet_forward_batch = 4 because CFG duplicates conditional/unconditional inputs
```

## 4. 固定输入

### 4.1 TTS 参考音频

两条 TTS 都使用同一个参考音频：

```text
LiveTalking/bilibili_downloads/DongQing_6s.wav
```

参考文本：

```text
那种快乐常常像一场梦，电影陪伴我们长大
```

### 4.2 两条测试文本

| id | 文本 |
|---|---|
| `dongqing_batch2_01` | 今天上午阳光很好，我们一起去公园慢慢散步吧。 |
| `dongqing_batch2_02` | 下午会议结束以后，请把今天测试结果整理清楚。 |

### 4.3 Talking Head 视频输入

第一轮内部 batch 测试建议两条样本使用同一个 stage1 视频，便于隔离变量：

```text
LiveTalking/LatentSync_test/api_stage1.mp4
```

即：

```text
video_batch = [api_stage1.mp4, api_stage1.mp4]
audio_batch = [dongqing_batch2_01.wav, dongqing_batch2_02.wav]
```

后续如需测不同人物或不同视频，再扩展为两条不同视频。

## 5. 现有代码差距

### 5.1 GPT-SoVITs

当前 `/tts` API 的 `batch_size` 是推理内部切分文本片段时使用的 batch 参数，但接口返回单个 wav。它不能直接完成：

```text
[text_1, text_2] -> [wav_1, wav_2]
```

因此需要新增或改造一个 TTS 内部 batch 测试入口，满足：

1. 一次传入两个独立 text。
2. 模型内部按 batch=2 推理。
3. 返回或保存两个独立 wav。
4. manifest 中记录 tensor batch 证据和总 wall time。

### 5.2 LatentSync / Talking Head

当前 LatentSync 入口：

```text
third_party/LatentSync/scripts/inference.py
```

只接受单个：

```text
--video_path
--audio_path
--video_out_path
```

当前 pipeline：

```text
third_party/LatentSync/latentsync/pipelines/lipsync_pipeline.py
```

存在明确单样本假设：

```text
assume batch size = 1
```

因此当前 `run_latentsync_dongqing_batch.py` 不能用于本次最终指标，只能作为顺序 baseline。必须新增模型内部 batch 版本，至少支持：

```text
video_paths: [str, str]
audio_paths: [str, str]
video_out_paths: [str, str]
```

并在 UNet denoising loop 中让样本维度真实为 2。

## 6. 推荐实施路线

### 6.1 TTS 内部 batch=2

建议新增脚本：

```text
LiveTalking/scripts/tts_precompute/run_dongqing_internal_batch_size_2_20char.py
```

目标行为：

```text
一次调用 GPT-SoVITs TTS pipeline
输入两个独立文本
内部 batch_size = 2
输出两个独立 wav
```

TTS 文本切分必须使用：

```text
text_split_method = cut0
```

原因是 `cut5` 会继续按逗号、句号等标点拆分文本，导致两条输入被切成 3 条或更多片段，模型内部 batch 不再等价于两条独立样本。

manifest 建议字段：

```json
{
  "stage": "tts",
  "batch_mode": "model_internal",
  "batch_size": 2,
  "is_true_model_batch": true,
  "total_seconds": 0.0,
  "evidence": {
    "input_text_count": 2,
    "observed_batch_size": 2,
    "instrumented_tensor_shapes": []
  },
  "items": [
    {
      "id": "dongqing_batch2_01",
      "wav_path": "...",
      "audio_duration_seconds": 0.0
    },
    {
      "id": "dongqing_batch2_02",
      "wav_path": "...",
      "audio_duration_seconds": 0.0
    }
  ]
}
```

验收标准：

| 检查项 | 必须满足 |
|---|---|
| 一次 TTS 执行 | 是 |
| 独立文本数量 | 2 |
| 独立 wav 输出 | 2 |
| 模型内部 batch 证据 | 有 tensor shape 或 hook log |
| manifest 标注 | `is_true_model_batch=true` |

### 6.2 Talking Head 内部 batch=2

建议新增脚本：

```text
LiveTalking/test/ck_time/run_latentsync_internal_batch_size_2.py
```

建议不要再调用当前单样本 CLI 两次，而是在一个 Python 进程内加载一次模型，并改造/包装 pipeline，使其支持：

```python
pipeline.batch_call(
    video_paths=[
        "LatentSync_test/api_stage1.mp4",
        "LatentSync_test/api_stage1.mp4",
    ],
    audio_paths=[
        "test/ck_time/tts_batch_size_2_dongqing_20char/dongqing_batch2_01.wav",
        "test/ck_time/tts_batch_size_2_dongqing_20char/dongqing_batch2_02.wav",
    ],
    video_out_paths=[
        "test/ck_time/outputs_internal_batch_size_2/<run_id>/dongqing_batch2_01.mp4",
        "test/ck_time/outputs_internal_batch_size_2/<run_id>/dongqing_batch2_02.mp4",
    ],
)
```

需要改造的关键点：

| 模块 | 当前问题 | batch=2 要求 |
|---|---|---|
| `prepare_mask_latents` | `f c h w -> 1 c f h w` 写死 batch=1 | 支持 `b f c h w -> b c f h w` |
| `prepare_image_latents` | 写死 `1 c f h w` | 支持 batch 维 |
| `prepare_latents` | 当前针对单样本 chunk 数 | 支持每个样本各自 chunk，并 pad 到同一长度或按 bucket 分组 |
| `audio_embeds` | 单条音频 chunks | 支持 `b f ...` |
| `faces / masks` | 单视频帧序列 | 支持两个样本的 faces batch |
| restore / mux | 单输出 | 分别还原并写出两个 mp4 |

建议第一版降低复杂度：

1. 两条 TTS 文本控制得尽量等长。
2. 若音频 chunk 数不同，先 pad 到同一 chunk 数。
3. 推理后按原始 chunk 数裁剪各自输出。
4. 同一个 video 可以复用预处理结果，但进入 UNet 前仍要构造 batch=2 tensor。

manifest 建议字段：

```json
{
  "stage": "talking_head",
  "engine": "LatentSync",
  "batch_mode": "model_internal",
  "batch_size": 2,
  "is_true_model_batch": true,
  "total_seconds": 0.0,
  "model_load_seconds": 0.0,
  "preprocess_seconds": 0.0,
  "denoise_seconds": 0.0,
  "postprocess_seconds": 0.0,
  "evidence": {
    "input_sample_count": 2,
    "observed_sample_batch": 2,
    "observed_unet_batch": 4,
    "cfg_enabled": true,
    "instrumented_tensor_shapes": []
  },
  "items": [
    {
      "id": "dongqing_batch2_01",
      "audio_duration_seconds": 0.0,
      "video_duration_seconds": 0.0,
      "output": "..."
    },
    {
      "id": "dongqing_batch2_02",
      "audio_duration_seconds": 0.0,
      "video_duration_seconds": 0.0,
      "output": "..."
    }
  ]
}
```

验收标准：

| 检查项 | 必须满足 |
|---|---|
| 模型只加载一次 | 是 |
| 一次 batch pipeline 调用处理样本数 | 2 |
| UNet denoising 输入体现 batch | sample batch=2，CFG 下 unet batch=4 |
| 输出视频数量 | 2 |
| manifest 标注 | `is_true_model_batch=true` |

## 7. 计时口径

建议分两种口径记录：

| 口径 | 含义 | 最终是否重点汇报 |
|---|---|---|
| cold wall time | 包含模型加载、预处理、推理、后处理 | 辅助 |
| inference wall time | 模型已加载后，单次 batch=2 推理总耗时 | 重点 |

最终核心公式：

```text
tts_internal_batch2_seconds =
    TTS 单次内部 batch=2 推理 wall time

talking_head_internal_batch2_seconds =
    Talking Head 单次内部 batch=2 推理 wall time

tts_plus_talking_head_internal_batch2_seconds =
    tts_internal_batch2_seconds + talking_head_internal_batch2_seconds

avg_seconds_per_result =
    tts_plus_talking_head_internal_batch2_seconds / 2
```

如果模型加载时间也要汇报，单独列：

```text
cold_total_seconds = tts_cold_seconds + talking_head_cold_seconds
```

不要把 cold time 和 warm inference time 混在同一个指标里。

## 8. 输出目录

TTS 内部 batch 输出：

```text
LiveTalking/test/ck_time/tts_internal_batch_size_2_dongqing_20char/
```

Talking Head 内部 batch 输出：

```text
LiveTalking/test/ck_time/outputs_internal_batch_size_2/<run_id>/
```

最终 summary：

```text
LiveTalking/test/ck_time/outputs_internal_batch_size_2/<run_id>/internal_batch_size_2_summary.md
```

## 9. 结果记录模板

```markdown
# TTS + Talking Head 模型内部 batch_size=2 推理时间结果

运行时间：YYYY-MM-DD HH:MM:SS +0800

## 结论

本次结果是否为模型内部 batch_size=2：是 / 否

TTS + Talking Head 内部 batch_size=2 总推理时间：待填 秒

平均每条结果耗时：待填 秒

## 核心结果

| 指标 | 秒 |
|---|---:|
| TTS internal batch_size=2 inference | 待填 |
| Talking Head internal batch_size=2 inference | 待填 |
| TTS + Talking Head internal batch_size=2 | 待填 |
| Average per result | 待填 |

## 内部 batch 证据

| 阶段 | 证据 |
|---|---|
| TTS | 待填，如 `observed_batch_size=2` 和 tensor shapes |
| Talking Head | 待填，如 `observed_sample_batch=2`, `observed_unet_batch=4` |

## 输入

| id | text | ref_audio | talking_head_video |
|---|---|---|---|
| dongqing_batch2_01 | 今天上午阳光很好，我们一起去公园慢慢散步吧。 | DongQing_6s.wav | api_stage1.mp4 |
| dongqing_batch2_02 | 下午会议结束以后，请把今天测试结果整理清楚。 | DongQing_6s.wav | api_stage1.mp4 |

## 输出

| id | audio duration | video duration | audio output | video output |
|---|---:|---:|---|---|
| dongqing_batch2_01 | 待填 | 待填 | 待填 | 待填 |
| dongqing_batch2_02 | 待填 | 待填 | 待填 | 待填 |

## 备注

- 是否包含模型加载：
- 是否启用 DeepCache：
- inference_steps：
- guidance_scale：
- GPU 显存峰值：
- 嘴型同步和画面稳定性：
```

## 10. 实施前检查清单

在开始重新跑测试前，必须先完成：

1. TTS 入口支持两个独立文本在同一次模型 batch 中推理。
2. TTS manifest 能证明 `observed_batch_size=2`。
3. LatentSync pipeline 去除 `assume batch size = 1` 的限制。
4. LatentSync UNet denoising loop 支持 sample batch=2。
5. Talking Head manifest 能证明 sample batch=2，CFG 下 UNet batch 可为 4。
6. 旧的顺序执行结果不能写入最终“内部 batch_size=2”结论。
