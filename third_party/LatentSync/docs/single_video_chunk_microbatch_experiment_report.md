# LatentSync Single-Video Chunk Micro-Batch 推理优化实验技术报告

生成时间：2026-06-16

## 1. 结论摘要

本报告研究一种不需要重新训练的 LatentSync 推理优化方案：将同一个视频内部多个相互独立的 16-frame chunk 合并到 batch 维度，在每个 diffusion timestep 上一次性送入 UNet。

核心结论：

- 该方案不改变模型权重、不改变 UNet 结构、不改变 denoise 数学过程，因此不需要重新训练。
- 该方案不是把连续 chunk 拼到帧维 `F`，而是拼到 batch 维 `B`。
- 同一个 chunk 内的 diffusion timestep 仍然必须串行，不能并行。
- 不同 chunk 之间没有 hidden state 依赖，具备 micro-batch 推理可行性。
- 该方案的主要收益来自提高 GPU 利用率、减少 Python 循环调度开销、扩大单次 UNet 前向的有效 batch。
- 主要风险是显存峰值上升、最后一组 chunk padding 处理、随机 latent 对齐、输出顺序恢复。

推荐实验路径：

1. 保持当前单视频 `pipeline.__call__()` 行为作为 baseline。
2. 新增实验性参数，例如 `chunk_batch_size`。
3. 将连续的多个 16-frame chunk 合并为 micro-batch。
4. 比较 `chunk_batch_size = 1, 2, 4, 8` 的总耗时、denoise 耗时、GPU 峰值显存和输出一致性。

## 2. 背景与问题定义

当前单视频推理路径位于：

```text
latentsync/pipelines/lipsync_pipeline.py::__call__()
```

当前 denoise 编排可以抽象为：

```text
for chunk_index in num_inferences:
    prepare audio / mask / ref latents for this 16-frame chunk

    for timestep in timesteps:
        unet_input = concat(latents, mask, masked_image_latents, ref_latents)
        noise_pred = UNet(unet_input, timestep, audio_embeds)
        latents = scheduler.step(noise_pred, timestep, latents)

    decode this chunk
```

其中 `num_frames` 默认是 16。每个 chunk 的输入形状大致为：

```text
latents:              [1, 4, 16, H, W]
mask_latents:         [1, 1, 16, H, W]
masked_image_latents: [1, 4, 16, H, W]
ref_latents:          [1, 4, 16, H, W]
unet_input:           [1, 13, 16, H, W]
audio_embeds:         [16, S, D]
```

如果开启 classifier-free guidance，进入 UNet 前 batch 维会翻倍：

```text
unet_input:   [2, 13, 16, H, W]
audio_embeds: [32, S, D] 或等价的 batch-folded 表达
```

问题是：单视频内部 chunk 目前按 Python for-loop 串行处理。每个 chunk 都完整跑一次所有 diffusion timesteps。对于长视频，UNet 调用次数为：

```text
num_chunks * num_inference_steps
```

如果 `num_chunks = 10`，`num_inference_steps = 20`，则需要 200 次 UNet forward。micro-batch 方案希望把多个 chunk 合在一起，使 UNet 调用次数变为：

```text
ceil(num_chunks / chunk_batch_size) * num_inference_steps
```

例如 `chunk_batch_size = 4` 时，理论 UNet 调用次数约降为原来的 1/4。实际墙钟加速不会线性等于 4 倍，因为单次 UNet batch 变大后计算量也增加，但 GPU 吞吐通常会更好。

## 3. 模型结构可行性分析

### 3.1 Chunk 之间是否有模型依赖

LatentSync UNet 的视频张量约定为：

```text
(B, C, F, H, W)
```

其中：

- `B`：batch，或 classifier-free guidance 后的 batch。
- `C`：通道。
- `F`：当前 chunk 内帧数，默认 16。
- `H/W`：latent 空间分辨率。

motion module 的 temporal attention 只在当前输入的 `F` 帧内部计算。代码中 temporal attention 会将 token 维和 frame 维重排：

```text
(b f) s c -> (b s) f c
```

这意味着 temporal layer 关注的是单个样本内部的 `f` 维。只要多个 chunk 被放到 batch 维 `B`，它们不会在 temporal attention 内互相混合。

因此：

```text
正确做法:
[chunk0, chunk1, chunk2] -> batch 维 B

错误做法:
chunk0 + chunk1 + chunk2 -> 帧维 F
```

错误地拼到 `F` 维会让 temporal attention 跨 chunk 建模，改变模型行为，可能影响唇形连续性和质量。

### 3.2 Diffusion timestep 是否可以并行

不能。

同一个 chunk 的 denoise 是递推过程：

```text
x_t -> x_(t-1) -> x_(t-2) -> ... -> x_0
```

`scheduler.step()` 的输出 `prev_sample` 是下一步 UNet 的输入。因此同一 chunk 内的 timestep 必须严格串行。

micro-batch 方案并行的是多个 chunk 在同一个 timestep 上的 UNet forward：

```text
timestep k:
    [chunk0.x_k, chunk1.x_k, chunk2.x_k] -> UNet -> [chunk0.noise, chunk1.noise, chunk2.noise]
    scheduler.step 分别更新每个 chunk
```

### 3.3 是否需要重新训练

不需要。

原因：

- 训练时模型本身已经支持 batch 维。
- UNet、attention、motion module 都以 batch 维作为独立样本维处理。
- micro-batch 只改变推理调度，不改变输入语义。
- 每个样本内部仍然是原来的 16 帧 chunk。

唯一需要保证的是：micro-batch 后的张量布局必须和模型期望一致。

## 4. 实验目标

本实验目标不是直接替换当前推理流程，而是验证 single-video chunk micro-batch 是否能在保持质量一致的前提下降低推理时间。

实验要回答以下问题：

1. `chunk_batch_size > 1` 是否能降低总推理时间？
2. denoise 阶段是否得到明显加速？
3. GPU 峰值显存增加多少？
4. 输出视频与 baseline 是否保持可接受一致？
5. 最优 `chunk_batch_size` 在当前 GPU 上是多少？

## 5. 实验变量设计

### 5.1 自变量

```text
chunk_batch_size: 1, 2, 4, 8
```

其中：

- `1` 是当前 baseline。
- `2` 是低风险实验值。
- `4` 是推荐重点测试值。
- `8` 可能显存压力较大，需要视 GPU 而定。

### 5.2 控制变量

所有实验保持以下参数一致：

```text
unet_config_path: configs/unet/stage2_512.yaml
inference_ckpt_path: checkpoints/latentsync_unet.pt
video_path: assets/demo1_video.mp4
audio_path: assets/demo1_audio.wav
num_frames: 16
inference_steps: 20
guidance_scale: 1.5
seed: 1247
resolution: 512
```

### 5.3 观测指标

性能指标：

```text
total_seconds
preprocess_seconds
denoise_seconds
postprocess_seconds
gpu_peak_memory_mb
num_unet_forward_calls
avg_unet_forward_ms
```

质量指标：

```text
输出视频是否成功生成
输出帧数量是否一致
音频长度是否一致
肉眼检查唇形同步
可选: SyncNet confidence
可选: pixel/latent 差异统计
```

稳定性指标：

```text
是否 OOM
是否出现 NaN/Inf
最后一个 chunk padding 是否正确丢弃
输出顺序是否正确
```

## 6. 具体技术方案

### 6.1 Baseline 单 chunk 流程

当前单样本流程对每个 chunk 独立处理：

```text
for chunk_idx:
    latents = all_latents[:, :, start:end]
    audio_embeds = whisper_chunks[start:end]
    mask_latents = encode mask for this chunk
    masked_image_latents = encode masked frames
    ref_latents = encode ref frames

    for timestep:
        run UNet once
```

### 6.2 Micro-batch 改造后的流程

将多个 chunk 按 `chunk_batch_size` 分组：

```text
chunk_groups = [
    [chunk0, chunk1, chunk2, chunk3],
    [chunk4, chunk5, chunk6, chunk7],
    ...
]
```

对每个 chunk group：

1. 分别准备每个 chunk 的 audio/mask/ref latents。
2. 沿 batch 维合并。
3. 对该 group 执行完整 denoise loop。
4. denoise 完成后按原 chunk 顺序拆分。
5. decode 并 append 到输出列表。

核心张量变化：

```text
原始单 chunk:
latents:              [1, 4, 16, H, W]
mask_latents:         [1, 1, 16, H, W]
masked_image_latents: [1, 4, 16, H, W]
ref_latents:          [1, 4, 16, H, W]
audio_embeds:         [16, S, D]

micro-batch, M 个 chunk:
latents:              [M, 4, 16, H, W]
mask_latents:         [M, 1, 16, H, W]
masked_image_latents: [M, 4, 16, H, W]
ref_latents:          [M, 4, 16, H, W]
audio_embeds:         [M, 16, S, D]
```

CFG 开启后：

```text
unet_input:   [2M, 13, 16, H, W]
audio_embeds: [2M, 16, S, D]
```

UNet 输出：

```text
noise_pred raw: [2M, 4, 16, H, W]
noise_pred CFG: [M, 4, 16, H, W]
latents next:   [M, 4, 16, H, W]
```

### 6.3 最后一个 chunk 的 padding 策略

如果视频总帧 chunk 数不是 `chunk_batch_size` 的整数倍，最后一组可能小于 M。

推荐策略：

```text
实际 M = len(current_chunk_group)
不强行 pad 到固定 chunk_batch_size
直接使用较小 batch 推理
```

如果某个 chunk 内不足 16 帧，则沿用当前逻辑：

```text
pad 到 16 帧参与 UNet
decode 后只保留 valid_len
```

### 6.4 随机 latent 对齐

为了验证输出一致性，必须保证 baseline 和 micro-batch 使用同一组初始 latents。

推荐方式：

```text
先一次性生成 all_latents: [1, 4, total_frames, H, W]
micro-batch 时从 all_latents 中切片
每个 chunk 的初始 latent 与 baseline 完全对应
```

不要在每个 chunk group 内重新调用随机生成，否则即使调度等价，输出也会因为初始噪声不同而无法对齐。

### 6.5 Decode 策略

有两个选择：

方案 A：每个 group denoise 后立刻 decode。

```text
group latents: [M, 4, 16, H, W]
reshape -> [M*16, 4, H, W]
VAE decode
reshape -> [M, 16, C, H_img, W_img]
按 chunk 顺序 append
```

方案 B：所有 group denoise 完成后统一 decode。

推荐先采用方案 A，因为：

- 显存峰值更低。
- 更接近当前流程。
- 便于定位错误。

## 7. 伪代码设计

以下是实验实现时的核心伪代码，仅作为技术方案说明：

```python
num_inferences = ceil(len(whisper_chunks) / num_frames)

for group_start in range(0, num_inferences, chunk_batch_size):
    group_chunk_ids = range(group_start, min(group_start + chunk_batch_size, num_inferences))

    batch_audio_embeds = []
    batch_mask_latents = []
    batch_masked_image_latents = []
    batch_ref_latents = []
    batch_latents = []
    batch_valid_lens = []
    batch_ref_pixels = []
    batch_masks = []

    for chunk_id in group_chunk_ids:
        start = chunk_id * num_frames
        end = min((chunk_id + 1) * num_frames, len(whisper_chunks))
        valid_len = end - start

        audio_embeds = slice_or_pad_audio(whisper_chunks, start, num_frames)
        ref_pixel_values, masked_pixel_values, masks = prepare_masks(...)
        mask_latents, masked_image_latents = prepare_mask_latents(...)
        ref_latents = prepare_image_latents(...)
        latents = all_latents[:, :, start:start + num_frames]
        latents = pad_to_16_if_needed(latents)

        batch_audio_embeds.append(audio_embeds)
        batch_mask_latents.append(mask_latents)
        batch_masked_image_latents.append(masked_image_latents)
        batch_ref_latents.append(ref_latents)
        batch_latents.append(latents)
        batch_valid_lens.append(valid_len)

    latents = cat(batch_latents, dim=0)
    audio_embeds = stack(batch_audio_embeds, dim=0)
    mask_latents = cat(batch_mask_latents, dim=0)
    masked_image_latents = cat(batch_masked_image_latents, dim=0)
    ref_latents = cat(batch_ref_latents, dim=0)

    if do_classifier_free_guidance:
        audio_embeds = cat([zeros_like(audio_embeds), audio_embeds], dim=0)
        mask_latents = cat([mask_latents, mask_latents], dim=0)
        masked_image_latents = cat([masked_image_latents, masked_image_latents], dim=0)
        ref_latents = cat([ref_latents, ref_latents], dim=0)

    for t in timesteps:
        unet_input = cat([latents, latents], dim=0) if cfg else latents
        unet_input = scheduler.scale_model_input(unet_input, t)
        unet_input = cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)

        noise_pred = unet(unet_input, t, encoder_hidden_states=audio_embeds).sample

        if cfg:
            noise_uncond, noise_audio = noise_pred.chunk(2)
            noise_pred = noise_uncond + guidance_scale * (noise_audio - noise_uncond)

        latents = scheduler.step(noise_pred, t, latents).prev_sample

    decoded = decode_latents(latents)
    split_and_append(decoded, batch_valid_lens)
```

## 8. 正确性验证方案

### 8.1 Shape 验证

对 `chunk_batch_size = 2`，期望关键 shape 为：

```text
latents before CFG: [2, 4, 16, H, W]
unet_input after CFG and concat: [4, 13, 16, H, W]
audio_embeds after CFG: [4, 16, S, D]
noise_pred raw: [4, 4, 16, H, W]
noise_pred after CFG: [2, 4, 16, H, W]
decoded frames: [2, 16, 3, resolution, resolution]
```

对 `chunk_batch_size = 4`，期望：

```text
unet_input after CFG and concat: [8, 13, 16, H, W]
```

### 8.2 数值一致性验证

如果希望严格比较 baseline 与 micro-batch，需要满足：

- 固定 `seed`。
- 使用相同 `all_latents` 切片。
- 关闭会改变执行路径的非确定性优化。
- 使用相同 dtype。
- 使用相同 `guidance_scale` 和 timesteps。

由于 GPU attention/conv 在不同 batch size 下可能存在非完全 bitwise 一致，建议使用容忍阈值：

```text
latent max_abs_diff < 1e-3 或按 fp16 误差重新设定
pixel mean_abs_diff 可接受
肉眼输出无明显差异
```

### 8.3 输出顺序验证

必须验证：

```text
chunk0 output -> synced_video_frames[0:16]
chunk1 output -> synced_video_frames[16:32]
chunk2 output -> synced_video_frames[32:48]
```

不能因为 batch split 顺序错误导致视频片段错位。

## 9. 性能实验设计

### 9.1 Baseline 命令

```bash
cd /home/qianustb/LiveTalking/third_party/LatentSync

/home/qianustb/envs/livetalking/bin/python -m scripts.inference \
  --unet_config_path configs/unet/stage2_512.yaml \
  --inference_ckpt_path checkpoints/latentsync_unet.pt \
  --video_path assets/demo1_video.mp4 \
  --audio_path assets/demo1_audio.wav \
  --video_out_path temp_microbatch_exp/baseline.mp4 \
  --inference_steps 20 \
  --guidance_scale 1.5 \
  --temp_dir temp_microbatch_exp/baseline_temp \
  --profile_summary_path temp_microbatch_exp/baseline_profile.json \
  --seed 1247
```

### 9.2 实验命令模板

未来实现 `--chunk_batch_size` 参数后，建议命令如下：

```bash
/home/qianustb/envs/livetalking/bin/python -m scripts.inference \
  --unet_config_path configs/unet/stage2_512.yaml \
  --inference_ckpt_path checkpoints/latentsync_unet.pt \
  --video_path assets/demo1_video.mp4 \
  --audio_path assets/demo1_audio.wav \
  --video_out_path temp_microbatch_exp/chunk_bs4.mp4 \
  --inference_steps 20 \
  --guidance_scale 1.5 \
  --temp_dir temp_microbatch_exp/chunk_bs4_temp \
  --profile_summary_path temp_microbatch_exp/chunk_bs4_profile.json \
  --seed 1247 \
  --chunk_batch_size 4
```

### 9.3 结果记录表

| chunk_batch_size | total_seconds | preprocess | denoise | postprocess | GPU peak MB | success | 备注 |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 1 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | baseline |
| 2 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 低风险 |
| 4 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 推荐重点 |
| 8 | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 | 可能 OOM |

### 9.4 加速比计算

```text
denoise_speedup = baseline_denoise_seconds / experiment_denoise_seconds
total_speedup = baseline_total_seconds / experiment_total_seconds
```

由于 preprocess/postprocess 没有完全被 micro-batch 优化，`total_speedup` 通常小于 `denoise_speedup`。

## 10. 显存估算与风险

设 `M = chunk_batch_size`。

主要张量近似随 `M` 线性增长：

```text
latents:              M * [4, 16, H, W]
mask_latents:         M * [1, 16, H, W]
masked_image_latents: M * [4, 16, H, W]
ref_latents:          M * [4, 16, H, W]
UNet activations:     M * 当前单 chunk activation
CFG:                  再乘约 2
```

风险：

- `chunk_batch_size=8` 在 512 配置下可能 OOM。
- CFG 打开时 UNet 实际 batch 为 `2M`。
- attention activation 也会随 batch 增长。
- 如果同时缓存多个 chunk 的 VAE latent，显存/CPU 内存都可能上升。

缓解策略：

- 从 `chunk_batch_size=2` 开始。
- 每个 group denoise 完立即 decode 和释放中间张量。
- 使用 `torch.cuda.empty_cache()` 作为实验阶段观察工具，不作为性能优化核心。
- 记录 `torch.cuda.max_memory_allocated()`。

## 11. 与现有 batch_inference 的关系

当前 `batch_inference()` 已经实现了多输入样本的 batch。它的 batch 维来自多个视频/音频样本：

```text
[sample0 chunk k, sample1 chunk k, sample2 chunk k]
```

本报告提出的 single-video chunk micro-batch 是另一种 batch 来源：

```text
[same video chunk0, same video chunk1, same video chunk2]
```

两者可以抽象成同一种机制：

```text
batch item = one independent 16-frame chunk
```

长期看，可以把两种路径统一成一个通用 chunk scheduler：

```text
sample_id, chunk_id -> batch item
```

这样既能支持多样本 batch，也能支持单样本多 chunk micro-batch。

## 12. 推荐实现阶段划分

### Phase 1：最小实验实现

目标：验证速度与质量。

实现范围：

- 只改 `pipeline.__call__()`。
- 新增 `chunk_batch_size` 参数。
- 保持当前 `batch_inference()` 不变。
- 不引入异步队列。
- 不做 CPU/GPU pipeline。

验收条件：

- `chunk_batch_size=1` 输出与当前 baseline 完全一致或近似一致。
- `chunk_batch_size=2/4` 可成功生成视频。
- profile JSON 中 denoise 时间下降。

### Phase 2：复用 batch_inference 的 cache 思路

目标：减少重复 VAE encode 和预处理。

可复用当前已有函数：

```text
get_chunk_vae_cache()
slice_or_pad_chunks()
prepare_batch_latents()
```

注意：当前这些函数主要服务多样本 batch，需要适配单样本多 chunk。

### Phase 3：统一 chunk scheduler

目标：形成一个通用调度器。

抽象：

```text
ChunkTask:
    sample_id
    chunk_id
    start
    valid_len
    audio_embeds
    ref_pixel_values
    masks
    mask_latents
    masked_image_latents
    ref_latents
    latents
```

调度器负责：

```text
collect tasks -> make micro-batch -> denoise -> split -> decode/restore
```

## 13. 预期结果

在 GPU 显存允许的情况下，预期：

- `chunk_batch_size=2`：denoise 时间有较稳定下降，显存约增加到接近 2 倍但不会严格线性。
- `chunk_batch_size=4`：可能是性价比较高的点。
- `chunk_batch_size=8`：可能受显存或 attention activation 限制。

实际加速取决于：

- GPU 型号。
- 当前 batch=1 时 GPU 利用率。
- 分辨率 256/512。
- inference steps 数量。
- 是否启用 DeepCache。
- CFG scale 是否大于 1。

## 14. 最终建议

建议优先进行 `chunk_batch_size=2` 和 `chunk_batch_size=4` 实验。该方案符合模型结构，不需要重新训练，风险主要集中在工程调度和显存。

不建议优先做 UNet down/mid/up 级 pipeline parallel，因为当前 UNet 内部存在大量 skip connection，且单 chunk 内 timestep 强依赖，工程复杂度高于收益预期。

推荐最终实验判定标准：

```text
如果 chunk_batch_size=4 的 denoise_speedup >= 1.4x
且输出视频无明显质量下降
且 GPU peak memory 在可接受范围内
则该方案值得进入实现阶段。
```

