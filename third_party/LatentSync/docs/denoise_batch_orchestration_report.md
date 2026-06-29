# LatentSync Denoise Batch Orchestration Report

生成时间：2026-06-11

## 1. 结论摘要

当前 `LatentSync` 的 `batch_inference()` 确实实现了样本维度的 GPU batch：同一 group、同一 16 帧时间窗口、同一 timestep 下，多个样本会被合成一个更大的 `unet_input`，只调用一次 `self.unet(...)`。因此它不是 Python 层逐样本调用 UNet 的伪 batch。

但是 denoise 阶段仍然有两层必须串行的循环：

- 按 group 串行：`groups = [samples[start:start + batch_size] ...]`
- 按视频 chunk 串行：每 `num_frames=16` 帧一个 chunk
- 按 diffusion timestep 串行：每个 chunk 跑 `num_inference_steps` 次 scheduler/UNet，后一步依赖前一步 latents

batch 并行只发生在“单个 group 的单个 chunk 的单个 timestep”内部。因为 UNet 前向的计算量、显存读写、attention/conv 规模会随 batch 增大，墙钟时间不一定接近 batch=1；你观察到的 batch=1 约 90s、batch=2 约 180s、batch=3 约 270s，从现有 profile 看主要是 denoise UNet 计算近似线性增长，而不是缓存/预处理没有生效。

## 2. CLI 是否真的进入多样本 batch

入口在 `scripts/inference.py`：

- `--video_paths`、`--audio_paths`、`--video_out_paths` 是多样本 batch 参数，逗号分隔。
- 如果没有提供这三个复数参数，代码会把单数参数包装成长度为 1 的列表。
- 分支条件是：
  - `len(audio_paths) == 1 and args.batch_size == 1`：走单样本 `pipeline(...)`
  - 其他情况：走 `pipeline.batch_inference(...)`

一个容易误判的点：

- 如果只传 `--video_path/--audio_path/--video_out_path --batch_size 2`，代码会进入 `batch_inference()`，但 `video_paths/audio_paths/video_out_paths` 长度仍然是 1。
- 此时 `groups` 里只有一个 sample，`sample_batch = 1`，实际 UNet batch 仍然是 1。
- 真正的 batch=2/3 需要使用 `--video_paths a,b --audio_paths c,d --video_out_paths o1,o2` 这类复数参数，或者一个 video 搭配多个 audio 时使用 `--video_paths single --audio_paths a,b,c`，代码会把单 video 复制到多个 audio。

## 3. 单样本 denoise 编排

位置：`latentsync/pipelines/lipsync_pipeline.py::__call__()`

核心流程：

1. 计算 `whisper_chunks`，读取 audio/video。
2. `loop_video()` 按音频 chunk 数裁剪或循环视频，并做 face affine transform。
3. `prepare_latents(len(whisper_chunks), ...)` 生成 `all_latents`，形状为：

```text
[1, latent_channels=4, total_chunks, latent_h, latent_w]
```

4. `num_inferences = ceil(len(whisper_chunks) / num_frames)`，默认 `num_frames=16`。
5. 对每个 16 帧 chunk：
   - `audio_embeds`: `[16, seq, dim]`，若 CFG 打开则拼成 `[32, seq, dim]`。
   - `latents`: `[1, 4, 16, latent_h, latent_w]`
   - `mask_latents`: `[1, 1, 16, latent_h, latent_w]`
   - `masked_image_latents`: `[1, 4, 16, latent_h, latent_w]`
   - `ref_latents`: `[1, 4, 16, latent_h, latent_w]`
6. 对每个 diffusion timestep：
   - CFG 关闭时，`unet_input` 最终 channel 拼接后为 `[1, 13, 16, latent_h, latent_w]`
   - CFG 开启时，batch 维翻倍为 `[2, 13, 16, latent_h, latent_w]`
   - `noise_pred = self.unet(unet_input, t, encoder_hidden_states=audio_embeds).sample`
   - `latents = self.scheduler.step(noise_pred, t, latents, ...).prev_sample`
7. chunk 结束后 VAE decode、paste、append 到输出帧列表。

单样本中 chunk 和 timestep 都是串行的，不能通过 `batch_size` 参数改变。

## 4. 多样本 batch denoise 编排

位置：`latentsync/pipelines/lipsync_pipeline.py::batch_inference()`

### 4.1 样本准备

所有输入样本先逐个预处理：

- `audio2feat()`、`feature2chunks()` 得到每个样本的 `whisper_chunks`。
- `get_prepared_video_cache()` 按视频绝对路径缓存原视频帧、faces、boxes、affine matrices。
- `loop_prepared_video()` 根据当前音频长度裁剪/循环视频。

这一步目前是 Python for-loop 串行执行，不是 denoise 计算本体。

### 4.2 group 切分

```text
groups = [samples[start:start + batch_size] for start in range(0, len(samples), batch_size)]
```

- 如果总样本数等于 batch_size，就只有一个 group。
- 如果总样本数大于 batch_size，会多个 group 串行跑。
- 最后一个 group 可能小于 batch_size，实际 `sample_batch = len(group)`。

### 4.3 每个 group 的 latents

对每个 group：

```text
max_chunks = max(len(sample["whisper_chunks"]) for sample in group)
all_latents = prepare_batch_latents(sample_batch, max_chunks, ...)
```

`all_latents` 形状：

```text
[sample_batch, 4, max_chunks, latent_h, latent_w]
```

这里使用 group 内最长音频长度作为时间轴。短样本会在每个 chunk 中被 pad，并在写回 `synced_chunks` 时用 `valid_len` 丢弃超出真实长度的帧。

### 4.4 每个 16 帧 chunk 的 batch 拼接

对每个 chunk：

- 对 group 内每个 sample 取 `start:start+num_frames` 的 audio chunk，不足 16 帧则补零。
- `get_chunk_vae_cache()` 取或生成当前 sample 当前 chunk 的 VAE latents。
- 将 group 内样本合并：

```text
audio_embeds:              [B, 16, seq, dim]
mask_latents:              [B, 1, 16, H, W]
masked_image_latents:      [B, 4, 16, H, W]
ref_latents:               [B, 4, 16, H, W]
latents:                   [B, 4, 16, H, W]
```

如果 `guidance_scale > 1.0`，classifier-free guidance 会把 batch 维翻倍：

```text
unet_input before channel concat: [2B, 4, 16, H, W]
mask/ref/masked latents:          [2B, ..., 16, H, W]
audio_embeds:                     [2B, 16, seq, dim]
```

当前你 profile 中 `guidance_scale=1.0`，因此 CFG 关闭，实际 UNet batch 就是 B。

### 4.5 timestep denoise 循环

核心循环：

```text
for t in timesteps:
    unet_input = latents
    unet_input = scheduler.scale_model_input(unet_input, t)
    unet_input = cat([unet_input, mask_latents, masked_image_latents, ref_latents], dim=1)
    noise_pred = unet(unet_input, t, encoder_hidden_states=audio_embeds).sample
    latents = scheduler.step(noise_pred, t, latents).prev_sample
```

对 `guidance_scale=1.0`、`stage2_512.yaml` 来说：

```text
latents:     [B, 4, 16, 64, 64]
mask:        [B, 1, 16, 64, 64]
masked img:  [B, 4, 16, 64, 64]
ref:         [B, 4, 16, 64, 64]
unet_input:  [B, 13, 16, 64, 64]
noise_pred:  [B, 4, 16, 64, 64]
```

这一层没有按 sample 写 Python 循环调用 UNet；B 个样本在一次 `self.unet(...)` 中前向。

## 5. UNet 内部 batch 是否被拆开

位置：`latentsync/models/unet.py`、`latentsync/models/resnet.py`、`latentsync/models/attention.py`

结论：UNet 内部保留了 batch 维参与一次大前向，没有显式按样本 for-loop。

关键点：

- `UNet3DConditionModel.forward()` 中 timestep 会 `expand(sample.shape[0])`，这里的 `sample.shape[0]` 就是 B 或 CFG 后的 2B。
- `InflatedConv3d.forward()` 会把 `b c f h w` reshape 成 `(b f) c h w` 后跑 2D conv，再 reshape 回来。
- `Transformer3DModel.forward()` 也会把 `b c f h w` reshape 成 `(b f) c h w`，再变成 `(b f) seq dim`。
- `BasicTransformerBlock.forward()` 中如果 `encoder_hidden_states` 是 `[B, F, S, D]`，会 reshape 成 `(B F) S D`，与视觉 token 对齐。
- attention 使用 `torch.nn.functional.scaled_dot_product_attention()`，不是手写逐样本 attention 循环。

所以 denoise 的 batch 是“真实的大张量 batch”，不是 `for sample in batch: unet(sample)`。

## 6. 现有 profile 数据解读

当前目录中已有 profile summary：

### 6.1 单样本

`temp_batch2_same_video_profile/profile_summary.json`

```text
mode: single
batch_size: 1
total: 1
preprocess: 4.201s
denoise: 62.810s
postprocess: 23.823s
total_seconds: 90.834s
gpu_peak_memory_mb: 10937 MB
```

### 6.2 batch=3，相同 video/audio

`temp_batch3_same_video_audio_profile/profile_summary.json`

```text
mode: gpu_batch
batch_size: 3
total: 3
preprocess: 3.984s
denoise: 196.677s
postprocess: 69.349s
total_seconds: 270.010s
gpu_peak_memory_mb: 21228 MB
cache.video_hits: 2
cache.video_misses: 1
cache.vae_hits: 32
cache.vae_misses: 16
```

解读：

- `video_hits=2` 说明相同视频的预处理缓存命中。
- `vae_hits=32`、`vae_misses=16` 说明 chunk 级 VAE 缓存也在工作。
- preprocess 没有乘以 3，说明预处理不是主要问题。
- denoise 从 62.8s 到 196.7s，约 3.13 倍，主要耗时增长来自 batch=3 的 UNet denoise 前向。
- postprocess 从 23.8s 到 69.3s，约 2.9 倍，restore/write/ffmpeg 是逐输出串行处理，线性增长正常。

### 6.3 batch=2，不同 demo

`temp_batch2_profile/profile_summary.json`

```text
mode: gpu_batch
batch_size: 2
total: 2
preprocess: 10.615s
denoise: 261.539s
postprocess: 95.833s
total_seconds: 367.988s
gpu_peak_memory_mb: 19279 MB
```

这个 profile 的 denoise 比 batch=3 相同 demo 更长，可能因为 group 内 `max_chunks` 由更长的样本决定。`batch_inference()` 按 group 最大音频长度跑 chunk，短样本被 pad；因此不同长度样本混 batch 时，短样本会陪最长样本一起跑完整 `max_chunks`，这会浪费 denoise 计算。

## 7. 为什么 batch 增大后不像理想并行

主要原因如下：

1. Diffusion timestep 天然串行：第 t 步的 latents 依赖第 t+1 步输出，不能把 20 个 timestep 并行成 1 个。
2. Chunk 天然串行：当前实现每 16 帧一组，逐 chunk 生成并 decode。
3. Batch 只减少了 Python 调度次数和部分固定开销，没有减少 UNet 总 FLOPs。
4. UNet 内部大部分卷积/attention 在 `(B * F)` 维度上计算，B 增大基本意味着每层处理更多 token/feature map。
5. 512 分辨率下 latent 是 `64x64`，`B=3,F=16` 已经是较大的前向；GPU 可能受显存带宽、attention kernel、cache locality 或 occupancy 限制，吞吐无法随着 B 增大明显提升。
6. `batch_inference()` 每个 chunk 之后调用 `torch.cuda.empty_cache()`，这可能增加额外同步/allocator 开销。它不是线性增长的根因，但会让 batch 场景更不干净。
7. 不同音频长度混 batch 时使用 `max_chunks`，短样本 pad 后仍参与部分无效计算。
8. postprocess 当前逐输出串行 restore、write video、ffmpeg mux，会随输出个数线性增长。

## 8. 当前 batch 实现的有效点

实现中已经有几个有价值的优化：

- 相同 `video_path` 会复用 video affine transform 结果。
- 相同视频、相同 chunk 会复用 VAE 编码结果。
- 多样本同 chunk/timestep 会合成一次 UNet forward。
- `guidance_scale=1.0` 时不会触发 CFG batch 翻倍，当前 profile 已经是比较省算力的设置。

## 9. 排查建议

如果要进一步确认瓶颈，可以按下面顺序做小实验：

1. 确认真实 batch：profile 里 `total` 必须等于输入样本数，progress bar 描述应是 `Doing batch inference x2...` 或 `x3...`，不是 `x1...`。
2. 固定相同 video/audio、相同 seed、相同 steps，分别跑 batch=1/2/3，并只比较 `phases.denoise`。
3. 输出每个 group 的 `sample_batch`、`max_chunks`、`num_inferences`，排除不同音频长度导致的 pad 浪费。
4. 用 `torch.profiler` 或 Nsight Systems 只包住 denoise loop，查看 UNet 前向、scheduler、allocator、empty_cache 的占比。
5. 临时去掉 chunk 末尾 `torch.cuda.empty_cache()` 做 A/B，看 allocator 开销是否明显。
6. 按音频长度排序/分桶后再 batch，减少 group 内 `max_chunks` pad 浪费。

## 10. 优化方向

可考虑的工程优化：

- 对输入按 `len(whisper_chunks)` 分桶，长度相近的样本组成 batch，降低 pad 浪费。
- 对完全相同 audio 的 `whisper_chunks` 做缓存，当前 video/VAE 有缓存，但 audio feature 仍逐样本算。
- 避免每个 chunk 调 `torch.cuda.empty_cache()`，改成异常/OOM 后再清，或只在 group 结束后清理。
- postprocess 可并行化到 CPU 多进程或线程池，尤其 restore/write/ffmpeg 对多输出是线性串行。
- 如果显存允许，评估是否能把多个 chunk 合并到更大的时间维或样本维中，但这会改变 motion module/temporal attention 的上下文假设，风险比单纯 batch 样本更高。
- 使用 profiler 验证 DeepCache 是否真的支持该自定义 pipeline/3D UNet。当前 `inference.sh` 开了 `--enable_deepcache`，但 profile 文件中 `enable_deepcache=false`，需要单独验证。

## 11. 最终判断

当前代码的 denoise batch 编排是“真实 batch，但不是免费并行”。如果 batch=2/3 的输入确实是复数路径，那么线性耗时主要来自 UNet 前向计算随 batch 增长，而不是代码在 denoise 内部逐样本串行调用 UNet。更值得优化的是：减少无效 pad、减少 `empty_cache()` 同步开销、缓存重复 audio feature、并行 postprocess，以及通过 profiler 找出 batch 下 UNet kernel 的具体瓶颈。

