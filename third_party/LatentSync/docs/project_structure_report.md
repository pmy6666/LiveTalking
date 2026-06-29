# LatentSync Project Structure Report

生成时间：2026-06-11

## 1. 总体定位

`LiveTalking/third_party/LatentSync/` 是一个完整的第三方 LatentSync 工程副本，包含推理、训练、数据预处理、评估、模型定义、配置、权重、demo 资产以及本地 profiling 临时输出。项目核心思路是：用 Whisper 提取音频特征，将音频 embedding 通过 cross-attention 注入 3D UNet；UNet 输入由 noised latents、mask、masked frame latents、reference frame latents 按 channel 维拼接；denoise 后经 VAE decode，再把嘴部区域贴回原视频。

## 2. 顶层目录和文件

- `README.md`：官方项目说明，包含方法概述、环境安装、推理、数据处理、训练、评估。
- `inference.sh`：命令行推理示例，当前使用 `configs/unet/stage2_512.yaml`、`checkpoints/latentsync_unet.pt`、`--enable_deepcache`、单视频单音频参数。
- `train_unet.sh`：`torchrun -m scripts.train_unet` 入口，默认配置 `configs/unet/stage1_512.yaml`。
- `train_syncnet.sh`：`torchrun -m scripts.train_syncnet` 入口，默认配置 `configs/syncnet/syncnet_16_pixel_attn.yaml`。
- `data_processing_pipeline.sh`：数据预处理 pipeline 的 shell 入口。
- `gradio_app.py`：Gradio 推理界面。
- `predict.py`、`cog.yaml`：Cog/Replicate 推理接口。
- `requirements.txt`、`setup_env.sh`：Python 依赖和权重下载脚本。
- `.git/`：LatentSync 子仓库自己的 git 元数据。

## 3. 配置与权重

- `configs/scheduler_config.json`：Diffusers DDIM scheduler 配置，由 `DDIMScheduler.from_pretrained("configs")` 读取。
- `configs/audio.yaml`：音频相关配置。
- `configs/unet/`：UNet 训练/推理配置。
  - `stage1.yaml`、`stage2.yaml`：256 分辨率配置。
  - `stage1_512.yaml`、`stage2_512.yaml`：512 分辨率配置。
  - `stage2_efficient.yaml`：较低显存的 stage2 配置。
  - 关键字段：`data.num_frames=16`、`data.resolution`、`data.mask_image_path`、`run.inference_steps`、`run.guidance_scale`、`model.in_channels=13`、`model.out_channels=4`、`model.add_audio_layer=true`、`model.use_motion_module=true`。
- `configs/syncnet/`：SyncNet 训练/评估配置，包括 latent/pixel、多种帧数和 attention 版本。
- `checkpoints/`：本地推理权重。
  - `latentsync_unet.pt`：UNet 推理 checkpoint。
  - `whisper/tiny.pt`：Whisper tiny 权重。
  - `auxiliary/`：README 中提到的 SyncNet、SFD、HyperIQA 等辅助模型应放在这里。

## 4. 推理入口

- `scripts/inference.py`：主要 CLI 推理入口。
  - 解析单样本参数：`--video_path`、`--audio_path`、`--video_out_path`。
  - 解析内部 batch 参数：`--video_paths`、`--audio_paths`、`--video_out_paths`，均为逗号分隔。
  - 载入 DDIM scheduler、Whisper `Audio2Feature`、VAE、`UNet3DConditionModel`，组装 `LipsyncPipeline`。
  - 当 `len(audio_paths) == 1 and batch_size == 1` 时调用 `pipeline.__call__()`；其他情况调用 `pipeline.batch_inference()`。
  - 输出 profile summary JSON，包含 preprocess、denoise、postprocess、GPU peak memory、cache hit/miss。
- `latentsync/pipelines/lipsync_pipeline.py`：推理主流程。
  - `__call__()`：单样本推理。
  - `batch_inference()`：多样本内部 batch 推理。
  - 还包含视频预处理缓存、chunk 级 VAE 缓存、VAE decode、restore、ffmpeg mux 等逻辑。

## 5. 模型代码

- `latentsync/models/unet.py`：`UNet3DConditionModel`，基于 AnimateDiff/Diffusers 风格的 3D 条件 UNet。
- `latentsync/models/unet_blocks.py`：Down/Mid/Up blocks，包含 cross-attention block 和 motion module 接入点。
- `latentsync/models/attention.py`：3D transformer 和 attention 实现。内部会把视频张量从 `b c f h w` reshape 到 `(b f) c h w` 或 `(b f) seq dim`。
- `latentsync/models/resnet.py`：InflatedConv3d、InflatedGroupNorm、ResnetBlock3D、Upsample3D/Downsample3D。
- `latentsync/models/motion_module.py`：temporal motion module。
- `latentsync/models/stable_syncnet.py`、`wav2lip_syncnet.py`：SyncNet 相关模型。
- `latentsync/models/utils.py`：模型辅助函数。

## 6. 数据与特征

- `latentsync/data/unet_dataset.py`：UNet 训练数据集。
- `latentsync/data/syncnet_dataset.py`：SyncNet 训练数据集。
- `latentsync/whisper/audio2feature.py`：音频到 Whisper feature/chunk 的封装。推理时 `audio2feat()` 后调用 `feature2chunks(..., fps=25)`，生成与视频帧对齐的 `whisper_chunks`。
- `latentsync/whisper/whisper/`：内置 Whisper 代码副本。

## 7. 图像、音视频、工具函数

- `latentsync/utils/image_processor.py`：face crop、mask、masked image、restore 相关处理。
- `latentsync/utils/affine_transform.py`：人脸仿射变换和反变换。
- `latentsync/utils/face_detector.py`：人脸检测封装。
- `latentsync/utils/audio.py`：音频特征、mel、overlap 等处理。
- `latentsync/utils/av_reader.py`、`util.py`：读写视频/音频、ffmpeg 检查、分布式训练辅助、loss 和 scheduler 工具。
- `latentsync/utils/mask*.png`：推理/训练使用的固定 mask。

## 8. 训练代码

- `scripts/train_unet.py`：UNet 训练入口。
  - 初始化 DDP、DDIM scheduler、VAE、Whisper encoder、UNet、SyncNet、TREPA/LPIPS 等。
  - 使用 `UNetDataset` 和 `DistributedSampler`。
  - 训练 batch size 来自 `config.data.batch_size`，与推理 CLI 的 `--batch_size` 是两套概念。
  - 训练中 validation 复用 `LipsyncPipeline`。
- `scripts/train_syncnet.py`：SyncNet 训练入口。
  - 初始化 DDP、SyncNet dataset、SyncNet 模型、optimizer、validation dataloader。
  - 支持 pixel/latent space，latent space 时可用 VAE 编码帧。

## 9. 数据预处理

- `preprocess/data_processing_pipeline.py`：总控入口。
  - 检查辅助模型。
  - 删除坏视频。
  - 重采样 FPS/音频采样率。
  - shot detect。
  - 视频分段。
  - affine transform。
  - SyncNet 对齐。
  - HyperIQA visual quality 过滤。
- 其他 `preprocess/*.py` 是上述每一步的具体实现，部分步骤支持 multiprocessing 或 multi-gpu。

## 10. 评估与工具

- `eval/eval_sync_conf.py`、`eval/eval_syncnet_acc.py`、`eval/eval_fvd.py`：同步置信度、SyncNet accuracy、FVD 等评估。
- `eval/syncnet/`、`eval/syncnet_detect.py`：SyncNet 推理/检测封装。
- `tools/`：数据列表、下载、统计、清理、占用 GPU 等辅助脚本。

## 11. 本地新增/临时内容观察

当前目录中存在一些本地 profiling 和备份内容：

- `temp_batch2_profile/`
- `temp_batch2_same_video_profile/`
- `temp_batch3_same_video_audio_profile/`
- `latentsync/pipelines/lipsync_pipeline.py.bak_video_vae_cache_20260610_201824`
- `latentsync/utils/affine_transform.py.bak_video_vae_cache_20260610_201824`

这些内容说明当前工作区已经有过 batch/profile/cache 相关实验。它们不是官方 README 中的基础结构，但对理解当前性能现象很有价值。

## 12. 推理主数据流速览

1. CLI 读取配置和路径。
2. 加载 scheduler、Whisper、VAE、UNet。
3. 构造 `LipsyncPipeline` 并放到 CUDA。
4. 音频经 Whisper 得到 `whisper_chunks`。
5. 视频读取、face affine crop、按音频长度 loop/裁剪。
6. 每 16 帧一个 chunk。
7. 对每个 chunk 准备 mask/ref/masked VAE latents。
8. DDIM denoise：每个 chunk 跑 `num_inference_steps` 次 UNet。
9. VAE decode，嘴部区域融合回 reference face。
10. restore 回原视频坐标。
11. 写临时视频、音频并用 ffmpeg mux 成最终输出。

