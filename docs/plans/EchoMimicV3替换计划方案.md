# LiveTalking Talking Head 替换为 EchoMimicV3 计划方案

生成日期：2026-04-26

## 1. 目标

将 `.` 当前工作流里的 Talking Head / Avatar 推理模块替换或扩展为 `antgroup/echomimic_v3`，优先使用官方开源预训练模型。建议先做成新增 `--model echomimicv3` 插件，而不是直接删除 `wav2lip`、`musetalk`、`ultralight`，这样可以保留现有实时链路作为回退。

## 2. 当前 LiveTalking 工作流理解

当前工程的核心链路是：

```text
浏览器 / API
  -> /offer 创建 WebRTC 会话
  -> SessionManager 构建 AvatarSession
  -> /human 文本 或 /humanaudio 音频
  -> TTS 输出 16k PCM 音频块
  -> ASR / audio_features 按 20ms 音频块累积特征
  -> Avatar.inference_batch 生成 batch 视频帧或口型区域
  -> paste_back_frame 合成完整帧
  -> streamout 推送 WebRTC / RTMP / virtualcam
```

关键文件：

- `app.py`：解析 `--model`，动态导入 `avatars.musetalk_avatar`、`avatars.wav2lip_avatar`、`avatars.ultralight_avatar`，加载全局模型和 avatar 缓存。
- `registry.py`：通过 `@register("avatar", "...")` 注册 avatar 插件，`registry.create("avatar", opt.model, ...)` 创建会话实例。
- `avatars/base_avatar.py`：定义统一运行时。`render()` 启动 TTS、推理线程和输出线程；`inference()` 从 `self.asr.feat_queue` 取特征；`process_frames()` 把生成帧和音频帧送给输出。
- `avatars/audio_features/base_asr.py`：统一音频输入协议。TTS 或上传音频被切成 16k、20ms 的 `AudioFrameData`。
- `server/routes.py`：`/human`、`/humanaudio`、`/interrupt_talk` 等业务入口不直接关心具体数字人模型。
- `streamout/webrtc.py`、`server/webrtc.py`：将 avatar 输出的视频帧、音频帧转成 aiortc track。

现有三个 avatar 插件的共同接口是：

```python
load_model(...)
load_avatar(avatar_id)
warm_up(...)

@register("avatar", "<model_name>")
class XxxReal(BaseAvatar):
    def __init__(self, opt, model, avatar): ...
    def inference_batch(self, index, audiofeat_batch): ...
    def paste_back_frame(self, pred_frame, idx): ...
```

这说明最合适的接入点是新增 `avatars/echomimicv3_avatar.py`，并在 `app.py` 和 `config.py` 里注册 `echomimicv3`。

## 3. EchoMimicV3 能力和约束

根据官方仓库 `https://github.com/antgroup/echomimic_v3` 和 HuggingFace 模型页：

- EchoMimicV3 是音频驱动的人像 / 半身 / 人体动画生成模型，输入通常是参考图片、音频、prompt，输出视频。
- 官方优先推荐的开源权重包括：
  - `Wan2.1-Fun-V1.1-1.3B-InP`：基础模型。
  - `EchoMimicV3-Flash` / `echomimicv3-flash-pro`：Flash 权重。
  - `chinese-wav2vec2-base`：Flash 音频编码器。
  - `EchoMimicV3-preview`：preview 权重。
  - `wav2vec2-base-960h`：preview 音频编码器。
- 官方 Flash 更新说明提到：8-step 高质量生成、无需人脸 mask、约 12G VRAM、支持最高 768x768。
- 官方 `infer_flash.py` 的形态是离线脚本：`image_path + audio_path + prompt -> output.mp4`，然后用 moviepy 合并音频。
- 官方 `infer_preview.py` 支持长视频分段生成和 overlap 融合，但仍是完整片段生成，不是逐 20ms 或逐帧低延迟流式生成。

主要差异：

| 项目 | LiveTalking 当前模型 | EchoMimicV3 官方推理 |
| --- | --- | --- |
| 输入 | 16k PCM 流式音频块 | 完整 wav 文件或音频片段 |
| 输出 | 每 batch 生成视频帧，持续推流 | 一次生成一个 mp4 / 视频张量 |
| 延迟 | 目标实时，25fps | 秒级到十几秒级分段延迟，取决于显卡和步数 |
| avatar 资产 | 预处理后的 full_imgs / face_imgs / coords / latents | 单张参考图 + prompt，Flash 无需 mask |
| 合成方式 | 口型区域贴回原帧 | 直接生成完整视频帧 |

结论：EchoMimicV3 可以先作为“分段生成式 avatar 后端”接入 LiveTalking，但不能直接等价替换当前 `inference_batch()` 的实时逐帧口型模型。若要求完全实时，需要二次改造 EchoMimicV3 pipeline，让它支持滑窗音频、缓存 latent、增量输出帧和中断。

## 4. 推荐技术路线

### 路线 A：MVP，新增非实时分段 EchoMimicV3 后端

这是最稳妥的第一阶段。

目标：

- 用户仍通过 `/human` 或 `/humanaudio` 输入。
- LiveTalking 仍负责 TTS、WebRTC、会话、打断、前端。
- EchoMimicV3 每次接收一段完整音频，生成一段视频帧，再按 25fps 推送给 WebRTC。
- 静音时仍推参考图或自定义 idle 视频。

实现方式：

1. 新增 `avatars/echomimicv3_avatar.py`。
2. `load_model(opt)` 加载 EchoMimicV3 Flash pipeline、Wav2Vec2、Wan VAE、T5、CLIP、Transformer。
3. `load_avatar(avatar_id)` 读取 `data/avatars/<avatar_id>/echomimicv3/`：
   - `ref.png` 或 `ref.jpg`
   - `prompt.txt`
   - 可选 `idle_imgs/`
   - 可选 `avatar_config.json`
4. 不复用 `BaseAvatar.inference()` 现有 `asr.feat_queue` 机制，建议在 `EchoMimicV3Real` 里覆盖 `render()`：
   - TTS 仍将音频块送入 avatar。
   - avatar 内部缓存一次回答的音频块，遇到 `status=end` 后拼成 wav。
   - 调用 EchoMimicV3 pipeline 生成视频。
   - 读取生成帧或直接拿 tensor 转 BGR frame，按 25fps 推给 `self.output.push_video_frame()`。
   - 同步把原音频按 20ms 推给 `self.output.push_audio_frame()`。
5. `/interrupt_talk` 调用 `flush_talk()` 时清空待生成音频、停止当前视频推送；如果 diffusion 推理已进入 GPU 计算，第一阶段可以只做到“推理完成后丢弃结果”，第二阶段再做子进程级中断。

优点：

- 对 LiveTalking 改动小。
- 能优先使用官方开源预训练 Flash 模型。
- 效果会比 Wav2Lip / MuseTalk 更接近生成式人物动画，而不只是嘴部贴回。

代价：

- 不是实时口型驱动，会有“先生成再播放”的等待。
- 一段回答越长，首帧等待越久。
- 并发能力明显低于当前 Wav2Lip / MuseTalk，需要限制 `max_session`。

### 路线 B：中期，分片滑窗伪流式

在 MVP 可跑后，把每次回答切成较短窗口生成，例如 2-4 秒一段：

- 音频缓存达到 `segment_seconds` 或句末标记后启动生成。
- 生成第 N 段时播放第 N-1 段。
- 使用 EchoMimicV3 preview 的 long video overlap 思路，或者 Flash 版本固定 `video_length=81`，每段约 3.24 秒。
- 通过首帧 / 尾帧衔接和 overlap crossfade 减少跳变。

这条路线可以把等待从整段回答压到首段生成耗时，但仍不是严格 20ms 级实时。

### 路线 C：长期，深度改造成低延迟流式生成

若目标是替代当前实时 Talking Head，需要深入改官方 pipeline：

- 将 `infer_flash.py` 中的模型加载和推理函数拆成可复用类。
- 让 Wav2Vec2 audio embeds 支持滑窗追加。
- 缓存 text prompt embedding、clip image embedding、VAE 条件 latent。
- 每次只生成较短 latent frames，并复用上一窗口末尾 latent / image frames。
- 支持推理过程中的 cooperative cancellation。
- 跳过 mp4 写盘，直接返回 RGB frame tensor。

这部分工作量较大，而且可能受扩散模型采样步数限制，最终延迟和吞吐未必达到现有 Wav2Lip 水平。

## 5. 目录和配置建议

新增目录：

```text
LiveTalking/
  third_party/
    echomimic_v3/                 # 官方代码，建议 git submodule 或 vendor
  models/
    echomimicv3/
      flash/
        Wan2.1-Fun-V1.1-1.3B-InP/
        chinese-wav2vec2-base/
        transformer/
          diffusion_pytorch_model.safetensors
      preview/
        Wan2.1-Fun-V1.1-1.3B-InP/
        wav2vec2-base-960h/
        transformer/
          diffusion_pytorch_model.safetensors
  avatars/
    echomimicv3_avatar.py
  data/
    avatars/
      echomimicv3_avatar1/
        echomimicv3/
          ref.png
          prompt.txt
          avatar_config.json
          idle_imgs/
```

新增命令行参数建议放到 `config.py`：

```text
--model echomimicv3
--echomimicv3_repo third_party/echomimic_v3
--echomimicv3_model_dir models/echomimicv3/flash
--echomimicv3_variant flash
--echomimicv3_sample_size 768 768
--echomimicv3_video_length 81
--echomimicv3_num_steps 8
--echomimicv3_guidance_scale 6.0
--echomimicv3_audio_guidance_scale 3.0
--echomimicv3_weight_dtype bfloat16
--echomimicv3_segment_seconds 3.0
```

`app.py` 中 `_avatar_modules` 增加：

```python
'echomimicv3': 'avatars.echomimicv3_avatar'
```

并添加启动分支：

```python
elif opt.model == 'echomimicv3':
    model = load_model(opt)
    global_avatars[opt.avatar_id] = load_avatar(opt.avatar_id)
    warm_up(opt, model, global_avatars[opt.avatar_id])
```

## 6. EchoMimicV3 Avatar 适配器设计

建议第一版不要继承当前 `BaseAvatar.inference()` 的 ASR 特征队列，而是复用 BaseAvatar 的 TTS、音频输入、输出和录制能力，覆盖渲染主循环。

核心类职责：

```text
EchoMimicV3Real(BaseAvatar)
  - 保存 ref_image、prompt、idle_frames
  - 接收 put_audio_frame 的 20ms PCM
  - 按 status=start/end 组装 utterance
  - 将 numpy PCM 写入临时 wav 或直接送入改造后的 pipeline
  - 调用 EchoMimicV3 生成 frames
  - 将 frames + 原始 audio stream 按时间戳推送到 output
```

第一版可以先写临时 wav，直接复用官方 `infer_flash.py` 的逻辑，但不要每次重新加载模型。应把官方脚本拆成：

```python
class EchoMimicV3FlashEngine:
    def __init__(self, paths, config, device): ...
    def generate(self, ref_image_path, audio_path, prompt, output_dir, params) -> list[np.ndarray]:
        ...
```

为了减少磁盘 IO，第二版再改成：

```python
def generate_frames(self, ref_image: PIL.Image, audio_np: np.ndarray, prompt: str) -> list[np.ndarray]:
    ...
```

## 7. 与现有实时协议的冲突点

需要重点处理这些问题：

1. 首帧延迟：EchoMimicV3 是扩散式视频生成，不能像 Wav2Lip 一样每 `batch_size` 快速吐口型帧。前端最好增加“生成中”状态，或者在生成期间播放 idle。
2. 中断：当前 `flush_talk()` 能清空队列，但不能天然打断一次 GPU diffusion。建议把生成任务放进独立 worker 线程或进程，用 token 丢弃旧结果；强中断可在第二阶段做进程 kill / restart。
3. 音画同步：官方输出 mp4 时已合并音频；LiveTalking 需要拆成“视频帧 + 原始音频帧”分别推 WebRTC。建议不用官方合成后的音频，直接使用原始 PCM 按 20ms 推。
4. 帧率：LiveTalking 固定 25fps，EchoMimicV3 参数也设为 `fps=25`，避免重采样复杂度。
5. 分辨率：WebRTC 编码 CPU 压力会随分辨率上升。初期建议 512x512 或 768x768 二选一测试；如果 finalfps 不足，优先降到 512。
6. 并发：Flash 官方约 12G VRAM，单卡建议先 `--max_session 1`；多会话要排队生成或多 GPU 分配。
7. License：LiveTalking 和 EchoMimicV3 模型均为 Apache 2.0 路线，但生成内容仍需遵守 EchoMimicV3 官方安全和责任声明。

## 8. 实施步骤

### 阶段 0：准备和验证官方模型

1. 将官方 `echomimic_v3` 放入 `third_party/echomimic_v3`。
2. 下载 Flash 权重，优先使用：
   - `BadToBest/EchoMimicV3` 的 `echomimicv3-flash-pro`
   - `alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`
   - `TencentGameMate/chinese-wav2vec2-base`
3. 在独立环境跑通官方 `run_flash.sh` 等价命令。
4. 确认本机 CUDA、PyTorch、diffusers、transformers、moviepy 版本兼容。

### 阶段 1：抽出 EchoMimicV3 引擎

1. 从官方 `infer_flash.py` 抽出模型加载和 `generate()`。
2. 保证模型只加载一次。
3. 输出从 mp4 文件改为 frame list，至少先支持从临时 mp4 读帧。
4. 增加最小单元测试或脚本：输入 `ref.png + wav`，输出帧数量、fps、首帧耗时、总耗时。

### 阶段 2：接入 LiveTalking 插件

1. 新增 `avatars/echomimicv3_avatar.py`。
2. 修改 `config.py` 支持 `--model echomimicv3` 和相关参数。
3. 修改 `app.py` 的 `_avatar_modules` 和启动分支。
4. 让 `/humanaudio` 首先跑通：上传 wav -> 生成视频 -> WebRTC 播放。
5. 再让 `/human` 跑通：文本 -> TTS -> PCM 缓存 -> 生成视频 -> WebRTC 播放。

### 阶段 3：体验优化

1. 生成期间播放 idle 帧或静态参考图。
2. 用 playback token 实现旧结果丢弃。
3. 支持 3 秒左右分段生成，减少长回答等待。
4. 支持 avatar 级 prompt 配置和分辨率配置。
5. 前端 dashboard 显示“生成中 / 播放中 / 可打断”。

### 阶段 4：性能和质量评估

需要记录：

- 首帧等待时间。
- 10 秒音频生成耗时。
- 播放阶段 finalfps。
- 显存峰值。
- 512 和 768 分辨率的效果差异。
- `num_inference_steps=5/8/12` 的口型同步和画质差异。
- `audio_guidance_scale=2.0/2.5/3.0` 的口型同步差异。

## 9. MVP 验收标准

第一版完成后应满足：

- `python app.py --transport webrtc --model echomimicv3 --avatar_id echomimicv3_avatar1` 可以启动。
- 浏览器打开 dashboard 后可以创建 WebRTC 会话。
- `/humanaudio` 上传 3-5 秒 wav 后，数字人能播放 EchoMimicV3 生成的视频和原始音频。
- `/human` 文本输入后，TTS 生成音频，EchoMimicV3 生成对应视频并播放。
- `/interrupt_talk` 至少能停止待播放视频和后续音频；正在生成的旧结果不会继续播放。
- 模型加载只发生在服务启动，不在每次请求重复加载。

## 10. 风险判断

最重要的风险是“实时性预期”。EchoMimicV3 官方 Flash 已经比 preview 更快，但它仍是生成式视频模型，不是 LiveTalking 当前 `batch_size -> mouth frames` 的实时补丁模型。因此建议产品上明确区分：

- 需要低延迟互动：继续保留 Wav2Lip / MuseTalk。
- 需要更高画质、更自然人体动画：使用 EchoMimicV3，但接受生成等待。

若最终目标是完全替换现有实时 Talking Head，建议先完成路线 A 和路线 B，用真实数据测延迟和显存，再决定是否投入路线 C 的深度流式改造。

## 11. 参考来源

- EchoMimicV3 GitHub：`https://github.com/antgroup/echomimic_v3`
- EchoMimicV3 HuggingFace：`https://huggingface.co/BadToBest/EchoMimicV3`
- Wan2.1-Fun-V1.1-1.3B-InP：`https://huggingface.co/alibaba-pai/Wan2.1-Fun-V1.1-1.3B-InP`
- LiveTalking 本地代码：`app.py`、`avatars/base_avatar.py`、`avatars/wav2lip_avatar.py`、`avatars/musetalk_avatar.py`、`server/routes.py`、`server/webrtc.py`

## 12. 当前实施状态与启动命令

更新时间：2026-04-26

已经完成的代码接入：

- 已新增 `avatars/echomimicv3_avatar.py`，注册 `@register("avatar", "echomimicv3")`。
- 已在 `app.py` 增加 `--model echomimicv3` 的动态导入和启动分支。
- 已在 `config.py` 增加 EchoMimicV3 相关参数。
- 已在 `requirements.txt` 增加 EchoMimicV3 Flash 推理依赖。
- 已拉取官方推理代码到 `third_party/echomimic_v3`。
- 已通过语法检查：`python -m py_compile config.py app.py avatars/echomimicv3_avatar.py`。

当前本机检测到的路径：

```text
third_party/echomimic_v3
EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors
EchoMimicV3/transformer/diffusion_pytorch_model.safetensors
Wan2.1-Fun-1.3B-InP
chinese-wav2vec2-base
wav2vec2-base-960h
```

当前 `Wan2.1-Fun-1.3B-InP` 已检测到约 19G 权重，并包含：

```text
Wan2.1_VAE.pth
models_t5_umt5-xxl-enc-bf16.pth
models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
diffusion_pytorch_model.safetensors
```

已验证：

- `chinese-wav2vec2-base` 已作为 Flash-Pro 的优先音频编码器路径；`wav2vec2-base-960h` 仅保留给 preview 或兜底场景。
- `wav2lip256_avatar1` 可作为 EchoMimicV3 参考图和 idle 帧来源，参考图为 `data/avatars/wav2lip256_avatar1/full_imgs/00000000.png`。
- `../envs/livetalking` 环境已补齐 `pyloudnorm`、`decord`、`moviepy==2.2.1`，并将 `huggingface-hub` 修正到 `transformers==4.46.2` 兼容范围。
- EchoMimicV3 路径校验和官方模块导入 dry-run 已通过。

当前本机 `nvidia-smi` 无法连接 NVIDIA driver，`torch.cuda.is_available()` 为 `False`。因此本机当前 shell 不能完成真实 EchoMimicV3 推理；需要在能看到 NVIDIA GPU/driver 的环境中启动。最终启动命令如下：

```bash
cd .

python app.py \
  --transport webrtc \
  --model echomimicv3 \
  --avatar_id wav2lip256_avatar1 \
  --max_session 1 \
  --echomimicv3_repo third_party/echomimic_v3 \
  --echomimicv3_model_dir EchoMimicV3 \
  --echomimicv3_base_model_dir Wan2.1-Fun-1.3B-InP \
  --echomimicv3_wav2vec_dir chinese-wav2vec2-base \
  --echomimicv3_transformer_path EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors \
  --echomimicv3_sample_size 768 768 \
  --echomimicv3_video_length 81 \
  --echomimicv3_num_steps 8 \
  --listenport 8010
```

如果显存或编码压力较大，建议先用 512 分辨率验证链路：

```bash
cd .

python app.py \
  --transport webrtc \
  --model echomimicv3 \
  --avatar_id wav2lip256_avatar1 \
  --max_session 1 \
  --echomimicv3_repo third_party/echomimic_v3 \
  --echomimicv3_model_dir EchoMimicV3 \
  --echomimicv3_base_model_dir Wan2.1-Fun-1.3B-InP \
  --echomimicv3_wav2vec_dir chinese-wav2vec2-base \
  --echomimicv3_transformer_path EchoMimicV3/echomimicv3-flash-pro/diffusion_pytorch_model.safetensors \
  --echomimicv3_sample_size 512 512 \
  --echomimicv3_video_length 81 \
  --echomimicv3_num_steps 8 \
  --listenport 8010
```

启动后访问：

```text
http://<serverip>:8010/dashboard.html
```

本机浏览器可访问：

```text
http://127.0.0.1:8010/dashboard.html
```
