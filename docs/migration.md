# LiveTalking 轻量迁移方案：仅保留 GPT-SoVITS 和 LatentSync

最后审阅日期：2026-06-29

## 1. 目标和边界

本方案用于把当前 `LiveTalking/` 迁移到另一台服务器，同时只迁移两类预训练模型：

- GPT-SoVITS：用于 TTS 和语音克隆。
- LatentSync：用于离线或两阶段流程里的唇形同步增强。

不迁移的权重包括 Wav2Lip、MuseTalk、EchoMimicV3、Wan2.1-Fun、StableAvatar、DWPose、SD-VAE 独立副本等。迁移后首轮验收只覆盖 GPT-SoVITS 服务和 LatentSync 推理；不要用 `--model wav2lip`、`--model musetalk`、`--model echomimicv3` 作为主应用验收，除非额外补齐对应权重。

## 2. 推荐目录布局

目标服务器建议使用下面布局。后续所有命令都从 `LiveTalking/` 项目根目录执行，避免硬编码旧机器路径。

```bash
workspace/
  LiveTalking/
  envs/
    livetalking/
```

项目里的 shell 入口会自动查找：

- `./.venv/bin/python`
- `../envs/livetalking/bin/python`
- `$HOME/envs/livetalking/bin/python`

也可以显式指定：

```bash
PYTHON_BIN=/path/to/python ./start_gpt_sovits_v2proplus.sh
```

## 3. 需要迁移的文件

### 3.1 必须迁移的代码目录

代码建议整体迁移，但排除大权重、缓存、输出和历史实验结果：

```bash
LiveTalking/
  app.py
  config.py
  requirements.txt
  start_gpt_sovits_v2proplus.sh
  scripts/
  server/
  tts/
  utils/
  choice/
  GPT-SoVITS/
  third_party/LatentSync/
  LatentSync_test/
  gpt_sovits_official_materials/   # 可选，保留测试音频和参考音频更方便验收
```

### 3.2 必须迁移的 GPT-SoVITS 权重

```bash
models/gpt-sovits-v2proplus/
  s1v3.ckpt
  s2Gv2ProPlus.pth
  s2Dv2ProPlus.pth

GPT-SoVITS/GPT_SoVITS/pretrained_models/
  chinese-hubert-base/
  chinese-roberta-wwm-ext-large/
  pretrained_eres2netv2w24s4ep4.ckpt
  fast_langdetect/
```

当前本机体积约：

- `models/gpt-sovits-v2proplus`：约 460 MB。
- `GPT-SoVITS/GPT_SoVITS/pretrained_models`：约 1.1 GB。

`start_gpt_sovits_v2proplus.sh` 会在启动时把 `models/gpt-sovits-v2proplus` 里的权重软链接到 GPT-SoVITS 官方目录的相对位置。

### 3.3 必须迁移的 LatentSync 权重

```bash
third_party/LatentSync/checkpoints/
  latentsync_unet.pt
  whisper/tiny.pt
  auxiliary/models/buffalo_l/
  auxiliary/models/buffalo_l.zip
```

当前本机体积约 5.4 GB。

## 4. 明确不迁移或可删除的权重目录

如果目标是节省传输时间和磁盘，这些目录不传输。当前服务器也可以在确认有备份后删除。

```bash
models/wav2lip.pth
models/musetalk/
models/musetalkV15/
models/dwpose/
models/face-parse-bisent/
models/sd-vae/
models/syncnet/
models/whisper/
EchoMimicV3/
Wan2.1-Fun-1.3B-InP/
third_party/StableAvatar/checkpoints/
third_party/echomimic_v3/asset/
wav2vec2-base-960h/
chinese-wav2vec2-base/
```

注意：`models/sd-vae` 不迁移后，LatentSync 可能会在首次推理时尝试从 Hugging Face 下载 `stabilityai/sd-vae-ft-mse`。如果目标服务器不能联网，建议要么保留 `models/sd-vae`，要么提前把 Hugging Face 缓存准备好。若严格只迁移 GPT-SoVITS 和 LatentSync 权重，则把这个风险写入验收条件。

## 5. 传输命令

推荐先传代码和必要资产，再单独传权重。

```bash
cd /path/to/workspace
rsync -a --info=progress2 \
  --exclude '.git/' \
  --exclude '**/__pycache__/' \
  --exclude '**/.cache/' \
  --exclude 'outputs/' \
  --exclude 'cache/' \
  --exclude 'Echo_mimicV3_test/' \
  --exclude 'MuseTalk_test/' \
  --exclude 'Wan2.1-Fun-1.3B-InP/' \
  --exclude 'EchoMimicV3/' \
  --exclude 'third_party/StableAvatar/checkpoints/' \
  --exclude 'models/wav2lip.pth' \
  --exclude 'models/musetalk/' \
  --exclude 'models/musetalkV15/' \
  --exclude 'models/dwpose/' \
  --exclude 'models/face-parse-bisent/' \
  --exclude 'models/sd-vae/' \
  --exclude 'models/syncnet/' \
  --exclude 'models/whisper/' \
  source_host:/home/qianustb/LiveTalking/ ./LiveTalking/
```

如果不想用复杂排除规则，也可以先传全量代码再删除目标服务器上的非必要权重目录。

## 6. 环境安装

### 6.1 系统依赖

目标服务器需要 NVIDIA driver、CUDA 运行环境和 ffmpeg。Ubuntu 示例：

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg git git-lfs build-essential cmake pkg-config libgl1 libglib2.0-0
```

建议 Python 版本使用 3.10。LatentSync 当前依赖锁定了 `torch==2.5.1`、`torchvision==0.20.1` 和 cu121 wheel。

### 6.2 推荐新建 conda 环境

```bash
cd /path/to/workspace/LiveTalking
conda create -p ../envs/livetalking python=3.10 -y
conda activate ../envs/livetalking
python -m pip install -U pip setuptools wheel
```

按顺序安装依赖：

```bash
# 先装 LatentSync，保留其 torch/cu121 版本约束。
python -m pip install -r third_party/LatentSync/requirements.txt

# 再装 LiveTalking 基础依赖。
python -m pip install -r requirements.txt

# 最后装 GPT-SoVITS 依赖。
python -m pip install -r GPT-SoVITS/requirements.txt
python -m pip install -r GPT-SoVITS/extra-req.txt
```

如果 `onnxruntime-gpu`、`insightface`、`mediapipe` 或 `opencc` 编译失败，优先检查 Python 版本、CUDA wheel 和系统编译工具，不建议随意升级 torch，否则 LatentSync 可能出现不可控兼容问题。

### 6.3 可选：使用当前环境导出文件

当前仓库已有环境导出脚本：

```bash
cd LiveTalking
./scripts/export_livetalking_env.sh
```

恢复时可以使用：

```bash
conda create -p ../envs/livetalking --file env_export/conda-explicit-spec.txt
```

如果目标机器 CUDA、glibc 或驱动差异较大，改用 6.2 的 requirements 安装更稳。

## 7. 相对路径约定

迁移后统一从 `LiveTalking/` 根目录运行命令。当前关键路径已经基本满足相对路径要求：

- `config.py` 使用 `PROJECT_ROOT = Path(__file__).resolve().parent`。
- `start_gpt_sovits_v2proplus.sh` 使用脚本所在目录作为 `ROOT_DIR`。
- `LatentSync_test/run_latentsync_dongqing_batch.py` 使用 `PROJECT_ROOT / "third_party" / "LatentSync"`。
- `scripts/common_env.sh` 会解析项目根目录和 Python 解释器。

需要注意的残留绝对路径：

- `third_party/LatentSync/configs/*`、`preprocess/*`、`tools/*` 中有上游训练数据路径，例如 `/mnt/bn/...`。这些属于训练和数据预处理示例，不参与当前推理迁移。
- 历史测试输出的 `manifest.json` 可能记录了 `/home/qianustb/...`。这些是旧结果元数据，不作为启动路径。
- 不要在新服务器用旧机器绝对路径传参，例如 `/home/qianustb/LiveTalking/...`。统一写成 `models/...`、`third_party/LatentSync/...` 或从项目根解析的相对路径。

## 8. 启动和验收

### 8.1 启动 GPT-SoVITS

在一个终端：

```bash
cd /path/to/workspace/LiveTalking
conda activate ../envs/livetalking
./start_gpt_sovits_v2proplus.sh
```

默认服务地址是：

```text
http://127.0.0.1:9880
```

如需改端口：

```bash
GPT_SOVITS_PORT=9881 ./start_gpt_sovits_v2proplus.sh
```

### 8.2 验证 GPT-SoVITS 权重路径

启动日志应显示：

```text
Using GPT model:    .../models/gpt-sovits-v2proplus/s1v3.ckpt
Using SoVITS model: .../models/gpt-sovits-v2proplus/s2Gv2ProPlus.pth
Using SV model:     .../GPT-SoVITS/GPT_SoVITS/pretrained_models/pretrained_eres2netv2w24s4ep4.ckpt
Using BERT dir:     .../GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
Using CNHuBERT dir: .../GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base
```

### 8.3 验证 LatentSync

另一个终端执行：

```bash
cd /path/to/workspace/LiveTalking
conda activate ../envs/livetalking
LatentSync_test/run_latentsync_dongqing_batch.sh --skip-dep-check --only 01_morning_breakfast
```

正式验收建议去掉 `--skip-dep-check`：

```bash
LatentSync_test/run_latentsync_dongqing_batch.sh --only 01_morning_breakfast
```

输出默认在：

```bash
LatentSync_test/generated_videos_dongqing_sync/
```

LatentSync 官方建议显存至少约 18 GB。若显存紧张，保留默认 `--enable-deepcache`，并确保没有其它大模型进程占用 GPU。

### 8.4 可选启动 LiveTalking 主服务

如果只验证 TTS 接入，不依赖本地 Avatar 权重，可以先运行业务里只调用 TTS 的脚本。若要启动 `app.py`，必须选择已有权重支持的 avatar 后端。当前“只迁移 GPT-SoVITS 和 LatentSync”方案下，不建议直接运行默认：

```bash
python app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1
```

因为 `models/wav2lip.pth` 已被列为不迁移。

## 9. 目标服务器健康检查

```bash
cd /path/to/workspace/LiveTalking
conda activate ../envs/livetalking

python - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())
print("cuda version", torch.version.cuda)
print("device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu")
PY

test -f models/gpt-sovits-v2proplus/s1v3.ckpt
test -f models/gpt-sovits-v2proplus/s2Gv2ProPlus.pth
test -f GPT-SoVITS/GPT_SoVITS/pretrained_models/pretrained_eres2netv2w24s4ep4.ckpt
test -d GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base
test -d GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
test -f third_party/LatentSync/checkpoints/latentsync_unet.pt
test -f third_party/LatentSync/checkpoints/whisper/tiny.pt
test -d third_party/LatentSync/checkpoints/auxiliary/models/buffalo_l
```

## 10. 删除非必要权重的建议

在当前服务器删除前，先生成清单并确认备份：

```bash
cd /home/qianustb/LiveTalking
du -h -d 2 models EchoMimicV3 Wan2.1-Fun-1.3B-InP third_party/StableAvatar/checkpoints 2>/dev/null
```

确认不再需要后再删除。删除命令不要写进自动化脚本，建议人工逐项执行并保留一份离线备份。

## 11. 风险和回退

- 只迁移 GPT-SoVITS 和 LatentSync 后，实时数字人后端不完整。需要 Wav2Lip、MuseTalk 或 EchoMimicV3 时，必须补齐对应权重。
- LatentSync 若无法联网且未迁移 `models/sd-vae`，可能在首次加载 VAE 时失败。严格离线部署建议额外迁移 `models/sd-vae`，但这会超出“只迁移 GPT-SoVITS 和 LatentSync 权重”的范围。
- 目标服务器 CUDA/driver 与 cu121 wheel 不兼容时，优先调整 PyTorch 安装源和 CUDA wheel，而不是改业务代码。
- 若 GPT-SoVITS 启动失败，先检查 `start_gpt_sovits_v2proplus.sh` 打印的五个模型路径，再检查 `GPT-SoVITS/GPT_SoVITS/configs/tts_infer_livetalking_v2proplus.yaml` 是否由启动脚本重新生成。

## 12. 2026-06-29 联网复核更新：避免重复从 Hugging Face 下载

本节根据上游仓库和本地代码再次复核，结论是：当前迁移清单已经覆盖 GPT-SoVITS v2ProPlus 和 LatentSync 推理主权重，但 GPT-SoVITS 中文推理还应额外迁移 `GPT-SoVITS/GPT_SoVITS/text/G2PWModel/`。这不是 v2ProPlus 声学模型权重，但当前代码默认启用，缺失时会影响中文文本前端。

参考源：

- GPT-SoVITS 官方 README：`https://github.com/RVC-Boss/GPT-SoVITS/blob/main/README.md`
- GPT-SoVITS HF v2Pro 权重目录：`https://huggingface.co/lj1995/GPT-SoVITS/tree/main/v2Pro`
- GPT-SoVITS HF speaker verification 权重目录：`https://huggingface.co/lj1995/GPT-SoVITS/tree/main/sv`
- LatentSync 官方 README：`https://github.com/ByteDance/LatentSync/blob/main/README.md`
- LatentSync 1.6 HF 权重目录：`https://huggingface.co/ByteDance/LatentSync-1.6/tree/main`

### 12.1 GPT-SoVITS 复核结果

本项目启动脚本固定使用 `GPT_SOVITS_MODEL_VERSION=v2ProPlus`，并生成：

```yaml
t2s_weights_path: GPT_SoVITS/pretrained_models/s1v3.ckpt
vits_weights_path: GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth
```

需要迁移并已在本机存在的文件：

```bash
models/gpt-sovits-v2proplus/s1v3.ckpt
models/gpt-sovits-v2proplus/s2Gv2ProPlus.pth
models/gpt-sovits-v2proplus/s2Dv2ProPlus.pth
GPT-SoVITS/GPT_SoVITS/pretrained_models/pretrained_eres2netv2w24s4ep4.ckpt
GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base/
GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large/
GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect/
```

追加迁移项：

```bash
GPT-SoVITS/GPT_SoVITS/text/G2PWModel/
```

原因：`GPT-SoVITS/GPT_SoVITS/text/chinese2.py` 当前默认 `is_g2pw = True`，并加载 `GPT_SoVITS/text/G2PWModel`。本地已经有完整目录，其中 `g2pW.onnx` 约 635 MB。迁移它可以避免中文 TTS 首次运行时再去下载 G2PW。

如果只使用非中文 TTS，可以不传 `G2PWModel/`；但当前项目配置默认 `TTS_TEXT_LANG=zh`、`TTS_PROMPT_LANG=zh`，所以建议传。

### 12.2 LatentSync 复核结果

LatentSync 官方安装脚本的推理权重下载项是：

```bash
huggingface-cli download ByteDance/LatentSync-1.6 whisper/tiny.pt --local-dir checkpoints
huggingface-cli download ByteDance/LatentSync-1.6 latentsync_unet.pt --local-dir checkpoints
```

本项目本地推理代码还会通过 `insightface.app.FaceAnalysis(root="checkpoints/auxiliary")` 使用 InsightFace 人脸检测和 106 点 landmark 模型，因此需要同时迁移本地已有的 `buffalo_l`。

需要迁移并已在本机存在的文件：

```bash
third_party/LatentSync/checkpoints/latentsync_unet.pt
third_party/LatentSync/checkpoints/whisper/tiny.pt
third_party/LatentSync/checkpoints/auxiliary/models/buffalo_l/
third_party/LatentSync/checkpoints/auxiliary/models/buffalo_l.zip
```

`ByteDance/LatentSync-1.6` HF 仓库里还包含 `stable_syncnet.pt` 和若干 `auxiliary/` 评估或训练辅助权重。当前迁移目标是推理，不做训练、数据预处理和评估，所以不需要迁移：

```bash
third_party/LatentSync/checkpoints/stable_syncnet.pt
third_party/LatentSync/checkpoints/auxiliary/syncnet_v2.model
third_party/LatentSync/checkpoints/auxiliary/sfd_face.pth
third_party/LatentSync/checkpoints/auxiliary/koniq_pretrained.pkl
third_party/LatentSync/checkpoints/auxiliary/i3d_torchscript.pt
third_party/LatentSync/checkpoints/auxiliary/vit_g_hybrid_pt_1200e_ssv2_ft.pth
```

只有在运行 LatentSync 的训练、预处理、评估脚本时才补这些辅助模型。

### 12.3 最终不联网权重校验命令

迁移完成后，先不要运行自动下载脚本。直接在目标服务器执行：

```bash
cd /path/to/workspace/LiveTalking

test -f models/gpt-sovits-v2proplus/s1v3.ckpt
test -f models/gpt-sovits-v2proplus/s2Gv2ProPlus.pth
test -f models/gpt-sovits-v2proplus/s2Dv2ProPlus.pth
test -f GPT-SoVITS/GPT_SoVITS/pretrained_models/pretrained_eres2netv2w24s4ep4.ckpt
test -d GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-hubert-base
test -d GPT-SoVITS/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large
test -d GPT-SoVITS/GPT_SoVITS/pretrained_models/fast_langdetect
test -d GPT-SoVITS/GPT_SoVITS/text/G2PWModel

test -f third_party/LatentSync/checkpoints/latentsync_unet.pt
test -f third_party/LatentSync/checkpoints/whisper/tiny.pt
test -d third_party/LatentSync/checkpoints/auxiliary/models/buffalo_l
```

为了强制暴露缺失文件，而不是静默联网下载，可以在验收时加离线环境变量：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

再启动：

```bash
./start_gpt_sovits_v2proplus.sh
LatentSync_test/run_latentsync_dongqing_batch.sh --only 01_morning_breakfast
```

如果离线模式下能启动和完成一次 LatentSync 推理，说明 GPT-SoVITS 和 LatentSync 当前迁移范围内的模型权重已经传齐。
