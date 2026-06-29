# LiveTalking GitHub 中转传输指令

本文档用于“目标服务器公网不可直接访问，无法从当前服务器 `rsync/ssh` 直传”的场景。流程改为：

```text
当前服务器 -> GitHub 仓库 -> 新服务器 git clone -> 新服务器下载模型权重
```

本次方案只上传代码，不把 GPT-SoVITS 和 LatentSync 的预训练模型权重上传到 GitHub。

占位说明：

- `GITHUB_URL`：GitHub 仓库地址，例如 `git@github.com:ORG/REPO.git` 或 `https://github.com/ORG/REPO.git`
- `PATH`：新服务器上的目标父目录，例如 `/data/workspace`
- `IP`：新服务器 IP。本方案不需要用 `IP` 直连传输；只有你能从某个内网跳板 SSH 到新服务器时才会用到。

最终目标目录：

```bash
PATH/LiveTalking/
```

## 1. 前提判断

这个方案要求：

- 当前服务器能访问 GitHub 并能 `git push`。
- 新服务器虽然公网入站不可访问，但能出站访问 GitHub 并能 `git clone`。
- 新服务器能访问 Hugging Face，或者已经配置好可用的 Hugging Face 镜像。
- 新服务器 Python 环境里有 Hugging Face CLI；如果没有，先执行 `python -m pip install -U huggingface_hub`。

由于权重不进入 GitHub，本方案不再需要 Git LFS，也不需要把权重打包分片。

## 2. 传输范围

代码会整体复制到一个临时迁移仓库，但排除缓存、输出、历史实验目录和所有模型权重。

克隆后由脚本下载的 GPT-SoVITS 权重会落到这些相对路径：

```bash
models/gpt-sovits-v2proplus/
GPT-SoVITS/GPT_SoVITS/pretrained_models/
GPT-SoVITS/GPT_SoVITS/text/G2PWModel/
```

克隆后由脚本下载的 LatentSync 权重会落到这些相对路径：

```bash
third_party/LatentSync/checkpoints/latentsync_unet.pt
third_party/LatentSync/checkpoints/whisper/tiny.pt
third_party/LatentSync/checkpoints/auxiliary/models/buffalo_l/
third_party/LatentSync/checkpoints/auxiliary/models/buffalo_l.zip
```

下载脚本：

```bash
bash scripts/download_required_weights.sh
```

## 3. 当前服务器：准备迁移仓库目录

在当前服务器执行。这里不直接在 `LiveTalking` 里初始化新仓库，而是在 `/tmp` 构建一个干净的迁移副本，避免把本地 `.git`、缓存、历史输出和权重带进去。

```bash
TRANSFER_DIR=/tmp/livetalking-github-transfer
SOURCE_DIR=LiveTalking

rm -rf "$TRANSFER_DIR"
mkdir -p "$TRANSFER_DIR"

rsync -a --info=progress2 \
  --exclude '.git/' \
  --exclude '**/.git/' \
  --exclude '**/__pycache__/' \
  --exclude '**/.cache/' \
  --exclude '.download_cache/' \
  --exclude '.env' \
  --exclude '*.log' \
  --exclude 'outputs/' \
  --exclude 'cache/' \
  --exclude 'logs/' \
  --exclude 'env_export/' \
  --exclude 'bilibili_downloads/' \
  --exclude 'data/' \
  --exclude '**/.torch_compile_cache/' \
  --exclude 'LatentSync_test/_work/' \
  --exclude 'LatentSync_test/outputs*/' \
  --exclude 'test/ck_time/_work*/' \
  --exclude 'test/ck_time/outputs*/' \
  --exclude 'test/ck_time/latentsync_internal_batch_compare/*/_work*/' \
  --exclude 'gpt_sovits_official_materials/generated*/' \
  --exclude 'third_party/LatentSync/temp*/' \
  --exclude 'Echo_mimicV3_test/' \
  --exclude 'MuseTalk_test/' \
  --exclude 'Wan2.1-Fun-1.3B-InP/' \
  --exclude 'EchoMimicV3/' \
  --exclude 'third_party/StableAvatar/checkpoints/' \
  --exclude 'third_party/echomimic_v3/asset/' \
  --exclude 'models/wav2lip.pth' \
  --exclude 'models/musetalk/' \
  --exclude 'models/musetalkV15/' \
  --exclude 'models/dwpose/' \
  --exclude 'models/face-parse-bisent/' \
  --exclude 'models/sd-vae/' \
  --exclude 'models/syncnet/' \
  --exclude 'models/whisper/' \
  --exclude 'models/gpt-sovits-v2proplus/' \
  --exclude 'GPT-SoVITS/GPT_SoVITS/pretrained_models/' \
  --exclude 'GPT-SoVITS/GPT_SoVITS/text/G2PWModel/' \
  --exclude 'GPT-SoVITS/GPT_SoVITS/text/G2PWModel_1.1.zip' \
  --exclude 'third_party/LatentSync/checkpoints/' \
  --exclude 'wav2vec2-base-960h/' \
  --exclude 'chinese-wav2vec2-base/' \
  "$SOURCE_DIR"/ "$TRANSFER_DIR"/
```

说明：

- 上面排除了所有权重目录，避免在 Git 普通对象里误提交大文件。
- `--exclude '**/.git/'` 用来排除 `GPT-SoVITS/`、`third_party/LatentSync/` 等第三方子仓库自己的 `.git`，避免 GitHub 迁移仓库里出现嵌套仓库或 submodule 混乱。
- `SOURCE_DIR=LiveTalking` 是相对路径；请在 `LiveTalking` 的父目录执行本节命令。

## 4. 当前服务器：初始化代码仓库并推送 GitHub

```bash
TRANSFER_DIR=/tmp/livetalking-github-transfer
GITHUB_URL=git@github.com:ORG/REPO.git

cd "$TRANSFER_DIR"
git init
git add .
git commit -m "Add LiveTalking code"
git branch -M main
git remote add origin "$GITHUB_URL"
git push -u origin main
```

如果 GitHub 仓库已经有内容，建议使用一个新分支：

```bash
git checkout -b livetalking-transfer
git push -u origin livetalking-transfer
```

## 5. 当前服务器：推送后检查

在当前服务器执行：

```bash
cd /tmp/livetalking-github-transfer
git status --short
find . -type f -size +100M
```

预期：

- `git status --short` 为空。
- `find . -type f -size +100M` 没有输出。

如果发现超过 100 MB 的文件，先补充 `rsync --exclude` 规则，再重新生成迁移目录；不要把大权重直接提交到 GitHub。

## 6. 新服务器：从 GitHub 克隆

在新服务器执行。这里不需要新服务器开放公网入站，只需要能出站访问 GitHub。

```bash
PATH=/data/workspace
GITHUB_URL=git@github.com:ORG/REPO.git

mkdir -p "$PATH"
cd "$PATH"
git clone "$GITHUB_URL" LiveTalking
cd LiveTalking
```

如果使用的是迁移分支：

```bash
git clone -b livetalking-transfer "$GITHUB_URL" LiveTalking
```

## 7. 新服务器：下载权重

在新服务器的 `PATH/LiveTalking` 目录执行：

```bash
bash scripts/download_required_weights.sh
```

脚本只使用项目内相对路径，下载完成后会自动校验 GPT-SoVITS 和 LatentSync 的必需权重。

如果目标服务器访问 Hugging Face 慢，可以先配置镜像后再运行脚本，例如：

```bash
export HF_ENDPOINT=https://hf-mirror.com
bash scripts/download_required_weights.sh
```

## 8. 新服务器：权重校验

脚本末尾已经包含以下校验。需要单独复查时，在 `PATH/LiveTalking` 执行：

```bash
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

echo "required weights: OK"
```

查看核心目录体积：

```bash
du -h -d 2 \
  models/gpt-sovits-v2proplus \
  GPT-SoVITS/GPT_SoVITS/pretrained_models \
  GPT-SoVITS/GPT_SoVITS/text/G2PWModel \
  third_party/LatentSync/checkpoints
```

## 9. 新服务器：离线验收建议

为了确认运行时不会隐式从 Hugging Face 下载，权重下载和校验完成后再设置：

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
```

启动 GPT-SoVITS：

```bash
cd PATH/LiveTalking
conda activate ../envs/livetalking
./start_gpt_sovits_v2proplus.sh
```

另一个终端验证 LatentSync：

```bash
cd PATH/LiveTalking
conda activate ../envs/livetalking
LatentSync_test/run_latentsync_dongqing_batch.sh --only 01_morning_breakfast
```

离线模式下这两步能通过，说明 GPT-SoVITS 和 LatentSync 权重已经完整。

## 10. 不建议上传到 GitHub 的目录

这些目录体积大，且不属于本次 GitHub 代码上传目标：

```bash
EchoMimicV3/
Wan2.1-Fun-1.3B-InP/
MuseTalk_test/
Echo_mimicV3_test/
models/wav2lip.pth
models/musetalk/
models/musetalkV15/
models/dwpose/
models/face-parse-bisent/
models/sd-vae/
models/syncnet/
models/whisper/
models/gpt-sovits-v2proplus/
GPT-SoVITS/GPT_SoVITS/pretrained_models/
GPT-SoVITS/GPT_SoVITS/text/G2PWModel/
third_party/LatentSync/checkpoints/
third_party/StableAvatar/checkpoints/
third_party/echomimic_v3/asset/
wav2vec2-base-960h/
chinese-wav2vec2-base/
```

如果目标服务器完全不能访问 Hugging Face，那么“只传代码、不传权重”的方案不可行，需要改用内网对象存储、离线硬盘、镜像仓库或内网 Git 服务。
