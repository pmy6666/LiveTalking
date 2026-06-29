# 运行指令：bash scripts/download_required_weights.sh
#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(dirname "${BASH_SOURCE[0]}")"
cd "$SCRIPT_DIR/.."

DOWNLOAD_CACHE=".download_cache"
GPT_PRETRAINED_DIR="GPT-SoVITS/GPT_SoVITS/pretrained_models"
GPT_MODEL_DIR="models/gpt-sovits-v2proplus"
G2PW_DIR="GPT-SoVITS/GPT_SoVITS/text/G2PWModel"
LATENTSYNC_CKPT_DIR="third_party/LatentSync/checkpoints"

if command -v hf >/dev/null 2>&1; then
  HF_DOWNLOAD=(hf download)
elif command -v huggingface-cli >/dev/null 2>&1; then
  HF_DOWNLOAD=(huggingface-cli download)
else
  echo "Missing Hugging Face CLI. Install it first: python -m pip install -U huggingface_hub" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Missing python3; it is required to extract zip files." >&2
  exit 1
fi

download_hf() {
  local repo_id="$1"
  local local_dir="$2"
  shift 2

  mkdir -p "$local_dir"
  "${HF_DOWNLOAD[@]}" "$repo_id" "$@" --local-dir "$local_dir"
}

extract_zip() {
  local zip_path="$1"
  local output_dir="$2"

  python3 - "$zip_path" "$output_dir" <<'PY'
import pathlib
import sys
import zipfile

zip_path = pathlib.Path(sys.argv[1])
output_dir = pathlib.Path(sys.argv[2])
output_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path) as archive:
    archive.extractall(output_dir)
PY
}

copy_if_missing() {
  local src="$1"
  local dst="$2"

  if [[ ! -f "$src" ]]; then
    echo "Expected source file is missing after download: $src" >&2
    exit 1
  fi

  if [[ -f "$dst" ]]; then
    echo "Exists, skip: $dst"
    return
  fi

  mkdir -p "$(dirname "$dst")"
  cp "$src" "$dst"
}

echo "Downloading GPT-SoVITS required weights..."
download_hf "lj1995/GPT-SoVITS" "$GPT_PRETRAINED_DIR" \
  --include "s1v3.ckpt" \
  --include "v2Pro/s2Gv2ProPlus.pth" \
  --include "v2Pro/s2Dv2ProPlus.pth" \
  --include "sv/pretrained_eres2netv2w24s4ep4.ckpt" \
  --include "chinese-hubert-base/*" \
  --include "chinese-roberta-wwm-ext-large/*" \
  --include "fast_langdetect/*"

mkdir -p "$GPT_MODEL_DIR"
copy_if_missing "$GPT_PRETRAINED_DIR/s1v3.ckpt" "$GPT_MODEL_DIR/s1v3.ckpt"
copy_if_missing "$GPT_PRETRAINED_DIR/v2Pro/s2Gv2ProPlus.pth" "$GPT_MODEL_DIR/s2Gv2ProPlus.pth"
copy_if_missing "$GPT_PRETRAINED_DIR/v2Pro/s2Dv2ProPlus.pth" "$GPT_MODEL_DIR/s2Dv2ProPlus.pth"

mkdir -p "$GPT_PRETRAINED_DIR"
copy_if_missing "$GPT_PRETRAINED_DIR/sv/pretrained_eres2netv2w24s4ep4.ckpt" \
  "$GPT_PRETRAINED_DIR/pretrained_eres2netv2w24s4ep4.ckpt"

if [[ ! -d "$G2PW_DIR" ]]; then
  echo "Downloading GPT-SoVITS G2PWModel..."
  G2PW_CACHE_DIR="$DOWNLOAD_CACHE/gpt-sovits-pretrained"
  G2PW_EXTRACT_DIR="$DOWNLOAD_CACHE/g2pw_extract"
  rm -rf "$G2PW_EXTRACT_DIR"

  download_hf "XXXXRT/GPT-SoVITS-Pretrained" "$G2PW_CACHE_DIR" "G2PWModel.zip"
  extract_zip "$G2PW_CACHE_DIR/G2PWModel.zip" "$G2PW_EXTRACT_DIR"

  if [[ -d "$G2PW_EXTRACT_DIR/G2PWModel" ]]; then
    mkdir -p "$(dirname "$G2PW_DIR")"
    mv "$G2PW_EXTRACT_DIR/G2PWModel" "$G2PW_DIR"
  elif [[ -d "$G2PW_EXTRACT_DIR/G2PWModel_1.1" ]]; then
    mkdir -p "$(dirname "$G2PW_DIR")"
    mv "$G2PW_EXTRACT_DIR/G2PWModel_1.1" "$G2PW_DIR"
  else
    echo "G2PWModel.zip did not contain G2PWModel or G2PWModel_1.1." >&2
    exit 1
  fi
else
  echo "Exists, skip: $G2PW_DIR"
fi

echo "Downloading LatentSync required weights..."
download_hf "ByteDance/LatentSync-1.6" "$LATENTSYNC_CKPT_DIR" \
  --include "latentsync_unet.pt" \
  --include "whisper/tiny.pt" \
  --include "auxiliary/models/buffalo_l.zip"

if [[ ! -d "$LATENTSYNC_CKPT_DIR/auxiliary/models/buffalo_l" ]]; then
  mkdir -p "$LATENTSYNC_CKPT_DIR/auxiliary/models/buffalo_l"
  extract_zip "$LATENTSYNC_CKPT_DIR/auxiliary/models/buffalo_l.zip" \
    "$LATENTSYNC_CKPT_DIR/auxiliary/models/buffalo_l"
else
  echo "Exists, skip: $LATENTSYNC_CKPT_DIR/auxiliary/models/buffalo_l"
fi

echo "Checking required weights..."
test -f "$GPT_MODEL_DIR/s1v3.ckpt"
test -f "$GPT_MODEL_DIR/s2Gv2ProPlus.pth"
test -f "$GPT_MODEL_DIR/s2Dv2ProPlus.pth"
test -f "$GPT_PRETRAINED_DIR/pretrained_eres2netv2w24s4ep4.ckpt"
test -d "$GPT_PRETRAINED_DIR/chinese-hubert-base"
test -d "$GPT_PRETRAINED_DIR/chinese-roberta-wwm-ext-large"
test -d "$GPT_PRETRAINED_DIR/fast_langdetect"
test -d "$G2PW_DIR"

test -f "$LATENTSYNC_CKPT_DIR/latentsync_unet.pt"
test -f "$LATENTSYNC_CKPT_DIR/whisper/tiny.pt"
test -d "$LATENTSYNC_CKPT_DIR/auxiliary/models/buffalo_l"

echo "Required GPT-SoVITS and LatentSync weights are ready."
