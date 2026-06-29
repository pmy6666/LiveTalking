#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST="${GPT_SOVITS_HOST:-127.0.0.1}"
PORT="${GPT_SOVITS_PORT:-9880}"
source "$ROOT_DIR/scripts/common_env.sh"
PYTHON_BIN="$(resolve_livetalking_python "$ROOT_DIR")"
MODEL_VERSION="${GPT_SOVITS_MODEL_VERSION:-v2ProPlus}"

if [[ -n "${GPT_SOVITS_REPO:-}" ]]; then
  GPT_SOVITS_REPO="$GPT_SOVITS_REPO"
elif [[ -d "$ROOT_DIR/GPT-SoVITS" ]]; then
  GPT_SOVITS_REPO="$ROOT_DIR/GPT-SoVITS"
elif [[ -d "$ROOT_DIR/../GPT-SoVITS" ]]; then
  GPT_SOVITS_REPO="$ROOT_DIR/../GPT-SoVITS"
else
  GPT_SOVITS_REPO="$ROOT_DIR/../GPT-SoVITS"
fi

DEFAULT_MODEL_DIR="$ROOT_DIR/models/gpt-sovits-v2proplus"
SOVITS_MODEL="${GPT_SOVITS_SOVITS_MODEL:-$DEFAULT_MODEL_DIR/s2Gv2ProPlus.pth}"
S2D_MODEL="${GPT_SOVITS_S2D_MODEL:-$DEFAULT_MODEL_DIR/s2Dv2ProPlus.pth}"
GPT_MODEL="${GPT_SOVITS_GPT_MODEL:-$DEFAULT_MODEL_DIR/s1v3.ckpt}"

if [[ ! -f "$SOVITS_MODEL" && -f "$ROOT_DIR/s2Gv2ProPlus.pth" ]]; then
  SOVITS_MODEL="$ROOT_DIR/s2Gv2ProPlus.pth"
fi
if [[ ! -f "$S2D_MODEL" && -f "$ROOT_DIR/s2Dv2ProPlus.pth" ]]; then
  S2D_MODEL="$ROOT_DIR/s2Dv2ProPlus.pth"
fi
if [[ ! -f "$GPT_MODEL" && -f "$ROOT_DIR/s1v3.ckpt" ]]; then
  GPT_MODEL="$ROOT_DIR/s1v3.ckpt"
fi
SV_MODEL="${GPT_SOVITS_SV_MODEL:-$GPT_SOVITS_REPO/GPT_SoVITS/pretrained_models/pretrained_eres2netv2w24s4ep4.ckpt}"
BERT_DIR="${GPT_SOVITS_BERT_DIR:-$GPT_SOVITS_REPO/GPT_SoVITS/pretrained_models/chinese-roberta-wwm-ext-large}"
CNHUBERT_DIR="${GPT_SOVITS_CNHUBERT_DIR:-$GPT_SOVITS_REPO/GPT_SoVITS/pretrained_models/chinese-hubert-base}"

PRETRAIN_DIR="$GPT_SOVITS_REPO/GPT_SoVITS/pretrained_models"
V2PRO_DIR="$PRETRAIN_DIR/v2Pro"
SV_DIR="$PRETRAIN_DIR/sv"
TEMP_CONFIG="$GPT_SOVITS_REPO/GPT_SoVITS/configs/tts_infer_livetalking_v2proplus.yaml"

if [[ ! -d "$GPT_SOVITS_REPO" ]]; then
  echo "GPT-SoVITS repo not found: $GPT_SOVITS_REPO" >&2
  echo "Set GPT_SOVITS_REPO, or place the official repo at ./GPT-SoVITS or ../GPT-SoVITS." >&2
  exit 1
fi

if [[ ! -f "$SOVITS_MODEL" ]]; then
  echo "Missing SoVITS inference checkpoint: $SOVITS_MODEL" >&2
  exit 1
fi

if [[ ! -f "$GPT_MODEL" ]]; then
  echo "Missing GPT/T2S checkpoint: $GPT_MODEL" >&2
  echo "GPT-SoVITS api_v2 inference for v2ProPlus still needs s1v3.ckpt." >&2
  exit 1
fi

if [[ ! -f "$SV_MODEL" ]]; then
  echo "Missing speaker verification checkpoint: $SV_MODEL" >&2
  echo "GPT-SoVITS v2ProPlus also needs pretrained_eres2netv2w24s4ep4.ckpt." >&2
  echo "Put it in LiveTalking/, or set GPT_SOVITS_SV_MODEL to its actual path." >&2
  exit 1
fi

if [[ ! -d "$BERT_DIR" ]]; then
  echo "Missing BERT model directory: $BERT_DIR" >&2
  echo "GPT-SoVITS needs the chinese-roberta-wwm-ext-large folder." >&2
  echo "Set GPT_SOVITS_BERT_DIR if it exists somewhere else." >&2
  exit 1
fi

if [[ ! -d "$CNHUBERT_DIR" ]]; then
  echo "Missing CNHuBERT model directory: $CNHUBERT_DIR" >&2
  echo "GPT-SoVITS needs the chinese-hubert-base folder." >&2
  echo "Set GPT_SOVITS_CNHUBERT_DIR if it exists somewhere else." >&2
  exit 1
fi

mkdir -p "$V2PRO_DIR"
mkdir -p "$SV_DIR"
ln -sfn "$GPT_MODEL" "$PRETRAIN_DIR/s1v3.ckpt"
ln -sfn "$SOVITS_MODEL" "$V2PRO_DIR/s2Gv2ProPlus.pth"
ln -sfn "$SV_MODEL" "$SV_DIR/pretrained_eres2netv2w24s4ep4.ckpt"

if [[ -f "$S2D_MODEL" ]]; then
  ln -sfn "$S2D_MODEL" "$V2PRO_DIR/s2Dv2ProPlus.pth"
  echo "Linked s2D checkpoint for completeness: $S2D_MODEL"
  echo "Note: api_v2 inference does not directly consume s2D; it is mainly a training-side checkpoint."
fi

echo "Using GPT model:    $GPT_MODEL"
echo "Using SoVITS model: $SOVITS_MODEL"
echo "Using SV model:     $SV_MODEL"
echo "Using BERT dir:     $BERT_DIR"
echo "Using CNHuBERT dir: $CNHUBERT_DIR"
echo "Using model version: $MODEL_VERSION"
echo "Starting GPT-SoVITS API at http://$HOST:$PORT"

cat > "$TEMP_CONFIG" <<EOF
custom:
  bert_base_path: ${BERT_DIR}
  cnhuhbert_base_path: ${CNHUBERT_DIR}
  device: cuda
  is_half: true
  t2s_weights_path: GPT_SoVITS/pretrained_models/s1v3.ckpt
  version: ${MODEL_VERSION}
  vits_weights_path: GPT_SoVITS/pretrained_models/v2Pro/s2Gv2ProPlus.pth
EOF

cd "$GPT_SOVITS_REPO"
"$PYTHON_BIN" api_v2.py -a "$HOST" -p "$PORT" -c "GPT_SoVITS/configs/$(basename "$TEMP_CONFIG")"
