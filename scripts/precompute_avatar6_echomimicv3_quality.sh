#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_env.sh"
PYTHON_BIN="$(resolve_livetalking_python "${PROJECT_ROOT}")"

FORCE="${FORCE:-1}"
TREE_ID="${TREE_ID:-default_choice_tree}"
QUALITY_STEPS="${QUALITY_STEPS:-20}"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" scripts/precompute_choice_echomimicv3_cache.py \
  --tree_id "${TREE_ID}" \
  --force "${FORCE}" \
  "$@" \
  --transport webrtc \
  --model echomimicv3 \
  --avatar_id avatar6 \
  --tts gpt-sovits \
  --TTS_SERVER http://127.0.0.1:9880 \
  --TTS_MEDIA_TYPE wav \
  --GPT_SOVITS_STREAMING_MODE 2 \
  --REF_FILE bilibili_downloads/SaBeining_enhanced.wav \
  --REF_TEXT "三位航天员你们好，我是朗读者撒贝宁" \
  --max_session 1 \
  --echomimicv3_repo third_party/echomimic_v3 \
  --echomimicv3_model_dir EchoMimicV3 \
  --echomimicv3_base_model_dir Wan2.1-Fun-1.3B-InP \
  --echomimicv3_wav2vec_dir chinese-wav2vec2-base \
  --echomimicv3_sample_size 768 768 \
  --echomimicv3_video_length 201 \
  --echomimicv3_num_steps "${QUALITY_STEPS}" \
  --echomimicv3_guidance_scale 5.5 \
  --echomimicv3_audio_guidance_scale 4.0 \
  --echomimicv3_gpu_memory_mode model_cpu_offload \
  --echomimicv3_weight_dtype bfloat16
