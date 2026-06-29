#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_AUDIO="${SOURCE_AUDIO:-$SCRIPT_DIR/gpt_sovits_fewshot_demo_audio_16k_mono.wav}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/reference_clips}"
START="${START:-5}"
DURATION="${DURATION:-9}"
OUT_FILE="${OUT_FILE:-$OUT_DIR/official_ref_${START}s_${DURATION}s.wav}"

mkdir -p "$OUT_DIR"

ffmpeg -y -hide_banner \
  -ss "$START" \
  -t "$DURATION" \
  -i "$SOURCE_AUDIO" \
  -ac 1 \
  -ar 16000 \
  -c:a pcm_s16le \
  "$OUT_FILE"

echo "$OUT_FILE"
