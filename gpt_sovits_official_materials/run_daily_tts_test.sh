#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SERVER="${SERVER:-http://127.0.0.1:9880}"
REF_START="${REF_START:-5}"
REF_DURATION="${REF_DURATION:-9}"
REF_AUDIO="${REF_AUDIO:-$SCRIPT_DIR/../bilibili_downloads/DongQing_6s.wav}"
OUT_DIR="${OUT_DIR:-$SCRIPT_DIR/generated_daily_tts_dongqing_fragment_0p1}"
PROMPT_TEXT="${PROMPT_TEXT:-那种快乐常常像一场梦，电影陪伴我们长大}"
SPEED_FACTOR="${SPEED_FACTOR:-1.08}"
FRAGMENT_INTERVAL="${FRAGMENT_INTERVAL:-0.1}"

if [[ ! -f "$REF_AUDIO" ]]; then
  START="$REF_START" DURATION="$REF_DURATION" OUT_FILE="$REF_AUDIO" "$SCRIPT_DIR/make_reference_clip.sh" >/dev/null
fi

"$PYTHON_BIN" "$SCRIPT_DIR/generate_daily_tts.py" \
  --server "$SERVER" \
  --ref_audio "$REF_AUDIO" \
  --prompt_text "$PROMPT_TEXT" \
  --speed_factor "$SPEED_FACTOR" \
  --fragment_interval "$FRAGMENT_INTERVAL" \
  --out_dir "$OUT_DIR" \
  "$@"
