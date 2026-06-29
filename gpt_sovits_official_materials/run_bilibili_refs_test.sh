#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SERVER="${SERVER:-http://127.0.0.1:9880}"
OUT_ROOT="${OUT_ROOT:-$SCRIPT_DIR/generated_bilibili_refs_tts}"
SPEED_FACTOR="${SPEED_FACTOR:-1.08}"
FRAGMENT_INTERVAL="${FRAGMENT_INTERVAL:-0.08}"

"$PYTHON_BIN" "$SCRIPT_DIR/run_bilibili_refs_test.py" \
  --server "$SERVER" \
  --out_root "$OUT_ROOT" \
  --speed_factor "$SPEED_FACTOR" \
  --fragment_interval "$FRAGMENT_INTERVAL" \
  "$@"
