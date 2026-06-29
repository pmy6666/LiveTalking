#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
source "$ROOT_DIR/scripts/common_env.sh"
PYTHON_BIN="$(resolve_livetalking_python "$ROOT_DIR")"
CHOICE_CONFIG="${CHOICE_CONFIG:-$ROOT_DIR/scripts/two_stage_pre/config.yaml}"

cd "$ROOT_DIR"

exec "$PYTHON_BIN" app.py \
  --choice_config "$CHOICE_CONFIG" \
  "$@"
