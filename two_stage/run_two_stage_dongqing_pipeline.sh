#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${PROJECT_ROOT}/scripts/common_env.sh"
PYTHON_BIN="$(resolve_livetalking_python "${PROJECT_ROOT}")"

cd "${PROJECT_ROOT}"
"${PYTHON_BIN}" \
  two_stage/run_two_stage_dongqing_pipeline.py "$@"
