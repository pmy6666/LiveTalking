#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
ENV_DIR="${ENV_DIR:-${PROJECT_ROOT}/../envs/livetalking}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/env_export}"
ARCHIVE="${ARCHIVE:-0}"

mkdir -p "${OUT_DIR}"

PYTHON_BIN="${ENV_DIR}/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python not found in ENV_DIR: ${ENV_DIR}" >&2
  echo "Set ENV_DIR to the conda/venv path you want to export." >&2
  exit 1
fi

"${PYTHON_BIN}" -m pip freeze > "${OUT_DIR}/requirements-freeze.txt"

if command -v conda >/dev/null 2>&1; then
  conda list -p "${ENV_DIR}" --explicit > "${OUT_DIR}/conda-explicit-spec.txt"
  conda env export -p "${ENV_DIR}" --no-builds > "${OUT_DIR}/conda-environment.yml"
  sed -i \
    -e 's#^name: .*#name: livetalking#' \
    -e '/^prefix: /d' \
    "${OUT_DIR}/conda-environment.yml"
else
  echo "conda command not found; skipped conda spec export." >&2
fi

cat > "${OUT_DIR}/README.md" <<EOF
# LiveTalking Environment Export

Recommended restore on another Linux machine:

\`\`\`bash
conda create -p ../envs/livetalking --file env_export/conda-explicit-spec.txt
./start_gpt_sovits_v2proplus.sh
\`\`\`

Fallback restore:

\`\`\`bash
conda env create -p ../envs/livetalking -f env_export/conda-environment.yml
../envs/livetalking/bin/python -m pip install -r env_export/requirements-freeze.txt
\`\`\`

All LiveTalking shell entrypoints also accept:

\`\`\`bash
PYTHON_BIN=/path/to/python ./some_script.sh
\`\`\`
EOF

if [[ "${ARCHIVE}" == "1" ]]; then
  tar -C "$(dirname "${ENV_DIR}")" -czf "${OUT_DIR}/livetalking-env.tar.gz" "$(basename "${ENV_DIR}")"
  echo "Wrote ${OUT_DIR}/livetalking-env.tar.gz"
fi

echo "Wrote environment export files to ${OUT_DIR}"
