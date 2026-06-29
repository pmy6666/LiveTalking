#!/usr/bin/env bash

resolve_livetalking_root() {
  local source_path="${BASH_SOURCE[1]:-${BASH_SOURCE[0]}}"
  local source_dir
  source_dir="$(cd "$(dirname "${source_path}")" && pwd)"

  while [[ "${source_dir}" != "/" ]]; do
    if [[ -f "${source_dir}/app.py" && -d "${source_dir}/scripts" ]]; then
      printf '%s\n' "${source_dir}"
      return 0
    fi
    source_dir="$(dirname "${source_dir}")"
  done

  return 1
}

resolve_livetalking_python() {
  local root_dir="$1"

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    if [[ "${PYTHON_BIN}" == */* ]]; then
      local python_dir
      python_dir="$(cd "$(dirname "${PYTHON_BIN}")" && pwd)"
      printf '%s\n' "${python_dir}/$(basename "${PYTHON_BIN}")"
    else
      printf '%s\n' "${PYTHON_BIN}"
    fi
    return 0
  fi

  for candidate in \
    "${root_dir}/.venv/bin/python" \
    "${root_dir}/../envs/livetalking/bin/python" \
    "${HOME}/envs/livetalking/bin/python"; do
    if [[ -x "${candidate}" ]]; then
      printf '%s\n' "${candidate}"
      return 0
    fi
  done

  printf '%s\n' "python3"
}
