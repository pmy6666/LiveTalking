#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./setup_all_envs.sh [options]

Options:
  --livetalking-only        Install only the LiveTalking runtime dependencies. Default.
  --with-gpt-sovits        Also install GPT-SoVITS dependencies that have wheels on Windows/Linux.
  --with-latentsync        Also install LatentSync dependencies that have wheels on Windows/Linux.
  --with-compile-deps      Try packages that usually need a C/C++ toolchain on Windows.
  --cpu                    Install CPU PyTorch wheels instead of CUDA wheels.
  --cuda121                Install PyTorch CUDA 12.1 wheels. Default.
  --skip-ffmpeg            Do not install ffmpeg via conda.
  --check                  Run import and pip consistency checks after installation.
  -h, --help               Show this help.

Environment variables:
  ENV_DIR                  Target conda env path.
                           Windows/Git Bash default: F:/miniforge3/envs/LiveTalking if present.
                           Other default: <repo-parent>/envs/livetalking
  CONDA_EXE                conda executable path. Default: conda, or F:/miniforge3/Scripts/conda.exe if present.
  PYTHON_BIN               Explicit Python executable inside the target env.

Examples:
  ./setup_all_envs.sh --check
  ENV_DIR=/opt/envs/livetalking ./setup_all_envs.sh --with-gpt-sovits --check
  ./setup_all_envs.sh --with-gpt-sovits --with-latentsync --with-compile-deps

Notes:
  - This script intentionally avoids creating or modifying any environment except ENV_DIR.
  - On Windows, insightface, pyopenjtalk, and jieba_fast often need Microsoft C++ Build Tools.
    They are skipped unless --with-compile-deps is passed.
  - GPT-SoVITS declares gradio<5, while LatentSync declares gradio==5.24.0. When both are
    requested, this script keeps gradio==5.24.0 so LatentSync remains usable.
EOF
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${SCRIPT_DIR}"

INSTALL_GPT_SOVITS=0
INSTALL_LATENTSYNC=0
INSTALL_COMPILE_DEPS=0
INSTALL_FFMPEG=1
RUN_CHECK=0
TORCH_FLAVOR="cuda121"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --livetalking-only)
      INSTALL_GPT_SOVITS=0
      INSTALL_LATENTSYNC=0
      ;;
    --with-gpt-sovits)
      INSTALL_GPT_SOVITS=1
      ;;
    --with-latentsync)
      INSTALL_LATENTSYNC=1
      ;;
    --with-compile-deps)
      INSTALL_COMPILE_DEPS=1
      ;;
    --cpu)
      TORCH_FLAVOR="cpu"
      ;;
    --cuda121)
      TORCH_FLAVOR="cuda121"
      ;;
    --skip-ffmpeg)
      INSTALL_FFMPEG=0
      ;;
    --check)
      RUN_CHECK=1
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
  shift
done

is_windows_path_available() {
  [[ -d "/f/miniforge3/envs/LiveTalking" || -d "F:/miniforge3/envs/LiveTalking" ]]
}

default_env_dir() {
  if is_windows_path_available; then
    printf '%s\n' "F:/miniforge3/envs/LiveTalking"
  else
    printf '%s\n' "$(cd "${PROJECT_ROOT}/.." && pwd)/envs/livetalking"
  fi
}

default_conda_exe() {
  if [[ -x "F:/miniforge3/Scripts/conda.exe" ]]; then
    printf '%s\n' "F:/miniforge3/Scripts/conda.exe"
  elif [[ -x "/f/miniforge3/Scripts/conda.exe" ]]; then
    printf '%s\n' "/f/miniforge3/Scripts/conda.exe"
  else
    printf '%s\n' "conda"
  fi
}

ENV_DIR="${ENV_DIR:-$(default_env_dir)}"
CONDA_EXE="${CONDA_EXE:-$(default_conda_exe)}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="${PYTHON_BIN}"
elif [[ -x "${ENV_DIR}/python.exe" ]]; then
  PYTHON_BIN="${ENV_DIR}/python.exe"
elif [[ -x "${ENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${ENV_DIR}/bin/python"
else
  PYTHON_BIN="${ENV_DIR}/bin/python"
fi

echo "Project root: ${PROJECT_ROOT}"
echo "Target env:   ${ENV_DIR}"
echo "Conda exe:    ${CONDA_EXE}"
echo "Python bin:   ${PYTHON_BIN}"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python was not found in ENV_DIR. Creating env with Python 3.10..."
  "${CONDA_EXE}" create -y -p "${ENV_DIR}" python=3.10
  if [[ -x "${ENV_DIR}/python.exe" ]]; then
    PYTHON_BIN="${ENV_DIR}/python.exe"
  fi
fi

pip_install() {
  "${PYTHON_BIN}" -m pip install "$@"
}

echo "Upgrading basic packaging tools inside target env..."
pip_install -U pip setuptools wheel packaging

echo "Installing PyTorch..."
if [[ "${TORCH_FLAVOR}" == "cpu" ]]; then
  pip_install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cpu
else
  pip_install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
fi

if [[ "${INSTALL_FFMPEG}" == "1" ]]; then
  echo "Installing ffmpeg into target conda env..."
  "${CONDA_EXE}" install -y -p "${ENV_DIR}" -c conda-forge ffmpeg
fi

echo "Installing shared pinned runtime packages..."
pip_install \
  numpy==1.26.4 \
  protobuf==3.20.3 \
  transformers==4.48.0 \
  huggingface-hub==0.30.2 \
  diffusers==0.32.2 \
  accelerate==0.26.1 \
  einops==0.8.2 \
  omegaconf==2.3.0 \
  opencv-python==4.9.0.80 \
  opencv-python-headless==4.11.0.86 \
  scipy scikit-learn pandas tqdm matplotlib rich \
  librosa==0.10.1 \
  soundfile==0.12.1 \
  ffmpeg-python==0.2.0 \
  imageio==2.31.1 \
  imageio-ffmpeg==0.5.1

echo "Installing LiveTalking dependencies..."
pip_install \
  torch-ema ninja trimesh tensorboardX PyMCubes dearpygui configargparse \
  face_alignment python_speech_features numba resampy lpips==0.1.4 \
  edge_tts flask flask_sockets flask-cors aiohttp aiohttp_cors aiortc \
  openai websocket-client websockets==12.0 tensorboard==2.14.1 \
  gevent==23.9.1 gevent-websocket==0.10.1 \
  onnxruntime-gpu==1.21.0 \
  typeguard==2.13.3

if [[ "${INSTALL_LATENTSYNC}" == "1" ]]; then
  echo "Installing LatentSync optional dependencies..."
  pip_install \
    decord==0.6.0 mediapipe==0.10.11 scenedetect==0.6.1 \
    kornia==0.8.0 DeepCache==0.1.1 gradio==5.24.0

  if [[ "${INSTALL_COMPILE_DEPS}" == "1" ]]; then
    echo "Installing LatentSync compile-dependent package: insightface..."
    pip_install insightface==0.7.3
  else
    echo "Skipped insightface==0.7.3. Pass --with-compile-deps after installing a C/C++ toolchain."
  fi
fi

if [[ "${INSTALL_GPT_SOVITS}" == "1" ]]; then
  echo "Installing GPT-SoVITS optional dependencies..."
  pip_install \
    "pytorch-lightning>=2.4" funasr==1.0.27 cn2an pypinyin g2p_en \
    modelscope sentencepiece "peft<0.18.0" chardet jieba split-lang \
    "fast_langdetect>=0.3.1" wordsegment rotary_embedding_torch ToJyutping \
    g2pk2 ko_pron opencc x_transformers "torchmetrics<=1.5" \
    pydantic==2.10.6 "ctranslate2>=4.0,<5" faster-whisper

  if [[ "${INSTALL_COMPILE_DEPS}" == "1" ]]; then
    echo "Installing GPT-SoVITS compile-dependent packages: pyopenjtalk jieba_fast..."
    pip_install "pyopenjtalk>=0.4.1" jieba_fast
  else
    echo "Skipped pyopenjtalk and jieba_fast. Pass --with-compile-deps after installing a C/C++ toolchain."
  fi

  if [[ "${INSTALL_LATENTSYNC}" == "1" ]]; then
    echo "Both GPT-SoVITS and LatentSync were requested; keeping gradio==5.24.0 for LatentSync."
    pip_install gradio==5.24.0
  fi
fi

echo "Restoring GPU onnxruntime if CPU onnxruntime was pulled by optional packages..."
if "${PYTHON_BIN}" -m pip show onnxruntime >/dev/null 2>&1; then
  "${PYTHON_BIN}" -m pip uninstall -y onnxruntime
fi
pip_install --force-reinstall --no-deps onnxruntime-gpu==1.21.0

if [[ "${RUN_CHECK}" == "1" ]]; then
  echo "Running pip check..."
  "${PYTHON_BIN}" -m pip check || true

  echo "Running LiveTalking import check..."
  "${PYTHON_BIN}" - <<'PY'
import torch
import numpy
import cv2
import flask
import aiohttp
import aiortc
import onnxruntime as ort

print("imports-ok")
print("torch", torch.__version__)
print("numpy", numpy.__version__)
print("onnxruntime", ort.__version__, ort.get_available_providers())
PY

  echo "Running app.py --help check..."
  (cd "${PROJECT_ROOT}" && "${PYTHON_BIN}" app.py --help >/dev/null)
fi

cat <<EOF

Done.

Start LiveTalking:
  cd "${PROJECT_ROOT}"
  "${PYTHON_BIN}" app.py --transport webrtc --model wav2lip --avatar_id wav2lip256_avatar1 --listenport 8010

Open:
  http://127.0.0.1:8010/dashboard.html

Send text after a WebRTC session is created:
  curl -X POST http://127.0.0.1:8010/human \\
    -H 'Content-Type: application/json' \\
    -d '{"sessionid":"YOUR_SESSION_ID","type":"echo","text":"hello","interrupt":false}'

EOF
