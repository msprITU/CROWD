#!/usr/bin/env bash
set -Eeuo pipefail

# Install the CLTR Python environment. Run from any directory; paths are
# resolved relative to this script. System NVIDIA drivers and CUDA Toolkit
# are deliberately not installed or modified by this script.

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-"$ROOT_DIR/.venv"}"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"

info() { printf '\n[CLTR setup] %s\n' "$*"; }
fail() { printf '\n[CLTR setup] ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$ROOT_DIR/CLTR" ]] || fail "Cannot find the CLTR/ directory next to setup.sh."
[[ -f "$ROOT_DIR/requirements.txt" ]] || fail "Cannot find requirements.txt next to setup.sh."
[[ -d "$ROOT_DIR/mmcv_m/utils" ]] || fail "Cannot find mmcv_m/utils/. This project-specific directory is required."

command -v "$PYTHON_BIN" >/dev/null 2>&1 || fail \
  "$PYTHON_BIN was not found. Install Python 3.11 and its venv support, or set PYTHON_BIN=/path/to/python3.11."

PYTHON_VERSION="$("$PYTHON_BIN" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
[[ "$PYTHON_VERSION" == "3.11" ]] || fail \
  "Python 3.11 is required; $PYTHON_BIN reports Python $PYTHON_VERSION."

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  info "Creating virtual environment at $VENV_DIR"
  "$PYTHON_BIN" -m venv "$VENV_DIR" || fail \
    "Could not create the virtual environment. On Ubuntu, install python3.11-venv first."
else
  info "Using existing virtual environment: $VENV_DIR"
fi

PY="$VENV_DIR/bin/python"
PIP=("$PY" -m pip)

info "Updating pip and installing build tools"
"${PIP[@]}" install --upgrade pip wheel ninja "setuptools<82"

info "Installing PyTorch 2.5.1 and torchvision 0.20.1 (CUDA 12.4 wheels)"
"${PIP[@]}" install --index-url https://download.pytorch.org/whl/cu124 \
  "torch==2.5.1" "torchvision==0.20.1"

info "Installing project Python dependencies"
# Torch/torchvision use the official PyTorch wheel index; MMCV is compiled
# separately below. Filter them out so pip does not replace those builds.
FILTERED_REQ="$(mktemp)"
trap 'rm -f "$FILTERED_REQ"' EXIT
awk '!/^[[:space:]]*(torch|torchvision|mmcv)([<=>~!;[:space:]]|$)/' \
  "$ROOT_DIR/requirements.txt" > "$FILTERED_REQ"
"${PIP[@]}" install -r "$FILTERED_REQ"

info "Checking CUDA Toolkit and compiler"
if [[ -x "$CUDA_HOME/bin/nvcc" ]]; then
  export CUDA_HOME
  export PATH="$CUDA_HOME/bin:$PATH"
elif command -v nvcc >/dev/null 2>&1; then
  CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
  export CUDA_HOME
else
  fail "CUDA Toolkit nvcc was not found. Install CUDA Toolkit 12.4 (the driver alone is not enough), then rerun: CUDA_HOME=/usr/local/cuda-12.4 ./setup.sh"
fi

NVCC_VERSION="$(nvcc --version | sed -n 's/.*release \([0-9][0-9]*\.[0-9][0-9]*\).*/\1/p' | tail -n 1)"
[[ "$NVCC_VERSION" == "12.4" ]] || fail \
  "This setup is tested with CUDA Toolkit 12.4; found nvcc version '${NVCC_VERSION:-unknown}' at $CUDA_HOME. Set CUDA_HOME to the 12.4 toolkit."

# Prefer GCC 12 when present; it is compatible with the tested CUDA 12.4 setup.
if command -v gcc-12 >/dev/null 2>&1 && command -v g++-12 >/dev/null 2>&1; then
  export CC="${CC:-gcc-12}"
  export CXX="${CXX:-g++-12}"
fi
export FORCE_CUDA=1

info "Building MMCV 2.1.0 with CUDA operations"
"${PIP[@]}" uninstall -y mmcv mmcv-lite >/dev/null 2>&1 || true
"${PIP[@]}" install --no-build-isolation --no-cache-dir "mmcv==2.1.0"

info "Applying the project's custom MMCV utility files"
MMCV_UTILS_DIR="$("$PY" -c 'import os, mmcv; print(os.path.join(os.path.dirname(mmcv.__file__), "utils"))')"
cp -a "$ROOT_DIR/mmcv_m/utils/." "$MMCV_UTILS_DIR/"

info "Running installation checks"
"$PY" - <<'PY'
import torch
import torchvision
import scipy
import h5py
import PIL
import imageio
import cv2
import mmcv
from mmcv.ops import nms
from torch.utils.tensorboard import SummaryWriter
import nni
import sklearn
import matplotlib

print("PyTorch:", torch.__version__)
print("Torchvision:", torchvision.__version__)
print("SciPy:", scipy.__version__)
print("h5py:", h5py.__version__)
print("Pillow:", PIL.__version__)
print("imageio:", imageio.__version__)
print("OpenCV:", cv2.__version__)
print("MMCV:", mmcv.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
print("Core dependencies: OK")
PY

info "Checking key CLTR imports"
cd "$ROOT_DIR"
PYTHONPATH="$ROOT_DIR/CLTR${PYTHONPATH:+:$PYTHONPATH}" "$PY" - <<'PY'
import importlib
modules = [
    "config", "utils", "image", "inference",
    "util.misc", "util.box_ops", "util.dataset2",
]
for name in modules:
    importlib.import_module(name)
    print(f"[OK] {name}")
PY

info "Testing MMCV NMS on the GPU when CUDA is available"
"$PY" - <<'PY'
import torch
from mmcv.ops import nms
if not torch.cuda.is_available():
    print("CUDA is not available to PyTorch; skipped GPU NMS runtime test.")
else:
    boxes = torch.tensor([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=torch.float32, device="cuda")
    scores = torch.tensor([0.9, 0.8], device="cuda")
    dets, keep = nms(boxes, scores, 0.5)
    assert keep.numel() == 1, f"Unexpected NMS result: {keep}"
    print("GPU:", torch.cuda.get_device_name(0))
    print("MMCV CUDA NMS: OK")
PY

cat <<EOF

[CLTR setup] Installation completed.

Activate the environment in your terminal with:
  source "$VENV_DIR/bin/activate"

Then run your CLTR scripts from the project root or follow the project-specific
training instructions in README.md. Dataset paths and training configuration
must still be set for your machine.
EOF
