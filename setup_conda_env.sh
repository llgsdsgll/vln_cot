#!/usr/bin/env bash
# ============================================================================
# Setup Conda Environment for NavGen 8-GPU Parallel Generation
# ============================================================================
# This script:
#   1. Installs Miniconda3 (if not already installed)
#   2. Creates the lhvln-cu128 conda environment
#   3. Installs PyTorch with CUDA 12.8 (for RTX 4090 sm_89)
#   4. Installs Habitat-Sim 0.3.1 (headless)
#   5. Installs all NavGen runtime dependencies
#
# Usage:
#   ./setup_conda_env.sh [--skip-miniconda] [--skip-pytorch] [--skip-habitat]
# ============================================================================

set -euo pipefail

CONDA_ROOT="/mnt/data/gengshuang/miniconda3"
CONDA_EXE="${CONDA_ROOT}/bin/conda"
CONDA_ENV="lhvln-cu128"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SKIP_MINICONDA=0
SKIP_PYTORCH=0
SKIP_HABITAT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --skip-miniconda) SKIP_MINICONDA=1; shift ;;
    --skip-pytorch)   SKIP_PYTORCH=1; shift ;;
    --skip-habitat)   SKIP_HABITAT=1; shift ;;
    --help|-h)        echo "Usage: $0 [--skip-miniconda] [--skip-pytorch] [--skip-habitat]"; exit 0 ;;
    *)                echo "Unknown option: $1"; exit 1 ;;
  esac
done

echo "============================================"
echo " NavGen Conda Environment Setup"
echo "============================================"
echo " Conda root : ${CONDA_ROOT}"
echo " Conda exe  : ${CONDA_EXE}"
echo " Env name   : ${CONDA_ENV}"
echo " Project    : ${PROJECT_ROOT}"
echo "============================================"

# --- Step 1: Install Miniconda ---
if [[ "$SKIP_MINICONDA" -eq 0 ]]; then
  if [[ -x "$CONDA_EXE" ]]; then
    echo "[INFO] Miniconda already installed at ${CONDA_ROOT}"
  else
    echo "[INFO] Downloading Miniconda installer..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/Miniconda3-latest-Linux-x86_64.sh

    echo "[INFO] Installing Miniconda to ${CONDA_ROOT}..."
    bash /tmp/Miniconda3-latest-Linux-x86_64.sh -b -p "$CONDA_ROOT"

    echo "[INFO] Running conda init bash..."
    "$CONDA_EXE" init bash

    echo "[INFO] Miniconda installed successfully."
    rm -f /tmp/Miniconda3-latest-Linux-x86_64.sh
  fi
else
  echo "[INFO] Skipping Miniconda installation (--skip-miniconda)"
fi

# --- Step 2: Create conda environment ---
if "$CONDA_EXE" env list | grep -q "^${CONDA_ENV} "; then
  echo "[INFO] Conda environment '${CONDA_ENV}' already exists"
else
  echo "[INFO] Creating conda environment '${CONDA_ENV}' with Python 3.9 (required by habitat-sim==0.3.1)..."
  "$CONDA_EXE" create -n "$CONDA_ENV" python=3.9 -y
  echo "[INFO] Environment created."
fi

# --- Step 3: Install PyTorch with CUDA 12.8 ---
if [[ "$SKIP_PYTORCH" -eq 0 ]]; then
  echo "[INFO] Installing PyTorch 2.7.0 + CUDA 12.8..."
  "$CONDA_EXE" run -n "$CONDA_ENV" pip install \
    torch==2.7.0 torchvision==0.22.0 \
    --index-url https://download.pytorch.org/whl/cu128

  echo "[INFO] Verifying PyTorch + CUDA..."
  "$CONDA_EXE" run -n "$CONDA_ENV" python -c \
    "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}'); print(f'Arch list: {torch.cuda.get_arch_list()}')"
else
  echo "[INFO] Skipping PyTorch installation (--skip-pytorch)"
fi

# --- Step 4: Install Habitat-Sim ---
if [[ "$SKIP_HABITAT" -eq 0 ]]; then
  echo "[INFO] Installing Habitat-Sim 0.2.4 (headless) via conda (requires Python 3.9)..."
  "$CONDA_EXE" install -n "$CONDA_ENV" habitat-sim==0.2.4 headless -c conda-forge -c aihabitat -y
  echo "[INFO] Removing conflicting libglvnd (Mesa GLdispatch overrides NVIDIA EGL)..."
  "$CONDA_EXE" remove -n "$CONDA_ENV" libglvnd --force -y 2>/dev/null || true
  echo "[INFO] Habitat-Sim installed."
else
  echo "[INFO] Skipping Habitat-Sim installation (--skip-habitat)"
fi

# --- Step 5: Install NavGen runtime dependencies ---
echo "[INFO] Installing NavGen runtime dependencies (with numpy fix for habitat-sim)..."
"$CONDA_EXE" run -n "$CONDA_ENV" pip install -r "${PROJECT_ROOT}/requirements_navgen_runtime_current.txt"
"$CONDA_EXE" run -n "$CONDA_ENV" pip install "numpy<1.24,>=1.20" "opencv-python-headless<4.10"

echo "[INFO] All dependencies installed."

# --- Step 6: Verify environment ---
echo ""
echo "============================================"
echo " Environment Verification"
echo "============================================"

"$CONDA_EXE" run -n "$CONDA_ENV" python -c "
import torch
print(f'PyTorch   : {torch.__version__}')
print(f'CUDA      : {torch.cuda.is_available()} ({torch.cuda.device_count()} devices)')
print(f'Arch list : {torch.cuda.get_arch_list()}')

try:
    import habitat_sim
    print(f'Habitat-Sim : {habitat_sim.__version__}')
except ImportError:
    print('Habitat-Sim : NOT INSTALLED')

try:
    from recognize_anything.ram.models import ram_plus
    print('RAM model   : importable')
except ImportError:
    print('RAM model   : NOT IMPORTABLE (need recognize_anything package)')
"

echo ""
echo "============================================"
echo " Setup Complete!"
echo "============================================"
echo " To activate the environment:"
echo "   source ${CONDA_ROOT}/etc/profile.d/conda.sh"
echo "   conda activate ${CONDA_ENV}"
echo ""
echo " To run 8-GPU parallel generation:"
echo "   ./run_navgen_8gpu.sh --total-target 1000 --training-data-mode --max-step 500"
echo ""