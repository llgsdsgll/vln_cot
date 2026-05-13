#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./scripts/setup_current_navgen_env.sh [options]

Options:
  --env-name NAME           Conda env name. Default: lhvln-cu128
  --python VERSION          Python version. Default: 3.9
  --conda-exe PATH          Conda executable path. Default: auto-detect
  --torch-index-url URL     Torch wheel index URL.
                            Default: https://download.pytorch.org/whl/cu128
  --repo-root DIR           Repo root. Default: parent of this script
  --skip-habitat            Skip habitat-sim installation
  --skip-sanity-check       Skip final import / CUDA sanity check
  --help                    Show this help message

Notes:
  1. This script mirrors the currently verified local NavGen runtime setup.
  2. It is based on the active lhvln-cu128 environment that is working with
     Habitat-Sim, RAM, and NavGen on the current machine.
  3. It keeps the default env name as lhvln-cu128 so existing batch scripts
     can continue to run without modification.
EOF
}

ENV_NAME="lhvln-cu128"
PYTHON_VERSION="3.9"
TORCH_INDEX_URL="https://download.pytorch.org/whl/cu128"
INSTALL_HABITAT=1
RUN_SANITY_CHECK=1
REPO_ROOT=""
CONDA_EXE="${CONDA_EXE:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env-name)
      ENV_NAME="${2:?missing value for --env-name}"
      shift 2
      ;;
    --python)
      PYTHON_VERSION="${2:?missing value for --python}"
      shift 2
      ;;
    --conda-exe)
      CONDA_EXE="${2:?missing value for --conda-exe}"
      shift 2
      ;;
    --torch-index-url)
      TORCH_INDEX_URL="${2:?missing value for --torch-index-url}"
      shift 2
      ;;
    --repo-root)
      REPO_ROOT="${2:?missing value for --repo-root}"
      shift 2
      ;;
    --skip-habitat)
      INSTALL_HABITAT=0
      shift
      ;;
    --skip-sanity-check)
      RUN_SANITY_CHECK=0
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "$CONDA_EXE" ]]; then
  CONDA_EXE="$(command -v conda || true)"
fi
if [[ -z "$CONDA_EXE" || ! -x "$CONDA_EXE" ]]; then
  echo "Unable to find a usable conda executable. Pass --conda-exe PATH." >&2
  exit 1
fi

if [[ -z "$REPO_ROOT" ]]; then
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fi
RUNTIME_REQUIREMENTS="$REPO_ROOT/requirements_navgen_runtime_current.txt"

if [[ ! -f "$RUNTIME_REQUIREMENTS" ]]; then
  echo "Missing runtime requirements file: $RUNTIME_REQUIREMENTS" >&2
  exit 1
fi

echo "[1/6] Preparing conda env: $ENV_NAME"
if "$CONDA_EXE" env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  echo "  Env $ENV_NAME already exists. Reusing it."
else
  "$CONDA_EXE" create -y -n "$ENV_NAME" "python=$PYTHON_VERSION"
fi

echo "[2/6] Installing base conda packages"
"$CONDA_EXE" install -y -n "$ENV_NAME" -c conda-forge \
  pip git ffmpeg cmake ninja pkg-config

if [[ "$INSTALL_HABITAT" -eq 1 ]]; then
  echo "[3/6] Installing habitat-sim 0.3.1 (headless)"
  "$CONDA_EXE" install -y -n "$ENV_NAME" habitat-sim==0.3.1 headless -c conda-forge -c aihabitat
else
  echo "[3/6] Skipping habitat-sim installation"
fi

echo "[4/6] Installing PyTorch wheels matching the current working setup"
"$CONDA_EXE" run -n "$ENV_NAME" python -m pip install --upgrade pip setuptools wheel
"$CONDA_EXE" run -n "$ENV_NAME" python -m pip install \
  --index-url "$TORCH_INDEX_URL" \
  torch==2.8.0 torchvision==0.23.0

echo "[5/6] Installing runtime dependencies pinned to the current environment"
"$CONDA_EXE" run -n "$ENV_NAME" python -m pip install -r "$RUNTIME_REQUIREMENTS"

echo "[6/6] Writing activation hints"
ACTIVATE_HINT="$REPO_ROOT/scripts/activate_${ENV_NAME}_current.sh"
cat > "$ACTIVATE_HINT" <<EOF
#!/usr/bin/env bash
set -euo pipefail
eval "\$($CONDA_EXE shell.bash hook)"
conda activate "$ENV_NAME"
cd "$REPO_ROOT"
export DASHSCOPE_BASE_URL="\${DASHSCOPE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export NAVGEN_LLM_MODEL="\${NAVGEN_LLM_MODEL:-qwen3.6-plus}"
export NAVGEN_VLM_MODEL="\${NAVGEN_VLM_MODEL:-\$NAVGEN_LLM_MODEL}"
echo "Activated $ENV_NAME at $REPO_ROOT"
EOF
chmod +x "$ACTIVATE_HINT"

if [[ "$RUN_SANITY_CHECK" -eq 1 ]]; then
  echo
  echo "Running sanity check..."
  "$CONDA_EXE" run -n "$ENV_NAME" python - <<'PY'
import sys

checks = [
    ("torch", "import torch; print('torch', torch.__version__)"),
    ("torchvision", "import torchvision; print('torchvision', torchvision.__version__)"),
    ("habitat_sim", "import habitat_sim; print('habitat_sim ok')"),
    ("openai", "import openai; print('openai', getattr(openai, '__version__', 'n/a'))"),
    ("timm", "import timm; print('timm', timm.__version__)"),
    ("transformers", "import transformers; print('transformers', transformers.__version__)"),
    ("numpy", "import numpy; print('numpy', numpy.__version__)"),
    ("PIL", "import PIL; print('pillow', PIL.__version__)"),
    ("PyQt5", "import PyQt5; print('PyQt5 ok')"),
    ("cv2", "import cv2; print('cv2', cv2.__version__)"),
    ("scipy", "import scipy; print('scipy', scipy.__version__)"),
    ("sentencepiece", "import sentencepiece; print('sentencepiece', sentencepiece.__version__)"),
    ("fairscale", "import fairscale; print('fairscale', fairscale.__version__)"),
]

for name, code in checks:
    try:
        exec(code, {})
    except Exception as exc:
        print(f'[FAIL] {name}: {exc}', file=sys.stderr)
        raise

try:
    import torch
    print('cuda_available', torch.cuda.is_available())
    print('torch_cuda', torch.version.cuda)
    if torch.cuda.is_available():
        print('gpu_name', torch.cuda.get_device_name(0))
except Exception as exc:
    print(f'[WARN] CUDA probe failed: {exc}', file=sys.stderr)
PY
fi

echo
echo "Setup finished."
echo "Activate with:"
echo "  source \"$ACTIVATE_HINT\""
echo
echo "This script mirrors the current working NavGen runtime."
echo "Before running NavGen, make sure these are ready:"
echo "  - data/hm3d/train"
echo "  - data/hm3d/val"
echo "  - data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
echo "  - data/models/ram_plus_swin_large_14m.pth"
echo "  - DASHSCOPE_API_KEY"
