#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
HABITAT_LAB_ROOT="${HABITAT_LAB_ROOT:-${PROJECT_ROOT}/habitat-lab}"

# 本地运行 Habitat 仿真，远端大模型通过 OpenAI 兼容接口访问。
# 典型用法：
#   1. 直接填写远端服务地址：
#      export QWEN_API_BASE_URL=http://<server-ip>:8000/v1
#   2. 或先做 SSH 端口映射，再复用本机地址：
#      ssh -fN -L 8000:localhost:8000 <user>@<server> -p <port>
#      export QWEN_API_BASE_URL=http://127.0.0.1:8000/v1
#   3. 模型名固定通过 curl http://localhost:8000/v1/models 自动探测，
#      不再从环境变量读取。

export PYTHONPATH=.

CONDA_ROOT="${CONDA_ROOT:-/home/gs/anaconda3}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-vlnce}"
export HABITAT_GPU="${HABITAT_GPU:-0}"

cd "${PROJECT_ROOT}"

source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME}"

export PYTHONPATH="${PROJECT_ROOT}:${HABITAT_LAB_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export CUDA_VISIBLE_DEVICES="${HABITAT_GPU:-0}"

resolved_api_provider="${QWEN_API_PROVIDER:-local}"
resolved_api_base_url="${QWEN_API_BASE_URL:-${OPENAI_BASE_URL:-${OPENAI_API_BASE:-}}}"

if [ -z "${resolved_api_base_url}" ]; then
  if [ "${resolved_api_provider}" = "dashscope" ]; then
    resolved_api_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
  else
    resolved_api_base_url="http://127.0.0.1:8000/v1"
  fi
fi

resolved_model=""

if [ "${resolved_api_provider}" = "local" ]; then
  models_endpoint="http://localhost:8000/v1/models"
  echo "Auto-detecting deployed model from ${models_endpoint}" >&2
  models_json="$(curl -fsS "${models_endpoint}")"
  resolved_model="$(
    MODELS_JSON="${models_json}" python3 - <<'PY'
import json
import os
import sys

payload = json.loads(os.environ["MODELS_JSON"])
models = payload.get("data") or []
if not models:
    sys.exit(1)

model_id = models[0].get("id")
if not model_id:
    sys.exit(1)

print(model_id)
PY
  )" || {
    echo "Failed to auto-detect a deployed model from ${models_endpoint}." >&2
    exit 1
  }
fi

cmd=(
  python3 qwen_navigation_test/test_qwen_vln.py
  --config "${QWEN_CONFIG:-vlnce_baselines/config/r2r_baselines/nonlearning.yaml}"
  --api-provider "${resolved_api_provider}"
  --api-base-url "${resolved_api_base_url}"
  --thinking-mode "${QWEN_THINKING_MODE:-off}"
  --max-retries "${QWEN_MAX_RETRIES:-3}"
  --history-len "${QWEN_HISTORY_LEN:-3}"
  --split "${QWEN_SPLIT:-val_unseen}"
  --num-episodes "${QWEN_NUM_EPISODES:-2}"
  --output-dir "${QWEN_OUTPUT_DIR:-qwen_navigation_test/results}"
  --dataset-root "${QWEN_DATASET_ROOT:-/home/gs/my_test/vln_dataset/data/datasets/r2r}"
  --scenes-dir "${QWEN_SCENES_DIR:-/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat}"
)

if [ -n "${resolved_model}" ]; then
  cmd+=(--model "${resolved_model}")
fi

if [ -n "${QWEN_API_KEY:-}" ]; then
  cmd+=(--api-key "${QWEN_API_KEY}")
fi

if [ -n "${QWEN_API_KEY_ENV:-}" ]; then
  cmd+=(--api-key-env "${QWEN_API_KEY_ENV}")
fi

"${cmd[@]}"
