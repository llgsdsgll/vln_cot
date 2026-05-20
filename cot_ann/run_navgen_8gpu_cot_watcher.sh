#!/usr/bin/env bash
# ============================================================================
# NavGen 8-GPU COT Watcher — parallel CoT annotation for 8 GPU runs
# ============================================================================
# Usage:
#   ./run_navgen_8gpu_cot_watcher.sh [options]
#
# Options:
#   --run-timestamp TS       Timestamp of the 8-GPU run (e.g. 20260515_161408)
#                           If omitted, auto-discovered from local directories
#   --run-name-prefix STR    Prefix for GPU run names. Default: navgen_8gpu
#   --remote-base-dir DIR    Remote base directory.
#                           Default: /mnt/data-cpfs/gengshuang/lhvln_dataset
#   --remote-host HOST       Remote SSH host. Default: 139.196.171.150
#   --remote-port PORT       Remote SSH port. Default: 6222
#   --remote-user USER       Remote SSH user. Default: root
#   --poll-interval N        Poll interval in seconds. Default: 120
#   --stable-seconds N       Stable seconds before processing. Default: 90
#   --max-item-attempts N    Max attempts per item. Default: 20
#   --stale-processing-seconds N  Stale processing timeout. Default: 1800
#   --model STR              Model ID. Default: qwen3.6-plus
#   --api-provider STR       API provider. Default: dashscope
#   --api-base-url URL       API base URL.
#                           Default: https://dashscope.aliyuncs.com/compatible-mode/v1
#   --api-key KEY            API key (or set DASHSCOPE_API_KEY env var)
#   --api-key-env STR        Env var name for API key. Default: DASHSCOPE_API_KEY
#   --thinking-mode STR      Thinking mode. Default: off
#   --invalid-frame-policy STR  Policy for invalid frames. Default: template
#   --max-retries N          Max API retries per frame. Default: 3
#   --delete-local-on-success  Delete local stage after success (default)
#   --keep-local-on-success  Keep local stage after success
#   --skip-existing-remote-output  Skip items with existing cot.jsonl (default)
#   --force-regenerate       Regenerate even if cot.jsonl exists
#   --num-gpus N             Number of GPUs. Default: 8
#   --help                   Show this help message
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEBUG_ROOT="${PROJECT_ROOT}/debug"
LOCAL_BATCH_ROOT="${PROJECT_ROOT}/nav_gen/remote_batches"
PYTHON_BIN="${PYTHON_BIN:-/mnt/data/gengshuang/miniconda3/envs/lhvln-cu128/bin/python}"
WATCHER_PY="${SCRIPT_DIR}/navgen_batch_cot_watcher.py"

# --- Defaults ---
RUN_TIMESTAMP=""
RUN_NAME_PREFIX="navgen_8gpu"
REMOTE_BASE_DIR="/mnt/data-cpfs/gengshuang/lhvln_dataset"
REMOTE_HOST="139.196.171.150"
REMOTE_PORT="6222"
REMOTE_USER="root"
NUM_GPUS=8

POLL_INTERVAL=120
STABLE_SECONDS=90
MAX_ITEM_ATTEMPTS=20
MAX_ITEMS_PER_CYCLE=5
WORKERS_PER_GPU=5
STALE_PROCESSING_SECONDS=1800

NETWORK_COMMAND_RETRIES=6
NETWORK_RETRY_DELAY_SECONDS=5
SSH_CONNECT_TIMEOUT_SECONDS=20
SSH_SERVER_ALIVE_INTERVAL_SECONDS=30
SSH_SERVER_ALIVE_COUNT_MAX=6

API_PROVIDER="dashscope"
API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_ID="qwen3.6-plus"
API_KEY_ENV="DASHSCOPE_API_KEY"
API_KEY=""
THINKING_MODE="off"
MAX_RETRIES=3
INVALID_FRAME_POLICY="template"

DELETE_LOCAL_ON_SUCCESS=1
SKIP_EXISTING_REMOTE_OUTPUT=1
EXPORT_MARKDOWN=0
INCLUDE_PROMPT_IN_MARKDOWN=0

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-timestamp)
      RUN_TIMESTAMP="${2:?missing value for --run-timestamp}"
      shift 2
      ;;
    --run-name-prefix)
      RUN_NAME_PREFIX="${2:?missing value for --run-name-prefix}"
      shift 2
      ;;
    --remote-base-dir)
      REMOTE_BASE_DIR="${2:?missing value for --remote-base-dir}"
      shift 2
      ;;
    --remote-host)
      REMOTE_HOST="${2:?missing value for --remote-host}"
      shift 2
      ;;
    --remote-port)
      REMOTE_PORT="${2:?missing value for --remote-port}"
      shift 2
      ;;
    --remote-user)
      REMOTE_USER="${2:?missing value for --remote-user}"
      shift 2
      ;;
    --poll-interval)
      POLL_INTERVAL="${2:?missing value for --poll-interval}"
      shift 2
      ;;
    --stable-seconds)
      STABLE_SECONDS="${2:?missing value for --stable-seconds}"
      shift 2
      ;;
    --max-item-attempts)
      MAX_ITEM_ATTEMPTS="${2:?missing value for --max-item-attempts}"
      shift 2
      ;;
    --stale-processing-seconds)
      STALE_PROCESSING_SECONDS="${2:?missing value for --stale-processing-seconds}"
      shift 2
      ;;
    --model)
      MODEL_ID="${2:?missing value for --model}"
      shift 2
      ;;
    --api-provider)
      API_PROVIDER="${2:?missing value for --api-provider}"
      shift 2
      ;;
    --api-base-url)
      API_BASE_URL="${2:?missing value for --api-base-url}"
      shift 2
      ;;
    --api-key)
      API_KEY="${2:?missing value for --api-key}"
      shift 2
      ;;
    --api-key-env)
      API_KEY_ENV="${2:?missing value for --api-key-env}"
      shift 2
      ;;
    --thinking-mode)
      THINKING_MODE="${2:?missing value for --thinking-mode}"
      shift 2
      ;;
    --invalid-frame-policy)
      INVALID_FRAME_POLICY="${2:?missing value for --invalid-frame-policy}"
      shift 2
      ;;
    --max-retries)
      MAX_RETRIES="${2:?missing value for --max-retries}"
      shift 2
      ;;
    --delete-local-on-success)
      DELETE_LOCAL_ON_SUCCESS=1
      shift
      ;;
    --keep-local-on-success)
      DELETE_LOCAL_ON_SUCCESS=0
      shift
      ;;
    --skip-existing-remote-output)
      SKIP_EXISTING_REMOTE_OUTPUT=1
      shift
      ;;
    --force-regenerate)
      SKIP_EXISTING_REMOTE_OUTPUT=0
      shift
      ;;
    --num-gpus)
      NUM_GPUS="${2:?missing value for --num-gpus}"
      shift 2
      ;;
    --help|-h)
      head -30 "$0" | grep '^#' | sed 's/^# //'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

# --- Validate API key ---
if [[ -z "${API_KEY}" && -z "${!API_KEY_ENV:-}" ]]; then
  echo "[ERROR] No API key detected. Set env var ${API_KEY_ENV}, or pass --api-key." >&2
  exit 1
fi

# --- Auto-discover timestamp ---
if [[ -z "$RUN_TIMESTAMP" ]]; then
  LATEST_GPU0_DIR="$(ls -d ${LOCAL_BATCH_ROOT}/${RUN_NAME_PREFIX}_*_gpu0 2>/dev/null | sort | tail -1)"
  if [[ -z "$LATEST_GPU0_DIR" ]]; then
    echo "[ERROR] No ${RUN_NAME_PREFIX}_*_gpu0 directory found in ${LOCAL_BATCH_ROOT}" >&2
    echo "Please specify --run-timestamp explicitly." >&2
    exit 1
  fi
  RUN_TIMESTAMP="$(basename "$LATEST_GPU0_DIR" | sed 's/^'${RUN_NAME_PREFIX}'_//' | sed 's/_gpu[0-9]*$//')"
  echo "[INFO] Auto-discovered run timestamp: ${RUN_TIMESTAMP} (from ${LATEST_GPU0_DIR})"
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo " NavGen 8-GPU COT Watcher"
echo "============================================"
echo " Run timestamp   : ${RUN_TIMESTAMP}"
echo " Remote base dir  : ${REMOTE_BASE_DIR}"
echo " GPUs             : 0-${NUM_GPUS}-1 (${NUM_GPUS})"
echo " Model            : ${MODEL_ID} (${API_PROVIDER})"
echo " Poll interval    : ${POLL_INTERVAL}s"
echo " Stable seconds   : ${STABLE_SECONDS}s"
echo " Max item attempts: ${MAX_ITEM_ATTEMPTS}"
echo " Workers/GPU      : ${WORKERS_PER_GPU}"
echo " Items/cycle/GPU  : 1"
echo "============================================"

# --- Create directories ---
COT_RUN_DIR="/dev/shm/navgen_8gpu_cot_${RUN_TIMESTAMP}"
LOG_DIR="${COT_RUN_DIR}/logs"
mkdir -p "$LOG_DIR"

PIDS=()

for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
  GPU_RUN_NAME="${RUN_NAME_PREFIX}_${RUN_TIMESTAMP}_gpu${GPU_ID}"
  REMOTE_ROOT="${REMOTE_BASE_DIR}/${GPU_RUN_NAME}"
  SUMMARY_PATH="${REMOTE_ROOT}/batch_summary.tsv"
  GPU_DIR="${COT_RUN_DIR}/gpu${GPU_ID}"
  STAGE_ROOT="${GPU_DIR}/stage"
  DB_PATH="${GPU_DIR}/cot_watcher.db"
  LOG_FILE="${LOG_DIR}/gpu${GPU_ID}.log"

  mkdir -p "$STAGE_ROOT"

  # --- Build watcher command ---
  CMD=(
    "${PYTHON_BIN}" "${SCRIPT_DIR}/navgen_batch_cot_watcher.py"
    --remote-root "${REMOTE_ROOT}"
    --summary-path "${SUMMARY_PATH}"
    --remote-host "${REMOTE_HOST}"
    --remote-port "${REMOTE_PORT}"
    --remote-user "${REMOTE_USER}"
    --state-db "${DB_PATH}"
    --local-stage-root "${STAGE_ROOT}"
    --poll-interval "${POLL_INTERVAL}"
    --stable-seconds "${STABLE_SECONDS}"
    --max-item-attempts "${MAX_ITEM_ATTEMPTS}"
    --max-items-per-cycle "${MAX_ITEMS_PER_CYCLE}"
    --max-concurrent-workers "${WORKERS_PER_GPU}"
    --stale-processing-seconds "${STALE_PROCESSING_SECONDS}"
    --api-provider "${API_PROVIDER}"
    --api-base-url "${API_BASE_URL}"
    --model "${MODEL_ID}"
    --api-key-env "${API_KEY_ENV}"
    --thinking-mode "${THINKING_MODE}"
    --max-retries "${MAX_RETRIES}"
    --invalid-frame-policy "${INVALID_FRAME_POLICY}"
    --network-command-retries "${NETWORK_COMMAND_RETRIES}"
    --network-retry-delay-seconds "${NETWORK_RETRY_DELAY_SECONDS}"
    --ssh-connect-timeout-seconds "${SSH_CONNECT_TIMEOUT_SECONDS}"
    --ssh-server-alive-interval-seconds "${SSH_SERVER_ALIVE_INTERVAL_SECONDS}"
    --ssh-server-alive-count-max "${SSH_SERVER_ALIVE_COUNT_MAX}"
  )

  if [[ "${DELETE_LOCAL_ON_SUCCESS}" -eq 1 ]]; then
    CMD+=(--delete-local-on-success)
  else
    CMD+=(--keep-local-on-success)
  fi

  if [[ "${SKIP_EXISTING_REMOTE_OUTPUT}" -eq 1 ]]; then
    CMD+=(--skip-existing-remote-output)
  else
    CMD+=(--force-regenerate-remote-output)
  fi

  if [[ -n "${API_KEY}" ]]; then
    CMD+=(--api-key "${API_KEY}")
  fi

  STAGGER_SECONDS=$((GPU_ID * 30))

  if [[ "$STAGGER_SECONDS" -gt 0 ]]; then
    echo "[INFO] GPU ${GPU_ID}: sleeping ${STAGGER_SECONDS}s before starting watcher..."
    sleep "$STAGGER_SECONDS"
  fi

  stdbuf -oL -eL env PYTHONUNBUFFERED=1 "${CMD[@]}" > "$LOG_FILE" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All ${NUM_GPUS} COT watchers launched. PIDs: ${PIDS[*]}"
echo "COT run dir: ${COT_RUN_DIR}"
echo "Log dir: ${LOG_DIR}"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_DIR}/gpu0.log"
echo "  # or check all:"
echo "  for i in 0 1 2 3 4 5 6 7; do echo '--- GPU \$i ---'; tail -3 ${LOG_DIR}/gpu\${i}.log; done"
echo ""
echo "To stop all processes:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Waiting for all watchers to complete..."

# --- Wait for all ---
FAIL=0
for idx in "${!PIDS[@]}"; do
  PID="${PIDS[$idx]}"
  GPU_ID="$idx"
  if ! wait "$PID" 2>/dev/null; then
    echo "[WARN] GPU ${GPU_ID} (PID ${PID}) exited with non-zero status"
    FAIL=$((FAIL + 1))
  fi
done

echo ""
echo "============================================"
echo " All COT watchers completed"
echo "============================================"
echo " Failed GPUs: ${FAIL} / ${NUM_GPUS}"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi