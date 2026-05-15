#!/usr/bin/env bash
# ============================================================================
# NavGen 8-GPU Parallel Data Generation Script
# ============================================================================
# Usage:
#   ./run_navgen_8gpu.sh [options]
#
# Options:
#   --total-target N        Total number of successful tasks across all 8 GPUs
#                           Default: 1000 (each GPU targets N/8 = 125)
#   --max-attempts-per-gpu N  Max attempts per GPU. Default: 10 * per-gpu-target
#   --max-step N            Max navigation steps per task. Default: 500
#   --training-data-mode    Enable training data mode (prune debug artifacts)
#   --resume                Resume interrupted runs
#   --conda-env NAME        Conda environment name. Default: lhvln-cu128
#   --conda-exe PATH        Conda executable path.
#                           Default: /mnt/data/gengshuang/miniconda3/bin/conda
#   --run-name-prefix STR   Prefix for each GPU's run name.
#                           Default: navgen_8gpu
#   --remote-user USER      Remote SSH user. Default: root
#   --remote-host HOST      Remote SSH host. Default: 139.196.171.150
#   --remote-port PORT      Remote SSH port. Default: 6222
#   --remote-base-dir DIR   Remote base directory.
#                           Default: /mnt/data-cpfs/gengshuang/lhvln_dataset
#   --no-remote             Skip remote sync (run locally only)
#   --help                  Show this help message
#
# Example:
#   ./run_navgen_8gpu.sh --total-target 1000 --training-data-mode --max-step 500
# ============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# --- Defaults ---
TOTAL_TARGET=1000
MAX_ATTEMPTS_PER_GPU=""
MAX_STEP=500
TRAINING_DATA_MODE=0
RESUME=0
CONDA_ENV="lhvln-cu128"
CONDA_EXE="/mnt/data/gengshuang/miniconda3/bin/conda"
PYTHON_BIN="/mnt/data/gengshuang/miniconda3/envs/lhvln-cu128/bin/python"
RUN_NAME_PREFIX="navgen_8gpu"
REMOTE_USER="root"
REMOTE_HOST="139.196.171.150"
REMOTE_PORT="6222"
REMOTE_BASE_DIR="/mnt/data-cpfs/gengshuang/lhvln_dataset"
NO_REMOTE=0
NUM_GPUS=8

# --- Parse arguments ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    --total-target)
      TOTAL_TARGET="${2:?missing value for --total-target}"
      shift 2
      ;;
    --max-attempts-per-gpu)
      MAX_ATTEMPTS_PER_GPU="${2:?missing value for --max-attempts-per-gpu}"
      shift 2
      ;;
    --max-step)
      MAX_STEP="${2:?missing value for --max-step}"
      shift 2
      ;;
    --training-data-mode)
      TRAINING_DATA_MODE=1
      shift
      ;;
    --resume)
      RESUME=1
      shift
      ;;
    --conda-env)
      CONDA_ENV="${2:?missing value for --conda-env}"
      shift 2
      ;;
    --conda-exe)
      CONDA_EXE="${2:?missing value for --conda-exe}"
      shift 2
      ;;
    --run-name-prefix)
      RUN_NAME_PREFIX="${2:?missing value for --run-name-prefix}"
      shift 2
      ;;
    --remote-user)
      REMOTE_USER="${2:?missing value for --remote-user}"
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
    --remote-base-dir)
      REMOTE_BASE_DIR="${2:?missing value for --remote-base-dir}"
      shift 2
      ;;
    --no-remote)
      NO_REMOTE=1
      shift
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

# --- Validate ---
if ! [[ "$TOTAL_TARGET" =~ ^[0-9]+$ ]] || [[ "$TOTAL_TARGET" -lt 1 ]]; then
  echo "--total-target must be a positive integer, got: $TOTAL_TARGET" >&2
  exit 1
fi

if [[ ! -x "$CONDA_EXE" ]]; then
  echo "Conda executable not found at: $CONDA_EXE" >&2
  echo "Please run setup_conda_env.sh first to install Miniconda and create the environment." >&2
  exit 1
fi

# --- Compute per-GPU target ---
PER_GPU_TARGET=$(( (TOTAL_TARGET + NUM_GPUS - 1) / NUM_GPUS ))
# The last GPU may have a smaller target to hit the exact total
LAST_GPU_TARGET=$(( TOTAL_TARGET - PER_GPU_TARGET * (NUM_GPUS - 1) ))
if [[ "$LAST_GPU_TARGET" -lt 1 ]]; then
  LAST_GPU_TARGET=1
fi

if [[ -z "$MAX_ATTEMPTS_PER_GPU" ]]; then
  MAX_ATTEMPTS_PER_GPU=$(( PER_GPU_TARGET * 10 ))
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"

echo "============================================"
echo " NavGen 8-GPU Parallel Data Generation"
echo "============================================"
echo " Total target     : ${TOTAL_TARGET} successful tasks"
echo " Per-GPU target   : ${PER_GPU_TARGET} (GPU 7: ${LAST_GPU_TARGET})"
echo " Max attempts/GPU : ${MAX_ATTEMPTS_PER_GPU}"
echo " Max step         : ${MAX_STEP}"
echo " Training mode    : ${TRAINING_DATA_MODE}"
echo " Resume           : ${RESUME}"
echo " Conda env        : ${CONDA_ENV}"
echo " Conda exe        : ${CONDA_EXE}"
echo " GPUs             : 0-${NUM_GPUS}-1 (${NUM_GPUS} GPUs)"
echo " Timestamp        : ${TIMESTAMP}"
echo "============================================"

# --- Check GPU availability ---
AVAILABLE_GPUS="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | wc -l)"
if [[ "$AVAILABLE_GPUS" -lt "$NUM_GPUS" ]]; then
  echo "[ERROR] Only ${AVAILABLE_GPUS} GPUs available, need ${NUM_GPUS}" >&2
  exit 1
fi

# --- Launch each GPU process ---
PIDS=()
LOG_DIR="${SCRIPT_DIR}/nav_gen/remote_batches/${RUN_NAME_PREFIX}_${TIMESTAMP}_logs"
mkdir -p "$LOG_DIR"

for GPU_ID in 0 1 2 3 4 5 6 7; do
  if [[ "$GPU_ID" -eq 7 ]]; then
    GPU_TARGET="$LAST_GPU_TARGET"
  else
    GPU_TARGET="$PER_GPU_TARGET"
  fi

  RUN_NAME="${RUN_NAME_PREFIX}_${TIMESTAMP}_gpu${GPU_ID}"
  LOG_FILE="${LOG_DIR}/gpu${GPU_ID}.log"

  COMMON_ARGS=(
    --run-name "$RUN_NAME"
    --sim-gpu-device "$GPU_ID"
    --ram-device "cuda:${GPU_ID}"
    --conda-env "$CONDA_ENV"
    --conda-exe "$CONDA_EXE"
    --loop "$GPU_TARGET"
    --max-attempts "$MAX_ATTEMPTS_PER_GPU"
    --max_step "$MAX_STEP"
    --progress-interval 5
  )

  if [[ "$TRAINING_DATA_MODE" -eq 1 ]]; then
    COMMON_ARGS+=(--training-data-mode)
  fi

  if [[ "$RESUME" -eq 1 ]]; then
    COMMON_ARGS+=(--resume)
  fi

  if [[ "$NO_REMOTE" -eq 1 ]]; then
    # Run locally without remote sync — use direct Python binary
    LOCAL_ROOT="${SCRIPT_DIR}/nav_gen/remote_batches/${RUN_NAME}"
    mkdir -p "$LOCAL_ROOT/success" "$LOCAL_ROOT/failure" "$LOCAL_ROOT/_running"

    nohup env \
      PYTHONUNBUFFERED=1 \
      MAGNUM_DEVICE=HeadlessEGL \
      NAVGEN_SIM_GPU_DEVICE="$GPU_ID" \
      NAVGEN_RAM_DEVICE="cuda:${GPU_ID}" \
      "$PYTHON_BIN" "${SCRIPT_DIR}/nav_gen/main.py" \
        --loop "$GPU_TARGET" \
        --render_sensor_height 1.0 \
        --max_step "$MAX_STEP" \
        > "$LOG_FILE" 2>&1 &
  else
    # Use the full batch wrapper with remote sync
    nohup env PYTHONUNBUFFERED=1 "$SCRIPT_DIR/run_navgen_remote_batch.sh" \
      "${COMMON_ARGS[@]}" \
      > "$LOG_FILE" 2>&1 &
  fi

  PIDS+=($!)
done

echo ""
echo "All 8 GPU processes launched. PIDs: ${PIDS[*]}"
echo "Log directory: ${LOG_DIR}"
echo ""
echo "To monitor progress:"
echo "  tail -f ${LOG_DIR}/gpu0.log"
echo "  # or check all:"
echo "  for i in 0 1 2 3 4 5 6 7; do echo '--- GPU \$i ---'; tail -3 ${LOG_DIR}/gpu\${i}.log; done"
echo ""
echo "To stop all processes:"
echo "  kill ${PIDS[*]}"
echo ""
echo "Waiting for all processes to complete..."

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
echo " All processes completed"
echo "============================================"
echo " Failed GPUs: ${FAIL} / ${NUM_GPUS}"

# --- Collect summary ---
echo ""
echo "Per-GPU summary:"
for GPU_ID in 0 1 2 3 4 5 6 7; do
  RUN_NAME="${RUN_NAME_PREFIX}_${TIMESTAMP}_gpu${GPU_ID}"
  SUMMARY="${SCRIPT_DIR}/nav_gen/remote_batches/${RUN_NAME}/batch_summary.tsv"
  if [[ -f "$SUMMARY" ]]; then
    SUCCESS_COUNT="$(awk -F'\t' 'NR > 1 && $2 == "ok" { count += 1 } END { print count + 0 }' "$SUMMARY")"
    TOTAL_ROWS="$(awk 'NR > 1 { count += 1 } END { print count + 0 }' "$SUMMARY")"
    echo "  GPU ${GPU_ID}: ${SUCCESS_COUNT} successes / ${TOTAL_ROWS} attempts"
  else
    echo "  GPU ${GPU_ID}: summary not found"
  fi
done

# --- Compute total success ---
TOTAL_SUCCESS=0
for GPU_ID in 0 1 2 3 4 5 6 7; do
  RUN_NAME="${RUN_NAME_PREFIX}_${TIMESTAMP}_gpu${GPU_ID}"
  SUMMARY="${SCRIPT_DIR}/nav_gen/remote_batches/${RUN_NAME}/batch_summary.tsv"
  if [[ -f "$SUMMARY" ]]; then
    SC="$(awk -F'\t' 'NR > 1 && $2 == "ok" { count += 1 } END { print count + 0 }' "$SUMMARY")"
    TOTAL_SUCCESS=$((TOTAL_SUCCESS + SC))
  fi
done

echo ""
echo "Total successes: ${TOTAL_SUCCESS} / ${TOTAL_TARGET} target"
echo "Log directory: ${LOG_DIR}"

if [[ "$FAIL" -gt 0 ]]; then
  exit 1
fi