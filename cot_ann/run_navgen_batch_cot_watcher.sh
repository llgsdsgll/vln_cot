#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEBUG_ROOT="${PROJECT_ROOT}/debug"

REMOTE_HOST="${REMOTE_HOST:-139.196.171.150}"
REMOTE_PORT="${REMOTE_PORT:-6222}"
REMOTE_USER="${REMOTE_USER:-root}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data-cpfs/gengshuang/lhvln_dataset/navgen_batch_20260514_train1000_step_assets_novideo_v2}"
SUMMARY_PATH="${SUMMARY_PATH:-${REMOTE_ROOT}/batch_summary.tsv}"

STATE_DB="${STATE_DB:-${DEBUG_ROOT}/navgen_batch_cot_state.sqlite3}"
LOCAL_STAGE_ROOT="${LOCAL_STAGE_ROOT:-${DEBUG_ROOT}/navgen_batch_live_stage}"

POLL_INTERVAL="${POLL_INTERVAL:-120}"
STABLE_SECONDS="${STABLE_SECONDS:-90}"
MAX_ITEM_ATTEMPTS="${MAX_ITEM_ATTEMPTS:-20}"
MAX_ITEMS_PER_CYCLE="${MAX_ITEMS_PER_CYCLE:-5}"
MAX_CONCURRENT_WORKERS="${MAX_CONCURRENT_WORKERS:-5}"
STALE_PROCESSING_SECONDS="${STALE_PROCESSING_SECONDS:-1800}"
NETWORK_COMMAND_RETRIES="${NETWORK_COMMAND_RETRIES:-6}"
NETWORK_RETRY_DELAY_SECONDS="${NETWORK_RETRY_DELAY_SECONDS:-5}"
SSH_CONNECT_TIMEOUT_SECONDS="${SSH_CONNECT_TIMEOUT_SECONDS:-20}"
SSH_SERVER_ALIVE_INTERVAL_SECONDS="${SSH_SERVER_ALIVE_INTERVAL_SECONDS:-30}"
SSH_SERVER_ALIVE_COUNT_MAX="${SSH_SERVER_ALIVE_COUNT_MAX:-6}"

API_PROVIDER="${API_PROVIDER:-dashscope}"
API_BASE_URL="${API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MODEL_ID="${MODEL_ID:-qwen3.6-plus}"
API_KEY_ENV="${API_KEY_ENV:-DASHSCOPE_API_KEY}"
API_KEY="${API_KEY:-}"
THINKING_MODE="${THINKING_MODE:-off}"
MAX_RETRIES="${MAX_RETRIES:-3}"
INVALID_FRAME_POLICY="${INVALID_FRAME_POLICY:-template}"

DELETE_LOCAL_ON_SUCCESS="${DELETE_LOCAL_ON_SUCCESS:-1}"
SKIP_EXISTING_REMOTE_OUTPUT="${SKIP_EXISTING_REMOTE_OUTPUT:-1}"
EXPORT_MARKDOWN="${EXPORT_MARKDOWN:-0}"
INCLUDE_PROMPT_IN_MARKDOWN="${INCLUDE_PROMPT_IN_MARKDOWN:-0}"

EXTRA_ARGS=("$@")

mkdir -p "${DEBUG_ROOT}" "${LOCAL_STAGE_ROOT}"

if [[ -z "${API_KEY}" && -z "${!API_KEY_ENV:-}" ]]; then
    echo "[ERROR] 未检测到 API Key。请设置环境变量 ${API_KEY_ENV}，或导出 API_KEY 后重试。" >&2
    exit 1
fi

echo "Remote root : ${REMOTE_ROOT}"
echo "Summary TSV : ${SUMMARY_PATH}"
echo "State DB    : ${STATE_DB}"
echo "Local stage : ${LOCAL_STAGE_ROOT}"
echo "Provider    : ${API_PROVIDER}"
echo "Base URL    : ${API_BASE_URL}"
echo "Model       : ${MODEL_ID}"
echo "Poll(sec)   : ${POLL_INTERVAL}"
echo "Stable(sec) : ${STABLE_SECONDS}"
echo "Retry/item  : ${MAX_ITEM_ATTEMPTS}"
echo "Workers     : ${MAX_CONCURRENT_WORKERS}"
echo "Stale(sec)  : ${STALE_PROCESSING_SECONDS}"
echo "SSH Retries : ${NETWORK_COMMAND_RETRIES}"
echo ""

CMD=(
    python3 "${SCRIPT_DIR}/navgen_batch_cot_watcher.py"
    --remote-root "${REMOTE_ROOT}"
    --summary-path "${SUMMARY_PATH}"
    --remote-host "${REMOTE_HOST}"
    --remote-port "${REMOTE_PORT}"
    --remote-user "${REMOTE_USER}"
    --state-db "${STATE_DB}"
    --local-stage-root "${LOCAL_STAGE_ROOT}"
    --poll-interval "${POLL_INTERVAL}"
    --stable-seconds "${STABLE_SECONDS}"
    --max-item-attempts "${MAX_ITEM_ATTEMPTS}"
    --max-items-per-cycle "${MAX_ITEMS_PER_CYCLE}"
    --max-concurrent-workers "${MAX_CONCURRENT_WORKERS}"
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

if [[ "${DELETE_LOCAL_ON_SUCCESS}" == "1" ]]; then
    CMD+=(--delete-local-on-success)
else
    CMD+=(--keep-local-on-success)
fi

if [[ "${SKIP_EXISTING_REMOTE_OUTPUT}" == "1" ]]; then
    CMD+=(--skip-existing-remote-output)
else
    CMD+=(--force-regenerate-remote-output)
fi

if [[ "${EXPORT_MARKDOWN}" == "1" ]]; then
    CMD+=(--export-markdown)
fi

if [[ "${INCLUDE_PROMPT_IN_MARKDOWN}" == "1" ]]; then
    CMD+=(--include-prompt-in-markdown)
fi

if [[ -n "${API_KEY}" ]]; then
    CMD+=(--api-key "${API_KEY}")
fi

CMD+=("${EXTRA_ARGS[@]}")

exec "${CMD[@]}"
