#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEBUG_ROOT="${PROJECT_ROOT}/debug"

REMOTE_HOST="${REMOTE_HOST:-139.196.171.150}"
REMOTE_PORT="${REMOTE_PORT:-6222}"
REMOTE_USER="${REMOTE_USER:-root}"
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
STAGE_ROOT="${STAGE_ROOT:-${DEBUG_ROOT}/navgen_remote_stage}"
EXPORT_MARKDOWN="${EXPORT_MARKDOWN:-1}"
INCLUDE_PROMPT_IN_MARKDOWN="${INCLUDE_PROMPT_IN_MARKDOWN:-0}"
API_PROVIDER="${API_PROVIDER:-dashscope}"
API_BASE_URL="${API_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
MODEL_ID="${MODEL_ID:-qwen3.6-plus}"
API_KEY_ENV="${API_KEY_ENV:-DASHSCOPE_API_KEY}"
API_KEY="${API_KEY:-}"
THINKING_MODE="${THINKING_MODE:-off}"
MAX_RETRIES="${MAX_RETRIES:-3}"
INVALID_FRAME_POLICY="${INVALID_FRAME_POLICY:-skip}"

REMOTE_FRAME_INFO=""
OUTPUT_JSONL=""
PASSTHROUGH_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote-frame-info)
            REMOTE_FRAME_INFO="${2:-}"
            shift 2
            ;;
        --remote-frame-info=*)
            REMOTE_FRAME_INFO="${1#*=}"
            shift
            ;;
        --output)
            OUTPUT_JSONL="${2:-}"
            shift 2
            ;;
        --output=*)
            OUTPUT_JSONL="${1#*=}"
            shift
            ;;
        --frame-info|--step-task-json)
            echo "[ERROR] 该包装脚本会自动处理本地 --frame-info/--step-task-json，请改用 --remote-frame-info。" >&2
            exit 1
            ;;
        *)
            PASSTHROUGH_ARGS+=("$1")
            shift
            ;;
    esac
done

if [[ -z "${REMOTE_FRAME_INFO}" ]]; then
    echo "[ERROR] 请通过 --remote-frame-info 指定远程 *.frame_info.json 路径。" >&2
    exit 1
fi

if [[ "${REMOTE_FRAME_INFO}" != *.frame_info.json ]]; then
    echo "[ERROR] --remote-frame-info 必须指向 *.frame_info.json 文件。" >&2
    exit 1
fi

REMOTE_STEP_TASK_JSON="${REMOTE_FRAME_INFO%.frame_info.json}.json"
REMOTE_FRAME_ASSETS="${REMOTE_FRAME_INFO%.frame_info.json}_frame_assets"
REMOTE_STEM="$(basename "${REMOTE_FRAME_INFO%.frame_info.json}")"
SAFE_STEM="$(printf '%s' "${REMOTE_STEM}" | tr -cs '[:alnum:]' '_' | sed 's/^_//; s/_$//')"

mkdir -p "${DEBUG_ROOT}" "${STAGE_ROOT}"
STAGE_DIR="${STAGE_ROOT}/${RUN_TIMESTAMP}"
mkdir -p "${STAGE_DIR}"

if [[ -z "${OUTPUT_JSONL}" ]]; then
    OUTPUT_JSONL="${DEBUG_ROOT}/navgen_${SAFE_STEM}_${RUN_TIMESTAMP}.jsonl"
fi
OUTPUT_DIR="$(dirname "${OUTPUT_JSONL}")"
OUTPUT_BASENAME="$(basename "${OUTPUT_JSONL}")"
OUTPUT_STEM="${OUTPUT_BASENAME%.jsonl}"
MARKDOWN_OUTPUT_DIR="${MARKDOWN_OUTPUT_DIR:-${OUTPUT_DIR}/${OUTPUT_STEM}_md}"

mkdir -p "${OUTPUT_DIR}"

sync_remote_path() {
    local remote_path="$1"
    rsync -a -s -e "ssh -p ${REMOTE_PORT}" \
        "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}" \
        "${STAGE_DIR}/"
}

echo "[1/4] 从远程同步 step_task 资源..."
echo "    host : ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_PORT}"
echo "    info : ${REMOTE_FRAME_INFO}"
echo "    json : ${REMOTE_STEP_TASK_JSON}"
echo "    asset: ${REMOTE_FRAME_ASSETS}"

sync_remote_path "${REMOTE_FRAME_INFO}"
sync_remote_path "${REMOTE_STEP_TASK_JSON}"
sync_remote_path "${REMOTE_FRAME_ASSETS}"

LOCAL_FRAME_INFO="${STAGE_DIR}/$(basename "${REMOTE_FRAME_INFO}")"
LOCAL_STEP_TASK_JSON="${STAGE_DIR}/$(basename "${REMOTE_STEP_TASK_JSON}")"

echo ""
echo "[2/4] 本地暂存目录:"
echo "    ${STAGE_DIR}"
echo "    frame_info : ${LOCAL_FRAME_INFO}"
echo "    step_json  : ${LOCAL_STEP_TASK_JSON}"

echo ""
echo "[3/4] 开始生成 NavGen CoT..."
echo "    输出 JSONL : ${OUTPUT_JSONL}"
echo "    Markdown   : ${MARKDOWN_OUTPUT_DIR}"
echo "    Provider   : ${API_PROVIDER}"
echo "    Base URL   : ${API_BASE_URL}"
echo "    模型       : ${MODEL_ID}"
echo "    Thinking   : ${THINKING_MODE}"

if [[ "${API_PROVIDER}" != "dashscope" ]]; then
    echo "[ERROR] 当前默认流程只支持直连 DashScope API；如需本地服务请显式修改脚本或传入兼容参数。" >&2
    exit 1
fi

if [[ -z "${API_KEY}" && -z "${!API_KEY_ENV:-}" ]]; then
    echo "[ERROR] 未检测到 API Key。请设置环境变量 ${API_KEY_ENV}，或导出 API_KEY 后重试。" >&2
    exit 1
fi

CMD=(
    python3 "${SCRIPT_DIR}/navgen_cot_synthesizer.py"
    --frame-info "${LOCAL_FRAME_INFO}"
    --step-task-json "${LOCAL_STEP_TASK_JSON}"
    --output "${OUTPUT_JSONL}"
    --api-provider "${API_PROVIDER}"
    --api-base-url "${API_BASE_URL}"
    --model "${MODEL_ID}"
    --api-key-env "${API_KEY_ENV}"
    --thinking-mode "${THINKING_MODE}"
    --max-retries "${MAX_RETRIES}"
    --invalid-frame-policy "${INVALID_FRAME_POLICY}"
)
if [[ -n "${API_KEY}" ]]; then
    CMD+=(--api-key "${API_KEY}")
fi
CMD+=("${PASSTHROUGH_ARGS[@]}")

"${CMD[@]}"

echo ""
echo "JSONL 完成！输出文件: ${OUTPUT_JSONL}"
echo "记录数: $(wc -l < "${OUTPUT_JSONL}")"

if [[ "${EXPORT_MARKDOWN}" == "1" ]]; then
    echo ""
    echo "[4/4] 导出逐帧 Markdown..."
    MD_CMD=(
        python3 "${SCRIPT_DIR}/jsonl_to_frame_markdown.py"
        --input "${OUTPUT_JSONL}"
        --output-dir "${MARKDOWN_OUTPUT_DIR}"
    )
    if [[ "${INCLUDE_PROMPT_IN_MARKDOWN}" == "1" ]]; then
        MD_CMD+=(--include-prompt)
    fi
    "${MD_CMD[@]}"
    echo ""
    echo "全部完成！"
    echo "JSONL 文件   : ${OUTPUT_JSONL}"
    echo "Markdown 目录: ${MARKDOWN_OUTPUT_DIR}"
else
    echo ""
    echo "[4/4] 已跳过 Markdown 导出（EXPORT_MARKDOWN=${EXPORT_MARKDOWN}）"
    echo "JSONL 文件: ${OUTPUT_JSONL}"
fi
