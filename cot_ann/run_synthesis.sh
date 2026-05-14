#!/usr/bin/env bash
# =============================================================================
# VLN CoT 数据合成运行脚本
#
# 使用方法：
#   bash scripts/cot_annotation/run_synthesis.sh
#   bash scripts/cot_annotation/run_synthesis.sh --episode-ids 1 6 7
#   API_PROVIDER=dashscope MODEL_ID=qwen3.5-plus THINKING_MODE=on \
#       bash scripts/cot_annotation/run_synthesis.sh --episode-ids 1
# =============================================================================
set -euo pipefail

# ---------------- 路径配置 ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
INPUT_JSON="${PROJECT_ROOT}/data/processed/gengshuang_1_H.264_0324_processed.json"
GT_JSON="/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json"
VIDEO_DIR="/home/gs/my_test/vln_dataset/vln_ce_video"
DEBUG_ROOT="${PROJECT_ROOT}/debug"

# ---------------- VLM API 配置 ----------------
API_PROVIDER="${API_PROVIDER:-local}"
API_BASE_URL="${API_BASE_URL:-}"
MODEL_ID="${MODEL_ID:-}"
MAX_RETRIES="${MAX_RETRIES:-3}"
THINKING_MODE="${THINKING_MODE:-off}"
INVALID_FRAME_POLICY="${INVALID_FRAME_POLICY:-skip}"
API_KEY_ENV="${API_KEY_ENV:-DASHSCOPE_API_KEY}"
API_KEY="${API_KEY:-}"

LOCAL_API_BASE_URL="http://localhost:8000/v1"
LOCAL_MODEL_ID="/mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8"
DASHSCOPE_API_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL_ID="qwen3.5-plus"

# ---------------- 输出命名配置 ----------------
RUN_TIMESTAMP="${RUN_TIMESTAMP:-$(date '+%Y%m%d_%H%M%S')}"
EXPORT_MARKDOWN="${EXPORT_MARKDOWN:-1}"
INCLUDE_PROMPT_IN_MARKDOWN="${INCLUDE_PROMPT_IN_MARKDOWN:-0}"

episode_ids_from_args=()
capture_episode_ids=0
for arg in "$@"; do
    if [[ "${arg}" == --episode-ids=* ]]; then
        episode_value="${arg#--episode-ids=}"
        if [[ -n "${episode_value}" ]]; then
            episode_ids_from_args+=("${episode_value}")
        fi
        capture_episode_ids=0
        continue
    fi

    if [[ "${capture_episode_ids}" -eq 1 ]]; then
        if [[ "${arg}" == --* ]]; then
            capture_episode_ids=0
        else
            episode_ids_from_args+=("${arg}")
            continue
        fi
    fi

    if [[ "${arg}" == "--episode-ids" ]]; then
        capture_episode_ids=1
    fi
done

if [[ "${#episode_ids_from_args[@]}" -eq 1 ]]; then
    EPISODE_TAG="$(printf 'ep%05d' "${episode_ids_from_args[0]}")"
elif [[ "${#episode_ids_from_args[@]}" -gt 1 ]]; then
    joined_episode_ids="$(printf '%s-' "${episode_ids_from_args[@]}")"
    joined_episode_ids="${joined_episode_ids%-}"
    EPISODE_TAG="eps_${joined_episode_ids}"
else
    EPISODE_TAG="all_episodes"
fi

RUN_LABEL="${API_PROVIDER}_${EPISODE_TAG}_${RUN_TIMESTAMP}"
DEFAULT_OUTPUT_JSONL="${DEBUG_ROOT}/${RUN_LABEL}.jsonl"
OUTPUT_JSONL="${OUTPUT_JSONL:-${DEFAULT_OUTPUT_JSONL}}"
OUTPUT_DIR="$(dirname "${OUTPUT_JSONL}")"
OUTPUT_BASENAME="$(basename "${OUTPUT_JSONL}")"
OUTPUT_STEM="${OUTPUT_BASENAME%.jsonl}"
DEFAULT_MARKDOWN_OUTPUT_DIR="${OUTPUT_DIR}/${OUTPUT_STEM}_md"
MARKDOWN_OUTPUT_DIR="${MARKDOWN_OUTPUT_DIR:-${DEFAULT_MARKDOWN_OUTPUT_DIR}}"

mkdir -p "${DEBUG_ROOT}"
mkdir -p "${OUTPUT_DIR}"

if [[ "${API_PROVIDER}" == "local" ]]; then
    API_BASE_URL="${API_BASE_URL:-${LOCAL_API_BASE_URL}}"
    MODEL_ID="${MODEL_ID:-${LOCAL_MODEL_ID}}"

    echo "[1/3] 检查 SSH 隧道（本地 8000 端口）..."
    if ! curl -sf --max-time 5 "${API_BASE_URL}/models" > /dev/null 2>&1; then
        echo "    隧道未连通，正在建立..."
        ssh -fN -L 8000:localhost:8000 root@139.196.171.150 -p 6222
        sleep 2
        if ! curl -sf --max-time 5 "${API_BASE_URL}/models" > /dev/null 2>&1; then
            echo "[ERROR] SSH 隧道建立失败，请检查网络连接或服务端状态。" >&2
            exit 1
        fi
        echo "    隧道建立成功。"
    else
        echo "    隧道已连通。"
    fi

    echo ""
    echo "[2/3] 服务端模型："
    curl -s "${API_BASE_URL}/models" | python3 -c \
        "import sys, json; [print('   ', m['id']) for m in json.load(sys.stdin)['data']]"
elif [[ "${API_PROVIDER}" == "dashscope" ]]; then
    API_BASE_URL="${API_BASE_URL:-${DASHSCOPE_API_BASE_URL}}"
    MODEL_ID="${MODEL_ID:-${DASHSCOPE_MODEL_ID}}"

    echo "[1/3] 检查成品 API 配置..."
    if [[ -z "${API_KEY}" && -z "${!API_KEY_ENV:-}" ]]; then
        echo "[ERROR] 未检测到 API Key。请设置环境变量 ${API_KEY_ENV}，或导出 API_KEY 后重试。" >&2
        exit 1
    fi
    echo "    provider: ${API_PROVIDER}"
    echo "    base_url: ${API_BASE_URL}"
    echo "    key_env : ${API_KEY_ENV}"

    echo ""
    echo "[2/3] 成品 API 模型："
    echo "    ${MODEL_ID}"
else
    echo "[ERROR] 不支持的 API_PROVIDER: ${API_PROVIDER}" >&2
    exit 1
fi

# ---------------- 运行合成脚本 ----------------
echo ""
echo "[3/4] 开始合成 CoT 数据..."
echo "    输入: ${INPUT_JSON}"
echo "    GT  : ${GT_JSON}"
echo "    视频: ${VIDEO_DIR}"
echo "    输出: ${OUTPUT_JSONL}"
echo "    Markdown: ${MARKDOWN_OUTPUT_DIR}"
echo "    Provider: ${API_PROVIDER}"
echo "    Base URL: ${API_BASE_URL}"
echo "    模型: ${MODEL_ID}"
echo "    Thinking: ${THINKING_MODE}"
echo "    Invalid : ${INVALID_FRAME_POLICY}"
echo ""

CMD=(
    python3 "${SCRIPT_DIR}/vln_data_synthesizer.py"
    --input "${INPUT_JSON}"
    --gt "${GT_JSON}"
    --video-dir "${VIDEO_DIR}"
    --output "${OUTPUT_JSONL}"
    --api-provider "${API_PROVIDER}"
    --api-base-url "${API_BASE_URL}"
    --model "${MODEL_ID}"
    --max-retries "${MAX_RETRIES}"
    --thinking-mode "${THINKING_MODE}"
    --invalid-frame-policy "${INVALID_FRAME_POLICY}"
    --api-key-env "${API_KEY_ENV}"
)

if [[ -n "${API_KEY}" ]]; then
    CMD+=(--api-key "${API_KEY}")
fi

CMD+=("$@")

"${CMD[@]}"

echo ""
echo "JSONL 完成！输出文件: ${OUTPUT_JSONL}"
echo "记录数: $(wc -l < "${OUTPUT_JSONL}")"

if [[ "${EXPORT_MARKDOWN}" == "1" ]]; then
    echo ""
    echo "[4/4] 导出逐帧 Markdown..."
    echo "    输入 JSONL : ${OUTPUT_JSONL}"
    echo "    输出目录   : ${MARKDOWN_OUTPUT_DIR}"

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
