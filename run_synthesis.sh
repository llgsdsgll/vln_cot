#!/usr/bin/env bash
# =============================================================================
# VLN CoT 数据合成运行脚本
#
# 使用方法：
#   bash run_synthesis.sh                 # 处理全部 100 个 episode
#   bash run_synthesis.sh --episode-ids 1 6 7   # 只处理指定 episode（调试用）
# =============================================================================
set -euo pipefail

# ---------------- 路径配置 ----------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_JSON="${SCRIPT_DIR}/gengshuang_1_H.264_0324_processed.json"
GT_JSON="/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json"
VIDEO_DIR="/home/gs/my_test/vln_dataset/vln_ce_video"
OUTPUT_JSONL="${SCRIPT_DIR}/output_cot.jsonl"

# ---------------- VLM API 配置 ----------------
API_BASE_URL="http://localhost:8000/v1"
MODEL_ID="/mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8"
MAX_RETRIES=3

# ---------------- SSH 隧道检查 ----------------
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

# 打印实际使用的模型
echo ""
echo "[2/3] 服务端模型："
curl -s "${API_BASE_URL}/models" | python3 -c \
    "import sys, json; [print('   ', m['id']) for m in json.load(sys.stdin)['data']]"

# ---------------- 运行合成脚本 ----------------
echo ""
echo "[3/3] 开始合成 CoT 数据..."
echo "    输入: ${INPUT_JSON}"
echo "    GT  : ${GT_JSON}"
echo "    视频: ${VIDEO_DIR}"
echo "    输出: ${OUTPUT_JSONL}"
echo "    模型: ${MODEL_ID}"
echo ""

python3 "${SCRIPT_DIR}/vln_data_synthesizer.py" \
    --input       "${INPUT_JSON}" \
    --gt          "${GT_JSON}" \
    --video-dir   "${VIDEO_DIR}" \
    --output      "${OUTPUT_JSONL}" \
    --api-base-url "${API_BASE_URL}" \
    --model       "${MODEL_ID}" \
    --max-retries "${MAX_RETRIES}" \
    "$@"   # 透传所有额外参数（如 --episode-ids 1 2 3）

echo ""
echo "完成！输出文件: ${OUTPUT_JSONL}"
echo "记录数: $(wc -l < "${OUTPUT_JSONL}")"
