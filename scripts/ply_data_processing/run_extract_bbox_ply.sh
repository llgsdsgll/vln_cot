#!/usr/bin/env bash
# 裁剪指定 episode 的 PLY，支持 Habitat 坐标系下各方向独立膨胀
#
# 用法示例：
#   bash scripts/ply_data_processing/run_extract_bbox_ply.sh
#
# 修改下方参数区即可

# ── 要处理的 episode id（空格分隔，留空则处理全部）──────────────────────────
EPISODE_IDS="4"

# ── Habitat 坐标系下各方向膨胀量（单位：米）────────────────────────────────
#   +X: 右   -X: 左   +Y: 上   -Y: 下   +Z: 后   -Z: 前
PAD_X_POS=3.0
PAD_X_NEG=3.0
PAD_Y_POS=2.5
PAD_Y_NEG=3.0
PAD_Z_POS=3.0
PAD_Z_NEG=3.0

# ── 路径（一般不需要改）────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

TRAIN_JSON="/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json"
GT_JSON="/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json"
PLY_DIR="/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat/mp3d"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed/episode_ply"

# ── 构建命令 ───────────────────────────────────────────────────────────────
CMD=(
    python3 "${SCRIPT_DIR}/extract_vln_episode_bbox_ply.py"
    --train-json "${TRAIN_JSON}"
    --gt-json    "${GT_JSON}"
    --ply-dir    "${PLY_DIR}"
    --output-dir "${OUTPUT_DIR}"
    --pad-x-pos  "${PAD_X_POS}"
    --pad-x-neg  "${PAD_X_NEG}"
    --pad-y-pos  "${PAD_Y_POS}"
    --pad-y-neg  "${PAD_Y_NEG}"
    --pad-z-pos  "${PAD_Z_POS}"
    --pad-z-neg  "${PAD_Z_NEG}"
)

if [ -n "${EPISODE_IDS}" ]; then
    CMD+=(--episode-ids ${EPISODE_IDS})
fi

echo "Running: ${CMD[*]}"
"${CMD[@]}"
