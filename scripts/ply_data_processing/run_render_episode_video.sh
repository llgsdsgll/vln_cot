#!/usr/bin/env bash
# 生成 VLN episode 轨迹视频（带 3D bounding box 标注）
#
# 用法：
#   bash scripts/ply_data_processing/run_render_episode_video.sh

# ── 参数区 ─────────────────────────────────────────────────────────────────
EPISODE_ID=4
LABEL_JSON="data/processed/labeled_episode_ply/ep00004_5LpN3gDmAk7.json"

# 相机参数
HFOV=86.0          # 水平 FOV（度）
VFOV=57.0          # 垂直 FOV（度）
SENSOR_HEIGHT=1.0  # 相机高度（米）
WIDTH=640          # 图像宽度（像素）
HEIGHT=426         # 图像高度（像素）
FPS=1              # 视频帧率

# 路径（一般不需要改）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
OUTPUT_DIR="${PROJECT_ROOT}/data/processed/episode_videos"
PYTHON="/home/gs/anaconda3/envs/vlnce/bin/python"

# ── 运行 ───────────────────────────────────────────────────────────────────
cd "${PROJECT_ROOT}"
"${PYTHON}" scripts/ply_data_processing/render_episode_video.py \
    --episode-id     "${EPISODE_ID}" \
    --label-json     "${LABEL_JSON}" \
    --output-dir     "${OUTPUT_DIR}" \
    --hfov           "${HFOV}" \
    --vfov           "${VFOV}" \
    --sensor-height  "${SENSOR_HEIGHT}" \
    --width          "${WIDTH}" \
    --height         "${HEIGHT}" \
    --fps            "${FPS}"
