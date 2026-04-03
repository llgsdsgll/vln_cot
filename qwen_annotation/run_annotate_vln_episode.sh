#!/usr/bin/env bash

# Example:
# REUSE_RUN_DIR=/home/gs/my_project/data_generation_pipline/output/qwen_annotations/episode_1_20260403_162938 \
# QWEN_THINKING=true QWEN_THINKING_BUDGET=4000 QWEN_TOP_P=0.8 QWEN_TEMPERATURE=0.7 QWEN_RESULT_FORMAT=message \
# ./run_annotate_vln_episode.sh
#
# Note:
# If REUSE_RUN_DIR points to an old run with existing frame_results_*, those
# frames will be treated as already completed and will be skipped.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/gs/anaconda3/bin/python}"

EPISODE_ID="${EPISODE_ID:-1}"
TRAIN_JSON="${TRAIN_JSON:-/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json}"
TRAIN_GT_JSON="${TRAIN_GT_JSON:-/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json}"
VIDEO_PATH="${VIDEO_PATH:-$PROJECT_ROOT/frames/1.mp4}"
FRAMES_DIR="${FRAMES_DIR:-$PROJECT_ROOT/frames}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/output/qwen_annotations}"
#REUSE_RUN_DIR="${REUSE_RUN_DIR:-/home/gs/my_project/data_generation_pipline/output/qwen_annotations/episode_1_20260403_162938}"
REUSE_RUN_DIR="${REUSE_RUN_DIR:-}"

SEGMENT_MODEL="${SEGMENT_MODEL:-qwen3.6-plus}"
LOCALIZE_MODEL="${LOCALIZE_MODEL:-qwen3.6-plus}"
VIDEO_FPS="${VIDEO_FPS:-5.0}"
QWEN_TEMPERATURE="${QWEN_TEMPERATURE:-0.0}"
QWEN_TOP_P="${QWEN_TOP_P:-}"
QWEN_RESULT_FORMAT="${QWEN_RESULT_FORMAT:-}"
QWEN_THINKING="${QWEN_THINKING:-false}"
QWEN_THINKING_BUDGET="${QWEN_THINKING_BUDGET:-}"

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "Error: DASHSCOPE_API_KEY is not set." >&2
  echo "Please run: export DASHSCOPE_API_KEY=your_key" >&2
  exit 1
fi

extra_args=()
if [[ -n "$REUSE_RUN_DIR" ]]; then
  extra_args+=(--reuse-run-dir "$REUSE_RUN_DIR")
fi
extra_args+=(--qwen-temperature "$QWEN_TEMPERATURE")
if [[ -n "$QWEN_TOP_P" ]]; then
  extra_args+=(--qwen-top-p "$QWEN_TOP_P")
fi
if [[ -n "$QWEN_RESULT_FORMAT" ]]; then
  extra_args+=(--qwen-result-format "$QWEN_RESULT_FORMAT")
fi
if [[ "$QWEN_THINKING" == "true" ]]; then
  extra_args+=(--thinking)
else
  extra_args+=(--no-thinking)
fi
if [[ -n "$QWEN_THINKING_BUDGET" ]]; then
  extra_args+=(--thinking-budget "$QWEN_THINKING_BUDGET")
fi

"$PYTHON_BIN" -u "$SCRIPT_DIR/annotate_vln_episode.py" \
  --train-json "$TRAIN_JSON" \
  --train-gt-json "$TRAIN_GT_JSON" \
  --episode-id "$EPISODE_ID" \
  --video-path "$VIDEO_PATH" \
  --frames-dir "$FRAMES_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --segment-model "$SEGMENT_MODEL" \
  --localize-model "$LOCALIZE_MODEL" \
  --video-fps "$VIDEO_FPS" \
  "${extra_args[@]}" \
  "$@"
