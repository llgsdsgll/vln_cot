#!/bin/bash
# 运行 Qwen3.5 VLN-CE 导航测试脚本
# 需在项目根目录 /mnt/data-cpfs/gengshuang/VLN-CE 下执行
# Habitat 使用 GPU 0，大模型在 GPU 1

export PYTHONPATH=.
export EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0

conda run -n habitat bash -c "
export PYTHONPATH=.
export EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0

python3 qwen_navigation_test/test_qwen_vln.py \
  --config vlnce_baselines/config/r2r_baselines/nonlearning.yaml \
  --server-url http://0.0.0.0:8000 \
  --model-name /mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8 \
  --split val_unseen \
  --num-episodes 2 \
  --output-dir qwen_navigation_test/results
"
