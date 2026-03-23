# Qwen3.5 VLN-CE 导航测试

测试 Qwen3.5 模型在 VLN-CE 任务上的导航能力（无需训练，通过 vLLM HTTP 接口调用模型）

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_qwen_vln.py` | 主测试脚本 |
| `run.sh` | 一键运行脚本（推荐使用） |
| `utils.py` | 工具函数 |
| `results/` | 测试结果输出目录 |
| `README.md` | 本文档 |
| `env_issues.md` | 环境问题排查记录 |

## 环境要求

- conda 环境：`habitat`
- vLLM 服务已启动，端口：`8000`
- 数据集路径：`/mnt/data-cpfs/gengshuang/vln_data/vln_ce/raw_data/r2r`
- 场景路径：`/mnt/data-cpfs/gengshuang/vln_data/scene_data/mp3d`

## 快速开始

**推荐方式：直接运行 run.sh**（需在项目根目录执行）

```bash
cd /mnt/data-cpfs/gengshuang/VLN-CE
bash qwen_navigation_test/run.sh
```

## 手动运行

```bash
cd /mnt/data-cpfs/gengshuang/VLN-CE

conda run -n habitat bash -c "
export PYTHONPATH=.
export EGL_DEVICE_ID=0
export CUDA_VISIBLE_DEVICES=0

python3 qwen_navigation_test/test_qwen_vln.py \
  --config vlnce_baselines/config/r2r_baselines/nonlearning.yaml \
  --server-url http://0.0.0.0:8000 \
  --model-name /mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8 \
  --split val_unseen \
  --num-episodes 10 \
  --output-dir qwen_navigation_test/results
"
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 必填 | VLN-CE 配置文件路径 |
| `--server-url` | `http://0.0.0.0:8000` | vLLM 服务地址 |
| `--model-name` | 自动检测 | 模型名称（vLLM 部署路径或别名） |
| `--split` | `val_unseen` | 数据集划分：`train` / `val_seen` / `val_unseen` |
| `--num-episodes` | `-1`（全部） | 测试 episode 数量 |
| `--output-dir` | `./results` | 结果保存目录 |

## 输出结果

测试完成后，结果保存在 `--output-dir` 目录下：

```
results/
└── results_val_unseen.json   # 包含各 episode 详细指标和平均指标
```

结果 JSON 结构：
```json
{
  "split": "val_unseen",
  "num_episodes": 10,
  "averaged_metrics": {
    "success": 0.xx,
    "spl": 0.xx,
    "ndtw": 0.xx,
    "distance_to_goal": x.xx,
    ...
  },
  "episode_results": [...]
}
```

## 环境问题

安装和运行过程中遇到的所有环境问题及解决方法，详见 [env_issues.md](env_issues.md)
