# Qwen3.5 VLN-CE 导航测试

测试 Qwen3.5 模型在 VLN-CE 任务上的导航能力。

当前脚本已调整为：
- Habitat 仿真在本地机器运行
- 大模型通过远端 OpenAI 兼容多模态接口访问
- 大模型接口组织方式参考 `human_ann_cot/scripts/cot_annotation/vln_data_synthesizer.py`

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_qwen_vln.py` | 主测试脚本 |
| `run.sh` | 一键运行脚本（推荐使用，支持从环境变量读取地址并自动探测模型名） |
| `utils.py` | 工具函数 |
| `results/` | 测试结果输出目录 |
| `README.md` | 本文档 |
| `env_issues.md` | 环境问题排查记录 |

## 环境要求

- 本地可运行 Habitat 的 Python 环境
- 远端可访问的大模型服务，需提供 OpenAI 兼容接口
- 数据集路径：`/mnt/data-cpfs/gengshuang/vln_data/vln_ce/raw_data/r2r`
- 场景路径：`/mnt/data-cpfs/gengshuang/vln_data/scene_data/mp3d`

如果本地不能直接访问服务器端口，可以先做 SSH 端口映射：

```bash
ssh -fN -L 8000:localhost:8000 <user>@<server> -p <port>
```

## 快速开始

**推荐方式：直接运行 `run.sh`**（需在项目根目录执行）

```bash
cd /mnt/data-cpfs/gengshuang/VLN-CE

export QWEN_API_PROVIDER=local
export QWEN_API_BASE_URL=http://127.0.0.1:8000/v1

bash qwen_navigation_test/run.sh
```

`run.sh` 会优先读取 `QWEN_API_BASE_URL`，其次兼容 `OPENAI_BASE_URL` / `OPENAI_API_BASE`。

模型名不会从环境变量读取，而是固定通过 `curl http://localhost:8000/v1/models` 自动获取已部署模型列表中的第一个模型。

如果你直接访问远端服务而不是做本地隧道，可以把 `QWEN_API_BASE_URL` 换成实际服务地址，例如：

```bash
export QWEN_API_BASE_URL=http://<server-ip>:8000/v1
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
  --api-provider local \
  --api-base-url http://127.0.0.1:8000/v1 \
  --model /mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8 \
  --thinking-mode off \
  --max-retries 3 \
  --split val_unseen \
  --num-episodes 10 \
  --output-dir qwen_navigation_test/results
"
```

使用阿里云百炼时，可改为：

```bash
export DASHSCOPE_API_KEY=<your-key>

python3 qwen_navigation_test/test_qwen_vln.py \
  --config vlnce_baselines/config/r2r_baselines/nonlearning.yaml \
  --api-provider dashscope \
  --model qwen3.5-plus \
  --thinking-mode off \
  --split val_unseen \
  --num-episodes 10 \
  --output-dir qwen_navigation_test/results
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--config` | 必填 | VLN-CE 配置文件路径 |
| `--api-provider` | `local` | 接口类型：`local` / `dashscope` |
| `--api-base-url` | `http://127.0.0.1:8000/v1` | OpenAI 兼容接口地址，兼容旧参数 `--server-url` |
| `--model` | local 自动探测 | 模型名称，兼容旧参数 `--model-name` |
| `--api-key` | `None` | 显式传入 API Key |
| `--api-key-env` | `DASHSCOPE_API_KEY` | 从环境变量读取 API Key 的变量名 |
| `--thinking-mode` | `off` | `auto` / `on` / `off` |
| `--max-retries` | `3` | 远端模型请求最大重试次数 |
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
