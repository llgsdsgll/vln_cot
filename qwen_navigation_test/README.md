# Qwen3.5 VLN-CE 导航测试

测试脚本已经调通为：
- Habitat 仿真在本地运行
- 大模型通过 OpenAI 兼容接口访问
- `run.sh` 默认使用 `vlnce` conda 环境
- 模型名固定通过 `curl http://localhost:8000/v1/models` 自动探测

## 文件说明

| 文件 | 说明 |
|------|------|
| `test_qwen_vln.py` | 主测试脚本 |
| `run.sh` | 一键运行脚本 |
| `results/` | 测试结果输出目录 |
| `README.md` | 当前说明文档 |
| `env_issues.md` | 历史环境问题记录 |

## 当前可用环境

- conda 环境：`vlnce`
- 项目根目录：`/home/gs/my_project/qwen_vln`
- `habitat-lab` 源码目录：`/home/gs/my_project/qwen_vln/habitat-lab`
- 模型服务地址默认值：`http://127.0.0.1:8000/v1`
- 模型列表探测地址：`http://localhost:8000/v1/models`

## 数据路径

当前默认数据目录已经和 `run.sh` 对齐：

- 数据集根目录：`/home/gs/my_test/vln_dataset/data/datasets/r2r`
- 场景根目录：`/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat`

其中：
- `val_unseen` 数据文件会解析为 `/home/gs/my_test/vln_dataset/data/datasets/r2r/val_unseen/val_unseen.json.gz`
- `scene_id=mp3d/...` 会拼到 `/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat`

## 环境准备

推荐先激活环境，再安装 `habitat-lab`：

```bash
conda activate vlnce
cd /home/gs/my_project/qwen_vln/habitat-lab
python setup.py develop --all
```

如果是全新环境，建议先确保以下版本存在：

```bash
conda activate vlnce
python -m pip install "msgpack==1.0.5"
```

说明：
- 仓库里已经把 `habitat_baselines/il/requirements.txt` 中的 `lmdb` 固定到了 `1.2.1`
- `msgpack==1.0.5` 是为了兼容当前 Python 3.6 环境，避免 `develop --all` 自动拉到不兼容的新版本

安装完成后可以快速确认：

```bash
conda run -n vlnce python -c "import habitat; print(habitat.__file__)"
curl http://localhost:8000/v1/models
```

## 快速开始

在项目根目录运行：

```bash
cd /home/gs/my_project/qwen_vln
bash qwen_navigation_test/run.sh
```

`run.sh` 当前默认行为：
- 自动 `source /home/gs/anaconda3/etc/profile.d/conda.sh`
- 自动 `conda activate vlnce`
- 自动把项目根目录和 `habitat-lab` 加入 `PYTHONPATH`
- 自动从 `http://localhost:8000/v1/models` 读取第一个已部署模型
- 默认评测 `val_unseen`
- 默认输出到 `qwen_navigation_test/results`

## 测评流程

整个测评链路由 `run.sh` 和 `test_qwen_vln.py` 两部分组成。

### 1. `run.sh` 做启动准备

`run.sh` 负责把运行环境准备好，然后再调用 Python 测试脚本：

1. 切到项目根目录 `/home/gs/my_project/qwen_vln`
2. `source conda.sh` 并激活 `vlnce` 环境
3. 设置 `PYTHONPATH`，把项目根目录和 `habitat-lab` 源码目录都加入模块搜索路径
4. 解析模型接口地址，默认使用 `http://127.0.0.1:8000/v1`
5. 通过 `curl http://localhost:8000/v1/models` 获取当前服务端部署的第一个模型名
6. 组装命令行参数，调用 `python3 qwen_navigation_test/test_qwen_vln.py`

### 2. `main()` 解析参数并初始化

`test_qwen_vln.py` 的 `main()` 负责准备配置和智能体：

1. 解析命令行参数，包括：
   - 模型接口参数：`--api-provider`、`--api-base-url`、`--model`
   - 评测参数：`--split`、`--num-episodes`、`--output-dir`
   - 数据路径参数：`--dataset-root`、`--scenes-dir`
2. 使用 `get_config()` 读取 `vlnce_baselines/config/r2r_baselines/nonlearning.yaml`
3. 创建输出目录
4. 设置 `torch` 和 `numpy` 随机种子
5. 初始化 `QwenNavigationAgent`

### 3. `QwenNavigationAgent` 如何请求模型

智能体每一步只做一件事：根据当前第一视角图像和导航指令预测下一动作。

初始化时会：

1. 解析 API 地址和 API key
2. 如果没有显式传 `--model`，在 local 模式下从 `http://localhost:8000/v1/models` 自动探测模型名

每次 `act()` 时会：

1. 用导航指令和当前步数构造 prompt
2. 取 `observations["rgb"]`
3. 把 RGB 图像编码成 base64 JPEG
4. 以多模态消息格式调用 `/v1/chat/completions`
5. 从模型返回文本中解析 `stop / forward / left / right`
6. 转成 Habitat 的动作 ID：
   - `stop -> HabitatSimActions.STOP`
   - `forward -> HabitatSimActions.MOVE_FORWARD`
   - `left -> HabitatSimActions.TURN_LEFT`
   - `right -> HabitatSimActions.TURN_RIGHT`

如果输出无法解析，脚本会默认返回 `MOVE_FORWARD`，避免中途直接停住。

### 4. `test_navigation()` 如何跑评测

`test_navigation()` 是真正的评测主循环：

1. 调用 `setup_config()` 把 split 同步到：
   - `DATASET.SPLIT`
   - `TASK.NDTW.SPLIT`
   - `TASK.SDTW.SPLIT`
2. 调用 `override_data_paths()` 用传入的 `dataset_root` 和 `scenes_dir` 覆盖：
   - `DATASET.DATA_PATH`
   - `TASK.NDTW.GT_PATH`
   - `DATASET.SCENES_DIR`
3. 创建 `Env(config=config.TASK_CONFIG)`，初始化 Habitat 环境
4. 根据 `num_episodes` 决定实际评测多少条 episode
5. 对每个 episode：
   - `env.reset()`
   - `agent.reset()`
   - 从 `env.current_episode.instruction.instruction_text` 取指令
   - 在 `while not env.episode_over` 循环中不断执行：
     - `action = agent.act(obs, instruction)`
     - `obs = env.step(action)`
6. episode 结束后，调用 `env.get_metrics()` 取本条轨迹指标
7. 把每条 episode 的结果保存到 `episode_results`
8. 所有 episode 结束后，对指标做平均

### 5. 最终输出什么

评测结束后脚本会做两件事：

1. 在日志里打印平均指标，例如：
   - `distance_to_goal`
   - `success`
   - `spl`
   - `ndtw`
   - `path_length`
   - `oracle_success`
   - `steps_taken`
2. 把完整结果写到：

```text
qwen_navigation_test/results/results_<split>.json
```

结果文件中同时包含：
- `averaged_metrics`：平均指标
- `episode_results`：每条 episode 的详细指标和指令文本

### 6. 一句话概括

整套流程可以概括为：

`run.sh` 准备环境和模型 -> `test_qwen_vln.py` 加载 Habitat 配置 -> 每个 episode 中用当前 RGB 图像和指令请求远端模型 -> 模型输出动作 -> Habitat 执行动作并返回新观察 -> episode 结束后统计指标 -> 最终保存 JSON 结果。

## 常用环境变量

`run.sh` 支持以下覆盖项：

| 环境变量 | 默认值 | 说明 |
|------|--------|------|
| `CONDA_ROOT` | `/home/gs/anaconda3` | conda 安装根目录 |
| `CONDA_ENV_NAME` | `vlnce` | 运行环境名 |
| `QWEN_API_PROVIDER` | `local` | 模型接口类型 |
| `QWEN_API_BASE_URL` | `http://127.0.0.1:8000/v1` | OpenAI 兼容接口地址 |
| `QWEN_THINKING_MODE` | `off` | `auto` / `on` / `off` |
| `QWEN_MAX_RETRIES` | `3` | 模型请求重试次数 |
| `QWEN_SPLIT` | `val_unseen` | 数据集 split |
| `QWEN_NUM_EPISODES` | `2` | 运行 episode 数 |
| `QWEN_OUTPUT_DIR` | `qwen_navigation_test/results` | 输出目录 |
| `QWEN_DATASET_ROOT` | `/home/gs/my_test/vln_dataset/data/datasets/r2r` | 数据集根目录 |
| `QWEN_SCENES_DIR` | `/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat` | 场景根目录 |
| `HABITAT_LAB_ROOT` | `<project_root>/habitat-lab` | Habitat-Lab 源码目录 |
| `HABITAT_GPU` | `0` | Habitat 使用的 GPU |

如果模型服务不在本机，也可以先做端口映射：

```bash
ssh -fN -L 8000:localhost:8000 <user>@<server> -p <port>
```

## 手动运行

```bash
conda activate vlnce
cd /home/gs/my_project/qwen_vln

export PYTHONPATH=/home/gs/my_project/qwen_vln:/home/gs/my_project/qwen_vln/habitat-lab

python3 qwen_navigation_test/test_qwen_vln.py \
  --config vlnce_baselines/config/r2r_baselines/nonlearning.yaml \
  --api-provider local \
  --api-base-url http://127.0.0.1:8000/v1 \
  --thinking-mode off \
  --max-retries 3 \
  --split val_unseen \
  --num-episodes 10 \
  --dataset-root /home/gs/my_test/vln_dataset/data/datasets/r2r \
  --scenes-dir /home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat \
  --output-dir qwen_navigation_test/results
```

说明：
- `--model` 不传时，local 模式会自动从 `http://localhost:8000/v1/models` 探测
- `--api-base-url` 兼容旧参数 `--server-url`
- `--model` 兼容旧参数 `--model-name`

## 结果输出

测试完成后，结果会写到：

```text
qwen_navigation_test/results/results_val_unseen.json
```

当前已经验证过一次 1-episode 运行，结果文件示例：
- `qwen_navigation_test/results/results_val_unseen.json`

结果结构如下：

```json
{
  "split": "val_unseen",
  "num_episodes": 10,
  "averaged_metrics": {
    "success": 0.xx,
    "spl": 0.xx,
    "ndtw": 0.xx
  },
  "episode_results": [...]
}
```

## 备注

- 当前 `test_qwen_vln.py` 已经改为直接使用 `requests` 调用 OpenAI 兼容 HTTP 接口，不依赖新版本 `openai` SDK
- 为兼容当前环境中的 `gym 0.26.2`，`habitat-lab/habitat/tasks/vln/vln.py` 中的旧版 `spaces.Discrete(0)` 已做兼容修正
