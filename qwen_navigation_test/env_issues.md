# 环境问题排查记录

这份文档整理的是本次把 `qwen_navigation_test` 跑通时，真实遇到的问题、原因和最终修复方式。

当前最终可用状态：
- conda 环境：`vlnce`
- 项目根目录：`/home/gs/my_project/qwen_vln`
- `habitat-lab` 已执行 `python setup.py develop --all`
- `run.sh` 已能成功跑通 `val_unseen` 的 1 个 episode

---

## 1. 旧前缀环境 `habitat_env` 已损坏，不能继续使用

### 现象

`run.sh` 早期使用：

```bash
conda run -p /home/gs/my_project/qwen_vln/habitat_env ...
```

运行时报错：

```text
Fatal Python error: Py_Initialize: Unable to get the locale encoding
ModuleNotFoundError: No module named 'encodings'
```

即使只执行：

```bash
conda run -p /home/gs/my_project/qwen_vln/habitat_env python3 -c "print(123)"
```

也会失败。

### 原因

这个前缀环境里的 Python 运行时已经异常，不能稳定执行 Python 代码。

### 解决

放弃旧前缀环境，改用新的命名环境：

```bash
conda activate vlnce
```

并且把 `run.sh` 改成：

```bash
source /home/gs/anaconda3/etc/profile.d/conda.sh
conda activate vlnce
```

---

## 2. `habitat` 模块缺失

### 现象

切到 `vlnce` 环境后，运行 `run.sh` 报错：

```text
ModuleNotFoundError: No module named 'habitat'
```

### 原因

`vlnce` 环境里没有安装 `habitat-lab`，或者没有把源码目录加入 Python 搜索路径。

### 解决

在 `habitat-lab` 目录下执行：

```bash
conda activate vlnce
cd /home/gs/my_project/qwen_vln/habitat-lab
python setup.py develop --all
```

同时在 `run.sh` 中把以下目录加入 `PYTHONPATH`：

```bash
/home/gs/my_project/qwen_vln
/home/gs/my_project/qwen_vln/habitat-lab
```

---

## 3. `python setup.py develop --all` 安装时 `lmdb` 编译失败

### 现象

执行：

```bash
python setup.py develop --all
```

报错类似：

```text
lmdb/cpython.c:194:5: error: unknown type name ‘PyThread_type_lock’
...
error: command 'gcc' failed with exit status 1
```

### 原因

`habitat-lab` 的 baseline 依赖里写的是：

```text
lmdb>=0.98
```

安装时会自动拉到过新的 `lmdb 2.2.0`，它和当前 Python 3.6 不兼容。

### 解决

把文件：

`/home/gs/my_project/qwen_vln/habitat-lab/habitat_baselines/il/requirements.txt`

中的：

```text
lmdb>=0.98
```

改成：

```text
lmdb==1.2.1
```

这个版本会安装到兼容 Python 3.6 的 wheel，不再走失败的源码编译路径。

---

## 4. `python setup.py develop --all` 安装时 `msgpack` 版本过新

### 现象

修完 `lmdb` 之后，继续安装又报：

```text
msgpack/_cmsgpack.c:15:6: error: #error Cython requires Python 3.8+.
```

### 原因

依赖链里自动拉到了过新的 `msgpack`，当前 Python 3.6 不兼容。

### 解决

先手动安装兼容版本：

```bash
conda activate vlnce
python -m pip install "msgpack==1.0.5"
```

然后再执行：

```bash
cd /home/gs/my_project/qwen_vln/habitat-lab
python setup.py develop --all
```

最终安装已经确认可以成功完成。

---

## 5. `openai` SDK 版本过老，不支持 `openai.OpenAI`

### 现象

运行 `run.sh` 报错：

```text
AttributeError: module 'openai' has no attribute 'OpenAI'
```

检查发现 `vlnce` 环境里的版本是：

```text
openai==0.8.0
```

### 原因

`test_qwen_vln.py` 之前按新版 SDK 写法使用：

```python
openai.OpenAI(...)
```

但当前环境里的 `openai` 太老，没有这个类。

### 解决

不再依赖 OpenAI Python SDK，而是直接用 `requests` 请求 OpenAI 兼容 HTTP 接口：

- 模型列表：`GET /v1/models`
- 推理接口：`POST /v1/chat/completions`

这样就兼容当前 Python 3.6 环境，不再受 SDK 版本限制。

---

## 6. 数据集默认路径已经失效

### 现象

运行时报错：

```text
FileNotFoundError: ... /mnt/data-cpfs/gengshuang/vln_data/vln_ce/raw_data/r2r/val_unseen/val_unseen.json.gz
```

### 原因

配置里仍然写的是旧机器的路径模板：

```text
/mnt/data-cpfs/gengshuang/vln_data/vln_ce/raw_data/r2r/{split}/{split}.json.gz
```

### 解决

当前实际可用的数据集根目录是：

```text
/home/gs/my_test/vln_dataset/data/datasets/r2r
```

已经在 `run.sh` 中作为默认值写入：

```bash
--dataset-root /home/gs/my_test/vln_dataset/data/datasets/r2r
```

并且 `test_qwen_vln.py` 会在启动时覆盖：

- `TASK_CONFIG.DATASET.DATA_PATH`
- `TASK_CONFIG.TASK.NDTW.GT_PATH`

---

## 7. 场景目录根路径多拼了一层

### 现象

运行时一度尝试加载：

```text
.../mp3d/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb
```

然后报错找不到场景文件，甚至触发段错误。

### 原因

数据里的 `scene_id` 本身已经带了前缀：

```text
mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb
```

所以 `SCENES_DIR` 不能再指到已经包含一层 `mp3d` 的更深目录，否则最终路径会重复。

### 正确路径

真实文件位置是：

```text
/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat/mp3d/zsNo4HB9uLZ/zsNo4HB9uLZ.glb
```

因此正确的场景根目录应该是：

```text
/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat
```

### 解决

`run.sh` 当前默认已更新为：

```bash
--scenes-dir /home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat
```

---

## 8. `gym 0.26.2` 与旧版 `habitat-lab` 的 `Discrete(0)` 不兼容

### 现象

任务初始化时报错：

```text
AssertionError: n (counts) have to be positive
```

定位到：

```python
self.observation_space = spaces.Discrete(0)
```

### 原因

旧版 `habitat-lab` 允许用 `spaces.Discrete(0)` 作为占位空间，但 `gym 0.26+` 已不允许 `n=0`。

### 解决

修改文件：

`/home/gs/my_project/qwen_vln/habitat-lab/habitat/tasks/vln/vln.py`

将：

```python
self.observation_space = spaces.Discrete(0)
```

改为：

```python
self.observation_space = spaces.Discrete(1)
```

这是一个占位兼容修复，不影响当前 `InstructionSensor` 返回的字典内容。

---

## 9. 当前最终可用的数据与模型配置

### 模型服务

模型接口默认：

```text
http://127.0.0.1:8000/v1
```

模型名通过：

```bash
curl http://localhost:8000/v1/models
```

自动探测第一个已部署模型。

### 数据与场景

数据集根目录：

```text
/home/gs/my_test/vln_dataset/data/datasets/r2r
```

场景根目录：

```text
/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat
```

### 运行命令

```bash
cd /home/gs/my_project/qwen_vln
bash qwen_navigation_test/run.sh
```

---

## 10. 已验证通过的结果

已经验证通过：

```bash
QWEN_NUM_EPISODES=1 CONDA_ENV_NAME=vlnce bash qwen_navigation_test/run.sh
```

结果文件已成功生成：

```text
/home/gs/my_project/qwen_vln/qwen_navigation_test/results/results_val_unseen.json
```

---

## 11. 相关修改文件

这次排障中实际改动过的关键文件：

| 文件 | 作用 |
|------|------|
| `qwen_navigation_test/run.sh` | 切换到 `vlnce` 环境，加入 `habitat-lab` 到 `PYTHONPATH`，默认数据/场景路径改为本机可用路径 |
| `qwen_navigation_test/test_qwen_vln.py` | 改成 `requests` 直连 OpenAI 兼容接口，并支持覆盖数据集与场景路径 |
| `qwen_navigation_test/README.md` | 同步当前可用环境与运行方式 |
| `habitat-lab/habitat_baselines/il/requirements.txt` | 将 `lmdb` 固定到 `1.2.1` |
| `habitat-lab/habitat/tasks/vln/vln.py` | 修复 `Discrete(0)` 与 `gym 0.26.2` 的兼容问题 |
