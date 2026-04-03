# Qwen VLN Annotation Pipeline

这份文档说明 `qwen_annotation/` 目录下这套纯 Qwen 标注流程的用法。

核心脚本：

- `qwen_annotation/run_annotate_vln_episode.sh`
- `qwen_annotation/annotate_vln_episode.py`
- `qwen_annotation/qwen_api.py`

当前默认模型是 `qwen3.6-plus`。

## 功能概览

这套流程会对一个 VLN episode 做三件事：

1. 读取完整第一人称视频，按 instruction 划分子任务。
2. 将每一帧分配到对应子任务。
3. 对每一帧中的子任务相关实体做视觉定位。

当前输出策略：

- `object`：保存 `visible`、`center`、`bbox`、`distance_m`
- `region`：保存 `visible`、`center`、`distance_range_m`
- 当前不再给 `region` 画 bounding box
- 每一帧处理后都会立刻保存结果，不需要等全部帧完成
- 单帧失败时最多重试 3 次，失败后记录错误并继续下一帧
- 每次运行都会带时间戳，方便调试

## 运行前准备

需要先设置 DashScope API Key：

```bash
export DASHSCOPE_API_KEY=your_key
```

当前 `run_annotate_vln_episode.sh` 默认使用：

- Python: `/home/gs/anaconda3/bin/python`
- `train.json`: `/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json`
- `train_gt.json`: `/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json`
- video: 项目根目录下的 `./frames/1.mp4`
- frames dir: 项目根目录下的 `./frames`

## 最常用运行方式

### 1. 不开启 thinking

这是当前默认方式：

```bash
./run_annotate_vln_episode.sh
```

这个根目录脚本会转发到：

```bash
./qwen_annotation/run_annotate_vln_episode.sh
```

等价于：

```bash
QWEN_THINKING=false \
QWEN_TEMPERATURE=0.0 \
./run_annotate_vln_episode.sh
```

特点：

- 输出更稳定
- 更适合结构化 JSON
- 更适合批量处理

### 2. 开启 thinking

```bash
QWEN_THINKING=true \
QWEN_THINKING_BUDGET=4000 \
QWEN_TOP_P=0.8 \
QWEN_TEMPERATURE=0.7 \
QWEN_RESULT_FORMAT=message \
./run_annotate_vln_episode.sh
```

适合：

- 想测试思考模式是否提升子任务分解
- 想测试单帧实体判断是否更稳

说明：

- `QWEN_THINKING_BUDGET` 只在 `QWEN_THINKING=true` 时生效
- `QWEN_RESULT_FORMAT` 会传到 `extra_body["result_format"]`
- 当前流程本身不是流式处理，所以没有接 `stream=true`

## 可配置环境变量

`run_annotate_vln_episode.sh` 支持这些环境变量：

- `EPISODE_ID`
- `TRAIN_JSON`
- `TRAIN_GT_JSON`
- `VIDEO_PATH`
- `FRAMES_DIR`
- `OUTPUT_DIR`
- `REUSE_RUN_DIR`
- `SEGMENT_MODEL`
- `LOCALIZE_MODEL`
- `VIDEO_FPS`
- `QWEN_TEMPERATURE`
- `QWEN_TOP_P`
- `QWEN_RESULT_FORMAT`
- `QWEN_THINKING`
- `QWEN_THINKING_BUDGET`
- `PYTHON_BIN`

## 常见示例

### 1. 指定 episode

```bash
EPISODE_ID=1 ./run_annotate_vln_episode.sh
```

### 2. 重新生成，不复用旧 run

默认现在就是不复用旧 run，只要直接运行即可：

```bash
./run_annotate_vln_episode.sh
```

### 3. 复用旧 run 的子任务结果

如果你已经确认某个 run 的分子任务结果没问题，可以复用：

```bash
REUSE_RUN_DIR=/home/gs/my_project/data_generation_pipline/output/qwen_annotations/episode_1_20260403_162938 \
./run_annotate_vln_episode.sh
```

这会从旧目录中读取：

- `subtasks`
- 已有的 `frame_results`

也就是说，`REUSE_RUN_DIR` 的行为不是“只复用 subtasks”，而是：

1. 跳过前面的子任务分解
2. 加载旧 run 中已经完成的 frame 结果
3. 对这些已经完成的帧直接跳过，不重复跑

它更适合下面两类场景：

- 上一次运行中断了，现在继续跑
- 你确认旧 run 前半段结果没问题，希望从已有进度继续

日志里你会看到类似：

```text
[paths] reuse_run_dir=/path/to/old_run
[reuse] Loaded subtasks: ...
[reuse] Loaded 12 existing frame results from the previous run.
[frame 01/39] already available from reused run; skipping
```

注意：

- 如果你刚修改了 region / subtask 规则，不要复用旧 run
- 否则你看到的还是旧规则生成的 subtasks
- 如果旧 run 里已经有很多 `frame_results`，这些帧也会被直接跳过
- `--overwrite` 不能覆盖这一点；它只影响缓存文件是否重生成，不会取消“已完成帧跳过”逻辑

### 4. 只想复用 subtasks，但想重新跑所有帧

当前代码里，`REUSE_RUN_DIR` 只要发现旧目录里存在 `frame_results_*`，就会把这些帧当成“已完成”并跳过。

所以如果你的目标是：

- 保留旧 subtasks
- 但重新测试新的单帧定位策略、thinking 配置或 prompt

那就不要直接指向一个已经包含完整 `frame_results_*` 的 run 目录。

更稳妥的做法有两种：

1. 不使用 `REUSE_RUN_DIR`，直接全新跑一遍。
2. 复制一个旧 run 目录，只保留 subtasks 相关文件，不保留 `frame_results_*` 目录，再把新的目录传给 `REUSE_RUN_DIR`。

### 5. 复用旧 run 并同时开启 thinking

```bash
REUSE_RUN_DIR=/home/gs/my_project/data_generation_pipline/output/qwen_annotations/episode_1_20260403_162938 \
QWEN_THINKING=true \
QWEN_THINKING_BUDGET=4000 \
QWEN_TOP_P=0.8 \
QWEN_TEMPERATURE=0.7 \
QWEN_RESULT_FORMAT=message \
./run_annotate_vln_episode.sh
```

注意：

- 这只会作用在“还没完成的那些帧”上
- 已经在旧 run 里完成的帧仍然会被跳过

### 6. 指定模型

```bash
SEGMENT_MODEL=qwen3.6-plus \
LOCALIZE_MODEL=qwen3.6-plus \
./run_annotate_vln_episode.sh
```

## 输出目录结构

每次运行都会生成一个带时间戳的目录，例如：

```text
output/qwen_annotations/episode_1_20260403_213404/
```

里面常见内容包括：

- `qwen_episode_annotations_<timestamp>.json`
- `progress_<timestamp>.json`
- `annotated_frames_<timestamp>/`
- `frame_results_<timestamp>/`
- `cache_<timestamp>/`
- `error_frame_XXXX_<timestamp>.json`

重点文件说明：

- `qwen_episode_annotations_<timestamp>.json`
  保存整次运行的总结果
- `progress_<timestamp>.json`
  保存当前进度
- `frame_results_<timestamp>/frame_0001_<timestamp>.json`
  保存单帧结构化结果
- `cache_<timestamp>/raw_responses_<timestamp>/*.txt`
  保存模型原始文本输出
- `cache_<timestamp>/raw_responses_<timestamp>/*_response.json`
  保存完整 response 结构，便于 debug

## 日志里会打印什么

启动时会打印类似：

```text
[start] run=20260403_213404 episode=1 frames=39 segment_model=qwen3.6-plus localize_model=qwen3.6-plus
[qwen] temperature=0.0 top_p=None result_format=None thinking=False thinking_budget=None
```

如果开启 thinking，会看到：

```text
[qwen] temperature=0.7 top_p=0.8 result_format=message thinking=True thinking_budget=4000
```

这部分配置也会写入最终输出 JSON 的 `qwen_request_settings` 字段。

## 直接运行 Python 脚本

如果你不想走 shell，也可以直接运行：

```bash
/home/gs/anaconda3/bin/python qwen_annotation/annotate_vln_episode.py \
  --episode-id 1 \
  --video-path /home/gs/my_project/data_generation_pipline/frames/1.mp4 \
  --frames-dir /home/gs/my_project/data_generation_pipline/frames \
  --output-dir /home/gs/my_project/data_generation_pipline/output/qwen_annotations \
  --segment-model qwen3.6-plus \
  --localize-model qwen3.6-plus \
  --qwen-temperature 0.0 \
  --no-thinking
```

开启 thinking 的版本：

```bash
/home/gs/anaconda3/bin/python qwen_annotation/annotate_vln_episode.py \
  --episode-id 1 \
  --segment-model qwen3.6-plus \
  --localize-model qwen3.6-plus \
  --qwen-temperature 0.7 \
  --qwen-top-p 0.8 \
  --qwen-result-format message \
  --thinking \
  --thinking-budget 4000
```

## `qwen_api.py` 单独测试

`qwen_annotation/qwen_api.py` 也可以单独测试多模态请求。

例如：

```bash
/home/gs/anaconda3/bin/python qwen_annotation/qwen_api.py \
  "What is in this picture?" \
  --image /path/to/test.jpg \
  --model qwen3.6-plus \
  --thinking \
  --thinking-budget 4000 \
  --top-p 0.8 \
  --temperature 0.7
```

如果想强制 JSON 输出：

```bash
/home/gs/anaconda3/bin/python qwen_annotation/qwen_api.py \
  "List the visible objects as JSON." \
  --image /path/to/test.jpg \
  --model qwen3.6-plus \
  --json
```

## 当前实现细节

单帧阶段不是写死某个框的位置，而是：

1. 把当前帧图片发给 Qwen
2. 同时附上 instruction、当前 subtask、当前帧需要检查的实体名单
3. 让模型返回结构化 JSON
4. 本地再做一次格式规范化和容错处理

当前本地硬规则主要是：

- 约束输出 JSON 字段
- 坐标裁剪到图像范围内
- 单帧最多重试 3 次
- `object` 画框，`region` 不画框

## 调试建议

如果结果异常，优先看这些文件：

- `cache_<timestamp>/raw_responses_<timestamp>/*.txt`
- `cache_<timestamp>/raw_responses_<timestamp>/*_response.json`
- `frame_results_<timestamp>/frame_XXXX_<timestamp>.json`
- `error_frame_XXXX_<timestamp>.json`

如果你修改了 prompt 或 region 规则，建议不要复用旧的 `REUSE_RUN_DIR`，直接新开一次 run。
