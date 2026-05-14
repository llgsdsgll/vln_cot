# NavGen CoT Workflow Files

## 说明范围

本文按当前这条 NavGen `step_task` CoT 合成流程整理，覆盖两个目录：

- `cot_ann/`：核心脚本
- `debug/`：本地暂存、测试输出、逐帧 Markdown

这里的“正在使用”指的是当前这轮基于远程 `*.frame_info.json + *_frame_assets`、直连 DashScope API、输出 JSONL 与逐帧 Markdown 的流程。

## 核心脚本

### 1. `cot_ann/navgen_cot_synthesizer.py`

当前主脚本，负责 NavGen 数据合成。

作用：

- 读取 `*.frame_info.json`
- 自动关联同名 `*.json`
- 读取 `*_frame_assets/frames_rgb`
- 构建当前帧 prompt
- 调用 DashScope `qwen3.6-plus`
- 生成每帧 CoT JSONL
- 对 `step1` 到 `step4` 做结构化校验

当前这版还包含：

- 当前目标 / 下一个目标优先的 `step2`
- 目标不可见时，结合历史 3 帧和动作轨迹推断目标方位
- 对 `left / right / upper-left / lower edge` 这类方向词的校验

### 2. `cot_ann/run_navgen_remote_synthesis.sh`

当前远程数据入口脚本。

作用：

- 从远程机器同步：
  - `*.frame_info.json`
  - 同名 `*.json`
  - 同名 `*_frame_assets`
- 暂存到本地 `debug/navgen_remote_stage/<timestamp>/`
- 调用 `navgen_cot_synthesizer.py`
- 可选继续导出逐帧 Markdown

当前这条 NavGen 流程，实际推荐从这个脚本启动。

### 3. `cot_ann/vln_data_synthesizer.py`

通用基础类脚本，仍然被当前 NavGen 脚本依赖。

主要提供：

- OpenAI-compatible API 调用封装
- 重试逻辑
- 通用文本清洗
- 结构化输出组装
- 通用 memory anchor 逻辑

`navgen_cot_synthesizer.py` 是在它的基础上做的 NavGen 定制。

### 4. `cot_ann/jsonl_to_frame_markdown.py`

逐帧导出工具。

作用：

- 读取生成好的 JSONL
- 按 `episode_id + frame` 拆成单独 Markdown
- 生成 `index.md`

当前检查单帧 CoT 时，主要用它。

### 5. `cot_ann/run_synthesis.sh`

旧的通用 VLN 合成入口，保留中。

说明：

- 它不是当前这条 NavGen `step_task` 远程同步流程的主入口
- 但仍对应旧的通用视频/GT 合成流程
- 当前不建议用它跑 NavGen `frame_info` 数据

因此暂时保留，不纳入这次清理。

## 当前保留的 debug 文件

### 1. `debug/navgen_remote_stage/20260514_192821/`

当前保留的本地暂存输入。

里面包含：

- 远程同步下来的 `*.frame_info.json`
- 对应 `*.json`
- 对应 `*_frame_assets/frames_rgb`

保留原因：

- 方便继续本地复跑
- 不需要重新 ssh/rsync 就能重复调试

### 2. `debug/navgen_test_item0001_16frames_dirv2.jsonl`

当前最新的 16 帧方向推理定向测试结果。

保留原因：

- 用来验证“不可见目标先写方位、再写距离”的新 prompt / 校验是否稳定

### 3. `debug/navgen_test_item0001_16frames_dirv2_md/`

上面这份 16 帧测试对应的逐帧 Markdown。

保留原因：

- 方便人工逐帧检查 `Step 2`
- 当前重点排查文件在这里：
  - `ep00001_frame0014.md`

### 4. `debug/navgen_full_item0001.jsonl`

整条 `step_task` 的完整输出。

保留原因：

- 作为长序列基线结果
- 便于观察 subtask 切换帧的稳定性

说明：

- 这是较早一次 full run 的结果
- 如果后续需要最新版本的整条结果，应基于当前脚本重新跑一遍覆盖或另存

### 5. `debug/navgen_full_item0001_md/`

完整 `step_task` 的逐帧 Markdown，对应上面的 full run。

保留原因：

- 方便按帧回看完整序列

## 本次已清理的旧文件

下面这些属于中间测试产物，已经被更新结果替代，因此删除：

- `debug/navgen_test_item0001_3frames_v2.jsonl`
- `debug/navgen_test_item0001_3frames_v3.jsonl`
- `debug/navgen_test_item0001_5frames.jsonl`
- `debug/navgen_test_item0001_5frames_md/`
- `debug/navgen_test_item0001_16frames_dirv1.jsonl`
- `debug/navgen_test_item0001_16frames_dirv1_md/`
- `debug/navgen_remote_stage/20260514_184835/`

清理原则：

- 删除明显被新版本替代的短序列调试文件
- 保留最新定向测试
- 保留完整长序列结果
- 保留一份可直接复跑的本地 stage 输入

## 当前推荐的最小工作流

### 远程同步并生成 CoT

```bash
bash cot_ann/run_navgen_remote_synthesis.sh \
  --remote-frame-info "<remote_step_task>.frame_info.json"
```

### 直接用本地 stage 输入复跑

```bash
python3 cot_ann/navgen_cot_synthesizer.py \
  --frame-info "debug/navgen_remote_stage/20260514_192821/<step_task>.frame_info.json" \
  --step-task-json "debug/navgen_remote_stage/20260514_192821/<step_task>.json" \
  --output debug/<new_output>.jsonl
```

### 导出逐帧 Markdown

```bash
python3 cot_ann/jsonl_to_frame_markdown.py \
  --input debug/<new_output>.jsonl
```

## 建议

如果后面继续迭代 prompt，建议沿用下面的保留策略：

- 永远保留一份最新 full run
- 永远保留一份最新定向小样本测试
- 中间版本用 `v1 / v2 / v3` 命名，但确认淘汰后及时清理
- `navgen_remote_stage` 只保留最新一份可复跑输入
