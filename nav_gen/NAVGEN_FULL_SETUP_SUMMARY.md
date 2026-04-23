# NavGen 从环境配置到验证与视频可视化的全流程总结

更新时间：2026-04-23

本文档总结本轮围绕 LH-VLN / NavGen 所做的完整工作，范围从最开始的环境与数据准备，一直到小规模验证、GPU 兼容处理、Qwen 接口切换，以及第一人称视频导出与 2D 目标框可视化。

说明：

- 本文档聚焦本轮为“跑通 NavGen 并便于人工核查结果”所做的改动。
- 仓库里还有一些其它未在本轮直接展开的改动或缓存文件，例如 `__pycache__`、部分 README 调整等，这里不逐项展开。
- 旧的 [NAVGEN_VIDEO_EXPORT_SUMMARY.md](./NAVGEN_VIDEO_EXPORT_SUMMARY.md) 保留，作为视频导出专项说明；本文档是更完整的总览版。

## 1. 最终达成的结果

本轮最终达成了以下目标：

1. 按项目 README 的思路完成了 NavGen 所需基础环境和依赖准备。
2. 将 HM3D 数据目录整理到代码所需结构下，使 NavGen 能直接读取。
3. 补齐了 NavGen 所需的 RAM 预训练权重路径。
4. 将原先调用 OpenAI 官方接口的逻辑切换为 DashScope 兼容模式下的 `qwen3.6-plus`。
5. 处理了 `split_task.py` 在 RTX 5080 上的 CUDA/架构兼容问题。
6. 新建并使用了可在该 GPU 上工作的 `lhvln-cu128` 环境来执行 GPU 相关阶段。
7. 做了一轮 NavGen 小规模 smoke run，并成功生成了 5 条可用轨迹。
8. 基于成功轨迹补充了多种视频导出能力：
   - 普通第一人称 RGB 视频
   - 修正朝向后的准确回放
   - `1 fps`、一帧一个动作的慢速视频
   - 带动作/目标/任务文本标注的视频
   - 对指令目标物体绘制 2D bounding box 和类别标签的视频

## 2. 环境与依赖配置

### 2.1 基础环境

基础安装思路遵循项目根目录 [README.md](../README.md)：

- Python 版本：`3.9`
- Habitat-Sim 版本：`0.3.1`
- 其余依赖来自 `requirements.txt`

在实际推进过程中，环境分成了两层理解：

1. 基础环境
   - 用于按 README 安装项目所需 Python 依赖、Habitat-Sim、NavGen 相关包。

2. GPU 兼容环境 `lhvln-cu128`
   - 后续为了让 `split_task.py` 在 RTX 5080 上真正使用 GPU，额外准备了 `lhvln-cu128` 环境。
   - 这个环境主要用于解决旧版 PyTorch / CUDA 组合对新 GPU 架构支持不足的问题。

### 2.2 为什么需要单独的 GPU 环境

在实际预检中，`split_task.py` 所依赖的 RAM 推理阶段会尝试使用 CUDA。
但在当前机器的 RTX 5080 上，旧环境中的 PyTorch 对对应 GPU 架构支持不足，会触发类似“当前 torch 不支持该 GPU 架构”的问题。

因此做了两层处理：

1. 在原始逻辑中加入运行时检测
   - 如果当前 CUDA 运行时不可用，或者当前 PyTorch 不支持该 GPU 架构，就回退到 CPU，避免流程直接崩溃。

2. 额外建立 `lhvln-cu128`
   - 用一套更合适的 CUDA / PyTorch 组合，让 `split_task.py` 这一步真正能跑在 GPU 上。

## 3. 数据与模型文件整理

### 3.1 HM3D 路径整理

根据 README，代码最终希望看到的关键目录结构是：

```text
LH-VLN/
├── data/
│   ├── hm3d/
│   │   ├── train/
│   │   ├── val/
│   │   └── hm3d_annotated_basis.scene_dataset_config.json
│   └── models/
```

本轮工作中，重点是把 HM3D 数据整理到代码默认读取的位置，也就是：

- `data/hm3d/train`
- `data/hm3d/val`
- `data/hm3d/hm3d_annotated_basis.scene_dataset_config.json`

这一步的意义是让以下默认路径都能直接成立：

- `project_path + "/data/hm3d/"`
- `project_path + "/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json"`

这些默认值目前会被 `nav_gen/main.py`、`nav_gen/run_smoke_test.py`、视频导出脚本等多个入口复用。

### 3.2 RAM 模型补齐

README 提到 NavGen 依赖 RAM 的预训练模型：

- `data/models/ram_plus_swin_large_14m.pth`

本轮已经将该权重路径补齐到代码预期位置。

这一步的意义是让下列默认参数不需要额外再改：

- `nav_gen/main.py` 中的 `--ram_model`
- `nav_gen/run_smoke_test.py` 中的 `ram_model`

## 4. 代码层的关键适配

## 4.1 LLM 接口切换到 Qwen3.6-Plus

你提出希望把项目里原先调用 OpenAI 官方接口的地方切换成阿里 DashScope 兼容模式下的 `qwen3.6-plus`，并且 API key 已经写入环境变量。

围绕这一点，本轮主要做了以下调整：

### `nav_gen/gpt.py`

改造后的逻辑已经不再假设必须走 OpenAI 官方域名，而是：

- 通过 `OpenAI(...)` 客户端走兼容接口
- 默认读取：
  - `DASHSCOPE_API_KEY`
  - `DASHSCOPE_BASE_URL`
  - `NAVGEN_LLM_MODEL`
  - `NAVGEN_VLM_MODEL`
- 默认模型改为：
  - 文本模型：`qwen3.6-plus`
  - 视觉模型：默认也跟随 `qwen3.6-plus`

这样做之后，原先 `gpt4o`、`gpt4o_mini`、`gpt4_vision` 这几个函数名虽然还保留，但底层已经不再强绑定 OpenAI 的具体模型名，而是统一走 DashScope 兼容模式的模型参数。

### `nav_gen/main.py`

主入口也同步改成了环境变量驱动：

- `--API_KEY` 默认读取 `DASHSCOPE_API_KEY`
- `--llm_model` 默认读取 `NAVGEN_LLM_MODEL`，默认值是 `qwen3.6-plus`
- `--vlm_model` 默认读取 `NAVGEN_VLM_MODEL`
- `--llm_base_url` 默认读取 `DASHSCOPE_BASE_URL`

这意味着后续直接跑 `nav_gen/main.py` 时，不需要在代码里硬编码模型或 base URL。

### `nav_gen/run_smoke_test.py`

为了让 smoke run 和主流程行为一致，这个辅助脚本也同步采用了相同的环境变量约定：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `NAVGEN_LLM_MODEL`
- `NAVGEN_VLM_MODEL`

## 4.2 GPU / CUDA 兼容处理

### `nav_gen/split_task.py`

这里加入了与 CUDA 运行时相关的保护逻辑，核心目标是：

- 先探测当前 CUDA 是否真的可用
- 再判断当前 PyTorch 是否支持当前 GPU 架构
- 如果不支持，则自动回退到 CPU

具体效果：

1. 避免在不兼容环境中直接崩溃。
2. 允许在旧环境中先保底跑通。
3. 为后续切换到 `lhvln-cu128` 提供平滑过渡。

### `nav_gen/run_split_steps.py`

新增了一个只执行以下两个阶段的辅助入口：

1. `split_traj(args)`
2. `gen_step_task(args)`

这个脚本的价值在于：

- 可以只跑“轨迹切分 + step task 生成”
- 不需要每次都从 task generation 和 trajectory generation 全流程重来
- 更适合在 GPU 兼容环境中单独验证 RAM/切分阶段

### `run_navgen_split_gpu.sh`

新增了一个 shell 脚本，专门用于在 `lhvln-cu128` 环境里跑上面的 split stages。

这个脚本会：

- 切到 `nav_gen/`
- 使用 `conda run -n lhvln-cu128`
- 自动注入：
  - `NAVGEN_SIM_GPU_DEVICE`
  - `NAVGEN_RAM_DEVICE`

它的作用可以概括为：

- 把 GPU 兼容环境、SIM GPU、RAM GPU 这几个变量封装起来
- 让 split 阶段更容易重复执行

## 4.3 Smoke Run 支撑脚本

### `nav_gen/run_smoke_test.py`

新增了一个更适合“快速验证是否能跑通”的辅助脚本。

它的功能包括：

1. 独立输出目录
   - 不污染默认 `nav_gen/task/` 结果

2. 自动循环生成任务并尝试轨迹生成

3. 记录详细 summary
   - 包括每次尝试的状态、耗时、成功 trial 目录等

4. 支持设置：
   - 目标成功条数
   - 最大尝试次数
   - 每条任务最大步数
   - 固定 scene

这个脚本最终用于完成本轮的 5 条成功轨迹 smoke run。

## 5. 视频导出与可视化增强

这一部分是后续为了更方便人工检查 NavGen 结果而新增的功能。

### 5.1 `nav_gen/habitat_base/config.py`

这里增加了导出视频时需要的相机配置能力：

- `front_rgb_only`
  - 仅启用前视 RGB
- `front_semantic_sensor`
  - 即使只开前视 RGB，也能保留前视语义图
- `render_width`
- `render_height`
- `render_sensor_height`
- `sensor_hfov`
- `sim_gpu_device`

这让导出脚本可以自由控制：

- 分辨率
- 相机高度
- 水平 FOV
- 是否启用语义图以支持 2D 框

### 5.2 `nav_gen/export_trajectory_rgb_video.py`

新增了完整的视频导出脚本，用来从：

- `success/trial_x/task.json`

重放为第一人称视频。

已实现的能力包括：

1. 轨迹回放
2. 朝向修正
3. 动作对齐视频
4. 文本标注
5. 2D 目标框
6. 元数据导出

#### 朝向修正

导出时发现原始 `yaw` 不能直接当作世界坐标角度使用，否则画面会“看起来像倒放”。

最终确认的正确映射为：

```text
world_angle_deg = saved_yaw_deg - 180
```

修正后，导出帧与原始 trial 中保存的 `front.png` 能基本对齐。

#### 动作对齐导出

新增 `frame_mode`：

- `state`
- `action`

其中 `action` 模式会让“一帧对应一个动作”，并且配合 `fps=1` 时，就可以做到“一秒一帧、一帧一个动作”。

这非常适合人工核查“动作是否和指令一致”。

#### 文本标注

开启 `--annotate` 后，画面左上角会显示：

- 当前帧编号
- 当前子 trial 编号
- 当前动作
- 当前目标
- 完整任务指令

#### 2D Bounding Box

开启 `--target_boxes` 后，会：

1. 保留前视 `semantic_sensor`
2. 读取前视语义图中的 `semantic_id`
3. 与 `semantic_scene.objects[*].semantic_id` 对齐
4. 仅保留“任务指令中出现的目标类别”
5. 将这些可见目标的语义区域转换成 2D 外接矩形
6. 在视频中绘制 bbox 和类别标签

注意：

- 这里只会框“当前画面里真正可见”的目标物体
- 这是基于语义 mask 的外接矩形，不是检测模型推理出的 bbox

## 6. 实际验证过程

## 6.1 NavGen 小规模 smoke run

本轮完成的一轮关键验证输出位于：

- `nav_gen/smoke_runs/5traj_20260423_160914/summary.json`

这次 smoke run 的结果是：

- 目标成功条数：`5`
- 实际尝试次数：`9`
- 成功轨迹数：`5`

从 summary 中可以看到：

- 有 `5` 次成功
- 有 `2` 次目标不可达
- 有 `1` 次 task generation error
- 有 `1` 次 task generation invalid

这说明：

- NavGen 主流程已经能够实际生成可用任务并跑出成功轨迹
- 但上游 LLM 任务生成阶段和场景可达性仍然存在正常范围内的失败样本

## 6.2 视频导出验证

针对这 5 条成功轨迹，已经导出了多套视频结果：

### 修正朝向后的普通第一人称视频

目录：

- `nav_gen/smoke_runs/5traj_20260423_160914/videos_corrected/`

### 慢速动作对齐视频

目录：

- `nav_gen/smoke_runs/5traj_20260423_160914/videos_action1fps/`

特点：

- `1 fps`
- 一帧一个动作
- 带动作/目标/任务文本标注

### 慢速动作对齐 + 目标框视频

目录：

- `nav_gen/smoke_runs/5traj_20260423_160914/videos_action1fps_boxes/`

特点：

- `1 fps`
- 一帧一个动作
- 带文本标注
- 带指令目标物体 2D 框

## 7. 已生成视频结果摘要

### 7.1 `videos_action1fps`

这组视频主要用于人工核对动作与指令：

1. `01_00241-h6nwVLpAKQz_action1fps.mp4`
   - `60` 帧，约 `60` 秒
2. `02_00410-v7DzfFFEpsD_action1fps.mp4`
   - `127` 帧，约 `127` 秒
3. `03_00179-MVVzj944atG_action1fps.mp4`
   - `30` 帧，约 `30` 秒
4. `04_00582-TYDavTf8oyy_action1fps.mp4`
   - `129` 帧，约 `129` 秒
5. `05_00177-VSxVP19Cdyw_action1fps.mp4`
   - `152` 帧，约 `152` 秒

### 7.2 `videos_action1fps_boxes`

这组视频主要用于同时核对：

- 动作是否合理
- 指令中的目标物体是否确实出现在正确视野中

统计如下：

1. `01_00241-h6nwVLpAKQz_action1fps_boxes.mp4`
   - 总帧数：`60`
   - 有框帧数：`37`
   - 类别：`box, couch, dish`

2. `02_00410-v7DzfFFEpsD_action1fps_boxes.mp4`
   - 总帧数：`127`
   - 有框帧数：`83`
   - 类别：`cup, table`

3. `03_00179-MVVzj944atG_action1fps_boxes.mp4`
   - 总帧数：`30`
   - 有框帧数：`30`
   - 类别：`alarm clock, bathroom cabinet, soap dispenser`

4. `04_00582-TYDavTf8oyy_action1fps_boxes.mp4`
   - 总帧数：`129`
   - 有框帧数：`22`
   - 类别：`bottle, desk`

5. `05_00177-VSxVP19Cdyw_action1fps_boxes.mp4`
   - 总帧数：`152`
   - 有框帧数：`111`
   - 类别：`desk, lamp, towel`

## 8. 当前推荐的运行方式

### 8.1 跑 NavGen 主流程

进入 `nav_gen/` 后，主流程入口仍然是：

```bash
python main.py
```

但现在建议通过环境变量显式控制：

- `DASHSCOPE_API_KEY`
- `DASHSCOPE_BASE_URL`
- `NAVGEN_LLM_MODEL`
- `NAVGEN_VLM_MODEL`
- `NAVGEN_SIM_GPU_DEVICE`
- `NAVGEN_RAM_DEVICE`

### 8.2 只跑 split stages

推荐优先使用：

```bash
./run_navgen_split_gpu.sh
```

它会自动切换到 `lhvln-cu128` 并设置 GPU 相关变量。

### 8.3 做 smoke run

可使用：

```bash
cd nav_gen
conda run -n lhvln-cu128 python run_smoke_test.py \
  --output_root /path/to/output \
  --target_successes 5 \
  --max_attempts 12 \
  --max_step 500
```

### 8.4 导出适合人工检查的视频

推荐：

```bash
cd nav_gen
conda run -n lhvln-cu128 python export_trajectory_rgb_video.py \
  --task_json "/path/to/success/trial_1/task.json" \
  --output_video "/path/to/output.mp4" \
  --fps 1 \
  --width 1280 \
  --height 744 \
  --sensor_height 1.0 \
  --hfov 86 \
  --frame_mode action \
  --annotate \
  --target_boxes
```

## 9. 本轮关键文件

本轮核心涉及文件如下：

- `README.md`
- `nav_gen/main.py`
- `nav_gen/gpt.py`
- `nav_gen/split_task.py`
- `nav_gen/run_smoke_test.py`
- `nav_gen/run_split_steps.py`
- `run_navgen_split_gpu.sh`
- `nav_gen/habitat_base/config.py`
- `nav_gen/export_trajectory_rgb_video.py`
- `nav_gen/NAVGEN_VIDEO_EXPORT_SUMMARY.md`
- `nav_gen/smoke_runs/5traj_20260423_160914/summary.json`

## 10. 当前仍需注意的限制

1. task generation 本身仍可能出现无效输出或 JSON 解析失败，这属于上游 LLM 生成质量问题的一部分。
2. 某些目标在当前 scene 中可能不可达，因此会出现 `unreachable_target`。
3. 2D 框只在目标进入当前前视画面且语义图中可见时显示。
4. 当前 2D 框是语义 mask 的外接矩形，不是检测模型预测框。
5. 目前更偏向“为了验证流程与结果可解释性”做增强，而不是对论文原始 NavGen 主体算法做结构性改写。

