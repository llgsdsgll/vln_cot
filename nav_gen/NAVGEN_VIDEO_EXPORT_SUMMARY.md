# NavGen 视频导出改动总结

更新时间：2026-04-23

本文档总结本轮对 NavGen 轨迹视频导出与可视化相关的改动。
说明：这里聚焦“视频导出、回放校正、慢速动作对齐、目标框可视化”这部分，不覆盖更早的环境配置、Qwen 接口切换、RAM 模型下载等其它改动。

## 1. 改动目标

本轮改动主要解决了以下问题：

1. 支持把 `success/trial_x/task.json` 重放成第一人称 RGB 视频。
2. 支持自定义相机高度、分辨率和水平 FOV。
3. 修正导出视频的朝向错误，使其与 NavGen 原始 rollout 一致。
4. 支持慢速检查模式：`1 fps`，一帧对应一个动作。
5. 支持在视频中叠加动作、目标、任务文本。
6. 支持对指令中出现且当前可见的目标物体绘制 2D bounding box，并标注类别。

## 2. 代码改动

### 2.1 `nav_gen/habitat_base/config.py`

新增和调整了导出时所需的相机配置能力：

- 新增 `front_rgb_only` 选项。
  作用：仅启用前视 RGB，相比原始多传感器导出更轻量。
- 新增 `front_semantic_sensor` 选项。
  作用：在“仅前视 RGB”模式下，仍然可以单独保留前视语义传感器，用于目标框绘制。
- 新增 `render_width` / `render_height`。
  作用：允许导出视频时指定渲染分辨率。
- 新增 `render_sensor_height`。
  作用：允许导出视频时指定相机高度，例如 `1.0m`。
- 新增 `sensor_hfov`。
  作用：允许导出时指定水平 FOV，例如 `86°`。
- 将 `gpu_device_id` 改为从 `sim_gpu_device` 读取，而不是写死。

本次实际使用的导出参数为：

- 相机高度：`1.0m`
- 分辨率：`1280x744`
- 水平 FOV：`86°`
- 对应垂直 FOV：约 `56.918°`

### 2.2 `nav_gen/export_trajectory_rgb_video.py`

新增了完整的视频导出脚本，核心能力如下：

- 从 `success/trial_x/task.json` 重放轨迹。
- 直接按保存的 `pos` 与 `yaw` 设置相机位姿。
- 输出 MP4 视频和同名 JSON 元数据。

#### 已实现功能

1. 轨迹重放
   - 支持遍历 `trial_0`, `trial_1`, ... 并按保存顺序回放。

2. 朝向修正
   - 新增 yaw 映射：
     - `world_angle_deg = saved_yaw_deg - 180`
   - 该修正用于解决“画面看起来像倒放”的问题。
   - 实际验证结果表明，修正后渲染画面与原始 trial 中保存的 `front.png` 基本一致。

3. `frame_mode`
   - `state`：每个保存状态导出一帧。
   - `action`：一帧对应一个已执行动作，会去掉初始空状态，适合人工核对动作与指令是否一致。

4. 文字标注
   - 通过 `--annotate` 可在画面左上角叠加：
     - 当前帧编号
     - 当前子 trial 编号
     - 当前动作
     - 当前目标
     - 完整任务指令

5. 目标物体 2D 框
   - 通过 `--target_boxes` 启用。
   - 使用前视 `semantic_sensor` 获取语义分割图。
   - 将语义图中的 `semantic_id` 与场景中对象的 `semantic_id` 对齐。
   - 只对“任务中出现的目标物体”进行筛选和绘制。
   - 当前物体进入视野时绘制 2D bbox，并标注类别名。

6. 元数据记录
   - 输出 JSON 中新增了：
     - `yaw_mapping`
     - `frame_mode`
     - `annotate`
     - `target_boxes`
     - `target_box_classes`
     - `frames_with_target_boxes`

## 3. 新增/增强的命令行参数

`export_trajectory_rgb_video.py` 当前支持：

- `--task_json`
- `--output_video`
- `--fps`
- `--width`
- `--height`
- `--sensor_height`
- `--hfov`
- `--sim_gpu_device`
- `--frame_mode {state,action}`
- `--annotate`
- `--target_boxes`

## 4. 生成结果目录

### 4.1 修正朝向后的常规视频

目录：

- `nav_gen/smoke_runs/5traj_20260423_160914/videos_corrected/`

特点：

- 第一人称 RGB
- 高度 `1.0m`
- 水平 FOV `86°`
- 朝向已修正

### 4.2 慢速动作对齐视频

目录：

- `nav_gen/smoke_runs/5traj_20260423_160914/videos_action1fps/`

特点：

- `1 fps`
- 一帧对应一个动作
- 可叠加动作/目标/任务文本

### 4.3 慢速动作对齐 + 目标框视频

目录：

- `nav_gen/smoke_runs/5traj_20260423_160914/videos_action1fps_boxes/`

特点：

- `1 fps`
- 一帧对应一个动作
- 带动作/目标/任务文本
- 带指令目标物体 2D bbox 和类别标签

## 5. 已导出的 5 条成功轨迹（带框版）

输出目录：

- `nav_gen/smoke_runs/5traj_20260423_160914/videos_action1fps_boxes/`

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

## 6. 关键验证结论

### 6.1 朝向问题

已确认原问题不是“帧顺序倒放”，而是 yaw 映射错误。
修正后，导出帧与原始 trial 保存的 `front.png` 基本一致。

### 6.2 目标框来源

目标框并非来自外部检测模型，而是来自 Habitat-Sim 的前视语义传感器：

- `semantic_sensor` 输出的 `semantic_id`
- 与 `semantic_scene.objects[*].semantic_id` 对齐
- 再筛选出任务中出现的目标类别

因此该框是“语义可见区域框”，不是检测模型预测框。

## 7. 当前限制

1. 只有目标物体真正进入当前第一视角、且语义图中可见时，才会显示框。
2. 遮挡严重或目标不在画面内时不会显示框。
3. 当前框是依据语义 mask 的最小外接矩形生成，不是实例检测器意义上的高质量检测框。
4. 当前默认会对任务中所有目标类别进行可见性检查，而不是只框“当前子目标”。

## 8. 推荐导出命令

如果希望得到最适合人工核对的版本，推荐：

```bash
cd /home/gs/my_project/LH-VLN/nav_gen
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

## 9. 涉及文件

- `nav_gen/habitat_base/config.py`
- `nav_gen/export_trajectory_rgb_video.py`
- `nav_gen/smoke_runs/5traj_20260423_160914/videos_corrected/`
- `nav_gen/smoke_runs/5traj_20260423_160914/videos_action1fps/`
- `nav_gen/smoke_runs/5traj_20260423_160914/videos_action1fps_boxes/`

