# `run_navgen_remote_batch.sh` 使用说明

## 1. 这个脚本是做什么的

`run_navgen_remote_batch.sh` 是一个面向批量生成 NavGen 数据的包装脚本。

它的核心作用不是直接改 NavGen 算法，而是帮你把下面几件事串起来：

1. 在 `lhvln-cu128` conda 环境里反复执行 `nav_gen/main.py`
2. 每次只跑 `--loop 1`，也就是一次只生成 1 条候选长任务
3. 每一轮结束后，立即把该轮产物同步到远程服务器
4. 可选地在每条成功轨迹生成后，立刻导出对应第一人称 RGB 视频
5. 默认在同步成功后删除本地这一轮的临时目录，减少本地占用
6. 自动重试，直到拿到你指定数量的成功 task

这个脚本适合下面这种场景：

- 你想连续生成很多条 NavGen 数据
- 你担心本地磁盘被轨迹图片撑爆
- 你希望结果一边生成一边写到远程服务器
- 你不想因为偶尔抽到不可达 task，就整批中断


## 2. 它实际跑的是哪一套流程

脚本内部调用的是：

```bash
python main.py --loop 1 ...
```

也就是说，每一次尝试都会完整执行 NavGen 的四步：

1. `gen_task`
2. `gen_traj`
3. `split_traj`
4. `gen_step_task`

如果你加上 `--export-videos`，那么每一轮在 `main.py` 成功之后，还会额外执行：

5. `export_trajectory_rgb_video.py`

但是这个脚本自己会在外层循环，所以：

- 传给脚本的 `--loop N`
- 表示“目标成功数是 N”
- 不是“只尝试 N 次”

如果某一轮生成失败，或者虽然退出码是 0 但没有产出有效 step task，脚本会继续下一轮，直到成功数达到目标，或者达到 `--max-attempts` 限制。


## 3. 默认本地和远程路径

默认情况下：

- 本地临时根目录：
  `/home/gs/my_project/LH-VLN/nav_gen/remote_batches`
- 远程根目录：
  `/mnt/data-cpfs/gengshuang/lhvln_dataset`
- 远程服务器：
  `root@139.196.171.150 -p 6222`

一次运行会产生一个批次目录：

```text
nav_gen/remote_batches/<run_name>/
```

远程也会对应生成：

```text
/mnt/data-cpfs/gengshuang/lhvln_dataset/<run_name>/
```


## 4. 如何使用

### 4.1 最常用的命令

```bash
./run_navgen_remote_batch.sh \
  --run-name batch_20260427 \
  --export-videos \
  --loop 100 \
  --max-attempts 1000 \
  --max_step 500
```

这条命令的意思是：

- 远程批次名叫 `batch_20260427`
- 目标是拿到 100 条成功 task
- 最多尝试 1000 次
- 每条长轨迹最多走 500 个动作


### 4.1.1 如果你希望每条成功轨迹都自动导视频

```bash
./run_navgen_remote_batch.sh \
  --run-name batch_20260427 \
  --export-videos \
  --video-fps 1 \
  --video-frame-mode action \
  --video-sensor-height 1.0 \
  --video-hfov 86 \
  --video-target-box-scope instance \
  --loop 10 \
  --max_step 500
```

这组参数的含义是：

- 每条成功长轨迹都会导出一个视频
- `1 fps`，基本等价于“一帧对应一个动作变化”
- 视角高度 `1.0m`
- 水平视场角 `86°`
- 默认会叠加文本标注
- 默认会画 2D 框
- `instance` 表示只框当前指令真正对应的那个目标实例，而不是同类所有物体


### 4.2 如果你想一直重试到成功数够

```bash
./run_navgen_remote_batch.sh \
  --run-name batch_20260427 \
  --loop 100 \
  --max-attempts 0 \
  --max_step 500
```

其中：

- `--max-attempts 0`
- 表示不设上限


### 4.3 如果你想保留本地每一轮的中间结果

```bash
./run_navgen_remote_batch.sh \
  --run-name batch_debug \
  --loop 5 \
  --keep-local-item \
  --max_step 500
```

默认行为是：

- 每轮同步成功后删除本地 `item_0001`、`item_0002` 这类目录

加上 `--keep-local-item` 后：

- 本地会保留每一轮的完整输出，方便排查


### 4.4 常用参数说明

- `--run-name`
  指定这批任务的名字
- `--loop`
  目标成功 task 数量
- `--max-attempts`
  最大尝试次数，默认是 `10 * --loop`
- `--max_step`
  单条长轨迹允许的最大动作数
- `--sim-gpu-device`
  Habitat-Sim 使用的 GPU 号
- `--ram-device`
  RAM 模型推理设备，例如 `cuda:0` 或 `cpu`
- `--export-videos`
  成功轨迹生成后，自动导出视频
- `--video-subdir`
  每轮本地视频输出目录名，默认 `videos_action1fps_boxes`
- `--video-fps`
  视频帧率
- `--video-frame-mode`
  `state` 或 `action`，推荐 `action`
- `--video-codec`
  `h264` 或 `mp4v`，默认 `h264`
- `--video-sensor-height`
  第一人称相机高度，单位米
- `--video-hfov`
  水平视场角
- `--video-annotate` / `--no-video-annotate`
  是否在视频上叠加文字说明
- `--video-target-boxes` / `--no-video-target-boxes`
  是否画 2D 目标框
- `--video-target-box-scope`
  `instance` 表示只框一个具体目标实例，`class` 表示框当前子目标同类物体
- `--keep-local-item`
  保留本地每轮输出
- `--cleanup-local`
  成功结束后进一步清理本地批次目录


## 5. 运行结束后会生成什么

一个典型远程批次目录大概是这样：

```text
<run_name>/
├── batch_summary.tsv
├── logs/
│   ├── item_0001/
│   ├── item_0002/
│   └── ...
├── step_task/
│   ├── xxx.json
│   ├── yyy.json
│   └── ...
├── videos/
│   ├── item_0001/
│   ├── item_0002/
│   └── ...
└── task/
    ├── 2/
    ├── 3/
    ├── ...
    └── trail_list.txt
```

其中最重要的是：

- `logs/`
- `step_task/`
- `videos/`
- `task/`
- `batch_summary.tsv`


## 6. `logs` 文件夹里是什么

`logs/` 是按尝试轮次保存的运行日志。

例如：

```text
logs/item_0001/
├── navgen_main.log
├── step_task_logs.txt
└── video_export.log
```

### `logs/item_xxxx/navgen_main.log`

这是整轮 `python main.py --loop 1` 的主日志。

里面通常会包含：

- 任务生成 prompt 和模型输出
- Habitat-Sim / OpenGL / scene dataset 的警告
- 长轨迹执行过程
- 每一步动作、朝向、目标距离
- 哪个子目标成功到达
- 是否完成整条长任务

你可以把它理解成：

- “这一轮从开始到结束的总流水日志”


### `logs/item_xxxx/step_task_logs.txt`

这是第 3 步和第 4 步专门的日志，也就是：

- `split_traj`
- `gen_step_task`

里面通常会包含：

- RAM 模型加载信息
- `Segment Trajectory` 进度
- `Make Task` 进度
- 每个切出来的 step task 内容

你可以把它理解成：

- “轨迹切分和短指令生成阶段的专项日志”


### `logs/item_xxxx/video_export.log`

如果你启用了 `--export-videos`，这里会记录：

- 每个 `success/trial_1/task.json` 对应的视频导出命令
- 视频导出是否成功
- 导出脚本返回的 metadata
- 目标框实例解析情况

你可以把它理解成：

- “视频回放与目标框选阶段的专项日志”


## 7. `task` 文件夹里是什么

`task/` 保存的是原始长任务以及轨迹原始数据。

它既包含：

- 第 1 步生成的长任务定义

也包含：

- 第 2 步录制得到的轨迹图片和轨迹状态


### 7.1 `task/<数字>/<长指令>/config.json`

例如：

```text
task/3/Please head to .../config.json
```

这个 `config.json` 是长任务定义文件。

它一般包含：

- `Task instruction`
  原始长指令
- `Subtask list`
  例如 `Move_to(...)`、`Grab(...)`、`Release(...)`
- `Robot`
  机器人类型
- `Scene`
  场景 ID
- `Object`
  涉及到的目标物体和区域

这里最前面的数字，例如 `2`、`3`，表示：

- 该长任务里涉及多少个 `Move_to` 导航目标


### 7.2 `task/<数字>/<长指令>/success/trial_1/`

如果这条长任务成功录制出完整轨迹，就会有这个目录。

这里面一般会包含：

- 一系列按动作步命名的图片目录
- `task.json`

其中每个动作目录大概长这样：

```text
12_move_forward_for_bag/
├── left.png
├── front.png
├── right.png
├── depth_left.png
├── depth_front.png
└── depth_right.png
```

这些是轨迹执行过程中的观测图像。

`task.json` 则会保存：

- 各段 trial 的位置序列
- yaw 序列
- action 序列
- geodesic distance
- 完整长任务执行记录


### 7.3 `task/<数字>/<长指令>/fail/`

如果长任务没有在 `max_step` 内完成，理论上会进入 `fail/`。

里面会保存：

- 失败轨迹对应的图片
- 对应的 `task.json`

不过在你的远程批次里，也可能看到：

- 只有 `config.json`
- 只有 `temp/`
- 没有 `success/`

这种通常说明：

- 这轮 task 本身不可达
- 或者这轮没有形成有效成功轨迹
- 但脚本仍然把这轮的中间结果同步上来了


### 7.4 `task/trail_list.txt`

这是第 3 步 `split_traj` 的中间结果文件。

它不是原始任务定义，而是“轨迹切分结果”。

文件里每一行都是一个 Python 风格的列表，描述一条切分后的轨迹片段序列。

每个片段大致包含：

- `trajectory`
  它来自哪条长轨迹
- `start`, `end`
  在原始轨迹中的起止步号
- `label`
  例如 `move_forward`、`turn_left`、`make a left turn`
- `target`
  当前片段服务的目标物体
- `scene tags`
  RAM 从该片段画面里识别出的场景标签

你可以把 `trail_list.txt` 理解成：

- “从长轨迹到短任务之间的中间桥梁文件”


## 8. `step_task` 文件夹里是什么

`step_task/` 是最终最像“训练数据”的结果目录。

这里存放的是：

- 从长轨迹切出来的短程导航任务 JSON

每个文件名通常就是这条短指令本身，例如：

```text
step_task/Move forward through the bedroom to the bag..json
```

每个 JSON 一般包含：

- `trajectory path`
  对应原始长轨迹的位置
- `start`, `end`
  这条短任务在原始轨迹中的步号范围
- `Robot`
- `Scene`
- `target`
- `Region`
- `start_pos`
  这条短任务开始时的机器人位置
- `start_yaw`
  这条短任务开始时的朝向
- `Task instruction`
  最终生成的短指令

你可以把它理解成：

- “最终可直接拿来做 step-level 数据集的样本”


## 9. `videos` 文件夹里是什么

只有在启用 `--export-videos` 时，远程目录下才会出现 `videos/`。

典型结构类似：

```text
videos/
└── item_0001/
    └── videos_action1fps_boxes/
        └── 3/
            └── Please head to .../
                └── success/
                    └── trial_1/
                        ├── trajectory_rgb.mp4
                        └── trajectory_rgb.json
```

其中：

- `trajectory_rgb.mp4`
  第一人称 RGB 轨迹视频
  现在默认会导出为更通用的 `H.264 / yuv420p / faststart` MP4
- `trajectory_rgb.json`
  视频导出 metadata

这个 metadata 里通常会包含：

- 帧数、分辨率、FOV、高度
- `frame_mode`
- 是否开启文字标注
- 是否开启目标框
- `trial_targets`
- `resolved_target_boxes`

其中 `resolved_target_boxes` 最重要。

在 `--video-target-box-scope instance` 模式下，它会记录：

- 当前 trial 的目标类别，例如 `bag`
- 当前指令引用，例如 `bag_6`
- 最终选中的具体 semantic object，例如 `bag_341`
- 它是通过什么策略选中的

目前脚本优先尝试按任务里的 `Object + Region` 理解目标；
如果 HM3D 语义 region 信息不稳定，则会自动退回成：

- 沿该 trial 轨迹回放
- 在语义图中找“当前子目标里最可见的那个实例”

所以它不会再像旧逻辑那样把同类多个物体一起框出来。


## 10. `batch_summary.tsv` 是什么

这个文件记录整个批次每一轮尝试的结果。

它通常包含这些列：

- `item_id`
- `status`
- `success_progress`
- `step_task_json_count`
- `video_count`
- `video_status`
- `local_item_dir`
- `remote_run_dir`

常见 `status` 包括：

- `ok`
  这一轮产出了有效成功 task
- `no_success_task`
  这一轮退出码正常，但没有产出有效成功样本
- `navgen_failed:<code>`
  这一轮运行时报错退出
- `video_failed:<detail>`
  长轨迹生成出来了，但视频导出没有成功

这个文件很重要，因为它能帮你快速判断：

- 一共尝试了多少轮
- 其中多少轮真正成功
- 哪一轮只生成了中间产物但没有形成有效 step task
- 哪一轮视频是否也顺利导出了


## 11. 需要特别注意的一点

对于这个脚本来说：

- `task/` 不是“纯成功数据目录”
- `logs/` 不是“只记录错误”
- `step_task/` 才最接近最终训练可用结果

更直白一点说：

- `task/` 更偏“原始生成痕迹 + 长轨迹原始记录”
- `step_task/` 更偏“清洗后的短任务数据”
- `videos/` 更偏“人工核查轨迹和目标框是否对齐”
- `logs/` 更偏“调试和追踪运行过程”


## 12. 推荐的检查顺序

如果你想判断这一批数据是不是可用，建议按这个顺序看：

1. 先看 `batch_summary.tsv`
2. 再看 `step_task/` 里有没有生成足够多的 JSON
3. 如果启用了视频导出，再看 `videos/` 和对应的 `trajectory_rgb.json`
4. 如果某轮异常，再看 `logs/item_xxxx/`
5. 如果要追原始轨迹，再去看 `task/.../success/trial_1/`


## 13. 一个最小使用示例

```bash
./run_navgen_remote_batch.sh \
  --run-name batch_demo \
  --export-videos \
  --video-fps 1 \
  --video-frame-mode action \
  --video-target-box-scope instance \
  --loop 10 \
  --max-attempts 100 \
  --max_step 500
```

运行结束后，重点查看：

- 远程：
  `/mnt/data-cpfs/gengshuang/lhvln_dataset/batch_demo/`
- 其中：
  `batch_summary.tsv`
- 以及：
  `step_task/`
- 如果启用了视频：
  `videos/`
