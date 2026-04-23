# NavGen 代码流程说明

更新时间：2026-04-23

本文档按 `nav_gen` 的实际执行顺序来讲解代码，而不是按文件列表逐个介绍。
如果你想回答“从运行 `python main.py` 开始，NavGen 到底做了什么”，这份文档就是为这个问题写的。

说明：

- 这里讲的是 `nav_gen` 主流水线。
- 文档最后会补充本轮新增的辅助脚本，例如 smoke run、split GPU 入口、视频导出脚本，但它们不是论文 NavGen 的主生成链。

## 1. 先看目录里哪些东西在主流程中会被用到

`nav_gen/` 里和主流程最相关的目录/文件可以先这样理解：

- `main.py`
  - NavGen 主入口，按顺序调用四个阶段。
- `task_gen.py`
  - 负责生成长任务，也就是 LH-VLN task。
- `dataset_gen.py`
  - 负责在 Habitat-Sim 中执行任务并记录轨迹。
- `split_task.py`
  - 负责把长轨迹切成更细的局部片段，并生成 step-by-step task。
- `dataset.py`
  - 负责把 `task/` 目录里的任务读回来，喂给轨迹生成阶段。
- `gpt.py`
  - LLM/VLM 调用封装。
- `habitat_base/`
  - Habitat-Sim 的包装层，尤其是 `config.py`、`simulation.py`、`visualization.py`。
- `prompt/`
  - 放 task generation 和 step task generation 用到的提示词模板。
- `scene/`
  - 放每个场景的对象列表文本，以及区域统计 CSV。
- `task/`
  - 主流程生成出来的长任务、轨迹、成功/失败记录都在这里。
- `step_task/`
  - 切分后的 step task 输出目录。
- `logs/`
  - RAM 识别日志、统计日志等。

## 2. 主流程总览

`nav_gen/main.py` 里把整个流程明确切成四步：

1. 生成 task
2. 生成 trajectory
3. 切分 trajectory
4. 生成 step-by-step task

也就是说，真正的主链是：

```text
main.py
  -> task_gen.gen_task()
  -> dataset_gen.gen_traj()
  -> split_task.split_traj()
  -> split_task.gen_step_task()
```

## 3. 第 0 步：程序启动时做了什么

### 文件：`nav_gen/main.py`

当你在 `nav_gen/` 目录下运行：

```bash
python main.py
```

代码一开始会先做三件事：

1. 计算路径
   - `nav_gen_path = os.getcwd()`
   - `project_path = os.path.dirname(nav_gen_path)`

2. 确保三个输出目录存在
   - `task/`
   - `step_task/`
   - `logs/`

3. 解析命令行参数
   - `read_args()`

### `read_args()` 里配置了哪些核心参数

可以按功能分成四组：

#### A. LLM / VLM 调用相关

- `API_KEY`
- `llm_model`
- `vlm_model`
- `llm_base_url`

当前这份代码已经适配了 DashScope 兼容接口，默认会读：

- `DASHSCOPE_API_KEY`
- `NAVGEN_LLM_MODEL`
- `NAVGEN_VLM_MODEL`
- `DASHSCOPE_BASE_URL`

#### B. task generation 相关

- `scene_path`
- `prompt_path`
- `task_path`
- `region_file`
- `loop`
- `scene_id`
- `sample_region`
- `sample_obj`

#### C. Habitat-Sim / 轨迹执行相关

- `scene`
- `scene_dataset`
- `sim_gpu_device`
- `max_step`
- `success_dis`

#### D. split / RAM 相关

- `step_task_path`
- `ram_model`
- `ram_device`
- `ram_logs`
- `split_save_path`

## 4. 第 1 步：`task_gen.py` 如何生成长任务

### 入口函数

主入口里会执行：

```python
for i in range(args.loop):
    gen_task(args)
```

也就是说，`gen_task(args)` 会被调用多次，每次尝试生成一个任务。

### 4.1 `gen_task(args)` 的输入依赖

它主要依赖两类输入：

1. 场景统计信息
   - `scene/Per_Scene_Region_Weighted_Votes.csv`
   - `scene/<scene_id>.txt`

2. 提示词模板
   - `prompt/system.txt`
   - `prompt/rule.txt`
   - `prompt/example.txt`
   - `prompt/spot.txt`
   - `prompt/stretch.txt`

### 4.2 `read_scene(file_path)` 先做什么

`read_scene()` 会读取 `Per_Scene_Region_Weighted_Votes.csv`，
建立一个：

```text
scene_id -> [region_1, region_2, ...]
```

的映射。

这一步的目标是先知道：

- 当前 scene 有哪些 room / region

### 4.3 `gen_task()` 如何准备 scene 输入

接下来 `gen_task()` 会做这些事：

1. 选 scene
   - 如果传了 `scene_id` 就固定场景
   - 否则随机抽一个 scene

2. 选 robot
   - 在 `spot` 和 `stretch` 之间随机选一个

3. 读取对应的 `scene/<scene_id>.txt`
   - 这个 txt 文件记录了每个 region 里有哪些 object，以及 object 的位置

4. 构造 `input_scene`
   - 结构大致是：

```text
Region x: room_name -> [obj1, obj2, obj3, ...]
```

5. 可选做 region sampling / object sampling
   - `sample_region()`
   - `sample_obj()`

这一步的本质是：

- 从场景的语义统计中，整理出一个“给 LLM 看得懂的房间-物体字典”

### 4.4 `gen_task()` 如何调用 LLM

它会读取：

- `prompt/rule.txt`
- `prompt/example.txt`
- `prompt/<robot>.txt`
- `prompt/system.txt`

然后拼出最终 prompt，交给：

```python
gpt4o_mini(args, args.prompt_path + "system.txt", prompt)
```

注意：

- 虽然函数名还叫 `gpt4o_mini`
- 但现在底层已经是通过 `gpt.py` 走 DashScope 兼容接口，默认模型是 `qwen3.6-plus`

### 4.5 LLM 输出什么格式

LLM 需要输出一个 JSON，里面至少包含：

- `Task instruction`
- `Subtask list`

随后代码会补上：

- `Robot`
- `Scene`

### 4.6 `gen_task()` 如何做合法性检查

生成结果不会直接保存，而是先过一轮校验：

1. 遍历 `Subtask list`
   - 提取所有 `Move_to('obj_region')`

2. 检查对象是否真的在对应 region 中

3. 检查 `Task instruction` 中是否直接出现 `region` / `Region`
   - 如果出现则判错

如果不合法：

- 返回 `False`

如果合法：

- 把任务保存到：

```text
task/<目标数>/<Task instruction>/config.json
```

### 4.7 第 1 步的输出是什么

这一阶段的输出是任务配置文件：

```text
nav_gen/task/<len>/<instruction>/config.json
```

这份 `config.json` 是后面轨迹执行阶段的输入。

## 5. 第 2 步：`dataset_gen.py` 如何执行任务并生成轨迹

### 入口函数

主入口里第二步调用的是：

```python
gen_traj(args)
```

### 5.1 `TaskDataset` 如何把任务读回来

`gen_traj(args)` 内部会先构造：

```python
dataset = TaskDataset(args)
```

`TaskDataset` 的工作是：

1. 遍历 `task/` 目录下所有 `config.json`
2. 读出 JSON
3. 从 `Subtask list` 中重新提取：
   - `Object`
   - `Region`
   - `Region Name`

也就是说：

- `task_gen.py` 负责把任务先存下来
- `dataset.py` 再把任务按轨迹执行需要的格式整理回来

### 5.2 `gen_traj()` 对每个任务做什么

对 dataset 中的每一条任务，都会调用：

```python
eval_for_one_task(args, config)
```

这个函数是“在 Habitat 里执行整条长任务”的核心。

## 6. `SceneSimulator`：轨迹执行阶段的核心包装器

### 文件：`habitat_base/simulation.py`

`eval_for_one_task()` 一开始就会构造：

```python
task_sim = SceneSimulator(args=args, config=config)
```

这是整个轨迹执行阶段最核心的对象。

### 6.1 `SceneSimulator.__init__()` 做了什么

它会依次完成：

1. 读取任务配置中的基础字段
   - `Scene`
   - `Robot`
   - `Object`
   - `Region`
   - `Task instruction`

2. 根据任务内容决定保存目录

```text
args.task_path + str(len(self.target)) + '/' + self.ins
```

3. 调用 `make_setting()` 和 `make_cfg()`
   - 生成 Habitat-Sim 的配置

4. 初始化 Habitat-Sim

5. 读取 scene managers / rigid object manager / pathfinder

6. 初始化 agent
   - 在可导航点随机采样一个起点

7. 初始化 yaw
   - 默认 `180`

8. 初始化 `GreedyGeodesicFollower`
   - 之后每一步动作都由它给出

9. 先做一次 `sim.step("move_forward")`
   - 用于初始化观测

### 6.2 `SceneSimulator` 提供了哪些关键方法

最关键的是这几个：

- `actor(action, step, success)`
  - 真正执行动作，并保存当前观察图像
- `get_coord(obj_target)`
  - 在当前 region 中找到目标物体中心坐标
- `geodesic_distance(position_b_list)`
  - 计算 agent 到多个候选目标点的 geodesic distance
- `get_info(success)`
  - 返回当前子目标、位置、yaw、geodesic distance
- `get_next_action(goal_pos)`
  - 调用 `GreedyGeodesicFollower` 给出下一步动作
- `return_state()`
  - 返回当前 agent 位置和 yaw

## 7. `eval_for_one_task()` 如何执行一条长任务

### 7.1 初始化阶段

`eval_for_one_task()` 会先做：

1. 初始化 `config['Geo dis']`
2. 初始化 `config['trial']`
3. 设置：
   - `success = 0`
   - `action = 'stop'`
4. 调用一次：

```python
task_sim.actor(action, -1, success)
```

这一步的作用是生成初始观测，并创建 `temp/` 图片目录。

### 7.2 每一步循环里做什么

主循环是：

```python
for step in range(args.max_step):
```

每一步会依次做：

1. 根据当前 `success` 确定当前子目标
2. 调用 `get_coord(obj_target)` 找目标候选坐标
3. 对候选坐标做 `snap_point`
4. 计算到目标的 geodesic distance
5. 调用 `return_state()` 记录当前状态
6. 把当前状态写入：

```text
config['trial']['trial_i']['pos'/'yaw'/'action']
```

### 7.3 成功判定

如果：

```python
geo_dis < args.success_dis
```

则认为当前子目标成功。

这时会：

1. `success += 1`
2. 再记录一次 `stop`
3. 调用 `actor('stop', ...)`
4. 统计当前子任务用了多少 nav step

如果所有子目标都成功：

- 直接 `break`

### 7.4 如果还没成功，如何决定下一步动作

如果当前子目标还没完成：

1. 调用 `get_info(success)` 获取目标位置
2. 调用：

```python
action = task_sim.get_next_action(coord)
```

3. 再调用：

```python
task_sim.actor(action, step, success)
```

也就是说，真正的动作执行策略不是 LLM 实时规划出来的，而是：

- LLM 先给长任务与子目标顺序
- 导航阶段使用 Habitat 自带的 `GreedyGeodesicFollower`

## 8. `actor()` 为什么很重要

`SceneSimulator.actor()` 不只是执行动作，还负责“落图”。

它会：

1. 在 `action != "stop"` 时调用 `sim.step(action)`
2. 更新内部维护的 yaw
3. 调用 `display_env(...)`

而 `display_env()` 会把当前观测保存到：

```text
temp/<step>_<action>_for_<target>/
```

里面包含：

- `front.png`
- `left.png`
- `right.png`
- `depth_front.png`
- `depth_left.png`
- `depth_right.png`

这一步非常关键，因为：

- 第 3 步 `split_task.py` 后面正是靠这些图片来做 RAM 识别和轨迹切分

## 9. 第 2 步结束时会产出什么

### 成功任务

如果整条长任务成功完成：

1. 把 `temp/` 挪到：

```text
success/trial_x/
```

2. 把完整任务记录保存为：

```text
success/trial_x/task.json
```

这个 `task.json` 里会包含：

- 原任务配置
- `Geo dis`
- `trial_0`, `trial_1`, ...
  - 每个 trial 里的 `pos`
  - `yaw`
  - `action`

### 失败任务

如果达到 `max_step` 还没完成：

- 挪到 `fail/trial_x_y-z/`
- 同样会保存 `task.json`

## 10. 第 3 步：`split_task.py` 如何切分长轨迹

### 入口函数

主入口的第三步是：

```python
split_traj(args)
```

### 10.1 `init_model(args)` 先做什么

切分阶段会先初始化 RAM 模型：

1. 根据 `ram_device` 解析设备
2. 通过 `_cuda_runtime_supported()` 判断 GPU 能否真的用
3. 加载：

```python
ram_plus(pretrained=args.ram_model, image_size=384, vit='swin_l')
```

4. 把日志重定向到 `args.ram_logs`

### 10.2 `TrainDataset` 读的是什么

这一阶段的 `TrainDataset` 不是读 `config.json` 内容，而是直接枚举：

```text
task/<len>/<instruction>/
```

返回的是“任务目录路径”。

因为切分阶段真正要读的是：

- `success/trial_1/` 里的动作图片目录

### 10.3 `get_trail(path)` 如何还原动作序列

它会进入：

```text
path/success/trial_1/
```

然后：

1. 读取所有 `step_action_for_target` 目录名
2. 解析出：
   - step index
   - action
   - target
3. 构造：
   - `action_dic`
   - `obs_dic`

其中 `obs_dic` 里保存了每个 step 对应的图片路径，后面 RAM 需要用到这些图片。

### 10.4 `split_traj(args)` 如何按 target 切段

它会遍历动作序列，并按“当前动作所属 target 是否改变”来切分。

一个 target 的连续轨迹，会被认为是一段候选轨迹。

但不是所有段都保留，代码里还做了长度过滤，例如：

- 只保留长度大于一定阈值的段

### 10.5 `segment_trajectory()` 如何把动作序列再切细

这是第 3 步最关键的逻辑。

它做了四层处理：

1. 找连续转向段
   - `turn_left`
   - `turn_right`

2. 合并相邻/重叠的同向转弯段

3. 处理不同方向重叠段
   - 合并成 `zigzag`

4. 把剩下的空隙补成 `move_forward`

于是，一段长轨迹会被切成多个小 segment，每个 segment 都有：

- 起止 step
- 动作标签
  - `move_forward`
  - `turn_left`
  - `turn_right`
  - `zigzag`
  - 或更自然语言化的转弯描述
- 对应观测图片
- 当前 target

### 10.6 `batch_ram(img_list)` 在这里扮演什么角色

每个 segment 会把自己的前视图片送进 RAM：

1. 对每张图做 transform
2. 用 RAM 识别场景标签
3. 统计频次
4. 取 top-5 tags

这些 tags 会作为“这段局部轨迹的场景语义摘要”。

### 10.7 `split_traj(args)` 最终输出什么

所有切好的 segment 序列会被写进：

```text
args.split_save_path
```

默认是：

```text
task/trail_list.txt
```

里面每一行都是一个切分后的轨迹列表。

## 11. 第 4 步：`gen_step_task(args)` 如何把切分结果变成 step task

### 入口函数

主入口第四步调用的是：

```python
gen_step_task(args)
```

### 11.1 它先读什么

先读取：

```text
task/trail_list.txt
```

把每一行重新解析回 Python 列表。

### 11.2 `make_task(args, trail_list)` 做了什么

对每条切分后的局部轨迹，会：

1. 收集每个 segment 的：
   - `action`
   - `scene tags`

2. 读取原始：

```text
.../success/trial_1/task.json
```

3. 还原这个局部轨迹在大轨迹里的元信息：
   - `trajectory path`
   - `start`
   - `end`
   - `Robot`
   - `Scene`
   - `target`
   - `Region`
   - `start_pos`
   - `start_yaw`

4. 再调用一次 LLM：

```python
gpt4o_mini(args, args.prompt_path + "gen_task.txt", json.dumps(tags))
```

让模型根据：

- 这段局部轨迹的动作模式
- 这段局部轨迹的 RAM scene tags

生成一句新的 step-by-step task instruction。

### 11.3 第 4 步最终输出什么

每条 step task 最终被保存到：

```text
step_task/<Task instruction>.json
```

这份 JSON 可以理解为：

- 从长任务中切出来的局部训练样本

## 12. 用一句话概括整个主链

NavGen 的主链其实可以概括成：

```text
场景语义统计 + 提示词
  -> LLM 生成长任务
  -> Habitat 执行长任务并记录图片/轨迹
  -> 用动作模式 + RAM 标签切分长轨迹
  -> 再用 LLM 把局部切片改写成 step task
```

也就是：

- 上游 LLM 负责“任务层级”
- Habitat 负责“导航执行层级”
- RAM 负责“局部视觉语义摘要”
- 下游 LLM 再把局部片段转成更细粒度的 step task

## 13. 数据文件是怎么在各阶段流动的

可以把主流程里的文件流总结成下面这样：

```text
scene/*.txt + region CSV + prompt/*
  -> task/<len>/<instruction>/config.json
  -> task/<len>/<instruction>/success/trial_x/task.json + 图片目录
  -> task/trail_list.txt
  -> step_task/*.json
```

如果你只是想抓住“看哪里”，最重要的几个产物就是：

1. `task/<len>/<instruction>/config.json`
   - 长任务定义
2. `success/trial_x/task.json`
   - 真实轨迹记录
3. `task/trail_list.txt`
   - 切分后的轨迹片段
4. `step_task/*.json`
   - 最终 step task 数据

## 14. 本轮新增的辅助脚本，不属于原始主链

下面这些文件是为了更容易验证和复用流程额外加的，不是原始 `main.py` 主链的一部分：

### `run_smoke_test.py`

用途：

- 隔离输出目录
- 快速做小规模尝试
- 收集 summary，方便判断“到底能不能跑通”

### `run_split_steps.py`

用途：

- 只跑：
  - `split_traj(args)`
  - `gen_step_task(args)`

适合在长任务轨迹已经准备好的情况下，单独重跑切分阶段。

### `run_navgen_split_gpu.sh`

用途：

- 专门在 `lhvln-cu128` 环境中运行 split 阶段
- 统一注入 GPU 环境变量

### `export_trajectory_rgb_video.py`

用途：

- 把 `success/trial_x/task.json` 回放成第一人称视频
- 支持：
  - 自定义相机高度/FOV/分辨率
  - 朝向修正
  - 一秒一帧动作对齐
  - 文本标注
  - 2D 目标框

这个脚本是本轮为了“人工核对 NavGen 输出是否合理”新增的可视化工具，不属于论文 NavGen 原始四步流水线。

## 15. 如果你继续读代码，建议按这个顺序

如果你现在要继续往下读源码，我建议顺序是：

1. `nav_gen/main.py`
2. `nav_gen/task_gen.py`
3. `nav_gen/gpt.py`
4. `nav_gen/dataset.py`
5. `nav_gen/dataset_gen.py`
6. `nav_gen/habitat_base/simulation.py`
7. `nav_gen/habitat_base/visualization.py`
8. `nav_gen/split_task.py`
9. `nav_gen/run_smoke_test.py`
10. `nav_gen/export_trajectory_rgb_video.py`

这样读下来，逻辑上最顺。

