# NavGen 论文 3.1 与代码实现对照说明

更新时间：2026-04-27

本文档的目标不是重复介绍 `nav_gen/` 目录，而是把论文
`Song 等 - 2025 - Towards Long-Horizon Vision-Language Navigation Platform, Benchmark and Method.pdf`
中的 **3.1 NavGen** 和附录中补充的 NavGen 细节，逐条对照到当前仓库里的实际代码实现。

如果你觉得 [NAVGEN_CODE_FLOW_EXPLANATION.md](./NAVGEN_CODE_FLOW_EXPLANATION.md) 按执行顺序讲得还不够“贴论文”，那这份文档就是它的补充版。

---

## 1. 先给结论

论文 3.1 把 NavGen 描述成一个 **bi-directional generation mechanism**：

1. **Forward Data Generation**
   - 先生成长任务 instruction
   - 再让 agent / expert 在 simulator 中跑出长轨迹

2. **Backward Data Generation**
   - 再把长轨迹拆成更细的动作片段
   - 用视觉标签 + 动作片段反向生成 step-by-step 导航指令

当前代码中，这个双向流程正好对应：

```text
main.py
  -> gen_task(args)       # forward: 先生成长任务
  -> gen_traj(args)       # forward: 再生成长轨迹
  -> split_traj(args)     # backward: 轨迹切分
  -> gen_step_task(args)  # backward: 反向生成 step task
```

也就是说：

- 论文里的 **forward generation** = `task_gen.py` + `dataset_gen.py`
- 论文里的 **backward generation** = `split_task.py`

---

## 2. 论文 3.1 的概念，分别落到哪些代码里

| 论文中的概念 | 代码里的对应位置 | 当前实现的含义 |
| --- | --- | --- |
| NavGen 平台总入口 | `nav_gen/main.py` | 按四阶段顺序执行整个数据生成链 |
| scene assets `S` | `scene/Per_Scene_Region_Weighted_Votes.csv` + `scene/<scene_id>.txt` | 提供 region 和 object 语义资源池 |
| robot configurations `R` | `prompt/spot.txt`、`prompt/stretch.txt` + `habitat_base/config.py` | 提供给 LLM 的机器人文字描述，以及 simulator 里的实际传感器参数 |
| prompt1 | `prompt/rule.txt` + `prompt/example.txt` + `prompt/system.txt` | forward generation 的任务生成提示词 |
| `D_ins = G(S, R, prompt1)` | `task_gen.py::gen_task()` + `gpt.py::gpt4o_mini()` | 生成长任务 instruction 和 `Subtask list` |
| `Sim(D_ins, S, A, OR(M, E))` | `dataset_gen.py::eval_for_one_task()` + `habitat_base/simulation.py` | 把长任务送进 Habitat-Sim，用 expert 轨迹跟随器跑出轨迹 |
| `D_traj` | `task/.../success/trial_1/task.json` + `task/.../success/trial_1/<step folders>/` | 长任务轨迹的状态序列和多视角观测 |
| trajectory splitting algorithm | `split_task.py::get_trail()` + `split_task.py::segment_trajectory()` | 把长轨迹切成 step-level 片段 |
| RAM image annotation model | `split_task.py::init_model()` + `split_task.py::batch_ram()` | 为每个片段补视觉标签 |
| prompt2 | `prompt/gen_task.txt` | backward generation 的 step 指令生成提示词 |
| step-by-step tasks | `split_task.py::make_task()` | 生成最终 `step_task/*.json` |

---

## 3. 对照论文 3.1.1：Forward Data Generation

论文 3.1.1 的核心意思可以压缩成一句话：

> 用场景资源和机器人配置做 prompt，先让大模型生成长任务，再让 simulator 用 expert 或 model 跑出轨迹。

当前代码里，这一步是分成两个文件完成的：

1. `task_gen.py`
2. `dataset_gen.py`

### 3.1 资源池 `S` 和 `R` 在代码里怎么准备

论文里说：

- scene assets 来自 HM3D
- robot configurations 来自 Spot / Stretch 等不同机器人设定

代码里对应是：

#### `S`: scene assets

- `scene/Per_Scene_Region_Weighted_Votes.csv`
  - 用于建立 `scene_id -> regions` 的映射
- `scene/<scene_id>.txt`
  - 用于读取某个具体场景里每个 region 中有哪些 object

对应函数：

- `task_gen.py::read_scene()`
- `task_gen.py::gen_task()`

在 `gen_task()` 中，代码会：

1. 随机或指定一个 `scene_id`
2. 读取 `scene/<scene_id>.txt`
3. 组装成 `input_scene`

最终给 LLM 的 scene 输入形态大致是：

```python
{
  "Region 11: Kitchen": ["box", "dish", ...],
  "Region 12: Living room": ["couch", ...],
}
```

#### `R`: robot configurations

论文里把 robot configuration 当成 prompt 条件的一部分。

代码里它有两层实现：

1. **文本层**
   - `prompt/spot.txt`
   - `prompt/stretch.txt`
   - 给 LLM 描述机器人能力、视角高度、动作类型

2. **仿真层**
   - `habitat_base/config.py::make_setting()`
   - `habitat_base/config.py::make_cfg()`
   - 真正定义传感器高度、FOV、动作步长、左右视角等

例如：

- `spot` 默认传感器高度 `0.5m`
- `stretch` 默认传感器高度 `1.0m`
- 左右相机朝向是 `+60°` / `-60°`
- `move_forward = 0.25m`
- `turn_left/turn_right = 30°`

这和论文正文、附录里提到的三视角观测和基础动作是对得上的。

### 3.2 论文里的 `D_ins = G(S, R, prompt1)` 对应哪段代码

这一步主要落在：

- `task_gen.py::gen_task()`
- `gpt.py::gpt4o_mini()`

具体流程是：

1. `gen_task()` 读入
   - `prompt/rule.txt`
   - `prompt/example.txt`
   - `prompt/<robot>.txt`
   - `prompt/system.txt`

2. 把 `scene` 和 `robot` 拼进 prompt

3. 调用：

```python
task = gpt4o_mini(args, args.prompt_path + "system.txt", prompt)
```

虽然函数名还叫 `gpt4o_mini`，但当前实现实际上已经改成了：

- 通过 `OpenAI` Python SDK
- 走 DashScope-compatible endpoint
- 默认模型是 `qwen3.6-plus`

也就是说：

- **论文写的是 GPT-4**
- **当前代码默认跑的是 Qwen 3.6 Plus**

这是论文方法和当前仓库实现之间最重要的一个现实差异。

### 3.3 论文里的“长任务 instruction + sub-task”在代码里长什么样

`gen_task()` 期待模型输出一个字典，至少包含：

- `Task instruction`
- `Subtask list`

例如：

```python
{
  "Task instruction": "Please head to the kitchen and stop at the box. Then go to the couch in the living room and stop there. After that, go back to the kitchen and stop at the dish",
  "Subtask list": [
    "Move_to('box_11')",
    "Stop_at('box')",
    "Move_to('couch_12')",
    "Stop_at('couch')",
    "Move_to('dish_11')",
    "Stop_at('dish')"
  ]
}
```

然后代码会补上：

- `Robot`
- `Scene`
- `Object`

最后保存到：

```text
task/<导航目标数>/<Task instruction>/config.json
```

### 3.4 代码如何做合法性检查

论文正文没有详细写 task validation，但代码里这一层很关键。

`task_gen.py::gen_task()` 会检查：

1. `Move_to('obj_region')` 里的 region id 是否真的存在
2. object 是否真的在对应 region 中
3. 指令文本里是否直接出现 `region/Region`

如果不合法：

- 直接 `return False`

这也是为什么 batch 跑的时候经常会出现：

- `no_success_task`
- 或者多次重试后才拿到一条成功 task

### 3.5 论文中的 `Sim(D_ins, S, A, OR(M, E))` 在代码里怎么落地

这一段主要落在：

- `dataset.py::TaskDataset`
- `dataset_gen.py::gen_traj()`
- `dataset_gen.py::eval_for_one_task()`
- `habitat_base/simulation.py::SceneSimulator`

#### 第一步：把 `config.json` 重新读回来

`TaskDataset` 会把 `task/` 目录里的 `config.json` 重新读成数据样本。

同时，它会把：

- `Subtask list`

进一步拆成：

- `Object`
- `Region Name`
- `Region`

这一步相当于把 forward generation 的 instruction 结果，变成 trajectory generation 的结构化输入。

#### 第二步：初始化 simulator

`eval_for_one_task()` 会构造：

```python
task_sim = SceneSimulator(args=args, config=config)
```

`SceneSimulator` 里做了几件事：

1. 用 `make_setting()` / `make_cfg()` 建 Habitat-Sim 配置
2. 初始化 agent
3. 随机采样一个 navigable 起点
4. 初始化 `GreedyGeodesicFollower`

这里的 `GreedyGeodesicFollower` 就是当前代码里的 **expert**

也就是说，论文公式 (1) 中的：

- `OR(M, E)`

在当前仓库里实际上只走了：

- `E = Habitat built-in greedy pathfinder expert`

并没有实现“在同一条数据生成链中切换到 learned navigation model”的分支。

### 3.6 论文里的 `D_traj` 在代码里是什么

长轨迹数据由两部分组成：

#### 1. 状态序列

保存在：

```text
task/.../success/trial_1/task.json
```

里面会记录：

- `trial_0`, `trial_1`, ...
- 每个 trial 的 `pos`
- `yaw`
- `action`
- `Geo dis`

#### 2. 多视角观测图像

保存在：

```text
task/.../success/trial_1/<step>_<action>_for_<target>/
```

每一步都会落 6 张图：

- `left.png`
- `front.png`
- `right.png`
- `depth_left.png`
- `depth_front.png`
- `depth_right.png`

对应保存逻辑在：

- `habitat_base/visualization.py::display_env()`

### 3.7 forward generation 和论文的一致点 / 差异点

#### 一致点

- 都是先生成长任务，再生成长轨迹
- 都以 HM3D scene assets 为核心资源池
- 都有 Spot / Stretch 两种机器人配置
- 都依赖 expert 轨迹来构造数据

#### 当前代码与论文的差异

1. 论文写 GPT-4，当前代码默认是 `qwen3.6-plus`
2. 论文写 `OR(M, E)`，当前代码只实现了 `E`
3. 论文里成功定义包含距离和视野条件；当前代码现在也默认要求“`geo_dis < success_dis` 且目标在前视角 semantic observation 中可见”才会 `stop`
4. 论文正文提到 Habitat 和 Isaac Sim，当前仓库这条主链只实现了 Habitat/HM3D 分支

---

## 4. 对照论文 3.1.2：Backward Data Generation

论文 3.1.2 的核心意思可以压缩成一句话：

> 先把长轨迹切成若干动作片段，再给每个片段打上视觉标签，最后反向生成更细粒度的 step-by-step instruction。

这一步几乎都落在：

- `split_task.py`

### 4.1 从长轨迹里先抽出单阶段导航段

对应函数：

- `split_task.py::get_trail()`
- `split_task.py::split_traj()`

`get_trail()` 会去读：

```text
task/.../success/trial_1/
```

然后恢复出两类信息：

1. `action_dic`
   - 每一步做了什么动作
   - 当前 step 对应哪个 target

2. `obs_dic`
   - 每一步对应的观测图片路径

`split_traj()` 再按 target 变化把整条长轨迹切成多个单目标段。

这一步对应论文里说的：

- “Within a single-stage navigation goal trajectory...”

也就是：

- 先从 multi-stage 任务切出 single-stage navigation trajectory

### 4.2 附录 Algorithm 1 和 `segment_trajectory()` 的逐项对照

论文附录 8.1 给了 Trajectory Splitting Algorithm。
当前代码中最直接的实现就是：

- `split_task.py::segment_trajectory()`

它的逻辑可以按“论文算法 -> 代码实现”这样理解。

#### A. 识别连续转向段

论文附录说：

- 用动态滑动窗口找连续动作段

代码里是：

1. 先扫描 `turn_left`
2. 再扫描 `turn_right`
3. 每次看长度为 `3` 的 window
4. 如果 window 中某个转向动作出现至少 `2` 次，就认为形成一个转向段

这就是：

- `segment_trajectory()` 里两段 `while i <= n - 3`

#### B. 合并相邻同类段

论文附录里有 merge 逻辑。

代码里是：

- 如果两个同标签段满足 `start <= current_end + 3`
- 就合并成一个更长段

这一层对应：

- `merged_segments`

#### C. 处理相互重叠的左右转

论文里有 “bypass forward” 这一类更复杂片段描述。

当前代码没有直接输出 `bypass_forward` 这个标签，而是采用：

- 如果左右转段重叠或紧挨，就标成 `zigzag`

也就是说：

- **论文的高层概念**：绕行 / 复杂局部动作
- **代码的具体标签**：`zigzag`

#### D. 用 `move_forward` 填补空白区间

论文算法会把连续转向段之间的其余部分视为前进类片段。

代码里是：

- 如果两段之间有空隙
- 就插入一个 `move_forward` segment

#### E. 用 RAM 为每段打视觉标签

论文 3.1.2 里说：

- 对每个 segment 用 RAM image annotation model 产生 high-confidence visual annotations

代码里是：

- `split_task.py::batch_ram()`

具体做法：

1. 对该 segment 的图像逐张做 RAM 识别
2. 统计 tag 频次
3. 去掉 `ceiling`、`floor` 一类低价值 tag
4. 取前 5 个 tag 作为 `scene tags`

这一步是 backward generation 的关键桥梁，因为后面 GPT 生成 step instruction 就靠这些 `scene tags`。

### 4.3 论文说多视角观测，代码里为什么 RAM 只看 front 图

这是一个很值得注意的实现细节。

论文和附录都强调：

- agent 在导航过程中会得到三视角观测
  - `+60°`
  - `0°`
  - `-60°`

代码里这个三视角确实存在：

- `color_sensor_l`
- `color_sensor_f`
- `color_sensor_r`

保存图像时也都落盘了：

- `left/front/right`

但是在 backward generation 里，`segment_trajectory()` 构造每个片段时只取：

```python
obs_dic[i]['front']
```

也就是说：

- **轨迹记录阶段**：保存三视角 RGB + depth
- **RAM 打标签阶段**：当前代码只使用 front 视角

所以如果你从论文回看代码，这里要记住：

- 三视角是完整记录了
- 但 step task 反向生成时没有把左右图一起送给 RAM

### 4.3.1 再结合附录看“NavGen 的视线”，代码里到底怎么实现

如果把论文正文和附录里关于 observation / viewpoint 的描述一起看，最容易对应到的其实是：

- `habitat_base/config.py::make_setting()`
- `habitat_base/config.py::make_cfg()`
- `habitat_base/visualization.py::display_env()`

这三处代码把“agent 看到了什么”定义得很具体。

#### A. 三个一人称视角不是抽象概念，而是 3 个并列相机

附录里提到 agent 会看到多个朝向的图像；代码中这不是后处理拼接，而是直接在 simulator 里注册了 3 个 RGB sensor：

- `color_sensor_l`
- `color_sensor_f`
- `color_sensor_r`

它们的朝向分别是：

- 左视角：`+60°`
- 前视角：`0°`
- 右视角：`-60°`

对应实现就在 `make_cfg()` 里：

- 左：`orientation = [0.0, math.pi / 3.0, 0.0]`
- 前：`orientation = [0.0, 0.0, 0.0]`
- 右：`orientation = [0.0, -math.pi / 3.0, 0.0]`

所以从论文附录回看代码时，可以直接把它理解成：

- 论文里的三视角观测
- 在代码里就是同一位置、同一高度、三个不同 yaw 的 pinhole camera

#### B. “视线高度”在代码里是机器人参数，不是写死常数

附录里会把 observation 和机器人形态放在一起讨论；代码里这一点也保留了。

`make_setting()` 里默认是：

- `spot`: `sensor_height = 0.5`
- `stretch`: `sensor_height = 1.0`

也就是说，NavGen 生成同一个场景数据时，机器人不同，实际第一人称视线高度也不同。

这和论文把 `R` 作为 generation condition 的说法是一致的：

- 机器人配置不仅影响 prompt
- 也影响 simulator 里的真实观测几何

#### C. FOV 在当前仓库里是可调参数，默认实现比论文表述更“工程化”

附录更偏方法描述，不会把每个实现参数都展开；但代码里已经把视场角做成了显式配置：

- `sensor_hfov`
- `render_width`
- `render_height`

默认情况下：

- `sensor_hfov = 90.0`
- `width = 512`
- `height = 512`

这意味着当前仓库的默认第一人称观测，是一个：

- 水平 FOV 90°
- 方形分辨率

的相机设置。

如果你后面继续做我们前面那种视频导出或 dataset 定制，这一层其实就是最底层的“视线定义入口”。

#### D. 附录里讲的是 observation stream，代码里实际还额外保存了 depth

除了 `left / front / right` 三张 RGB 图，`display_env()` 还会把下面三张深度图一起存下来：

- `depth_left.png`
- `depth_front.png`
- `depth_right.png`

因此当前 `task/.../trial_1/...` 目录中的单步观测，实际是：

- 3 张 RGB
- 3 张 depth

这比论文正文里直接写“visual observations”更具体。

#### E. 但 backward 阶段没有完整利用这套“多视线”

这也是最值得单独指出的一个实现差异：

- forward rollout 阶段确实采集了三视角 RGB 和三视角 depth
- backward step-task generation 阶段却只取 `front` 图送进 RAM

具体就在 `segment_trajectory()`：

```python
"obs": [obs_dic[i]['front'] for i in ...]
```

所以如果你问：

- “附录里强调的多视角，在最终 step task 生成里是否都被用了？”

当前仓库答案是：

- **没有**
- **多视角被记录了，但反向生成只消费了前视图**

#### F. 从论文到代码，关于“视线”的一句话总结

可以把它记成下面这句话：

> 论文和附录把 NavGen 的 observation 说成三视角一人称观测；而代码中这一点被具体实现成 `+60° / 0° / -60°` 三个 RGB 相机、对应三路 depth、随机器人变化的传感器高度，以及一个当前只在 backward 阶段使用 front-view 的简化消费逻辑。

### 4.4 论文中的 prompt2 对应什么

论文附录 9.1 说 NavGen 有：

- forward task generation prompt
- backward task generation prompt

当前 backward prompt 对应：

- `prompt/gen_task.txt`

这个文件虽然名字叫 `gen_task.txt`，但实际用途是：

- 根据 `action + scene tags + target`
- 生成自然语言 step-by-step navigation instruction

也就是：

- 论文里的 prompt2

### 4.5 论文里的“反向生成 step-by-step task”在代码里怎么落盘

对应函数：

- `split_task.py::make_task()`
- `split_task.py::gen_step_task()`

代码会为每条切分后的片段轨迹构造一个 step task，里面包括：

- 原始长轨迹路径
- `start` / `end`
- `Robot`
- `Scene`
- `target`
- `Region`
- `start_pos`
- `start_yaw`
- `Task instruction`

然后保存到：

```text
step_task/<instruction>.json
```

这就是最终最像“训练数据样本”的一层。

---

## 5. 附录补充后，再看代码会更容易理解的 4 个点

### 5.1 NavGen 的真正核心不是一个脚本，而是“前向生成 + 反向生成”这两个闭环

如果只看 `main.py`，容易觉得它只是四个函数顺序执行。

但从论文 3.1 的视角看，它更像两个闭环：

#### 闭环 A：forward

```text
scene/robot assets
-> prompt1
-> 长任务 instruction
-> Habitat expert rollout
-> 长轨迹 D_traj
```

#### 闭环 B：backward

```text
长轨迹 D_traj
-> trajectory splitting
-> RAM tags
-> prompt2
-> step-by-step tasks
```

这个“双向”结构，正是 NavGen 和普通“只生成 instruction”脚本最大的区别。

### 5.2 论文把 simulator 写得很抽象，但代码里其实非常具体

论文中：

- `Sim(D_ins, S, A, OR(M, E))`

代码中其实就是：

1. `SceneSimulator`
2. `GreedyGeodesicFollower`
3. `get_next_action()`
4. `actor()`

所以如果你想问：

- “这里的 expert 到底是谁？”

答案不是一个单独模型文件，而是：

- Habitat-Sim 的 `GreedyGeodesicFollower`

### 5.3 论文附录算法比正文更接近 `split_task.py`

正文 3.1.2 只说：

- split trajectory
- RAM annotate
- GPT generate step task

真正和代码一一对得上的，其实是附录 8.1 的算法描述。

尤其是：

- 滑动窗口
- merge 同类段
- 用 RAM 给 segment 打标签

这三步在 `segment_trajectory()` 里几乎都能直接找到实现影子。

### 5.4 附录的数据生成说明也解释了为什么当前代码看起来“偏 expert”

附录 9.2 说得很清楚：

- 在 Habitat 中，NavGen 用的是 built-in greedy pathfinder

这和当前代码完全一致。

所以当前仓库的 `nav_gen/` 更适合理解成：

- **NavGen 数据生成实现**

而不是：

- **论文里所有 NavGen + MGDM 训练和推理能力的完整统一实现**

---

## 6. 用一个真实样例把整条链串起来

下面用一个已经存在于仓库中的样例，把论文 3.1 对应到具体文件。

样例路径：

```text
nav_gen/smoke_runs/5traj_20260423_160914/task/3/Please head to the kitchen and pick up the box. Bring it over to the couch in the living room and set it down. After that, go back to the kitchen and pick up the dish/
```

### 第一步：forward task generation 结果

文件：

```text
config.json
```

这里能看到：

- `Task instruction`
- `Subtask list`

这就是论文里的：

- `D_ins`

### 第二步：forward trajectory generation 结果

文件：

```text
success/trial_1/task.json
```

这里能看到：

- `trial_0`
- `trial_1`
- `trial_2`
- 每段的 `pos / yaw / action`

这就是论文里的：

- `D_traj`

### 第三步：backward splitting 的中间结果

文件通常会落到：

```text
task/trail_list.txt
```

这里每一行就是一个切分后的片段序列。

这一步对应：

- 论文附录算法的输出 `Seg`

### 第四步：backward-generated step tasks

文件会落到：

```text
step_task/*.json
```

这里面的 `Task instruction` 就不再是长任务，而是：

- 更短的局部导航 instruction

这就是论文里说的：

- step-by-step guidance task

---

## 7. 如果你之后要继续读代码，推荐按这个顺序

### 第一轮：先建立论文到代码的整体映射

1. `main.py`
2. `task_gen.py::gen_task()`
3. `dataset_gen.py::eval_for_one_task()`
4. `split_task.py::split_traj()`
5. `split_task.py::make_task()`

### 第二轮：再看底层细节

1. `gpt.py`
2. `dataset.py`
3. `habitat_base/config.py`
4. `habitat_base/simulation.py`
5. `habitat_base/visualization.py`

### 第三轮：带着论文附录回看

重点回看：

1. 附录 8.1 的 Trajectory Splitting Algorithm
2. 附录 9.1 的 prompt used
3. 附录 9.2 的 data generation notes

这样你会更容易看懂：

- 为什么 `split_task.py` 要先切再生成
- 为什么 `SceneSimulator` 看起来更像 expert rollout 工具
- 为什么三视角图都保存了，但 RAM 阶段只用了 front 图

---

## 8. 最后一句总结

如果只用一句话概括：

> 论文 3.1 里的 NavGen，本质上是“先用 LLM + scene/robot assets 正向生成长任务和长轨迹，再用 trajectory splitting + RAM + LLM 反向生成 step-by-step 任务”的双向数据生成平台；而当前仓库里的实现，正好把这个想法拆成了 `task_gen.py`、`dataset_gen.py`、`split_task.py` 三个核心模块，并以 Habitat 的 greedy pathfinder 作为 forward rollout 的 expert。
