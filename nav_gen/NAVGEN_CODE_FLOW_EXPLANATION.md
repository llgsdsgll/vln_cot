# NavGen 代码流程说明

更新时间：2026-05-11

本文档按 `nav_gen` 的实际执行顺序来讲解代码，而不是按文件列表逐个介绍。
如果你想回答"从运行 `python main.py` 开始，NavGen 到底做了什么"，这份文档就是为这个问题写的。

说明：

- 这里讲的是 `nav_gen` 主流水线。
- 文档最后会补充本轮新增的辅助脚本，例如 smoke run、单任务重跑、目标视点可视化、视频导出脚本，但它们不是论文 NavGen 的主生成链。

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

##### C1. 命令行参数

- `scene`
- `scene_dataset`
- `sim_gpu_device`
- `render_sensor_height`
- `max_step`
- `success_dis`
- `success_visible_pixels`
- `allow_stop_without_visibility`
- `allow_occluded_goal_fallback`
  - 兼容保留参数；当前严格模式下即使传入也不会再回退到物体中心 goal
- `success_view_radius_min`
- `success_view_radius_max`
- `success_view_radius_step`
- `success_view_angle_step`

##### C2. 整体架构

轨迹生成不是 LLM 实时规划出来的，而是采用了 **"LLM 定义目标序列 + Habitat 自动导航执行"** 的分层策略：

```text
LLM → 定义子目标序列 (Move_to / Stop_at)
Habitat GreedyGeodesicFollower → 逐步导航执行
```

整个轨迹执行的入口是 `dataset_gen.py` 中的 `eval_for_one_task(args, config)`，对每个任务构造一个 `SceneSimulator` 对象后在 Habitat 中循环执行导航动作。

##### C3. 起点采样策略

`SceneSimulator.__init__()`（`simulation.py:49-51`）：

1. 在 NavMesh 上调用 `pathfinder.get_random_navigable_point()` 随机采样一个可导航点
2. 位置做了偏移修正：`sample_navigable_point - [0, 0, -0.25]`
3. 初始 yaw 固定为 `180°`
4. 初始化时先执行一次 `move_forward`，获取初始观测

这意味着**每次任务的起点是随机的**，同一任务在不同 trial 可能从不同位置出发。

##### C4. 动作空间定义

`config.py:132-143` 定义了三个离散动作：

| 动作 | 位移/旋转量 |
|------|------------|
| `move_forward` | 0.25m |
| `turn_left` | 30° |
| `turn_right` | 30° |

所有导航都基于这个固定步长的动作空间，yaw 通过手动维护（`self.yaw`），范围 `[-180, 180]`。

##### C5. 目标选择策略（核心）

这是轨迹生成中最关键的策略，涉及三层递进：

###### C5.1 候选物体枚举（`get_target_candidates`，`simulation.py:217-248`）

- 遍历场景语义信息，在指定 **region** 内找到所有同名物体实例
- 收集每个实例的：`center`（AABB 中心）、`semantic_id`、`object_id`
- 结果通过 `target_candidate_cache` 缓存，同一目标只计算一次

###### C5.2 可见视点采样（`get_visible_viewpoints` → `_sample_visible_viewpoints_for_candidate`，`simulation.py:318-381`）

围绕每个候选物体，在多圈多角度采样视点：

```text
半径: 0.75m → 2.5m（步长 0.5m）
角度: 0° → 360°（步长 30°）
```

具体流程：

1. 在 `(goal_center + radius·sin(angle), goal_center_y, goal_center + radius·cos(angle))` 生成候选点
2. 通过 `pathfinder.snap_point()` 将点映射到 NavMesh 上的可导航点
3. 对每个可导航点，计算**朝向物体中心的 yaw**（`_goal_center_to_saved_yaw`）
4. **关键步骤**：将 agent "瞬移"到该位置 + 朝向，通过前视 semantic sensor 计算 `semantic_id` 的像素数（`_semantic_pixels_for_pose`）
5. 只保留像素数 ≥ `success_visible_pixels`（默认 25）的视点
6. 视点排序：先按像素数降序，再按距离升序

这一步会做 **pose capture/restore**（`simulation.py:300-310`），模拟完后恢复 agent 原状态，不影响真实执行。

###### C5.3 最优目标选择（`get_goal_info`，`simulation.py:399-444`）

- 从所有可见视点中，对每个视点计算从当前 agent 位置的 **geodesic distance**
- 选择 geodesic distance **最小**的视点作为导航目标
- 如果完全找不到可见视点 → 返回 `(inf, None)`，任务判为 **unreachable**，直接终止

**策略意义**：不是导航到物体几何中心，而是导航到"能看见物体"的最近位置。这确保了 agent 到达目标时确实能看到目标，而不只是距离够近。

##### C6. 动作决策策略

导航过程中有三种动作决策模式，按优先级排列：

###### C6.1 远距离导航：`GreedyGeodesicFollower`

当 `geo_dis >= success_dis` 时，调用 `follower.next_action_along(goal_pos)`。这是 Habitat 自带的贪心最短路径跟随器，使用 NavMesh 进行路径规划。

> 源码路径：
> - Python 层：`habitat_sim/nav/greedy_geodesic_follower.py`（位于 `habitat_sim-0.3.1` egg 包内）
> - C++ 核心实现：`esp/nav/GreedyFollower.cpp`（位于 `/home/gs/my_test/habitat/habitat-sim/src/esp/nav/`）

###### C6.1.1 项目中的初始化参数

`simulation.py:56-64` 构造 follower 时传入的参数：

| 参数 | 值 | 说明 |
|------|-----|------|
| `goal_radius` | 1.0m | agent 距目标 <1m 时 follower 返回 STOP |
| `forward_key` | `"move_forward"` | 前进动作名 |
| `left_key` | `"turn_left"` | 左转动作名 |
| `right_key` | `"turn_right"` | 右转动作名 |
| `fix_thrashing` | True（默认） | 启用振荡修复 |
| `thrashing_threshold` | 16（默认） | 连续 16 步交替转向判为振荡 |

结合 `config.py:132-143` 的动作空间定义：
- `move_forward` 步长 = 0.25m → `forwardAmount_` = 0.25m
- `turn_left/right` 旋转 = 30° → `turnAmount_` = 30°（≈0.5236 rad）

###### C6.1.2 Python 层入口（`next_action_along`）

`greedy_geodesic_follower.py:148-167`：

```python
def next_action_along(self, goal_pos: np.ndarray) -> Any:
    # 1. 如果目标位置变了，先 reset 内部状态
    if self.last_goal is None or not np.allclose(goal_pos, self.last_goal):
        self.reset()
        self.last_goal = goal_pos

    # 2. 获取当前 agent 状态
    state = self.agent.state

    # 3. 调用 C++ impl 的核心方法
    next_act = self.impl.next_action_along(
        quat_to_magnum(state.rotation), state.position, goal_pos
    )

    # 4. 把 C++ CODES 枚举映射回 Python action 字符串
    if next_act == GreedyFollowerCodes.ERROR:
        raise errors.GreedyFollowerError()
    return self.action_mapping[next_act]
```

`action_mapping` 将 C++ 的枚举码映射为实际动作字符串：

- `STOP` → `"stop"`
- `FORWARD` → `"move_forward"`
- `LEFT` → `"turn_left"`
- `RIGHT` → `"turn_right"`

###### C6.1.3 C++ 核心实现（`nextActionAlong`）

`GreedyFollower.cpp:158-186`，核心逻辑分五步：

1. 通过 pathfinder 计算当前到目标的 geodesic shortest path
2. 如果有振荡修复（thrashing fix）待执行的动作序列，优先消耗它
3. 否则调用 `nextBestPrimAlong` 计算当前最佳 motion primitive
4. 如果检测到振荡，把最佳 primitive 反转存入 `thrashingActions_`，从反转序列中取下一个动作
5. 正常情况，返回 primitive 的第一个动作

```cpp
CODES nextActionAlong(const RigidState& start, const Vector3& end) {
    // Step 1: 计算 geodesic path
    ShortestPath path;
    path.requestedStart = start.translation;
    path.requestedEnd = end;
    pathfinder_->findPath(path);

    CODES nextAction;
    if (fixThrashing_ && thrashingActions_.size() > 0) {
        // Step 2: 优先消耗振荡修复序列
        nextAction = thrashingActions_.back();
        thrashingActions_.pop_back();
    } else {
        // Step 3: 计算最佳 primitive
        const auto nextActions = nextBestPrimAlong(start, path);
        if (nextActions.empty()) {
            nextAction = CODES::ERROR;
        } else if (fixThrashing_ && isThrashing()) {
            // Step 4: 检测到振荡 → 反转 primitive 作为修复
            thrashingActions_ = {nextActions.rbegin(), nextActions.rend()};
            nextAction = thrashingActions_.back();
            thrashingActions_.pop_back();
        } else {
            // Step 5: 返回 primitive 的第一个动作
            nextAction = nextActions[0];
        }
    }
    actions_.push_back(nextAction);
    return actions_.back();
}
```

###### C6.1.4 核心算法：基于 Motion Primitive 的贪心规划（`nextBestPrimAlong`）

`GreedyFollower.cpp:81-138`，这是 follower 最核心的方法。

**Motion Primitive 的定义**：所有 primitive 的形式是 `[turn_left]*n + move_forward` 或 `[turn_right]*n + move_forward`，其中 `0 <= n < π/turnAmount`。

在本项目配置下（`turnAmount` = 30°），最大转向次数 = π/0.5236 ≈ 6，primitive 集合为：

```text
forward              (0° 转向 + 前进)
left + forward       (30° 左转 + 前进)
left*2 + forward     (60° 左转 + 前进)
left*3 + forward     (90° 左转 + 前进)
left*4 + forward     (120° 左转 + 前进)
left*5 + forward     (150° 左转 + 前进)
right + forward      (30° 右转 + 前进)
right*2 + forward    (60° 右转 + 前进)
right*3 + forward    (90° 右转 + 前进)
right*4 + forward    (120° 右转 + 前进)
right*5 + forward    (150° 右转 + 前进)
```

共 **12 个 primitive**。

**评估流程**：算法维护两个 dummy node（`leftDummyNode_` 和 `rightDummyNode_`），从当前 agent 状态出发，逐步模拟左转/右转，在每个转向角度下都尝试前进一步并计算奖励：

```cpp
for (float angle = 0; angle < π; angle += turnAmount) {
    // 对 leftDummyNode_ 当前朝向，尝试 + forward，计算 reward
    float leftReward = computeReward(leftDummyNode_, path, leftPrim.size());
    // 对 rightDummyNode_ 当前朝向，尝试 + forward，计算 reward
    float rightReward = computeReward(rightDummyNode_, path, rightPrim.size());

    // 更新最佳 primitive
    if (leftReward > bestReward) {
        bestPrim = leftPrim; bestPrim.push_back(FORWARD);
    }
    if (rightReward > bestReward) {
        bestPrim = rightPrim; bestPrim.push_back(FORWARD);
    }

    // 早停：如果 reward 已经足够好 (>0.99)，不再继续搜索更大转向角度
    if (bestReward > 0.99) break;

    // 为下一轮准备：给 dummy node 各做一次转向
    leftPrim.push_back(LEFT);  turnLeft_(&leftDummyNode_);
    rightPrim.push_back(RIGHT); turnRight_(&rightDummyNode_);
}
```

###### C6.1.5 奖励函数（`computeReward`）

`GreedyFollower.cpp:60-79`，这是 primitive 选择的决策依据。奖励函数有四个维度：

```cpp
float computeReward(const SceneNode& node, const ShortestPath& path, size_t primLen) {
    TryStepResult tryStepRes = tryStep(node, path.requestedEnd);

    return (
        // 维度 1: geodesic distance 减少量（越大越好，说明沿这个方向前进更接近目标）
        (path.geodesicDistance - tryStepRes.postGeodesicDistance) / forwardAmount_
    )
    + (
        // 维度 2: 倾向更短的 primitive（避免不必要的转向）
        -0.0125 * primLen
        // 维度 3: 碰撞惩罚
        - (tryStepRes.didCollide ? collisionCost_ : 0.0)    // collisionCost_ = 0.25
        // 维度 4: 靠近障碍物惩罚（距障碍物 <0.2m 时额外惩罚）
        - (tryStepRes.postDistanceToClosestObstacle < closeToObsThreshold_ ? 0.05 : 0.0)
    );
}
```

| 维度 | 权重/值 | 作用 |
|------|---------|------|
| geodesic 进展 | `1/forwardAmount_`（≈4.0） | 推动向目标靠近 |
| primitive 长度 | `-0.0125` | 倾向更少的转向 |
| 碰撞惩罚 | `-0.25` | 避免撞墙 |
| 近障碍惩罚 | `-0.05`（阈值 0.2m） | 避免贴墙走 |

`tryStep` 在 dummy node 上模拟前进一步，然后计算：
- 模拟后的 geodesic distance 到目标
- 模拟后位置到最近障碍物的距离（`distanceToClosestObstacle`）
- 是否碰撞（`didCollide`）

###### C6.1.6 振荡修复机制（`isThrashing`）

`GreedyFollower.cpp:140-156`：如果最近 `thrashingThreshold_`（默认 16）步动作都是左-右交替（`LEFT, RIGHT, LEFT, RIGHT...`），判定为振荡。

修复方式：当检测到振荡时，把计算出的下一个 best primitive **反转**（`rbegin` → `rend`），让 agent 执行反转后的动作序列来跳出振荡。

例如：如果最佳 primitive 是 `[LEFT, FORWARD]`，反转后变成 `[FORWARD, LEFT]`，先前进再左转，打破振荡循环。

###### C6.1.7 关键行为特征总结

1. **不是简单路径跟踪**：follower 不是"先找完整路径再沿路径走"，而是**每一步都重新在所有 motion primitive 上做贪心选择**——模拟每一种可能的转向+前进组合，选 reward 最高的
2. **奖励函数是核心**：选择 primitive 的依据是"沿这个方向前进一步后，geodesic distance 减少最多，且不碰撞、不贴墙、转向最少"
3. **早停优化**：如果某个 primitive 的 reward 已经 > 0.99（接近最优），不再继续搜索更大的转向角度
4. **振荡修复**：检测到连续左右交替时，反转 primitive 序列来跳出
5. **goal_radius = 1.0m**：follower 自身认为到达的条件是 geodesic distance < 1m，但项目在外层用 `success_dis` + visibility 做了更严格的二次判定

###### C6.2 近距离朝向对齐：`get_goal_pose_alignment_action`（`simulation.py:525-542`）

当 `geo_dis < success_dis` 但目标不可见时触发：

- 如果 `planar_distance > 0.6m`，退回到 `get_visibility_search_action`
- 否则计算 `goal_yaw - current_yaw` 的差值
- 偏差 > 15° → `turn_left` / `turn_right`
- 偏差 ≤ 15° → 返回 `None`（表示对齐已完成，交给下一级策略）

###### C6.3 近距离局部搜索：`get_visibility_search_action`（`simulation.py:502-523`）

朝向对齐仍不够时：

- 计算当前 yaw 与"朝向物体中心"方向的偏差
- 偏差 > 15° → `turn_left` / `turn_right`
- 偏差 ≤ 15° → `move_forward`（微调逼近）

##### C7. 成功判定策略

`dataset_gen.py:68-93`，双重条件：

```python
geo_dis < success_dis  AND  target_visible_in_front_view
```

- `geo_dis` 是到**可见目标视点**的 geodesic distance，不是到物体中心
- `target_visible` 通过前视 semantic sensor 检查目标实例像素数 ≥ `success_visible_pixels`（默认 25）
- 只有两者都满足才判子目标成功
- 如果开了 `--allow_stop_without_visibility`，退回旧逻辑（只看距离）

##### C8. 轨迹执行主循环

`dataset_gen.py:35-137`，完整流程：

```text
for step in range(max_step):
    1. 确定当前子目标 (target[success])
    2. get_goal_info(success) → 选择导航目标视点
    3. 如果 coord=None → unreachable, 终止任务
    4. 如果 geo_dis=inf → 终止任务
    5. 记录 pos/yaw/action 到 trial
    6. 成功判定：
       - 距离够 & 可见 → success++, 记录 stop, 下一个子目标
       - 距离够 & 不可见 → 局部搜索（对齐 / 搜索动作）
       - 距离不够 → GreedyGeodesicFollower 导航
    7. actor() 执行动作 + 保存 6 视角图片
```

##### C9. 轨迹生成策略总结

| 策略环节 | 方法 | 核心思想 |
|---------|------|---------|
| 起点 | NavMesh 随机采样 | 随机探索起点 |
| 目标选择 | 可见视点最小 geodesic 距离 | 导航到"看得见"目标的位置 |
| 远距离动作 | GreedyGeodesicFollower | 贪心最短路径 |
| 近距离动作 | 朝向对齐 → 局部搜索 | 先对齐 yaw 再微调 |
| 成功判定 | 距离 + 可见像素数双条件 | 确保"真看见"而非"刚好靠近" |
| 失败处理 | unreachable 直接终止 | 避免无效轨迹 |

核心设计理念是：**轨迹不只是"走到附近"，而是要"走到能看见目标的位置"**，这是 LH-VLN 数据集对导航质量的更高要求。

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

- 从场景的语义统计中，整理出一个"给 LLM 看得懂的房间-物体字典"

### 4.4 `gen_task()` 如何调用 LLM

它会读取：

- `prompt/rule.txt`
- `prompt/example.txt`
- `prompt/<robot>.txt`
- `prompt/system.txt`

当前这套 prompt 规则的核心已经不是 `Grab` / `Release` 了，而是要求模型用：

- `Move_to('object_region_id')`
- `Stop_at('object')`

来组织任务。

也就是说，当前长任务更像"去某个环境物体附近并停下"的序列，而不是"抓取-搬运-释放"的操作序列。

然后拼出最终 prompt，交给：

```python
gpt4o_mini(args, args.prompt_path + "system.txt", prompt)
```

注意：

- 虽然函数名还叫 `gpt4o_mini`
- 但现在底层已经是通过 `gpt.py` 赆 DashScope 兼容接口，默认模型是 `qwen3.6-plus`

### 4.5 LLM 输出什么格式

代码期望 LLM 输出一个字典结构，优先按 JSON 解析；如果模型返回了：

- Markdown code fence
- 单独的 `json` / `python` 语言提示
- Python 字典字面量

当前 `task_gen.py` 里的 `_parse_task_output()` 也会先做清洗再解析。

最终这个结构里至少要包含：

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

3. 检查所有 `Stop_at('obj')`
   - 目标 object 是否真的出现在当前 `input_scene` 中

4. 检查 `Task instruction` 中是否直接出现 `region` / `Region`
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

这个函数是"在 Habitat 里执行整条长任务"的核心。

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
   - 这里还会把前视 semantic sensor、高度、FOV、GPU device 等参数一起写进 simulator config

4. 初始化 Habitat-Sim

5. 读取 scene managers / rigid object manager / pathfinder

6. 初始化 agent
   - 在可导航点随机采样一个起点

7. 初始化 yaw
   - 默认 `180`

8. 初始化 `GreedyGeodesicFollower`
   - 之后每一步动作都由它给出

9. 初始化两个缓存
   - `target_candidate_cache`
   - `target_viewpoint_cache`

10. 先做一次 `sim.step("move_forward")`
   - 用于初始化观测
   - 这一步也会让 agent 从采样起点向前执行一次基础动作

### 6.2 `SceneSimulator` 提供了哪些关键方法

最关键的是这几个：

- `actor(action, step, success)`
  - 真正执行动作，并保存当前观察图像
- `get_target_candidates(target_index)`
  - 在目标 region 内枚举同类物体实例，收集 `center`、`semantic_id`、`object_id`
- `get_visible_viewpoints(target_index, min_pixels)`
  - 围绕候选物体采样可导航视点，并筛掉 semantic pixel 不足的视点
- `get_goal_info(target_index)`
  - 选择当前真正要导航的 goal：优先选"可见目标视点"，必要时再决定是否回退到物体中心 snap 点
- `get_target_visibility(target_index, min_pixels)`
  - 检查当前 front semantic sensor 里目标实例到底有没有真正出现
- `get_info(success)`
  - 返回当前子目标、位置、yaw、geodesic distance
- `get_goal_pose_alignment_action(goal_center, goal_yaw)`
  - 已经靠近目标但还没满足可见性时，优先做朝向对齐
- `get_visibility_search_action(goal_center)`
  - 朝向对齐还不够时，做近距离局部搜索
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
2. 调用 `get_goal_info(success)` 选择当前要追踪的导航目标
   - 先在目标物体周围采样"能看见目标"的候选视点
   - 对每个候选视点计算 geodesic distance
   - 选 geodesic distance 最短的那个视点作为当前 goal
   - 如果完全找不到满足 `success_visible_pixels` 阈值的可见视点，则直接判 unreachable
3. 调用 `return_state()` 记录当前状态
4. 把当前状态写入：

```text
config['trial']['trial_i']['pos'/'yaw'/'action']
```

### 7.3 成功判定

当前默认不是只看距离了，而是同时满足：

```python
geo_dis < args.success_dis
and target_is_visible_in_front_view
```

其中 `target_is_visible_in_front_view` 是通过前视角 semantic sensor 判断的，
默认要求目标实例在前视角里至少有 `25` 个 semantic pixel。

需要注意一点：

- 这里的 `geo_dis` 默认是"到选中的可见目标视点"的 geodesic distance
- 不是简单地到物体几何中心的距离

如果命令行打开：

- `--allow_stop_without_visibility`

则会退回旧逻辑，只要距离满足就允许 stop success。

只有这两个条件都满足，当前子目标才会被判成功。

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
2. 如果已经进入 `success_dis` 范围，但前视角里还看不到目标：

```python
action = task_sim.get_goal_pose_alignment_action(goal_center, goal_yaw)
```

如果朝向已经基本对齐，但还是没看见目标，才会退到：

```python
action = task_sim.get_visibility_search_action(goal_center)
```

也就是说，当前实现不是"距离够了就盲目 stop"，而是会在目标附近继续做局部朝向调整 / 搜索，直到目标真正进入 front view，或者命中旧逻辑回退开关。

3. 否则调用：

```python
action = task_sim.get_next_action(coord)
```

4. 再调用：

```python
task_sim.actor(action, step, success)
```

也就是说，真正的动作执行策略不是 LLM 实时规划出来的，而是：

- LLM 先给长任务与子目标顺序
- 导航阶段使用 Habitat 自带的 `GreedyGeodesicFollower`

## 8. `actor()` 为什么很重要

`SceneSimulator.actor()` 不只是执行动作，还负责"落图"。

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

返回的是"任务目录路径"。

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

它会遍历动作序列，并按"当前动作所属 target 是否改变"来切分。

一个 target 的连续轨迹，会被认为是一段候选轨迹。

但不是所有段都保留，代码里还做了长度过滤，例如：

- 只保留长度大于一定阈值的段

### 10.5 `segment_trajectory()` 如何把动作序列再切细

这是第 3 步最关键的逻辑。函数位于 `split_task.py:507-708`。

输入参数：

- `trajectory`：当前子目标对应的 `(action, target)` 列表
- `obs_dic`：每个 step 对应的图片路径字典（`step_index -> {'front': path, ...}`）
- `task`：任务目录路径
- `key`：当前子轨迹在整个长轨迹中的全局起始 step offset

输出是一组 segment 字典列表，每个 segment 包含 `trajectory`、`start`、`end`、`label`、`target`、`scene tags` 等字段。

#### 10.5.1 Step 1：识别连续同向转向段（`split_task.py:521-543`）

算法用滑动窗口（长度 3）扫描动作序列，寻找"3 步中有至少 2 步是同向转向"的片段：

```python
i = 0
while i <= n - 3:
    window = trajectory[i:i + 3]
    if window.count('turn_left') >= 2:
        left_indices = [idx for idx, action in enumerate(window) if action == 'turn_left']
        start = i + left_indices[0]
        end = i + left_indices[-1]
        segments.append((start, end, 'turn_left'))
        i = end + 1  # 跳过已识别的段
    else:
        i += 1
```

同样的逻辑对 `turn_right` 重复一次。

**关键细节**：
- 窗口大小固定为 3，阈值固定为 2（即 3 步中 2 步同向转向即可触发）
- 找到匹配后，`i` 直接跳到 `end + 1`，避免重叠识别
- 如果不匹配，`i` 只前进 1 步，逐位扫描
- 输出的 segment 格式为 `(start_index, end_index, label)`
- 两次扫描独立进行：先扫 `turn_left`，再扫 `turn_right`

**举例**：假设动作序列为 `[forward, left, left, forward, left, right, left, right]`

- 扫描 `turn_left`：窗口 `[left, left, forward]` 在 i=1 处触发 → segment `(1, 2, 'turn_left')`
- 扫描 `turn_right`：窗口 `[right, left, right]` 在 i=5 处触发 → segment `(5, 7, 'turn_right')`

#### 10.5.2 Step 2：合并相邻/重叠的同向转弯段（`split_task.py:545-559`）

将 Step 1 中找到的同向段排序后，逐个合并相邻或距离 ≤3 的同 label 段：

```python
merged_segments = []
segments.sort()
current_start, current_end, current_label = segments[0]
for segment in segments[1:]:
    start, end, label = segment
    if start <= current_end + 3 and label == current_label:
        # 合并：延伸当前段的 end
        current_end = max(current_end, end)
    else:
        # 不合并：保存当前段，开始新段
        merged_segments.append((current_start, current_end, current_label))
        current_start, current_end, current_label = segment
merged_segments.append((current_start, current_end, current_label))
```

**关键细节**：
- 合并条件是 `start <= current_end + 3`，即两个同向段之间最多间隔 3 步仍会合并
- 只有同 label 的段才能合并（`turn_left` 不会和 `turn_right` 合并）
- 合并方式是取最大的 `end`，即将两段之间的间隔也纳入合并后的段

**举例**：假设 Step 1 输出 `(1, 2, left)` 和 `(4, 5, left)`：
- `4 <= 2 + 3` → 合并为 `(1, 5, left)`，中间的 step 3 也被纳入

#### 10.5.3 Step 3：处理不同方向的重叠段 → zigzag（`split_task.py:561-577`）

对合并后的段列表，如果连续两个不同方向的段有重叠或紧邻（`curr_end + 1 >= next_start`），合并为 `zigzag`：

```python
while len(temp) > 1:
    curr_start, curr_end, curr_label = temp.pop(0)
    next_start, next_end, next_label = temp.pop(0)
    if curr_end + 1 >= next_start:
        # 重叠或紧邻 → 合并为 zigzag
        new_start = min(curr_start, next_start)
        new_end = max(curr_end, next_end)
        result_segments.append((new_start, new_end, "zigzag"))
    else:
        # 不重叠 → 保留当前段，把下一个段放回队列头部
        result_segments.append((curr_start, curr_end, curr_label))
        temp.insert(0, (next_start, next_end, next_label))
if temp:
    result_segments += temp
```

**关键细节**：
- 重叠判断条件是 `curr_end + 1 >= next_start`，即只要两个段之间间隔 ≤1 就算重叠
- zigzag 段的 range 取两个段的 min(start) ~ max(end)
- 不重叠的段直接保留
- 处理完后 `temp` 中可能还剩一个段，追加到结果

**举例**：假设 Step 2 输出 `(1, 5, left)` 和 `(6, 8, right)`：
- `5 + 1 >= 6` → 合并为 `(1, 8, zigzag)`，代表"左右交替转向"

#### 10.5.4 Step 4：填充 move_forward 空隙 + 构建最终 segment（`split_task.py:579-611`）

遍历所有转向/zigzag 段，在它们之间的空隙填充 `move_forward` 段：

```python
final_segments = []
last_end = -1
for seg_start, seg_end, label in result_segments:
    # 如果两个段之间有空隙，补一个 move_forward 段
    if last_end + 2 < seg_start:
        final_segments.append({
            "trajectory": task,
            "start": last_end + 1 + key,
            "end": seg_start - 2 + key,
            "obs": [obs_dic[i]['front'] for i in range(last_end + 1 + key, seg_start - 1 + key)],
            "label": "move_forward",
            "target": target
        })
    # 添加当前转向/zigzag 段
    final_segments.append({
        "trajectory": task,
        "start": seg_start - 1 + key,
        "end": seg_end + key,
        "obs": [obs_dic[i]['front'] for i in range(seg_start - 1 + key, seg_end + key + 1)],
        "label": label if trajectory[seg_start - 1 : seg_end + 1].count(label) <= 3
                 else "make a " + label.split("_")[-1] + " turn",
        "target": target
    })
    last_end = seg_end
```

**关键细节**：

1. **move_forward 空隙判定**：`last_end + 2 < seg_start`，即两个段之间至少间隔 2 步才补 `move_forward`。如果间隔只有 1 步（紧邻），不补。

2. **全局 offset 修正**：所有 `start/end` 都加上 `key`（子轨迹在长轨迹中的全局起始 offset），使得索引映射到全局 step 序号。

3. **label 的自然语言化**：转向段中如果该转向动作出现 >3 次，label 从 `turn_left` 变为 `"make a left turn"`。这是一个简单的阈值规则：
   - 出现 ≤3 次 → 保持原始标签（如 `turn_left`、`turn_right`、`zigzag`）
   - 出现 >3 次 → 改为自然语言描述（如 `"make a left turn"`）

4. **转向段的 start 前移 1 步**：转向段的 `start` 用 `seg_start - 1`，即把转向前的一个 step 也纳入段中（可能是触发转向前的最后一步前进）。

5. **尾部 move_forward**：如果最后一个转向段之后还有剩余步骤，补一个尾部 `move_forward` 段：

```python
if last_end < n - 1:
    final_segments.append({
        "trajectory": task,
        "start": last_end + 1 + key,
        "end": n - 1 + key,
        "obs": [obs_dic[i]['front'] for i in range(last_end + 1 + key, n + key)],
        "label": "move_forward",
        "target": target
    })
```

#### 10.5.5 完整处理流程示意

假设动作序列为 `[forward, left, left, forward, right, right, forward, forward]`，全局 offset `key=10`：

```text
Step 1 (识别连续转向段):
  - 扫描 turn_left: 窗口 [left, left, forward] 在 i=1 → segment (1, 2, 'turn_left')
  - 扫描 turn_right: 窗口 [right, right, forward] 在 i=4 → segment (4, 5, 'turn_right')

Step 2 (合并同向段):
  - 只有两个段且方向不同，无法合并同向段
  - merged_segments = [(1, 2, 'turn_left'), (4, 5, 'turn_right')]

Step 3 (处理不同方向重叠 → zigzag):
  - curr=(1,2,left), next=(4,5,right)
  - 2+1 >= 4? → False (间隔1步不重叠)
  - result_segments = [(1, 2, 'turn_left'), (4, 5, 'turn_right')]

Step 4 (填充 move_forward):
  - last_end=-1, seg=(1,2,left): -1+2 < 1 → 补 move_forward (0+10, -1+10)
  - 添加 turn_left 段 (0+10, 2+10)
  - last_end=2, seg=(4,5,right): 2+2 < 4 → 补 move_forward (3+10, 2+10)
  - 添加 turn_right 段 (3+10, 5+10)
  - last_end=5, 尾部剩余 step 6,7 → 补 move_forward (6+10, 7+10)

最终 segments:
  [{start:10, end:10, label:"move_forward"},
   {start:10, end:12, label:"turn_left"},
   {start:13, end:12, label:"move_forward"},
   {start:13, end:15, label:"turn_right"},
   {start:16, end:17, label:"move_forward"}]
```

#### 10.5.6 每个 segment 的最终数据结构

`split_task.py:613-650` 为每个 segment 补充元信息：

```python
for step_index, seg in enumerate(final_segments):
    seg["step_index"] = step_index              # 在局部段中的序号
    seg["debug_subtraj_dir"] = subtraj_debug_dir  # 调试输出目录
    seg["scene tags"] = []                      # 后续由 RAM / scene instance 填充
    seg["all_tag_rankings"] = []                # 后续由 scene instance 填充
```

如果开启了调试模式（`subtraj_debug_dir` 不为 None），还会：

1. 为每个 step 的 front 图片做一份拷贝，存到 `result3_step_images/step_xxx_label/` 目录
2. 保存 `result2_segmented_steps.json`（包含压缩后的 step 列表）
3. 保存 `result3_step_tags_and_images.json`（包含图片路径和标签）
4. 如果 `scene_instance_options` 不为 None，调用 `generate_scene_instance_box_debug()` 做场景实例识别，将结果写回每个 segment 的 `scene tags` 和 `all_tag_rankings`

#### 10.5.7 `segment_trajectory()` 的调用上下文

`split_traj()` 函数（`split_task.py:788-886`）是外层调用者：

1. 先按 target 切段（`action_dic[end][1] != target` 时切分）
2. 对每个子轨迹做长度过滤（`end - start > 5`，至少 6 步）
3. 调用 `segment_trajectory(trail, obs_dic, task, start, ...)` 做细切分
4. 结果追加到全局 `trail_list`

长度过滤的意义：太短的子轨迹（≤5 步）信息量不足，直接跳过不生成 step task。

### 10.6 `batch_ram(img_list)` 在这里扮演什么角色

每个 segment 会把自己的前视图片送进 RAM：

1. 对每张图做 transform
2. 用 RAM 识别场景标签
3. 统计频次
4. 取 top-5 tags

这些 tags 会作为"这段局部轨迹的场景语义摘要"。

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

补充两个当前实现里的细节：

- 如果 `Task instruction` 太长，文件名会被截断到前 `100` 个字符
- 如果 instruction 末尾是空格，保存前会先去掉末尾空格

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

- 上游 LLM 负责"任务层级"
- Habitat 负责"导航执行层级"
- RAM 负责"局部视觉语义摘要"
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

如果你只是想抓住"看哪里"，最重要的几个产物就是：

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
- 收集 summary，方便判断"到底能不能跑通"

### `run_split_steps.py`

用途：

- 只跑：
  - `split_traj(args)`
  - `gen_step_task(args)`

适合在长任务轨迹已经准备好的情况下，单独重跑切分阶段。

### `rerun_single_task.py`

用途：

- 对已有 `task.json` / 单条任务配置做单任务重跑
- 方便复现某个失败 case，或者验证某个成功轨迹在新参数下会不会行为变化

### `visualize_goal_viewpoints_topdown.py`

用途：

- 把某个目标物体周围采样到的"可见目标视点"画到 top-down map 上
- 专门用来调试：
  - `success_visible_pixels`
  - `success_view_radius_*`
  - `render_sensor_height`

### `export_trajectory_rgb_video.py`

用途：

- 把 `success/trial_x/task.json` 回放成第一人称视频
- 支持：
  - 自定义相机高度/FOV/分辨率
  - 朝向修正
  - 一秒一帧动作对齐
  - 文本标注
  - 2D 目标框

这个脚本是本轮为了"人工核对 NavGen 输出是否合理"新增的可视化工具，不属于论文 NavGen 原始四步流水线。

说明：

- 旧文档里提到过 `run_navgen_split_gpu.sh`
- 但当前 `nav_gen/` 目录里已经没有这个脚本了，所以这里按现有代码改成了 `rerun_single_task.py` 和 `visualize_goal_viewpoints_topdown.py`

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
10. `nav_gen/rerun_single_task.py`
11. `nav_gen/visualize_goal_viewpoints_topdown.py`
12. `nav_gen/export_trajectory_rgb_video.py`

这样读下来，逻辑上最顺。