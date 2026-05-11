# NavGen Debug Notes

## 1. `scene_id` 字符串未包装成列表

**文件：** `task_gen.py:188`

**现象：** 传入 `--scene_id 00006-HkseAnWCgqk` 时，所有 attempt 报 `task_generation_error`，日志显示 `FileNotFoundError: .../scene/0.txt`。

**原因：** `gen_task` 中直接 `sample_scene = args.scene_id`（字符串），后续 `sample_scene[0]` 取的是字符串第一个字符 `"0"` 而非场景 ID。

**修复：**
```python
# 修复前
sample_scene = args.scene_id
# 修复后
sample_scene = [args.scene_id]
```

---

## 2. LLM API 调用无重试机制

**文件：** `gpt.py:76`（`_chat_completion`）

**现象：** 批量生成时偶发 `httpcore.ConnectError: [Errno 104] Connection reset by peer`，导致 `task_generation_error`，即使 API 连通性正常（curl 返回 200）。

**原因：** `_chat_completion` 直接调用一次 API，无任何重试逻辑，网络抖动即失败。

**修复：** 加入指数退避重试（最多 3 次）：
```python
def _chat_completion(args, model, messages, max_retries=3):
    import time
    client = _make_client(args)
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(model=model, messages=messages)
            if not response.choices:
                raise RuntimeError("The model returned no choices.")
            return _extract_message_text(response.choices[0].message)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
            client = _make_client(args)
```

---

## 3. 大量场景 `unreachable_target`：随机采样命中无效场景

**现象：** 不指定 `scene_id` 时，20 次 attempt 全部返回 `unreachable_target`（geodesic distance = inf 或无可见视点）。

**原因：** `task_gen.py` 从 `Per_Scene_Region_Weighted_Votes.csv` 的所有 key 中随机采样场景，但 `hm3d/train/` 有 803 个场景，而 `nav_gen/scene/` 下只有 181 个场景有对应的 `.txt` 物体列表文件。命中无 txt 文件的场景时，Habitat 导航图中目标物体路径不连通。

**建议：** 批量生成时通过 `--scene_id` 限定在 `nav_gen/scene/` 下有 txt 文件的场景内：

```bash
# 获取有效场景列表
ls nav_gen/scene/*.txt | xargs -I{} basename {} .txt > valid_scenes.txt

# 或在 run_smoke_test.py / run_navgen_remote_batch.sh 中随机选取有效场景
```

也可修改 `task_gen.py` 中的采样逻辑，将候选场景限定为 `scene_path` 下有 txt 文件的场景：

```python
import glob, os
valid_scenes = {os.path.basename(p)[:-4] for p in glob.glob(args.scene_path + "*.txt")}
sample_scene = random.sample(sorted(valid_scenes & set(scene_region.keys())), 1)
```

**影响：** 限定后有效场景从 ~803 降至 181，但 `unreachable_target` 率可大幅降低。

---

## 5. Bounding box 比 mask 大：同一 semantic_id 包含多个分散实例

**文件：** `export_trajectory_rgb_video.py:611`（`_extract_target_boxes`）

**现象：** 视频中目标物体的 bbox 远大于 mask 覆盖区域（如 `dishes` mask 仅 3754px，但 bbox 达 635×258px）。

**原因：** 原逻辑对整个 semantic_id 取 min/max 像素坐标，当同一类别有多个分散实例（如多个盘子分布在台面上）时，bbox 会包含所有实例之间的空白区域。

**修复：** 改用连通域分析，对每个连通区域单独生成 bbox：
```python
mask = (semantic_obs == semantic_id).astype(np.uint8)
n, labels = cv2.connectedComponents(mask)
for comp_id in range(1, n):
    ys, xs = np.where(labels == comp_id)
    if xs.size < min_pixels:
        continue
    boxes.append({"bbox": (xs.min(), ys.min(), xs.max(), ys.max()), ...})
```

---

## 6. 所有 item 返回 `no_success_task`：目标物体无可见视点

**文件：** `habitat_base/simulation.py:397`（`get_goal_info`）

**现象：** 批量生成时所有 item 均返回 `no_success_task`，日志显示 `The target of task ... is un reachable!`。

**原因：** `get_goal_info` 在半径 0.75–2.5m 范围内采样视点并检查语义传感器像素数。若目标物体周围无可导航且能看到目标的视点（`pixel_count >= 1`），则返回 `(inf, None, {goal_type: "no_visible_viewpoint"})`，`dataset_gen.py` 立即将任务标记为不可达。

**修复：** 在 `run_navgen_remote_batch.sh` 调用中加入 `--allow_occluded_goal_fallback`，回退到 snap 到物体中心的导航目标：
```bash
./run_navgen_remote_batch.sh \
  --run-name batch_xxx \
  --export-videos \
  --loop 5 \
  --max-attempts 50 \
  --max_step 500 \
  --allow_occluded_goal_fallback
```

**另：** `task_gen.py` 中 `os.mkdir` 改为 `os.makedirs(..., exist_ok=True)`，避免 task_path 子目录不存在时报 `FileNotFoundError`。


**现象：** 从项目根目录运行时，`region_file` 路径解析为 `/home/gs/my_project/LH-VLN/scene/Per_Scene_Region_Weighted_Votes.csv`（缺少 `nav_gen`），报 `FileNotFoundError`。

**原因：** `build_args()` 中 `nav_gen_path = os.getcwd()`，依赖当前工作目录为 `nav_gen/`。

**正确运行方式：**
```bash
cd /path/to/LH-VLN/nav_gen
python run_smoke_test.py --output_root smoke_out_single ...
```

---

## 7. 文档文件名调整

**变更：** 原 `DEBUG_NOTES.md` 已调整为 `DEBUG_AND_CHANGES_NOTES.md`。

**说明：** 旧内容保持原样保留，后续调试记录和代码改动记录统一继续追加在这个文件里。

---

## 8. 严格可见视点模式：取消放宽版视点与中心点回退

**文件：**

- `habitat_base/simulation.py:397`（`get_goal_info`）
- `main.py:44`
- `rerun_single_task.py:46`
- `visualize_goal_viewpoints_topdown.py:68`
- `NAVGEN_CODE_FLOW_EXPLANATION.md`

**变更前：**

- 先按 `success_visible_pixels` 查找可见目标视点。
- 如果找不到，并且阈值大于 1，则自动再尝试一次放宽版可见视点筛选：`pixel_count >= 1`。
- 如果仍然找不到，并且打开 `--allow_occluded_goal_fallback`，则回退到目标物体中心附近的 `snap_point` 继续导航。

**现改为：**

- 只接受满足 `success_visible_pixels` 阈值的可见目标视点。
- 不再做 `pixel_count >= 1` 的放宽版可见视点尝试。
- 不再回退到物体中心 `snap_point` 作为导航 goal。
- 只要当前目标没有满足阈值的可见视点，就直接返回 `no_visible_viewpoint`，并在轨迹生成阶段判为不可达。

**兼容性处理：**

- `allow_occluded_goal_fallback` 参数入口仍保留，避免旧脚本直接报错。
- 但该参数现在仅作为兼容保留项，实际行为已固定为严格可见视点模式。
- `main.py`、`rerun_single_task.py`、`visualize_goal_viewpoints_topdown.py` 中的帮助文字已同步标注为 deprecated / ignored。

**同步修改：**

- `visualize_goal_viewpoints_topdown.py` 不再把 `>= 1 px` 的 relaxed probes 当作候选可见视点展示。
- `NAVGEN_CODE_FLOW_EXPLANATION.md` 已同步更新为“没有满足阈值的可见视点就直接判 unreachable”的说明。

**目的：**

- 收紧成功前的目标可见性约束。
- 避免模型在“目标其实不可见”时，依赖放宽版筛选或中心点回退继续生成不够严格的轨迹。

---

## 9. 任务生成子任务从 `Grab/Release` 切换为 `Stop_at`

**文件：**

- `prompt/rule.txt`
- `prompt/example.txt`
- `prompt/spot.txt`
- `prompt/stretch.txt`
- `task_gen.py`
- `NAVGEN_CODE_FLOW_EXPLANATION.md`
- `NAVGEN_PAPER_3_1_APPENDIX_CODE_MAPPING.md`

**变更前：**

- task generation prompt 允许三种子任务：
  - `Move_to("object_region id")`
  - `Grab("object")`
  - `Release("object")`
- prompt 中还要求任务更接近“把一个区域里的物体带到另一个区域，再取回另一个物体”的搬运式结构。
- prompt 还限制被抓取对象应当是 portable、合理可抓的。

**现改为：**

- task generation prompt 现在只保留两种子任务：
  - `Move_to("object_region id")`
  - `Stop_at("object")`
- 不再要求生成抓取、搬运、释放类任务。
- 不再要求 object 必须 portable 或适合抓取。
- 只要 object 真实存在于当前 scene / `input_scene` 中，就可以作为任务目标。

**代码层同步：**

- `task_gen.py` 仍然对 `Move_to('obj_region')` 做 region/object 合法性检查。
- 另外新增了对 `Stop_at('object')` 的检查：object 必须真实存在于当前 `input_scene` 中，否则任务直接判 invalid。

**影响：**

- 长任务定义从“操作型任务”收敛为“导航到环境物体并停下”的任务。
- 机器人描述里的 grabbing 偏向也已去掉，避免继续把 prompt 往抓取/搬运方向带偏。

---

## 10. `debug_scene_instance_boxes.py` 增加更强的实例过滤表

**文件：**

- `debug_scene_instance_boxes.py`

**背景：**

- 在 `split_traj` 的 debug 可视化里，原先按 semantic instance 统计每一步 front 图中的高频实例时，容易把 `wall`、`Unknown`、`door frame` 这类背景/结构类实例排到前面。
- 这样虽然技术上是“可见实例”，但对调试 step-level 目标理解帮助不大，前景可交互或更有语义价值的物体容易被淹没。

**本次改动：**

- 新增两档过滤配置：
  - `basic`：只过滤 `ceiling`、`floor`
  - `strong`：过滤更大一批背景/结构类类别
- 默认过滤模式已改为 `strong`。
- 新增 `--extra-ignore-substring` 参数，允许在命令行继续追加自定义过滤词。

**`strong` 过滤表示例：**

- `unknown`
- `wall`
- `frame`
- `floor`
- `ceiling`
- `window`
- `curtain`
- `sheet`
- `stairs`
- `stairwell`
- `beam`
- `decoration`
- `door frame`
- `doorway`
- `door`
- `pillar`
- `column`
- `arch`
- `railing`
- `rail`
- `balustrade`
- `banister`
- `trim`
- `baseboard`
- `vent`
- `outlet`
- `switch`
- `socket`
- `ceiling lamp`
- `light fixture`
- `pillow`

**筛选逻辑：**

- 仍然保持“对每张 front 图统计可见 semantic instance，按 step 内出现帧数和像素面积排序”的主流程不变。
- 改动仅发生在 instance 候选进入排序前：
  - 若实例 category 名称中包含过滤表里的任一子串，则该实例直接跳过，不参与 top-k 排名。

**入口参数：**

```bash
python nav_gen/debug_scene_instance_boxes.py \
  --debug-json /path/to/result3_step_tags_and_images.json \
  --filter-profile strong
```

如需切回旧的弱过滤行为，可显式指定：

```bash
python nav_gen/debug_scene_instance_boxes.py \
  --debug-json /path/to/result3_step_tags_and_images.json \
  --filter-profile basic
```

如需再额外屏蔽某些类别：

```bash
python nav_gen/debug_scene_instance_boxes.py \
  --debug-json /path/to/result3_step_tags_and_images.json \
  --extra-ignore-substring mirror \
  --extra-ignore-substring cabinet
```

**验证结果：**

- 已重新对一条真实 debug 样本执行脚本并生成新的 `result4_scene_instance_boxes.json`。
- 新结果中，前几步的 top instances 已明显从：
  - `wall / Unknown / door frame`
  收紧为更接近前景语义物体的：
  - `radiator / wardrobe / sofa / desk / chair`

**目的：**

- 让 `result4_scene_instance_boxes.json` 与对应标框图更适合人工调试 step-level 语义内容。
- 降低背景结构类实例在 top-k 中的占比，让真正影响任务理解的前景物体更容易被看见。
