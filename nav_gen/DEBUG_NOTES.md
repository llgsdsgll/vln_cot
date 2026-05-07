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
