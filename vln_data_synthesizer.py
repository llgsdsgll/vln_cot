#!/usr/bin/env python3
"""
VLN CoT 数据合成脚本

功能：
  读取人工标注后的 processed.json（含 subtask/object/bbox）和 train_gt.json（含真值动作序列），
  结合视频帧，调用 VLM API（mock 占位）生成 Chain-of-Thought 推理文本，
  输出为 OpenAI/HuggingFace 标准的对话微调格式（JSONL）。

使用方法：
  python3 vln_data_synthesizer.py \\
      --input   gengshuang_1_H.264_0324_processed.json \\
      --gt      /home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json \\
      --video-dir /home/gs/my_test/vln_dataset/vln_ce_video \\
      --output  output_cot.jsonl

  可选参数：
      --max-retries   VLM API 最大重试次数（默认 3）
      --episode-ids   只处理指定 episode（如 --episode-ids 1 6 7）
"""

import argparse
import base64
import json
import logging
import os
import time
from typing import Any, Optional

import cv2

# ---------------------------------------------------------------------------
# Prompt 模板 —— 抽离为全局变量，方便随时修改
# ---------------------------------------------------------------------------

# 有目标物体可见时使用
PROMPT_TEMPLATE = """\
You are an expert embodied AI data synthesis specialist. Your task is to generate a rigorous "Chain-of-Thought" (CoT) reasoning text for a Vision-Language Navigation (VLN) agent, based on provided ground-truth data.

This generated text will be used to fine-tune a smaller VLM. Therefore, your output MUST adhere strictly to the following rules:
1. Absolute Data Loyalty: You must completely rely on the provided ground-truth data (history, spatial coordinates, next action). Never hallucinate objects or reasons that contradict the ground truth.
2. Bottom-Up Reasoning: Your logic must flow naturally: Deconstruct Task -> Perceive Environment -> Analyze Spatial Relationship -> Decide Action -> Update Memory.
3. Coordinate Format: You must strictly use the bounding box format `<box>[x_min, y_min, x_max, y_max]</box>` with values normalized to the [0, 1000] range. `[500, 500]` represents the center of the image.

[Current Ground-Truth State (For your reference only, integrate this into your natural language output)]:
- Global Instruction: "{global_instruction}"
- Pre-defined Sub-tasks: {global_subtasks_list}
- Current Completed Sub-task Index: {completed_index}
- Historical Trajectory Memory: "{script_memory}"
- Target Landmark/Object in Current Frame: {target_object_name}
- Ground-Truth Target BBox (Normalized 0-1000): <box>{bbox_norm}</box>
- Ground-Truth Next Action: <action>{next_action}</action>

[Visual Input]:
<Image>

Based on the image and the ground-truth state above, simulate the first-person perspective of the navigation agent and generate the CoT reasoning text using the exact XML structure below:

<Think>
[Task Decomposition & Progress (Read History)]:
According to the global instruction, the overall task consists of the following phases: {global_subtasks_list}. Reviewing my historical memory: "{script_memory}", I can confirm that I have completed step {completed_index}. Therefore, my current core objective is to execute the next logical phase.

[Current Environment & Spatial Perception]:
To fulfill the current objective, I am scanning my current visual field. In the current observation, I have successfully identified the key target: {target_object_name}. Its exact spatial bounding box is <box>{bbox_norm}</box>. (Describe its relative position based on the coordinates, e.g., "It is located in the lower-right area of my view, taking up a significant portion of the screen.")

[Decision Reasoning Logic]:
Based on the target's position and my current orientation, (Explain why the next action is necessary using the numerical coordinates. e.g., "Since the target's center is shifted to the left (x_min and x_max < 500), I need to turn left to align with it..."). Therefore, taking the specific action is the most logical choice to proceed.

[Memory Summary & Update (Update History)]:
(Briefly summarize the key observation and decision made in this step in ONE concise sentence, serving as a memory anchor for future steps.)
In this step, I observed the {target_object_name} located at the (describe relative direction), and I decided to execute {next_action} to adjust my navigation path.
</Think>
<Action>
{next_action}
</Action>\
"""

# 目标物体不可见时使用（无 objects 标注帧）
PROMPT_TEMPLATE_NO_TARGET = """\
You are an expert embodied AI data synthesis specialist. Your task is to generate a rigorous "Chain-of-Thought" (CoT) reasoning text for a Vision-Language Navigation (VLN) agent, based on provided ground-truth data.

This generated text will be used to fine-tune a smaller VLM. Therefore, your output MUST adhere strictly to the following rules:
1. Absolute Data Loyalty: You must completely rely on the provided ground-truth data (history, spatial coordinates, next action). Never hallucinate objects or reasons that contradict the ground truth.
2. Bottom-Up Reasoning: Your logic must flow naturally: Deconstruct Task -> Perceive Environment -> Explore -> Decide Action -> Update Memory.
3. Since the target object is NOT visible in the current frame, do NOT fabricate any bounding box coordinates.

[Current Ground-Truth State (For your reference only, integrate this into your natural language output)]:
- Global Instruction: "{global_instruction}"
- Pre-defined Sub-tasks: {global_subtasks_list}
- Current Completed Sub-task Index: {completed_index}
- Historical Trajectory Memory: "{script_memory}"
- Target Landmark/Object: (Not visible in current frame)
- Ground-Truth Next Action: <action>{next_action}</action>

[Visual Input]:
<Image>

Based on the image and the ground-truth state above, simulate the first-person perspective of the navigation agent and generate the CoT reasoning text using the exact XML structure below:

<Think>
[Task Decomposition & Progress (Read History)]:
According to the global instruction, the overall task consists of the following phases: {global_subtasks_list}. Reviewing my historical memory: "{script_memory}", I can confirm that I have completed step {completed_index}. Therefore, my current core objective is to execute the next logical phase.

[Current Environment & Spatial Perception]:
The target object is currently not visible in my view. Based on the layout, I need to explore further to locate the target associated with the current sub-task.

[Decision Reasoning Logic]:
Since the target is not yet visible, I must continue navigating based on the current trajectory and instruction context. The action {next_action} is the most appropriate step to bring the target into view and make progress toward the goal.

[Memory Summary & Update (Update History)]:
(Briefly summarize the key observation and decision made in this step in ONE concise sentence, serving as a memory anchor for future steps.)
In this step, the target was not visible; I decided to execute {next_action} to continue exploring the environment.
</Think>
<Action>
{next_action}
</Action>\
"""

# 动作编码映射
ACTION_MAP: dict[int, str] = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VLNDataSynthesizer")


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class VLNDataSynthesizer:
    """
    VLN Chain-of-Thought 数据合成器。

    处理流程：
      1. 读取 processed.json（帧级标注）和 train_gt.json（真值动作）。
      2. 按 episode 遍历，对每一帧提取视频帧、构建 prompt、调用 VLM API。
      3. 将结果保存为 JSONL 格式，供后续微调使用。
    """

    def __init__(self, max_retries: int = 3) -> None:
        """
        初始化合成器。

        Args:
            max_retries: VLM API 调用失败时的最大重试次数。
        """
        self.max_retries = max_retries

    # ------------------------------------------------------------------
    # 静态/工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_bbox(bbox: dict[str, float]) -> list[int]:
        """
        将百分比格式的 bbox 归一化到 [0, 1000] 整数区间。

        Args:
            bbox: 标注数据中的 bbox 字典，格式为
                  ``{"x": float, "y": float, "width": float, "height": float}``，
                  各字段均为百分比值（0-100）。

        Returns:
            ``[xmin, ymin, xmax, ymax]`` 归一化到 [0, 1000] 的整数列表。
        """
        xmin = int(round(bbox["x"] * 10))
        ymin = int(round(bbox["y"] * 10))
        xmax = int(round((bbox["x"] + bbox["width"]) * 10))
        ymax = int(round((bbox["y"] + bbox["height"]) * 10))
        # 限制在合法范围内
        xmin = max(0, min(1000, xmin))
        ymin = max(0, min(1000, ymin))
        xmax = max(0, min(1000, xmax))
        ymax = max(0, min(1000, ymax))
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def update_script_memory(
        script_memory: list[str],
        frame_idx: int,
        action_str: str,
    ) -> None:
        """
        将本步骤的真值动作追加到历史记忆列表（in-place 修改）。

        使用真值动作维护历史，防止模型幻觉在步骤间累积。

        Args:
            script_memory: 存储历史动作文本的列表（将被修改）。
            frame_idx:      当前帧序号（1-indexed）。
            action_str:     当前帧执行的动作字符串（如 ``"MOVE_FORWARD"``）。
        """
        script_memory.append(f"Step {frame_idx}: Executed {action_str}.")

    def build_prompt(
        self,
        frame_data: dict[str, Any],
        script_memory: list[str],
        global_instruction: str,
        global_subtasks: list[str],
        completed_index: int,
        next_action: str,
    ) -> str:
        """
        根据当前帧的真值数据构建填充后的 prompt 字符串。

        当帧中没有 objects 标注时，使用无目标物体版本的模板，
        跳过坐标输出并改为探索性描述。

        Args:
            frame_data:        processed.json 中单帧的字典数据。
            script_memory:     截至上一步的历史动作文本列表。
            global_instruction: 当前 episode 的全局导航指令。
            global_subtasks:   当前 episode 的子任务列表（有序、去重）。
            completed_index:   当前子任务在列表中的索引（之前的均已完成）。
            next_action:       当前帧的真值动作字符串。

        Returns:
            填充完毕的 prompt 字符串，可直接传给 VLM API。
        """
        memory_str = " ".join(script_memory) if script_memory else "None"
        subtasks_str = str(global_subtasks)

        objects = frame_data.get("objects", [])

        if objects:
            # 取第一个 object 作为当前帧的主要目标
            obj = objects[0]
            target_name = obj["label"]
            bbox_norm = self.normalize_bbox(obj["bbox"])
            bbox_str = str(bbox_norm)

            prompt = PROMPT_TEMPLATE.format(
                global_instruction=global_instruction,
                global_subtasks_list=subtasks_str,
                completed_index=completed_index,
                script_memory=memory_str,
                target_object_name=target_name,
                bbox_norm=bbox_str,
                next_action=next_action,
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_TARGET.format(
                global_instruction=global_instruction,
                global_subtasks_list=subtasks_str,
                completed_index=completed_index,
                script_memory=memory_str,
                next_action=next_action,
            )

        return prompt

    @staticmethod
    def extract_frame(video_path: str, frame_number: int) -> Optional[str]:
        """
        从视频文件中提取指定帧，返回 base64 编码的 JPEG 字符串。

        Args:
            video_path:   视频文件的绝对路径。
            frame_number: 目标帧号（1-indexed）。

        Returns:
            base64 编码的 JPEG 图像字符串；提取失败时返回 ``None``。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("无法打开视频文件: %s", video_path)
            return None

        try:
            # OpenCV 帧索引从 0 开始
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(
                    "无法读取帧 %d，视频: %s", frame_number, video_path
                )
                return None

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning("JPEG 编码失败，帧 %d，视频: %s", frame_number, video_path)
                return None

            return base64.b64encode(buffer).decode("utf-8")
        finally:
            cap.release()

    def call_vlm_api(self, prompt: str, image_b64: Optional[str]) -> Optional[str]:
        """
        调用 VLM API 生成 CoT 推理文本（当前为 mock 实现）。

        包含完整的 retry 逻辑（指数退避）和错误处理。
        替换此函数体即可接入真实的 Qwen-VL 或其他 VLM API。

        Args:
            prompt:     填充好的文本 prompt。
            image_b64:  base64 编码的图像字符串；为 ``None`` 时跳过图像输入。

        Returns:
            模型生成的文本（CoT + Action）；所有重试失败后返回 ``None``。
        """
        for attempt in range(1, self.max_retries + 1):
            try:
                # ----------------------------------------------------------------
                # TODO: 替换以下 mock 逻辑为真实 API 调用，例如：
                #
                #   import openai
                #   client = openai.OpenAI(api_key=..., base_url=...)
                #   messages = [{"role": "user", "content": [...]}]
                #   response = client.chat.completions.create(
                #       model="qwen-vl-max",
                #       messages=messages,
                #   )
                #   return response.choices[0].message.content
                # ----------------------------------------------------------------
                logger.debug("VLM API mock 调用 (attempt %d/%d)", attempt, self.max_retries)
                mock_response = (
                    "<Think>\n"
                    "[Task Decomposition & Progress (Read History)]:\n"
                    "Mock CoT response — replace call_vlm_api() with real API.\n"
                    "</Think>\n"
                    "<Action>\nMOCK_ACTION\n</Action>"
                )
                return mock_response

            except Exception as exc:  # noqa: BLE001
                wait = 2 ** attempt
                logger.warning(
                    "VLM API 调用失败 (attempt %d/%d): %s — 等待 %ds 后重试",
                    attempt,
                    self.max_retries,
                    exc,
                    wait,
                )
                if attempt < self.max_retries:
                    time.sleep(wait)
                else:
                    logger.error("VLM API 重试耗尽，跳过本帧。错误: %s", exc)
                    return None

        return None  # 不可达，但满足类型检查

    def process_trajectory(
        self,
        episode_data: dict[str, Any],
        gt_actions: list[int],
        video_dir: str,
    ) -> list[dict[str, Any]]:
        """
        处理单条轨迹（一个 episode）的所有帧，生成 CoT 标注记录列表。

        Args:
            episode_data: processed.json 中单个 episode 的字典。
            gt_actions:   该 episode 来自 train_gt.json 的动作序列（整数列表）。
            video_dir:    视频文件所在目录。

        Returns:
            每帧对应一条标注记录的列表（OpenAI/HF 对话格式）。
        """
        episode_id: int = episode_data["episode_id"]
        instruction: str = episode_data["instruction"]
        frames: list[dict[str, Any]] = episode_data["frames"]

        # 按首次出现顺序收集去重的子任务列表
        global_subtasks: list[str] = list(
            dict.fromkeys(
                f["subtask"] for f in frames if f.get("subtask") is not None
            )
        )

        video_path = os.path.join(video_dir, f"ep{episode_id:05d}.mp4")
        if not os.path.exists(video_path):
            logger.warning("视频文件不存在: %s，episode %d 将跳过图像输入", video_path, episode_id)

        script_memory: list[str] = []
        records: list[dict[str, Any]] = []

        logger.info(
            "开始处理 episode %d（%d 帧，%d 个子任务）",
            episode_id,
            len(frames),
            len(global_subtasks),
        )

        for frame_data in frames:
            frame_number: int = frame_data["frame"]
            subtask: Optional[str] = frame_data.get("subtask")

            # 跳过缺少 subtask 标注的帧
            if subtask is None:
                logger.warning(
                    "episode %d frame %d 缺少 subtask 标注，跳过", episode_id, frame_number
                )
                continue

            # 获取真值动作（列表为 1-indexed，对应 frame_number）
            action_idx = frame_number - 1
            if action_idx >= len(gt_actions):
                logger.warning(
                    "episode %d frame %d 超出 gt_actions 范围（长度 %d），跳过",
                    episode_id,
                    frame_number,
                    len(gt_actions),
                )
                continue

            action_code: int = gt_actions[action_idx]
            next_action: str = ACTION_MAP.get(action_code, f"UNKNOWN_{action_code}")

            # 当前子任务在全局列表中的索引（之前的均已完成）
            try:
                completed_index = global_subtasks.index(subtask)
            except ValueError:
                completed_index = 0

            # 提取视频帧
            image_b64: Optional[str] = None
            if os.path.exists(video_path):
                image_b64 = self.extract_frame(video_path, frame_number)
                if image_b64 is None:
                    logger.warning(
                        "episode %d frame %d 图像提取失败", episode_id, frame_number
                    )

            # 构建 prompt
            prompt = self.build_prompt(
                frame_data=frame_data,
                script_memory=script_memory,
                global_instruction=instruction,
                global_subtasks=global_subtasks,
                completed_index=completed_index,
                next_action=next_action,
            )

            # 调用 VLM API
            vlm_response = self.call_vlm_api(prompt, image_b64)
            if vlm_response is None:
                logger.error(
                    "episode %d frame %d VLM 调用失败，跳过", episode_id, frame_number
                )
                # 即使 API 失败也更新记忆，保证历史真值不断链
                self.update_script_memory(script_memory, frame_number, next_action)
                continue

            # 构建用户侧 content（含图像 + 文本）
            user_content: list[dict[str, Any]]
            if image_b64 is not None:
                user_content = [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                    },
                    {"type": "text", "text": prompt},
                ]
            else:
                user_content = [{"type": "text", "text": prompt}]

            record: dict[str, Any] = {
                "episode_id": episode_id,
                "frame": frame_number,
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": vlm_response},
                ],
            }
            records.append(record)

            # 更新真值历史记忆（必须在记录写入后进行，避免当前步污染自身 prompt）
            self.update_script_memory(script_memory, frame_number, next_action)

        logger.info(
            "episode %d 处理完成，生成 %d 条记录（跳过 %d 帧）",
            episode_id,
            len(records),
            len(frames) - len(records),
        )
        return records

    def run(
        self,
        input_json: str,
        gt_json: str,
        video_dir: str,
        output_jsonl: str,
        episode_ids: Optional[list[int]] = None,
    ) -> None:
        """
        主入口：遍历所有（或指定）episodes，合成 CoT 数据并写入 JSONL 文件。

        Args:
            input_json:   processed.json 文件路径。
            gt_json:      train_gt.json 文件路径。
            video_dir:    视频文件所在目录。
            output_jsonl: 输出 JSONL 文件路径。
            episode_ids:  若指定，则只处理列表中的 episode_id；为 ``None`` 时处理全部。
        """
        logger.info("加载标注数据: %s", input_json)
        with open(input_json, encoding="utf-8") as f:
            episodes: list[dict[str, Any]] = json.load(f)

        logger.info("加载真值动作数据: %s", gt_json)
        with open(gt_json, encoding="utf-8") as f:
            gt_data: dict[str, Any] = json.load(f)

        if episode_ids is not None:
            episodes = [ep for ep in episodes if ep["episode_id"] in episode_ids]
            logger.info("过滤后处理 %d 个 episode: %s", len(episodes), episode_ids)
        else:
            logger.info("共处理 %d 个 episode", len(episodes))

        total_records = 0
        with open(output_jsonl, "w", encoding="utf-8") as out_f:
            for ep_idx, episode in enumerate(episodes, start=1):
                episode_id = episode["episode_id"]
                ep_key = str(episode_id)

                if ep_key not in gt_data:
                    logger.warning(
                        "episode %d 在 train_gt.json 中未找到，跳过", episode_id
                    )
                    continue

                gt_actions: list[int] = gt_data[ep_key]["actions"]

                logger.info(
                    "[%d/%d] 处理 episode %d ...", ep_idx, len(episodes), episode_id
                )

                records = self.process_trajectory(episode, gt_actions, video_dir)

                for record in records:
                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

                total_records += len(records)

        logger.info(
            "全部处理完成！共生成 %d 条 CoT 记录，输出文件: %s",
            total_records,
            output_jsonl,
        )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="VLN CoT 数据合成脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="gengshuang_1_H.264_0324_processed.json",
        help="processed.json 文件路径",
    )
    parser.add_argument(
        "--gt",
        default="/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json",
        help="train_gt.json 文件路径",
    )
    parser.add_argument(
        "--video-dir",
        default="/home/gs/my_test/vln_dataset/vln_ce_video",
        help="视频文件目录（包含 ep00001.mp4 等文件）",
    )
    parser.add_argument(
        "--output",
        default="output_cot.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="VLM API 最大重试次数",
    )
    parser.add_argument(
        "--episode-ids",
        type=int,
        nargs="+",
        default=None,
        help="只处理指定 episode_id（空格分隔），不指定则处理全部",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主函数。"""
    args = parse_args()
    synthesizer = VLNDataSynthesizer(max_retries=args.max_retries)
    synthesizer.run(
        input_json=args.input,
        gt_json=args.gt,
        video_dir=args.video_dir,
        output_jsonl=args.output,
        episode_ids=args.episode_ids,
    )


if __name__ == "__main__":
    main()
