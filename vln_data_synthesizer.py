#!/usr/bin/env python3
"""
VLN CoT 数据合成脚本

功能：
  读取人工标注后的 processed.json（含 subtask/object/bbox）和 train_gt.json（含真值动作序列），
  结合视频帧，调用本地部署的 Qwen VLM（OpenAI 兼容接口）生成 Chain-of-Thought 推理文本，
  输出为 OpenAI/HuggingFace 标准的对话微调格式（JSONL）。

使用方法：
  python3 vln_data_synthesizer.py \\
      --input   gengshuang_1_H.264_0324_processed.json \\
      --gt      /home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json \\
      --video-dir /home/gs/my_test/vln_dataset/vln_ce_video \\
      --output  output_cot.jsonl

  可选参数：
      --api-base-url  本地 VLM 服务地址（默认 http://localhost:8000/v1）
      --model         模型名称，与服务端部署名一致（默认 /mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8）
      --max-retries   VLM API 最大重试次数（默认 3）
      --episode-ids   只处理指定 episode（如 --episode-ids 1 6 7）

本地服务映射命令（参考）：
  ssh -fN -L 8000:localhost:8000 root@139.196.171.150 -p 6222
"""

import argparse
import base64
import json
import logging
import os
import re
import time
from typing import Any, Optional

import cv2
import openai

# ---------------------------------------------------------------------------
# Prompt 模板 —— 抽离为全局变量，方便随时修改
# ---------------------------------------------------------------------------

# 有目标物体可见时使用
PROMPT_TEMPLATE = """\
You are generating supervised Chain-of-Thought (CoT) data for a Vision-Language Navigation (VLN) agent.

Rules:
1. Use only the provided ground-truth state and the current image. Do not hallucinate.
2. Output must match the Markdown template exactly.
3. Keep each step concise: 1-2 sentences only.
4. Do not output XML, code fences, notes, or any extra text.
5. Under **Final Action:** output exactly the token "{next_action}" and nothing else.

[Current Ground-Truth State]:
- Global Instruction: "{global_instruction}"
- Pre-defined Sub-tasks: {global_subtasks_list}
- Previously Completed Sub-tasks: {completed_subtasks}
- Current Active Sub-task Index: {completed_index}
- Current Active Sub-task: "{current_subtask}"
- Historical Trajectory Memory: "{script_memory}"
- Target Landmark/Object in Current Frame: {target_object_name}
- Ground-Truth Target BBox (Normalized 0-1000): {bbox_norm}
- Ground-Truth Next Action: {next_action}

[Visual Input]:
<Image>

[Task Requirements]:
Based on the image and the ground-truth state, generate the reasoning text strictly following the Markdown format below. Do NOT output any other conversational text.

**Reasoning Process:**
Step 1: Task Progress. State what has been completed from the memory and name the current active sub-task "{current_subtask}".
Step 2: Spatial Perception. Explicitly say that you see the '{target_object_name}' and that it is located at the exact bounding box {bbox_norm}. Describe its relative screen position.
Step 3: Decision Logic. Explain why the action '{next_action}' is necessary right now using the visible target and the current sub-task.
Step 4: Memory Update. Provide one concise sentence summarizing this observation and decision.

**Final Action:**
{next_action}\
"""

# 目标物体不可见时使用
PROMPT_TEMPLATE_NO_TARGET = """\
You are generating supervised Chain-of-Thought (CoT) data for a Vision-Language Navigation (VLN) agent.

Rules:
1. Use only the provided ground-truth state and the current image. Do not hallucinate.
2. Output must match the Markdown template exactly.
3. Keep each step concise: 1-2 sentences only.
4. Do not output XML, code fences, notes, or any extra text.
5. Under **Final Action:** output exactly the token "{next_action}" and nothing else.

[Current Ground-Truth State]:
- Global Instruction: "{global_instruction}"
- Pre-defined Sub-tasks: {global_subtasks_list}
- Previously Completed Sub-tasks: {completed_subtasks}
- Current Active Sub-task Index: {completed_index}
- Current Active Sub-task: "{current_subtask}"
- Historical Trajectory Memory: "{script_memory}"
- Target Landmark/Object: (Not visible in current frame)
- Ground-Truth Next Action: {next_action}

[Visual Input]:
<Image>

[Task Requirements]:
Based on the image and the ground-truth state, generate the reasoning text strictly following the Markdown format below. Do NOT output any other conversational text.

**Reasoning Process:**
Step 1: Task Progress. State what has been completed from the memory and name the current active sub-task "{current_subtask}".
Step 2: Spatial Perception. State that the target object is currently not visible in the field of view. Do NOT hallucinate any bounding boxes.
Step 3: Decision Logic. Explain why exploring via the action '{next_action}' is the most reasonable choice.
Step 4: Memory Update. Provide one concise sentence stating the target was unseen and the exploration decision made.

**Final Action:**
{next_action}\
"""

SYSTEM_PROMPT = """\
You are a careful VLN data generator.
Return only the requested Markdown template.
Every field must be faithful to the provided ground-truth state.
The final action token must exactly match the requested action.
"""

# 动作编码映射
ACTION_MAP: dict[int, str] = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}

VALID_ACTIONS = set(ACTION_MAP.values())
MAX_STEP_CHARS = 400
MAX_TOTAL_CHARS = 1600

RESPONSE_PATTERN = re.compile(
    r"\*\*Reasoning Process:?\*\*\s*"
    r"Step 1:\s*Task Progress\.\s*(?P<step1>.*?)\s*"
    r"Step 2:\s*Spatial Perception\.\s*(?P<step2>.*?)\s*"
    r"Step 3:\s*Decision Logic\.\s*(?P<step3>.*?)\s*"
    r"Step 4:\s*Memory Update\.\s*(?P<step4>.*?)\s*"
    r"\*\*Final Action:?\*\*\s*(?P<action>[A-Za-z_ -]+)\s*$",
    re.DOTALL,
)

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

    def __init__(
        self,
        max_retries: int = 3,
        api_base_url: str = "http://localhost:8000/v1",
        model_name: str = "/mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8",
    ) -> None:
        """
        初始化合成器。

        Args:
            max_retries:   VLM API 调用失败时的最大重试次数。
            api_base_url:  本地 VLM 服务的 OpenAI 兼容接口地址。
                           默认指向 SSH 隧道映射的 8000 端口。
            model_name:    服务端部署的模型名称，需与 vLLM --served-model-name 一致。
                           可通过 GET /v1/models 确认实际名称。
        """
        self.max_retries = max_retries
        self._client = openai.OpenAI(
            api_key="EMPTY",        # vLLM 本地服务不校验 API key，填任意非空字符串
            base_url=api_base_url,
        )
        self._model_name = model_name
        logger.info(
            "VLM 客户端初始化完成 | base_url=%s | model=%s",
            api_base_url,
            model_name,
        )

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

    @staticmethod
    def _collapse_whitespace(text: str) -> str:
        """将连续空白压缩为单个空格。"""
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _normalize_action_token(action_text: str) -> Optional[str]:
        """
        将模型输出的动作字符串标准化为 ``MOVE_FORWARD`` 等 canonical token。

        仅接受大小写、空格、连字符差异；若存在额外解释文本，则返回 ``None``。
        """
        normalized = VLNDataSynthesizer._collapse_whitespace(action_text)
        normalized = normalized.replace("-", "_").replace(" ", "_").upper()
        return normalized if normalized in VALID_ACTIONS else None

    @staticmethod
    def _relative_position_from_bbox(bbox_norm: list[int]) -> str:
        """根据归一化 bbox 粗略描述目标在屏幕中的相对位置。"""
        x_center = (bbox_norm[0] + bbox_norm[2]) / 2
        y_center = (bbox_norm[1] + bbox_norm[3]) / 2

        if x_center < 333:
            horizontal = "left"
        elif x_center > 667:
            horizontal = "right"
        else:
            horizontal = "center"

        if y_center < 333:
            vertical = "upper"
        elif y_center > 667:
            vertical = "lower"
        else:
            vertical = "middle"

        if horizontal == "center" and vertical == "middle":
            return "center of the view"
        if horizontal == "center":
            return f"{vertical} center of the view"
        if vertical == "middle":
            return f"{horizontal} side of the view"
        return f"{vertical}-{horizontal} area of the view"

    @staticmethod
    def _action_phrase(action: str) -> str:
        """将动作 token 转为自然语言短语。"""
        return action.lower().replace("_", " ")

    @staticmethod
    def _action_mentions(action: str) -> tuple[str, ...]:
        """返回动作在文本中允许出现的几种写法。"""
        phrase = action.lower().replace("_", " ")
        return (
            action.lower(),
            phrase,
        )

    def _build_current_task_state(
        self,
        frame_data: dict[str, Any],
        global_instruction: str,
        global_subtasks: list[str],
        completed_index: int,
    ) -> tuple[str, str]:
        """返回当前活动子任务与已完成子任务的字符串表示。"""
        current_subtask = frame_data.get("subtask")
        if current_subtask is None:
            if 0 <= completed_index < len(global_subtasks):
                current_subtask = global_subtasks[completed_index]
            else:
                current_subtask = global_instruction

        completed_subtasks = global_subtasks[:completed_index]
        completed_subtasks_str = str(completed_subtasks) if completed_subtasks else "[]"
        return current_subtask, completed_subtasks_str

    def _normalize_model_output(self, raw_text: str) -> str:
        """移除常见包装噪声，尽量抽取出 Markdown 主体。"""
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"```(?:markdown|md|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(
            r"\[ANNOTATION_THINK_START\].*?\[ANNOTATION_THINK_END\]",
            "",
            text,
            flags=re.DOTALL,
        ).strip()

        start_idx = text.find("**Reasoning Process")
        if start_idx != -1:
            text = text[start_idx:].strip()

        return text

    def _validate_and_format_response(
        self,
        raw_text: str,
        frame_data: dict[str, Any],
        next_action: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """
        校验模型输出是否满足约束，并重建为规范化 Markdown。

        Returns:
            (formatted_response, error_reason)
        """
        text = self._normalize_model_output(raw_text)
        if not text:
            return None, "empty response"

        match = RESPONSE_PATTERN.match(text)
        if not match:
            return None, "markdown structure mismatch"

        step1 = self._collapse_whitespace(match.group("step1"))
        step2 = self._collapse_whitespace(match.group("step2"))
        step3 = self._collapse_whitespace(match.group("step3"))
        step4 = self._collapse_whitespace(match.group("step4"))
        action = self._normalize_action_token(match.group("action"))

        if not all([step1, step2, step3, step4]):
            return None, "empty reasoning step"

        if action is None:
            return None, "invalid action token"

        if action != next_action:
            return None, f"action mismatch: {action} != {next_action}"

        steps = [step1, step2, step3, step4]
        if any(len(step) > MAX_STEP_CHARS for step in steps):
            return None, "step too long"

        if sum(len(step) for step in steps) > MAX_TOTAL_CHARS:
            return None, "response too long"

        objects = frame_data.get("objects", [])
        if objects:
            obj = objects[0]
            target_name = obj["label"]
            bbox_norm = str(self.normalize_bbox(obj["bbox"]))
            if target_name.lower() not in step2.lower():
                return None, "target name missing in step 2"
            if bbox_norm not in step2:
                return None, "bbox missing in step 2"
        else:
            if "not visible" not in step2.lower():
                return None, "step 2 must mention target is not visible"

        if not any(mention in step3.lower() for mention in self._action_mentions(next_action)):
            return None, "step 3 does not mention the next action"

        formatted_response = (
            "**Reasoning Process:**\n"
            f"Step 1: Task Progress. {step1}\n"
            f"Step 2: Spatial Perception. {step2}\n"
            f"Step 3: Decision Logic. {step3}\n"
            f"Step 4: Memory Update. {step4}\n\n"
            "**Final Action:**\n"
            f"{next_action}"
        )
        return formatted_response, None

    def _build_template_response(
        self,
        frame_data: dict[str, Any],
        script_memory: list[str],
        global_instruction: str,
        global_subtasks: list[str],
        completed_index: int,
        next_action: str,
    ) -> str:
        """当模型多次返回不合格时，使用真值信息生成严格合规的兜底答案。"""
        current_subtask, completed_subtasks_str = self._build_current_task_state(
            frame_data=frame_data,
            global_instruction=global_instruction,
            global_subtasks=global_subtasks,
            completed_index=completed_index,
        )
        action_phrase = self._action_phrase(next_action)
        memory_clause = (
            f"The memory shows {len(script_memory)} completed action steps so far."
            if script_memory
            else "The memory is still empty at this step."
        )

        if completed_subtasks_str == "[]":
            step1 = (
                f"{memory_clause} No sub-tasks have been completed yet. "
                f"The current active sub-task is '{current_subtask}'."
            )
        else:
            step1 = (
                f"{memory_clause} The completed sub-tasks so far are {completed_subtasks_str}. "
                f"The current active sub-task is '{current_subtask}'."
            )

        objects = frame_data.get("objects", [])
        if objects:
            obj = objects[0]
            target_name = obj["label"]
            bbox_norm = self.normalize_bbox(obj["bbox"])
            bbox_str = str(bbox_norm)
            position = self._relative_position_from_bbox(bbox_norm)
            step2 = (
                f"I see the '{target_name}' at the exact bounding box {bbox_str}. "
                f"It is in the {position}."
            )
            if next_action == "MOVE_FORWARD":
                step3 = (
                    f"The target is already visible in front of the agent, so move forward "
                    f"is necessary to keep approaching it and continue the sub-task."
                )
            elif next_action == "TURN_LEFT":
                step3 = (
                    f"The target or path needs better leftward alignment, so turn left is "
                    f"the correct action to continue the sub-task."
                )
            elif next_action == "TURN_RIGHT":
                step3 = (
                    f"The target or path needs better rightward alignment, so turn right is "
                    f"the correct action to continue the sub-task."
                )
            else:
                step3 = (
                    f"The navigation goal for the current sub-task has been reached, so stop "
                    f"is the correct action now."
                )
            step4 = (
                f"This step observes the '{target_name}' at {bbox_str} and decides to "
                f"{action_phrase} for the current sub-task."
            )
        else:
            step2 = "The target object is not visible in the current field of view."
            if next_action == "STOP":
                step3 = (
                    "Even though the target is not visible, the trajectory indicates that "
                    "stopping is the correct final action at this point."
                )
            else:
                step3 = (
                    f"Because the target is not visible, {action_phrase} is the most "
                    f"reasonable exploration action to continue the current sub-task."
                )
            step4 = (
                f"This step notes that the target is unseen and chooses to {action_phrase} "
                f"to maintain progress."
            )

        return (
            "**Reasoning Process:**\n"
            f"Step 1: Task Progress. {step1}\n"
            f"Step 2: Spatial Perception. {step2}\n"
            f"Step 3: Decision Logic. {step3}\n"
            f"Step 4: Memory Update. {step4}\n\n"
            "**Final Action:**\n"
            f"{next_action}"
        )

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
        current_subtask, completed_subtasks_str = self._build_current_task_state(
            frame_data=frame_data,
            global_instruction=global_instruction,
            global_subtasks=global_subtasks,
            completed_index=completed_index,
        )

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
                completed_subtasks=completed_subtasks_str,
                completed_index=completed_index,
                current_subtask=current_subtask,
                script_memory=memory_str,
                target_object_name=target_name,
                bbox_norm=bbox_str,
                next_action=next_action,
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_TARGET.format(
                global_instruction=global_instruction,
                global_subtasks_list=subtasks_str,
                completed_subtasks=completed_subtasks_str,
                completed_index=completed_index,
                current_subtask=current_subtask,
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

    def call_vlm_api(
        self,
        prompt: str,
        image_b64: Optional[str],
        frame_data: dict[str, Any],
        script_memory: list[str],
        global_instruction: str,
        global_subtasks: list[str],
        completed_index: int,
        next_action: str,
    ) -> Optional[str]:
        # 构建 user message content：图像（若有）+ 文本
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

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]
        saw_invalid_response = False

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "调用 VLM API (attempt %d/%d) model=%s image=%s",
                    attempt,
                    self.max_retries,
                    self._model_name,
                    "有" if image_b64 else "无",
                )

                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=0.1,
                    top_p=0.9,
                    max_tokens=512,
                    extra_body={"repetition_penalty": 1.02},
                )

                msg = response.choices[0].message
                raw_text = msg.content or ""

                formatted_response, format_error = self._validate_and_format_response(
                    raw_text=raw_text,
                    frame_data=frame_data,
                    next_action=next_action,
                )
                if formatted_response is not None:
                    return formatted_response

                saw_invalid_response = True
                logger.warning(
                    "VLM 输出格式不合格 (attempt %d/%d): %s | raw前200字符: %s",
                    attempt,
                    self.max_retries,
                    format_error,
                    self._normalize_model_output(raw_text)[:200],
                )
                if attempt < self.max_retries:
                    time.sleep(1)

            except openai.APIConnectionError as exc:
                wait = 2 ** attempt
                logger.warning("VLM 连接失败 (attempt %d/%d): %s — %ds 后重试", attempt, self.max_retries, exc, wait)
                if attempt < self.max_retries:
                    time.sleep(wait)
                    continue
                break

            except Exception as exc:  # noqa: BLE001
                wait = 2 ** attempt
                logger.warning("VLM 未知异常 (attempt %d/%d): %s — %ds 后重试", attempt, self.max_retries, exc, wait)
                if attempt < self.max_retries: time.sleep(wait)
                else:
                    break

        if saw_invalid_response:
            logger.warning("多次生成后仍不合格，改用模板兜底以保证输出格式稳定。")
            return self._build_template_response(
                frame_data=frame_data,
                script_memory=script_memory,
                global_instruction=global_instruction,
                global_subtasks=global_subtasks,
                completed_index=completed_index,
                next_action=next_action,
            )

        return None

    def process_trajectory(
        self,
        episode_data: dict[str, Any],
        gt_actions: list[int],
        video_dir: str,
        out_f,
    ) -> int:
        episode_id: int = episode_data["episode_id"]
        instruction: str = episode_data["instruction"]
        frames: list[dict[str, Any]] = episode_data["frames"]

        global_subtasks: list[str] = list(
            dict.fromkeys(f["subtask"] for f in frames if f.get("subtask") is not None)
        )

        video_path = os.path.join(video_dir, f"ep{episode_id:05d}.mp4")
        # 帧图像保存目录：与输出 JSONL 同级的 frames/ 子目录
        frames_dir = os.path.join(
            os.path.dirname(os.path.abspath(out_f.name)), "frames"
        )
        os.makedirs(frames_dir, exist_ok=True)

        script_memory: list[str] = []
        success_count = 0

        for frame_data in frames:
            frame_number: int = frame_data["frame"]
            subtask: Optional[str] = frame_data.get("subtask")

            if subtask is None:
                continue
            action_idx = frame_number - 1
            if action_idx >= len(gt_actions):
                continue

            action_code: int = gt_actions[action_idx]
            next_action: str = ACTION_MAP.get(action_code, f"UNKNOWN_{action_code}")

            # 帧级兜底：任何未预期异常只跳过本帧，不崩溃整个进程
            try:
                try:
                    completed_index = global_subtasks.index(subtask)
                except ValueError:
                    completed_index = 0

                # 提取帧：base64 仅用于 API 调用，同时保存 JPEG 文件供 JSONL 引用
                image_b64: Optional[str] = None
                image_path: Optional[str] = None
                if os.path.exists(video_path):
                    image_b64 = self.extract_frame(video_path, frame_number)
                    if image_b64 is not None:
                        fname = f"ep{episode_id:05d}_frame{frame_number:04d}.jpg"
                        image_path = os.path.join(frames_dir, fname)
                        with open(image_path, "wb") as img_f:
                            img_f.write(base64.b64decode(image_b64))

                prompt = self.build_prompt(
                    frame_data=frame_data,
                    script_memory=script_memory,
                    global_instruction=instruction,
                    global_subtasks=global_subtasks,
                    completed_index=completed_index,
                    next_action=next_action,
                )

                formatted_response = self.call_vlm_api(
                    prompt=prompt,
                    image_b64=image_b64,
                    frame_data=frame_data,
                    script_memory=script_memory,
                    global_instruction=instruction,
                    global_subtasks=global_subtasks,
                    completed_index=completed_index,
                    next_action=next_action,
                )

                if formatted_response is None:
                    logger.error("episode %d frame %d VLM 调用失败，跳过", episode_id, frame_number)
                    continue

                # JSONL 中 user_content 只存路径引用，不内嵌 base64
                user_content: list[dict[str, Any]]
                if image_path is not None:
                    user_content = [
                        {"type": "image_path", "image_path": image_path},
                        {"type": "text", "text": prompt},
                    ]
                else:
                    user_content = [{"type": "text", "text": prompt}]

                record: dict[str, Any] = {
                    "episode_id": episode_id,
                    "frame": frame_number,
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": formatted_response},
                    ],
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                success_count += 1
                logger.info("  => 成功写入 Episode %d Frame %d", episode_id, frame_number)

            except Exception as exc:  # noqa: BLE001
                # 单帧任何异常均只跳过本帧，不中断整个 episode
                logger.error(
                    "episode %d frame %d 处理异常，跳过本帧: %s: %s",
                    episode_id, frame_number, type(exc).__name__, exc,
                )
            finally:
                # 无论成功或失败，始终用真值更新记忆，保证后续帧历史不断链
                self.update_script_memory(script_memory, frame_number, next_action)

        return success_count

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

                # 将 out_f 传给 process_trajectory，并获取成功条数
                records_count = self.process_trajectory(episode, gt_actions, video_dir, out_f)
                total_records += records_count

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
        "--api-base-url",
        default="http://localhost:8000/v1",
        help="本地 VLM 服务的 OpenAI 兼容接口地址（SSH 隧道映射后的地址）",
    )
    parser.add_argument(
        "--model",
        default="/mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8",
        help="服务端部署的模型名称，需与 vLLM --served-model-name 一致；"
             "可通过 curl http://localhost:8000/v1/models 确认",
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
    synthesizer = VLNDataSynthesizer(
        max_retries=args.max_retries,
        api_base_url=args.api_base_url,
        model_name=args.model,
    )
    synthesizer.run(
        input_json=args.input,
        gt_json=args.gt,
        video_dir=args.video_dir,
        output_jsonl=args.output,
        episode_ids=args.episode_ids,
    )


if __name__ == "__main__":
    main()
