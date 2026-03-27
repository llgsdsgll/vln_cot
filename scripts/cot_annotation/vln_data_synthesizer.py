#!/usr/bin/env python3
"""
VLN CoT 数据合成脚本

功能：
  读取人工标注后的 processed.json（含 subtask/object/bbox）和 train_gt.json（含真值动作序列），
  结合视频帧，调用本地部署的 Qwen VLM（OpenAI 兼容接口）生成 Chain-of-Thought 推理文本，
  输出为 OpenAI/HuggingFace 标准的对话微调格式（JSONL）。

使用方法：
  python3 scripts/cot_annotation/vln_data_synthesizer.py \\
      --input   data/processed/gengshuang_1_H.264_0324_processed.json \\
      --gt      /home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json \\
      --video-dir /home/gs/my_test/vln_dataset/vln_ce_video \\
      --output  data/processed/output_cot.jsonl

  使用阿里云百炼成品 API（示例）：
  python3 scripts/cot_annotation/vln_data_synthesizer.py \\
      --api-provider dashscope \\
      --model qwen3.5-plus \\
      --thinking-mode on

  可选参数：
      --api-provider  后端类型（local / dashscope）
      --api-base-url  API Base URL；不传时按 provider 使用默认值
      --model         模型名称；不传时按 provider 使用默认值
      --api-key-env   成品 API 的密钥环境变量名（默认 DASHSCOPE_API_KEY）
      --thinking-mode thinking 模式（auto / on / off）
      --max-retries   VLM API 最大重试次数（默认 3）
      --invalid-frame-policy 无效输出的处理策略（默认 skip）
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
from pathlib import Path
from typing import Any, Optional

import cv2
import openai
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

# ---------------------------------------------------------------------------
# Prompt 模板 —— 抽离为全局变量，方便随时修改
# ---------------------------------------------------------------------------

# 有目标物体可见时使用
PROMPT_TEMPLATE = """\
You are generating teacher Chain-of-Thought (CoT) data for a Vision-Language Navigation (VLN) agent.

Return exactly one compact JSON object and nothing else.
Required keys:
- step1_task_progress
- step2_spatial_perception
- step3_decision_logic
- step4_memory_update

Rules:
1. Use only the provided ground-truth state and the current image. Do not hallucinate.
2. Keep each text field concise: 1-2 sentences only.
3. Summarize the trajectory memory instead of copying every past step verbatim.
4. Write actual reasoning content, not meta-instructions or copied placeholders.
5. `step2_spatial_perception` should describe the target's relative screen position and visible appearance; the exact object name and bbox will be inserted programmatically.
6. `step3_decision_logic` must explain why the action '{next_action}' is correct right now.
7. The final action label is already known, so do not add any extra action field beyond the four required reasoning keys.
8. Do not output Markdown, code fences, comments, or any extra keys.

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
Based on the image and the ground-truth state, return a compact JSON object such as:
{{
  "step1_task_progress": "<actual concise reasoning here>",
  "step2_spatial_perception": "<actual concise reasoning here>",
  "step3_decision_logic": "<actual concise reasoning here>",
  "step4_memory_update": "<actual concise reasoning here>"
}}\
"""

# 目标物体不可见时使用
PROMPT_TEMPLATE_NO_TARGET = """\
You are generating teacher Chain-of-Thought (CoT) data for a Vision-Language Navigation (VLN) agent.

Return exactly one compact JSON object and nothing else.
Required keys:
- step1_task_progress
- step2_spatial_perception
- step3_decision_logic
- step4_memory_update

Rules:
1. Use only the provided ground-truth state and the current image. Do not hallucinate.
2. Keep each text field concise: 1-2 sentences only.
3. Summarize the trajectory memory instead of copying every past step verbatim.
4. Write actual reasoning content, not meta-instructions or copied placeholders.
5. `step2_spatial_perception` must explicitly state that the target is not visible in the current frame and must not invent a bounding box.
6. `step3_decision_logic` must explain why the action '{next_action}' is correct right now.
7. The final action label is already known, so do not add any extra action field beyond the four required reasoning keys.
8. Do not output Markdown, code fences, comments, or any extra keys.

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
Based on the image and the ground-truth state, return a compact JSON object such as:
{{
  "step1_task_progress": "<actual concise reasoning here>",
  "step2_spatial_perception": "<actual concise reasoning here>",
  "step3_decision_logic": "<actual concise reasoning here>",
  "step4_memory_update": "<actual concise reasoning here>"
}}\
"""

SYSTEM_PROMPT = """\
You are a careful VLN data generator.
Return only one compact JSON object with the requested keys.
Every field must be faithful to the provided ground-truth state.
Do not output Markdown or any explanatory text.
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
LOCAL_API_BASE_URL = "http://localhost:8000/v1"
LOCAL_MODEL_NAME = "/mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8"
DASHSCOPE_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL_NAME = "qwen3.5-plus"
API_PROVIDER_CHOICES = ("local", "dashscope")
THINKING_MODE_CHOICES = ("auto", "on", "off")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_INPUT_JSON = PROCESSED_DATA_DIR / "gengshuang_1_H.264_0324_processed.json"
DEFAULT_OUTPUT_JSONL = PROCESSED_DATA_DIR / "output_cot.jsonl"


def collapse_whitespace(text: str) -> str:
    """将连续空白压缩为单个空格。"""
    return re.sub(r"\s+", " ", text).strip()


def strip_leading_label(text: str, label: str) -> str:
    """移除模型偶尔重复输出的字段标签。"""
    pattern = rf"^(?:step\s*\d+\s*[:.-]?\s*)?(?:{re.escape(label)}\s*[:.-]?\s*)?"
    return re.sub(pattern, "", text, flags=re.IGNORECASE).strip()


class StructuredTeacherTrace(BaseModel):
    """教师模型的结构化思维链输出。"""

    model_config = ConfigDict(extra="forbid")

    step1_task_progress: str = Field(...)
    step2_spatial_perception: str = Field(...)
    step3_decision_logic: str = Field(...)
    step4_memory_update: str = Field(...)

    @field_validator(
        "step1_task_progress",
        "step2_spatial_perception",
        "step3_decision_logic",
        "step4_memory_update",
    )
    @classmethod
    def validate_text_field(cls, value: str, info: ValidationInfo) -> str:
        text = collapse_whitespace(value)
        text = strip_leading_label(text, info.field_name.replace("_", " "))
        if not text:
            raise ValueError("must not be empty")
        if len(text) > MAX_STEP_CHARS:
            raise ValueError(f"must be <= {MAX_STEP_CHARS} chars")
        return text

    @model_validator(mode="after")
    def validate_semantics(self, info: ValidationInfo) -> "StructuredTeacherTrace":
        context = info.context or {}
        target_visible = context.get("target_visible", False)
        action_mentions = context.get("action_mentions", ())

        total_len = (
            len(self.step1_task_progress)
            + len(self.step2_spatial_perception)
            + len(self.step3_decision_logic)
            + len(self.step4_memory_update)
        )
        if total_len > MAX_TOTAL_CHARS:
            raise ValueError(f"reasoning text too long: {total_len} > {MAX_TOTAL_CHARS}")

        if target_visible:
            step2_lower = self.step2_spatial_perception.lower()
            if any(phrase in step2_lower for phrase in ("not visible", "not in view", "unseen")):
                raise ValueError("step2 should describe a visible target, not say it is unseen")
        else:
            step2_lower = self.step2_spatial_perception.lower()
            if not any(phrase in step2_lower for phrase in ("not visible", "not in view", "unseen")):
                raise ValueError("step2 must say the target is not visible")

        if action_mentions and not any(
            mention in self.step3_decision_logic.lower() for mention in action_mentions
        ):
            raise ValueError("step3 must mention the chosen action")

        return self


STRUCTURED_TRACE_JSON_SCHEMA = StructuredTeacherTrace.model_json_schema()

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
        api_provider: str = "local",
        api_base_url: Optional[str] = None,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        api_key_env: str = "DASHSCOPE_API_KEY",
        thinking_mode: str = "off",
        invalid_frame_policy: str = "skip",
    ) -> None:
        """
        初始化合成器。

        Args:
            max_retries:   VLM API 调用失败时的最大重试次数。
            api_provider:  API 提供方类型。``local`` 表示自部署 OpenAI 兼容接口，
                           ``dashscope`` 表示阿里云百炼兼容接口。
            api_base_url:  OpenAI 兼容接口地址。不传时根据 provider 使用默认值。
            model_name:    模型名称。不传时根据 provider 使用默认值。
            api_key:       显式传入的 API Key。通常建议优先使用环境变量。
            api_key_env:   当 provider 需要鉴权时，用于读取 API Key 的环境变量名。
            thinking_mode: thinking 模式开关。``auto`` 保持后端默认行为，
                           ``on`` / ``off`` 根据 provider 使用对应参数控制。
            invalid_frame_policy:
                           当模型多次返回不合格结果时的处理方式。
                           ``"skip"`` 表示跳过该帧，``"template"`` 表示使用脚本兜底模板。
        """
        if api_provider not in API_PROVIDER_CHOICES:
            raise ValueError(f"unsupported api_provider: {api_provider}")
        if thinking_mode not in THINKING_MODE_CHOICES:
            raise ValueError(f"unsupported thinking_mode: {thinking_mode}")

        self.max_retries = max_retries
        self.api_provider = api_provider
        self.api_key_env = api_key_env
        self.thinking_mode = thinking_mode
        self.invalid_frame_policy = invalid_frame_policy
        self._api_base_url = self._resolve_api_base_url(api_provider, api_base_url)
        self._model_name = self._resolve_model_name(api_provider, model_name)
        resolved_api_key = self._resolve_api_key(
            api_provider=api_provider,
            api_key=api_key,
            api_key_env=api_key_env,
        )
        self._client = openai.OpenAI(
            api_key=resolved_api_key,
            base_url=self._api_base_url,
        )
        self._use_json_object_response_format = api_provider == "local"
        logger.info(
            "VLM 客户端初始化完成 | provider=%s | base_url=%s | model=%s | thinking_mode=%s | invalid_policy=%s",
            api_provider,
            self._api_base_url,
            self._model_name,
            thinking_mode,
            invalid_frame_policy,
        )

    # ------------------------------------------------------------------
    # 静态/工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_api_base_url(api_provider: str, api_base_url: Optional[str]) -> str:
        """根据 provider 解析默认 base_url。"""
        if api_base_url:
            return api_base_url
        if api_provider == "dashscope":
            return DASHSCOPE_API_BASE_URL
        return LOCAL_API_BASE_URL

    @staticmethod
    def _resolve_model_name(api_provider: str, model_name: Optional[str]) -> str:
        """根据 provider 解析默认模型名称。"""
        if model_name:
            return model_name
        if api_provider == "dashscope":
            return DASHSCOPE_MODEL_NAME
        return LOCAL_MODEL_NAME

    @staticmethod
    def _resolve_api_key(
        api_provider: str,
        api_key: Optional[str],
        api_key_env: str,
    ) -> str:
        """根据 provider 解析 API Key。"""
        if api_provider == "local":
            return "EMPTY"
        if api_key:
            return api_key
        env_value = os.getenv(api_key_env)
        if env_value:
            return env_value
        raise ValueError(
            f"provider={api_provider} 需要 API Key，请设置环境变量 {api_key_env} 或通过 --api-key 显式传入。"
        )

    def _build_request_extra_body(self) -> dict[str, Any]:
        """根据 provider 构造额外请求参数。"""
        if self.api_provider == "dashscope":
            extra_body: dict[str, Any] = {}
        else:
            extra_body = {"repetition_penalty": 1.01}
        if self.thinking_mode == "auto":
            return extra_body
        enable_thinking = self.thinking_mode == "on"
        if self.api_provider == "dashscope":
            extra_body["enable_thinking"] = enable_thinking
        else:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return extra_body

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

    def _extract_json_object(self, raw_text: str) -> str:
        """移除包装噪声并尽量提取出 JSON 对象主体。"""
        text = raw_text.replace("\r\n", "\n").replace("\r", "\n").strip()
        text = re.sub(r"```(?:json|markdown|md|text)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx:end_idx + 1]
        return text

    def _parse_structured_response(
        self,
        raw_text: str,
        frame_data: dict[str, Any],
        current_subtask: str,
        next_action: str,
    ) -> tuple[Optional[StructuredTeacherTrace], Optional[str]]:
        """
        解析并校验模型返回的结构化 JSON。

        Returns:
            (structured_trace, error_reason)
        """
        text = self._extract_json_object(raw_text)
        if not text:
            return None, "empty response"

        objects = frame_data.get("objects", [])
        context: dict[str, Any] = {
            "current_subtask": current_subtask,
            "next_action": next_action,
            "action_mentions": self._action_mentions(next_action),
            "target_visible": bool(objects),
        }
        if objects:
            obj = objects[0]
            context["target_name"] = obj["label"]
            context["bbox_str"] = str(self.normalize_bbox(obj["bbox"]))

        try:
            structured_trace = StructuredTeacherTrace.model_validate_json(
                text,
                context=context,
            )
        except ValidationError as exc:
            first_error = exc.errors(include_url=False)[0]
            return None, f"{first_error['loc']}: {first_error['msg']}"
        except json.JSONDecodeError as exc:
            return None, f"invalid json: {exc.msg}"

        return structured_trace, None

    @staticmethod
    def _render_structured_response(
        structured_trace: StructuredTeacherTrace,
        frame_data: dict[str, Any],
        next_action: str,
    ) -> str:
        """将通过校验的结构化教师输出渲染为最终 Markdown 模板。"""
        objects = frame_data.get("objects", [])
        if objects:
            obj = objects[0]
            target_name = obj["label"]
            bbox = VLNDataSynthesizer.normalize_bbox(obj["bbox"])
            step2 = (
                f"I see the '{target_name}' and it is located at the exact bounding box {bbox}. "
                f"{structured_trace.step2_spatial_perception}"
            )
        else:
            step2 = structured_trace.step2_spatial_perception

        return (
            "**Reasoning Process:**\n"
            f"Step 1: Task Progress. {structured_trace.step1_task_progress}\n"
            f"Step 2: Spatial Perception. {step2}\n"
            f"Step 3: Decision Logic. {structured_trace.step3_decision_logic}\n"
            f"Step 4: Memory Update. {structured_trace.step4_memory_update}\n\n"
            "**Final Action:**\n"
            f"{next_action}"
        )

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

        current_subtask, _ = self._build_current_task_state(
            frame_data=frame_data,
            global_instruction=global_instruction,
            global_subtasks=global_subtasks,
            completed_index=completed_index,
        )
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

                request_kwargs: dict[str, Any] = {
                    "model": self._model_name,
                    "messages": messages,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "max_tokens": 1200,
                    "extra_body": self._build_request_extra_body(),
                }
                if self._use_json_object_response_format:
                    request_kwargs["response_format"] = {"type": "json_object"}

                response = self._client.chat.completions.create(
                    **request_kwargs,
                )

                msg = response.choices[0].message
                raw_text = msg.content or ""

                structured_trace, format_error = self._parse_structured_response(
                    raw_text=raw_text,
                    frame_data=frame_data,
                    current_subtask=current_subtask,
                    next_action=next_action,
                )
                if structured_trace is not None:
                    return self._render_structured_response(
                        structured_trace,
                        frame_data,
                        next_action,
                    )

                saw_invalid_response = True
                logger.warning(
                    "VLM 输出格式不合格 (attempt %d/%d): %s | raw前200字符: %s",
                    attempt,
                    self.max_retries,
                    format_error,
                    self._extract_json_object(raw_text)[:200],
                )
                if format_error == "empty response" and self._use_json_object_response_format:
                    logger.warning(
                        "response_format=json_object 在当前多模态请求上返回空内容，后续改用纯提示词 JSON 模式。"
                    )
                    self._use_json_object_response_format = False
                if attempt < self.max_retries:
                    assistant_feedback = raw_text if raw_text.strip() else "{}"
                    messages = messages + [
                        {"role": "assistant", "content": assistant_feedback[:4000]},
                        {
                            "role": "user",
                            "content": (
                                "Your previous JSON failed validation. "
                                f"Validation error: {format_error}. "
                                "Return one corrected compact JSON object with exactly these keys: "
                                "step1_task_progress, step2_spatial_perception, "
                                "step3_decision_logic, step4_memory_update. "
                                "Do not include any extra keys. "
                                "Do not repeat prompt instructions or placeholder text. "
                                f"Step 3 must clearly justify the action '{next_action}'."
                            ),
                        },
                    ]
                    time.sleep(1)

            except openai.BadRequestError as exc:
                error_text = str(exc)
                if self._use_json_object_response_format and "response_format" in error_text:
                    logger.warning(
                        "服务端似乎不支持 response_format=json_object，后续退化为纯提示词 JSON 模式。"
                    )
                    self._use_json_object_response_format = False
                    continue
                wait = 2 ** attempt
                logger.warning("VLM 请求参数异常 (attempt %d/%d): %s — %ds 后重试", attempt, self.max_retries, exc, wait)
                if attempt < self.max_retries:
                    time.sleep(wait)
                    continue
                break

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
            if self.invalid_frame_policy == "template":
                logger.warning("多次生成后仍不合格，按策略改用模板兜底。")
                return self._build_template_response(
                    frame_data=frame_data,
                    script_memory=script_memory,
                    global_instruction=global_instruction,
                    global_subtasks=global_subtasks,
                    completed_index=completed_index,
                    next_action=next_action,
                )
            logger.warning("多次生成后仍不合格，按策略跳过该帧，不写入伪造思维链。")

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
        default=str(DEFAULT_INPUT_JSON),
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
        default=str(DEFAULT_OUTPUT_JSONL),
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--api-provider",
        choices=API_PROVIDER_CHOICES,
        default="local",
        help="API 提供方类型：local=自部署 OpenAI 兼容接口，dashscope=阿里云百炼兼容接口",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="OpenAI 兼容接口地址；不传时根据 --api-provider 自动选择默认值",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="模型名称；不传时 local 默认为自部署模型名，dashscope 默认为 qwen3.5-plus",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="显式传入 API Key；通常建议优先通过环境变量提供",
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="当 provider 需要鉴权时，读取 API Key 的环境变量名",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=THINKING_MODE_CHOICES,
        default="off",
        help="thinking 模式：auto=保持后端默认行为，on=开启，off=关闭",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="VLM API 最大重试次数",
    )
    parser.add_argument(
        "--invalid-frame-policy",
        choices=("skip", "template"),
        default="skip",
        help="当模型多次返回不合格结果时的处理策略：skip=跳过该帧，template=使用脚本模板兜底",
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
        api_provider=args.api_provider,
        api_base_url=args.api_base_url,
        model_name=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        thinking_mode=args.thinking_mode,
        invalid_frame_policy=args.invalid_frame_policy,
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
