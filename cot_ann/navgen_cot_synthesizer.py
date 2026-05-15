#!/usr/bin/env python3
"""
基于 NavGen step_task frame_info 的 VLN CoT 数据合成脚本。

特点：
  1. 直接读取 <step_task>.frame_info.json 与配套的 *_frame_assets/frames_rgb 图片。
  2. CoT 编写重点围绕当前 subtask 目标与下一个 subtask 目标。
  3. 当当前/下一个目标不可见时，向模型提供历史可见帧与后续动作序列，帮助推断可能方位。
  4. 输出格式与现有 CoT JSONL 保持兼容，便于继续用 jsonl_to_frame_markdown.py 检查。
"""

from __future__ import annotations

import argparse
import base64
import json
import re
from pathlib import Path
from typing import Any, Optional

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    ValidationInfo,
    field_validator,
    model_validator,
)

from vln_data_synthesizer import (
    API_PROVIDER_CHOICES,
    MAX_STEP_CHARS,
    MAX_TOTAL_CHARS,
    PROJECT_ROOT,
    THINKING_MODE_CHOICES,
    FrameGenerationResult,
    VLNDataSynthesizer,
    collapse_whitespace,
    logger,
    strip_leading_label,
)

NAVGEN_ACTION_MAP: dict[str, str] = {
    "stop": "STOP",
    "move_forward": "MOVE_FORWARD",
    "turn_left": "TURN_LEFT",
    "turn_right": "TURN_RIGHT",
}

DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "debug"
RECENT_HISTORY_WINDOW = 3

VIEW_TOKEN_TO_GRID: dict[str, tuple[int, int]] = {
    "top_left": (0, 0),
    "top_center": (1, 0),
    "top_right": (2, 0),
    "middle_left": (0, 1),
    "middle_center": (1, 1),
    "middle_right": (2, 1),
    "bottom_left": (0, 2),
    "bottom_center": (1, 2),
    "bottom_right": (2, 2),
}

DIRECTION_TEXT_BY_GRID: dict[tuple[int, int], str] = {
    (0, 0): "upper-left",
    (1, 0): "upper",
    (2, 0): "upper-right",
    (0, 1): "left",
    (1, 1): "center",
    (2, 1): "right",
    (0, 2): "lower-left",
    (1, 2): "lower",
    (2, 2): "lower-right",
}

DIRECTION_KEYWORDS = (
    "upper-left",
    "upper right",
    "upper-right",
    "lower-left",
    "lower right",
    "lower-right",
    "left edge",
    "right edge",
    "upper edge",
    "lower edge",
    "left",
    "right",
    "upper",
    "lower",
)

VAGUE_DIRECTION_PHRASES = (
    "ahead",
    "straight ahead",
    "just ahead",
    "along the current trajectory",
    "along the trajectory",
    "to the side",
    "off to the side",
    "forward-left",
    "forward right",
    "forward-right",
    "front-left",
    "front right",
    "front-right",
)

NAVGEN_PROMPT_TEMPLATE = """\
You are generating teacher Chain-of-Thought (CoT) data for a Vision-Language Navigation (VLN) agent.

Return exactly one compact JSON object and nothing else.
Required keys:
- step1_task_progress
- step2_spatial_perception
- step3_decision_logic
- step4_memory_update

Rules:
1. Use only the provided ground-truth state, historical hints, and current image. Do not hallucinate.
2. Keep each text field concise: 1-2 sentences only.
3. Treat the Historical Trajectory Memory as the memory anchor carried from the previous frame. Use it as immediate past context instead of listing all past actions.
4. Write actual reasoning content, not meta-instructions or copied placeholders.
5. `step1_task_progress` must state the current instruction progress: which predefined sub-tasks are already done, which sub-task the agent is currently on, and which step should be executed right now.
6. `step2_spatial_perception` must, by default, describe only the current sub-task target(s) first and the next sub-task target(s) second. For each visible focus target, mention its exact name, exact bbox in `[xmin, ymin, xmax, ymax]`, relative screen position, and approximate distance in meters.
7. If a current or next focus target is not visible, explicitly state that it is not visible. Then explicitly conclude its most likely current direction using a phrase such as `left`, `right`, `upper-left`, `lower-right`, `left edge`, or `right edge`. Put the direction conclusion before the distance estimate, for example: `not visible ... most likely toward the lower-left edge ... about 1.09m away`. Use the recent previous-3-frame trace first, then the older historical sighting summary, together with the executed actions and current distance estimate, to infer that direction. Do not invent a current bbox for an invisible target.
8. For invisible focus targets, prefer exactly one concrete screen-direction category or one edge variant. Avoid vague phrasing such as `ahead`, `to the side`, `along the trajectory`, or `forward-left`.
9. Do not mention other visible sub-task objects in `step2_spatial_perception` unless no usable historical sighting exists for the invisible current/next focus target and the supporting objects provide concrete orientation support.
10. `step3_decision_logic` must follow this pattern: sentence 1 must start with `The action '{next_action}' is correct because ...` and explain the current target first; sentence 2 is optional and may mention the next target or historical clue.
11. `step4_memory_update` must be exactly one concise sentence that reuses the task-progress conclusion from `step1_task_progress` and records the current active task plus the action '{next_action}', so it can be reused as the next frame's Historical Trajectory Memory.
12. The final action label is already known, so do not add any extra action field beyond the four required reasoning keys.
13. Do not output Markdown, code fences, comments, or any extra keys.

[Current Ground-Truth State]:
- Global Instruction: "{global_instruction}"
- Pre-defined Sub-tasks: {global_subtasks_list}
- Previously Completed Sub-tasks: {completed_subtasks}
- Current Active Sub-task Index: {completed_index}
- Current Active Sub-task: "{current_subtask}"
- Next Sub-task: "{next_subtask}"
- Historical Trajectory Memory: "{script_memory}"
- Current Sub-task Target(s): {current_targets}
- Next Sub-task Target(s): {next_targets}
- Focus Targets to Prioritize in Step2: {focus_targets}
- Visible Focus Targets in Current Frame (Normalized 0-1000): {visible_focus_targets}
- Current-Frame Focus Target Distance Estimates: {focus_target_distances}
- Recent Previous-3-Frame Focus Trace: {recent_focus_trace}
- Historical Hints for Invisible Focus Targets: {historical_focus_hints}
- Likely Current Direction Hints for Invisible Focus Targets: {likely_direction_hints}
- Optional Supporting Objects in Current Frame (Use Only If No Historical Sighting Exists): {other_visible_objects}
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


def normalize_bbox_xyxy(
    bbox_xyxy: list[int] | tuple[int, int, int, int],
    image_width: int,
    image_height: int,
) -> list[int]:
    """将像素坐标 bbox 归一化到 [0, 1000]。"""
    xmin, ymin, xmax, ymax = bbox_xyxy
    if image_width <= 1 or image_height <= 1:
        raise ValueError("image size must be > 1")

    return [
        max(0, min(1000, int(round(xmin * 1000 / (image_width - 1))))),
        max(0, min(1000, int(round(ymin * 1000 / (image_height - 1))))),
        max(0, min(1000, int(round(xmax * 1000 / (image_width - 1))))),
        max(0, min(1000, int(round(ymax * 1000 / (image_height - 1))))),
    ]


def format_distance_text(distance_m: Any) -> Optional[str]:
    try:
        value = float(distance_m)
    except (TypeError, ValueError):
        return None
    if value < 0:
        return None
    return f"{value:.2f}m"


def normalize_view_position_token(view_position: Optional[str]) -> Optional[str]:
    if not view_position:
        return None
    normalized = view_position.strip().lower().replace("-", "_")
    alias_map = {
        "left": "middle_left",
        "right": "middle_right",
        "center": "middle_center",
        "top": "top_center",
        "bottom": "bottom_center",
    }
    return alias_map.get(normalized, normalized)


def direction_text_from_grid(x: int, y: int) -> str:
    if (x, y) in DIRECTION_TEXT_BY_GRID:
        return DIRECTION_TEXT_BY_GRID[(x, y)]
    if x < 0 and y <= 0:
        return "upper-left edge"
    if x < 0 and y >= 2:
        return "lower-left edge"
    if x < 0:
        return "left edge"
    if x > 2 and y <= 0:
        return "upper-right edge"
    if x > 2 and y >= 2:
        return "lower-right edge"
    if x > 2:
        return "right edge"
    if y < 0:
        return "upper edge"
    if y > 2:
        return "lower edge"
    return "center"


def shift_grid_for_action(x: int, y: int, action: str) -> tuple[int, int]:
    if action == "TURN_LEFT":
        return x + 1, y
    if action == "TURN_RIGHT":
        return x - 1, y
    if action == "MOVE_FORWARD":
        if y < 2:
            return x, y + 1
        if x == 0:
            return -1, 2
        if x == 2:
            return 3, 2
        return 1, 3
    return x, y


class NavGenStructuredTeacherTrace(BaseModel):
    """NavGen 场景下的结构化教师 CoT。"""

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
    def validate_semantics(self, info: ValidationInfo) -> "NavGenStructuredTeacherTrace":
        context = info.context or {}
        action_mentions = context.get("action_mentions", ())
        required_visible_labels = context.get("required_visible_labels", ())
        required_visible_bboxes = context.get("required_visible_bbox_texts", ())
        require_focus_unseen_statement = context.get("require_focus_unseen_statement", False)
        expected_action_label = str(context.get("expected_action_label", "")).lower()
        require_directional_inference = context.get("require_directional_inference", False)
        unseen_phrases = (
            "not visible",
            "not in view",
            "outside the current view",
            "out of view",
            "not currently visible",
            "unseen",
        )

        total_len = (
            len(self.step1_task_progress)
            + len(self.step2_spatial_perception)
            + len(self.step3_decision_logic)
            + len(self.step4_memory_update)
        )
        if total_len > MAX_TOTAL_CHARS:
            raise ValueError(f"reasoning text too long: {total_len} > {MAX_TOTAL_CHARS}")

        step2_lower = self.step2_spatial_perception.lower()
        if required_visible_labels:
            missing_labels = [
                label for label in required_visible_labels
                if label not in step2_lower
            ]
            if missing_labels:
                raise ValueError("step2 must mention every visible current/next focus target")

            compact_step2 = "".join(self.step2_spatial_perception.split())
            missing_bboxes = [
                bbox for bbox in required_visible_bboxes
                if "".join(bbox.split()) not in compact_step2
            ]
            if missing_bboxes:
                raise ValueError("step2 must include every visible current/next focus target bbox")

        if require_focus_unseen_statement and not any(
            phrase in step2_lower for phrase in unseen_phrases
        ):
            raise ValueError("step2 must state that the current/next focus target is not visible")
        if require_directional_inference and not any(
            keyword in step2_lower for keyword in DIRECTION_KEYWORDS
        ):
            raise ValueError("step2 must infer a likely direction for the invisible target")
        if require_directional_inference and any(
            phrase in step2_lower for phrase in VAGUE_DIRECTION_PHRASES
        ):
            raise ValueError(
                "step2 must use concrete screen directions like lower-left or right edge, not vague forward/side phrasing"
            )

        step3_lower = self.step3_decision_logic.lower()
        if not step3_lower.startswith("the action"):
            raise ValueError("step3 must start with 'The action ...'")
        if "because" not in step3_lower:
            raise ValueError("step3 must contain 'because'")
        if action_mentions and not any(
            mention in self.step3_decision_logic.lower() for mention in action_mentions
        ):
            raise ValueError("step3 must mention the chosen action")
        if expected_action_label and expected_action_label not in step3_lower:
            raise ValueError("step3 must include the exact chosen action label")

        step3_sentence_endings = re.findall(r"[.!?](?:\s|$)", self.step3_decision_logic)
        if len(step3_sentence_endings) > 2:
            raise ValueError("step3 should be at most two concise sentences")

        if action_mentions and not any(
            mention in self.step4_memory_update.lower() for mention in action_mentions
        ):
            raise ValueError("step4 must mention the chosen action")

        sentence_endings = re.findall(r"[.!?](?:\s|$)", self.step4_memory_update)
        if len(sentence_endings) > 1:
            raise ValueError("step4 should be one concise sentence")

        return self


class NavGenFrameInfoSynthesizer(VLNDataSynthesizer):
    """面向 NavGen step_task frame_info 的 CoT 合成器。"""

    @staticmethod
    def _truncate_template_text(text: str, max_chars: int) -> str:
        text = collapse_whitespace(text)
        if len(text) <= max_chars:
            return text
        if max_chars <= 3:
            return text[:max_chars]
        return text[: max_chars - 3].rstrip(" ,;:.") + "..."

    @staticmethod
    def _template_target_prefix(roles: list[str]) -> str:
        has_current = "current" in roles
        has_next = "next" in roles
        if has_current and has_next:
            return "Cur/next"
        if has_current:
            return "Cur"
        if has_next:
            return "Next"
        return "Obj"

    @staticmethod
    def _compact_bbox_text(bbox_norm: Any) -> str:
        if not isinstance(bbox_norm, list) or len(bbox_norm) != 4:
            return "[]"
        return "[" + ",".join(str(int(value)) for value in bbox_norm) + "]"

    @staticmethod
    def _template_position_short(
        view_position: Optional[str],
        bbox_norm: Optional[list[int]],
    ) -> str:
        normalized = normalize_view_position_token(view_position)
        if normalized in VIEW_TOKEN_TO_GRID:
            x, y = VIEW_TOKEN_TO_GRID[normalized]
            return direction_text_from_grid(x, y)
        if isinstance(bbox_norm, list) and len(bbox_norm) == 4:
            xmin, ymin, xmax, ymax = bbox_norm
            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2
            grid_x = 0 if center_x < 333 else 1 if center_x < 667 else 2
            grid_y = 0 if center_y < 333 else 1 if center_y < 667 else 2
            return direction_text_from_grid(grid_x, grid_y)
        return "center"

    def _template_direction_from_status(self, status: dict[str, Any]) -> str:
        direction_hint = collapse_whitespace(str(status.get("direction_hint") or ""))
        if direction_hint:
            return direction_hint
        return "lower"

    def _format_template_focus_status(
        self,
        status: dict[str, Any],
        *,
        include_distance: bool = True,
        separator: str = ", ",
    ) -> str:
        prefix = self._template_target_prefix(list(status.get("roles") or []))
        label = collapse_whitespace(str(status.get("label") or "object")) or "object"
        distance_text = collapse_whitespace(str(status.get("distance_text") or ""))
        if status.get("visible"):
            bbox_text = self._compact_bbox_text(status.get("bbox_norm"))
            position = collapse_whitespace(str(status.get("position_short") or "center"))
            parts = [f"{prefix} '{label}' {bbox_text}", position]
            if include_distance and distance_text:
                parts.append(distance_text)
            return separator.join(parts) + "."

        direction_text = self._template_direction_from_status(status)
        parts = [f"{prefix} '{label}' unseen", direction_text]
        if include_distance and distance_text:
            parts.append(distance_text)
        return separator.join(parts) + "."

    def _format_template_focus_status_compact(self, status: dict[str, Any]) -> str:
        return self._format_template_focus_status(
            status,
            include_distance=False,
            separator=",",
        )

    def _format_template_focus_status_minimal(self, status: dict[str, Any]) -> str:
        label = collapse_whitespace(str(status.get("label") or "object")) or "object"
        if status.get("visible"):
            bbox_text = self._compact_bbox_text(status.get("bbox_norm"))
            position = collapse_whitespace(str(status.get("position_short") or "center"))
            return f"{label} {bbox_text},{position}."
        direction_text = self._template_direction_from_status(status)
        return f"{label} unseen,{direction_text}."

    @staticmethod
    def _template_focus_label_text(
        labels: list[str],
        *,
        fallback_text: str,
        max_chars: int,
    ) -> str:
        cleaned = [collapse_whitespace(label) for label in labels if collapse_whitespace(label)]
        if not cleaned:
            return fallback_text
        first = cleaned[0]
        if len(cleaned) == 1:
            return NavGenFrameInfoSynthesizer._truncate_template_text(first, max_chars)
        remainder = len(cleaned) - 1
        compact = f"{first} +{remainder}"
        return NavGenFrameInfoSynthesizer._truncate_template_text(compact, max_chars)

    @staticmethod
    def _select_template_priority_statuses(
        statuses: list[dict[str, Any]],
        max_items: int = 2,
    ) -> list[dict[str, Any]]:
        if max_items <= 0:
            return []
        selected: list[dict[str, Any]] = []
        for role in ("current", "next"):
            for status in statuses:
                if status in selected:
                    continue
                if role in list(status.get("roles") or []):
                    selected.append(status)
                    break
        for status in statuses:
            if len(selected) >= max_items:
                break
            if status not in selected:
                selected.append(status)
        return selected[:max_items]

    @staticmethod
    def _normalize_navgen_action(action: str) -> Optional[str]:
        token = collapse_whitespace(action).lower()
        return NAVGEN_ACTION_MAP.get(token)

    @staticmethod
    def _resolve_step_task_json(frame_info_path: Path, step_task_json: Optional[str]) -> Path:
        if step_task_json:
            return Path(step_task_json).expanduser().resolve()
        if not frame_info_path.name.endswith(".frame_info.json"):
            raise ValueError(
                "--frame-info 必须指向 *.frame_info.json，或显式传入 --step-task-json。"
            )
        inferred = frame_info_path.with_name(
            frame_info_path.name.removesuffix(".frame_info.json") + ".json"
        )
        return inferred.resolve()

    @staticmethod
    def _infer_item_id_from_strings(*candidates: str) -> int:
        for raw in candidates:
            for candidate in Path(raw).parts[::-1]:
                match = re.fullmatch(r"item_(\d+)", candidate)
                if match:
                    return int(match.group(1))
        return 0

    @classmethod
    def _infer_item_id_from_path(
        cls,
        frame_info_path: Path,
        step_task_data: dict[str, Any],
        frame_info_meta: dict[str, Any],
    ) -> int:
        search_values = [
            str(frame_info_path),
            str(frame_info_meta.get("step_task_json") or ""),
            str(frame_info_meta.get("source_task_json") or ""),
            str(step_task_data.get("trajectory path") or ""),
        ]
        inferred = cls._infer_item_id_from_strings(*search_values)
        if inferred:
            return inferred
        for candidate in (frame_info_path.parent.name, frame_info_path.parent.parent.name, frame_info_path.parent.parent.parent.name):
            match = re.fullmatch(r"item_(\d+)", candidate)
            if match:
                return int(match.group(1))
        return 0

    @staticmethod
    def _pick_all_subtask_states(frame: dict[str, Any]) -> list[dict[str, Any]]:
        candidates = (
            frame.get("all_subtask_target_object_states"),
            frame.get("target_object_states"),
            frame.get("current_subtask_target_object_states"),
        )
        for candidate in candidates:
            if isinstance(candidate, list):
                return [item for item in candidate if isinstance(item, dict)]
        return []

    @staticmethod
    def _target_key(obj: dict[str, Any]) -> str:
        object_id = collapse_whitespace(str(obj.get("object_id") or ""))
        if object_id:
            return object_id
        return collapse_whitespace(str(obj.get("category") or "object"))

    @staticmethod
    def _role_text(roles: list[str]) -> str:
        ordered_roles = [role for role in ("current", "next") if role in roles]
        if not ordered_roles:
            return "other"
        if len(ordered_roles) == 1:
            return ordered_roles[0]
        return "current and next"

    @staticmethod
    def _view_position_to_text(view_position: Optional[str], bbox_norm: list[int]) -> str:
        mapping = {
            "left": "left side of the view",
            "right": "right side of the view",
            "center": "center of the view",
            "top": "upper side of the view",
            "bottom": "lower side of the view",
            "top_left": "upper-left area of the view",
            "top_center": "upper-center area of the view",
            "top_right": "upper-right area of the view",
            "middle_left": "middle-left area of the view",
            "middle_center": "center of the view",
            "middle_right": "middle-right area of the view",
            "bottom_left": "lower-left area of the view",
            "bottom_center": "lower-center area of the view",
            "bottom_right": "lower-right area of the view",
        }
        if view_position:
            normalized = view_position.strip().lower().replace("-", "_")
            if normalized in mapping:
                return mapping[normalized]
        return VLNDataSynthesizer._relative_position_from_bbox(bbox_norm)

    def _state_to_prompt_object(
        self,
        state: dict[str, Any],
        roles: list[str],
        image_width: int,
        image_height: int,
    ) -> Optional[dict[str, Any]]:
        if not state.get("visible_in_current_view"):
            return None
        bbox_xyxy = state.get("bbox_xyxy")
        if not isinstance(bbox_xyxy, list) or len(bbox_xyxy) != 4:
            return None

        label = collapse_whitespace(str(state.get("category") or "object")) or "object"
        bbox_norm = normalize_bbox_xyxy(
            bbox_xyxy=bbox_xyxy,
            image_width=image_width,
            image_height=image_height,
        )
        return {
            "key": self._target_key(state),
            "label": label,
            "bbox_norm": bbox_norm,
            "position": self._view_position_to_text(state.get("view_position_3x3"), bbox_norm),
            "distance_text": format_distance_text(state.get("distance_m")),
            "roles": list(roles),
            "role_text": self._role_text(list(roles)),
        }

    def _format_prompt_objects(self, objects: list[dict[str, Any]]) -> str:
        if not objects:
            return "[]"

        summaries: list[str] = []
        reference_label: Optional[str] = None
        reference_bbox: Optional[list[int]] = None

        for index, obj in enumerate(objects):
            summary = (
                f"[{obj['role_text']}] '{obj['label']}' at {obj['bbox_norm']} "
                f"({obj['position']})"
            )
            if obj.get("distance_text"):
                summary += f", about {obj['distance_text']} away"
            if index > 0 and reference_label is not None and reference_bbox is not None:
                relation = self._relative_relation_between_bboxes(reference_bbox, obj["bbox_norm"])
                if relation == "near":
                    summary += f", near '{reference_label}'"
                else:
                    summary += f", {relation} '{reference_label}'"

            summaries.append(summary)
            if index == 0:
                reference_label = obj["label"]
                reference_bbox = obj["bbox_norm"]

        return "; ".join(summaries)

    def _build_focus_targets(
        self,
        subtasks: list[dict[str, Any]],
        current_subtask_index: int,
    ) -> tuple[list[dict[str, Any]], str, str, str]:
        focus_targets: dict[str, dict[str, Any]] = {}

        def register_targets(role: str, subtask_index: int) -> None:
            if subtask_index < 0 or subtask_index >= len(subtasks):
                return
            subtask = subtasks[subtask_index]
            for anchor in subtask.get("anchor_instances", []):
                if not isinstance(anchor, dict):
                    continue
                key = self._target_key(anchor)
                descriptor = focus_targets.setdefault(
                    key,
                    {
                        "key": key,
                        "object_id": collapse_whitespace(str(anchor.get("object_id") or "")),
                        "category": collapse_whitespace(str(anchor.get("category") or "object")) or "object",
                        "roles": [],
                        "role_to_instruction": {},
                        "role_to_subtask_index": {},
                    },
                )
                if role not in descriptor["roles"]:
                    descriptor["roles"].append(role)
                descriptor["role_to_instruction"][role] = collapse_whitespace(
                    str(subtask.get("instruction") or "")
                )
                descriptor["role_to_subtask_index"][role] = subtask_index

        register_targets("current", current_subtask_index)
        register_targets("next", current_subtask_index + 1)

        ordered_targets = sorted(
            focus_targets.values(),
            key=lambda item: (
                0 if "current" in item["roles"] else 1,
                item["role_to_subtask_index"].get("current", item["role_to_subtask_index"].get("next", 10**9)),
                item["category"],
            ),
        )

        current_targets = [
            f"'{item['category']}'"
            for item in ordered_targets
            if "current" in item["roles"]
        ]
        next_targets = [
            f"'{item['category']}'"
            for item in ordered_targets
            if "next" in item["roles"]
        ]
        next_subtask = (
            collapse_whitespace(str(subtasks[current_subtask_index + 1].get("instruction") or ""))
            if current_subtask_index + 1 < len(subtasks)
            else "None"
        )

        return (
            ordered_targets,
            ", ".join(current_targets) if current_targets else "[]",
            ", ".join(next_targets) if next_targets else "[]",
            next_subtask,
        )

    def _find_state_in_frame(
        self,
        frame: dict[str, Any],
        target_key: str,
    ) -> Optional[dict[str, Any]]:
        for state in self._pick_all_subtask_states(frame):
            if self._target_key(state) == target_key:
                return state
        return None

    def _find_last_visible_state(
        self,
        frames: list[dict[str, Any]],
        current_frame_pos: int,
        target_key: str,
    ) -> tuple[Optional[int], Optional[dict[str, Any]]]:
        for previous_pos in range(current_frame_pos - 1, -1, -1):
            for state in self._pick_all_subtask_states(frames[previous_pos]):
                if self._target_key(state) != target_key:
                    continue
                if state.get("visible_in_current_view") and state.get("bbox_xyxy"):
                    return previous_pos, state
        return None, None

    def _infer_likely_direction_from_history(
        self,
        target_key: str,
        frames: list[dict[str, Any]],
        current_frame_pos: int,
        fallback_state: Optional[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> Optional[str]:
        del image_width, image_height
        last_pos, last_state = self._find_last_visible_state(
            frames=frames,
            current_frame_pos=current_frame_pos,
            target_key=target_key,
        )
        state_for_direction = last_state or fallback_state
        if state_for_direction is None:
            return None

        view_token = normalize_view_position_token(
            state_for_direction.get("view_position_3x3")
        )
        if view_token is None or view_token not in VIEW_TOKEN_TO_GRID:
            return None

        x, y = VIEW_TOKEN_TO_GRID[view_token]
        if last_pos is None:
            last_pos = current_frame_pos
        for pos in range(last_pos + 1, current_frame_pos):
            action = self._normalize_navgen_action(str(frames[pos].get("next_action") or ""))
            if action is None:
                continue
            x, y = shift_grid_for_action(x, y, action)

        if last_pos == current_frame_pos - 1:
            if x == 0:
                x = -1
            elif x == 2:
                x = 3
            elif y == 0:
                y = -1
            elif y == 2:
                y = 3

        return direction_text_from_grid(x, y)

    def _build_history_hint(
        self,
        target: dict[str, Any],
        frames: list[dict[str, Any]],
        current_frame_pos: int,
        image_width: int,
        image_height: int,
    ) -> str:
        last_pos, last_state = self._find_last_visible_state(
            frames=frames,
            current_frame_pos=current_frame_pos,
            target_key=target["key"],
        )
        current_state = self._find_state_in_frame(frames[current_frame_pos], target["key"])
        role_text = self._role_text(target["roles"])
        label = target["category"]
        current_distance_text = format_distance_text(
            current_state.get("distance_m") if current_state else None
        )
        likely_direction = self._infer_likely_direction_from_history(
            target_key=target["key"],
            frames=frames,
            current_frame_pos=current_frame_pos,
            fallback_state=current_state,
            image_width=image_width,
            image_height=image_height,
        )

        if last_pos is None or last_state is None:
            direction_clause = (
                f" most likely toward the {likely_direction} of the view,"
                if likely_direction
                else ""
            )
            current_distance_clause = (
                f" current estimated distance is about {current_distance_text},"
                if current_distance_text
                else ""
            )
            return (
                f"[{role_text}] '{label}' has not been visible in earlier frames,"
                f"{direction_clause}{current_distance_clause} so infer its likely side only from the recent action trend."
            )

        bbox_norm = normalize_bbox_xyxy(
            bbox_xyxy=last_state["bbox_xyxy"],
            image_width=image_width,
            image_height=image_height,
        )
        position_text = self._view_position_to_text(last_state.get("view_position_3x3"), bbox_norm)
        action_tokens = [
            action
            for action in (
                self._normalize_navgen_action(str(frames[pos].get("next_action") or ""))
                for pos in range(last_pos + 1, current_frame_pos)
            )
            if action is not None
        ]
        action_text = ", ".join(action_tokens) if action_tokens else "none"
        frame_number = frames[last_pos].get("frame_index", last_pos)
        direction_clause = (
            f" most likely toward the {likely_direction} of the view now;"
            if likely_direction
            else ""
        )
        current_distance_clause = (
            f" current estimated distance is about {current_distance_text};"
            if current_distance_text
            else ""
        )
        return (
            f"[{role_text}] '{label}'{direction_clause}{current_distance_clause} was last visible at frame {frame_number} at {bbox_norm} "
            f"({position_text}); actions since then: {action_text}."
        )

    def _build_recent_focus_trace(
        self,
        target: dict[str, Any],
        frames: list[dict[str, Any]],
        current_frame_pos: int,
        image_width: int,
        image_height: int,
    ) -> str:
        label = target["category"]
        role_text = self._role_text(target["roles"])
        start_pos = max(0, current_frame_pos - RECENT_HISTORY_WINDOW)
        entries: list[str] = []

        for pos in range(start_pos, current_frame_pos):
            frame = frames[pos]
            frame_number = frame.get("frame_index", pos)
            action = self._normalize_navgen_action(str(frame.get("next_action") or ""))
            action_text = action if action is not None else collapse_whitespace(str(frame.get("next_action") or "unknown"))
            matched_state = None
            for state in self._pick_all_subtask_states(frame):
                if self._target_key(state) == target["key"]:
                    matched_state = state
                    break

            if matched_state and matched_state.get("visible_in_current_view") and matched_state.get("bbox_xyxy"):
                bbox_norm = normalize_bbox_xyxy(
                    bbox_xyxy=matched_state["bbox_xyxy"],
                    image_width=image_width,
                    image_height=image_height,
                )
                position_text = self._view_position_to_text(
                    matched_state.get("view_position_3x3"),
                    bbox_norm,
                )
                distance_text = format_distance_text(matched_state.get("distance_m"))
                distance_clause = f", about {distance_text} away" if distance_text else ""
                entries.append(
                    f"frame {frame_number}: visible at {bbox_norm} ({position_text}){distance_clause}, then action {action_text}"
                )
            elif matched_state is not None:
                distance_text = format_distance_text(matched_state.get("distance_m"))
                distance_clause = f", current estimate {distance_text}" if distance_text else ""
                entries.append(
                    f"frame {frame_number}: not visible{distance_clause}, then action {action_text}"
                )

        if not entries:
            return (
                f"[{role_text}] '{label}' has no exported trace in the previous "
                f"{RECENT_HISTORY_WINDOW} frames."
            )
        return (
            f"[{role_text}] '{label}' previous-{RECENT_HISTORY_WINDOW}-frame trace: "
            + "; ".join(entries)
            + "."
        )

    def _format_focus_targets_for_prompt(self, focus_targets: list[dict[str, Any]]) -> str:
        if not focus_targets:
            return "[]"

        summaries = []
        for target in focus_targets:
            role_text = self._role_text(target["roles"])
            current_instruction = target["role_to_instruction"].get("current")
            next_instruction = target["role_to_instruction"].get("next")
            role_details: list[str] = []
            if current_instruction:
                role_details.append(f"current='{current_instruction}'")
            if next_instruction:
                role_details.append(f"next='{next_instruction}'")
            detail_text = ", ".join(role_details) if role_details else "role details unavailable"
            summaries.append(f"[{role_text}] '{target['category']}' ({detail_text})")
        return "; ".join(summaries)

    def _build_navgen_context(
        self,
        frames: list[dict[str, Any]],
        current_frame_pos: int,
        frame: dict[str, Any],
        subtasks: list[dict[str, Any]],
        image_width: int,
        image_height: int,
    ) -> dict[str, Any]:
        current_subtask_index = int(frame.get("subtask_index", 0))
        focus_targets, current_targets, next_targets, next_subtask = self._build_focus_targets(
            subtasks=subtasks,
            current_subtask_index=current_subtask_index,
        )
        focus_target_keys = {target["key"] for target in focus_targets}

        visible_state_lookup = {
            self._target_key(state): state
            for state in self._pick_all_subtask_states(frame)
            if state.get("visible_in_current_view")
        }
        all_visible_states = list(visible_state_lookup.values())
        current_focus_distance_estimates: list[str] = []
        likely_direction_hints: list[str] = []
        invisible_targets_with_history: list[dict[str, Any]] = []
        invisible_targets_without_history: list[dict[str, Any]] = []

        visible_focus_objects: list[dict[str, Any]] = []
        invisible_focus_targets: list[dict[str, Any]] = []
        template_focus_statuses: list[dict[str, Any]] = []
        for target in focus_targets:
            current_state = self._find_state_in_frame(frame, target["key"])
            current_distance_text = format_distance_text(
                current_state.get("distance_m") if current_state else None
            )
            if current_distance_text:
                current_focus_distance_estimates.append(
                    f"[{self._role_text(target['roles'])}] '{target['category']}' is about {current_distance_text} from the agent"
                )
            state = visible_state_lookup.get(target["key"])
            prompt_object = None
            direction_hint: Optional[str] = None
            if state is not None:
                prompt_object = self._state_to_prompt_object(
                    state=state,
                    roles=target["roles"],
                    image_width=image_width,
                    image_height=image_height,
                )
            if prompt_object is not None:
                visible_focus_objects.append(prompt_object)
            else:
                invisible_focus_targets.append(target)
                last_pos, _ = self._find_last_visible_state(
                    frames=frames,
                    current_frame_pos=current_frame_pos,
                    target_key=target["key"],
                )
                if last_pos is None:
                    invisible_targets_without_history.append(target)
                else:
                    invisible_targets_with_history.append(target)
                direction_hint = self._infer_likely_direction_from_history(
                    target_key=target["key"],
                    frames=frames,
                    current_frame_pos=current_frame_pos,
                    fallback_state=current_state,
                    image_width=image_width,
                    image_height=image_height,
                )
                if direction_hint:
                    likely_direction_hints.append(
                        f"[{self._role_text(target['roles'])}] '{target['category']}' is most likely toward the {direction_hint} of the view"
                    )
            template_focus_statuses.append(
                {
                    "key": target["key"],
                    "label": target["category"],
                    "roles": list(target["roles"]),
                    "visible": prompt_object is not None,
                    "bbox_norm": prompt_object["bbox_norm"] if prompt_object is not None else None,
                    "position": prompt_object["position"] if prompt_object is not None else None,
                    "position_short": self._template_position_short(
                        current_state.get("view_position_3x3") if current_state else None,
                        prompt_object["bbox_norm"] if prompt_object is not None else None,
                    ) if prompt_object is not None else None,
                    "distance_text": current_distance_text,
                    "direction_hint": direction_hint,
                }
            )

        other_visible_objects: list[dict[str, Any]] = []
        if invisible_focus_targets and invisible_targets_without_history:
            for state in all_visible_states:
                state_key = self._target_key(state)
                if state_key in focus_target_keys:
                    continue
                prompt_object = self._state_to_prompt_object(
                    state=state,
                    roles=["other"],
                    image_width=image_width,
                    image_height=image_height,
                )
                if prompt_object is not None:
                    other_visible_objects.append(prompt_object)

        recent_focus_traces = [
            self._build_recent_focus_trace(
                target=target,
                frames=frames,
                current_frame_pos=current_frame_pos,
                image_width=image_width,
                image_height=image_height,
            )
            for target in invisible_focus_targets
        ]
        history_hints = [
            self._build_history_hint(
                target=target,
                frames=frames,
                current_frame_pos=current_frame_pos,
                image_width=image_width,
                image_height=image_height,
            )
            for target in invisible_focus_targets
        ]

        return {
            "focus_targets": focus_targets,
            "current_targets": current_targets,
            "next_targets": next_targets,
            "next_subtask": next_subtask,
            "visible_focus_objects": visible_focus_objects,
            "visible_focus_targets_str": self._format_prompt_objects(visible_focus_objects),
            "other_visible_objects": other_visible_objects,
            "other_visible_objects_str": self._format_prompt_objects(other_visible_objects[:2]),
            "focus_target_distance_estimates": current_focus_distance_estimates,
            "focus_target_distance_estimates_str": "; ".join(current_focus_distance_estimates)
            if current_focus_distance_estimates else "[]",
            "recent_focus_traces": recent_focus_traces,
            "recent_focus_trace_str": "; ".join(recent_focus_traces) if recent_focus_traces else "[]",
            "history_hints": history_hints,
            "history_hints_str": "; ".join(history_hints) if history_hints else "[]",
            "likely_direction_hints": likely_direction_hints,
            "likely_direction_hints_str": "; ".join(likely_direction_hints)
            if likely_direction_hints else "[]",
            "focus_targets_str": self._format_focus_targets_for_prompt(focus_targets),
            "template_focus_statuses": template_focus_statuses,
            "validation_context": {
                "required_visible_labels": tuple(
                    collapse_whitespace(obj["label"]).lower() for obj in visible_focus_objects
                ),
                "required_visible_bbox_texts": tuple(
                    str(obj["bbox_norm"]) for obj in visible_focus_objects
                ),
                "require_focus_unseen_statement": len(visible_focus_objects) == 0 and bool(focus_targets),
                "require_directional_inference": len(invisible_focus_targets) > 0,
                "expected_action_label": "",
            },
        }

    def build_prompt(
        self,
        frame_data: dict[str, Any],
        history_memory: Optional[str],
        global_instruction: str,
        global_subtasks: list[str],
        completed_index: int,
        next_action: str,
    ) -> str:
        """构造 NavGen frame_info 场景下的 prompt。"""
        current_subtask, completed_subtasks_str = self._build_current_task_state(
            frame_data=frame_data,
            global_instruction=global_instruction,
            global_subtasks=global_subtasks,
            completed_index=completed_index,
        )
        navgen_context = frame_data["navgen_context"]
        return NAVGEN_PROMPT_TEMPLATE.format(
            global_instruction=global_instruction,
            global_subtasks_list=str(global_subtasks),
            completed_subtasks=completed_subtasks_str,
            completed_index=completed_index,
            current_subtask=current_subtask,
            next_subtask=navgen_context["next_subtask"],
            script_memory=self._format_history_memory_input(history_memory),
            current_targets=navgen_context["current_targets"],
            next_targets=navgen_context["next_targets"],
            focus_targets=navgen_context["focus_targets_str"],
            visible_focus_targets=navgen_context["visible_focus_targets_str"],
            focus_target_distances=navgen_context["focus_target_distance_estimates_str"],
            recent_focus_trace=navgen_context["recent_focus_trace_str"],
            historical_focus_hints=navgen_context["history_hints_str"],
            likely_direction_hints=navgen_context["likely_direction_hints_str"],
            other_visible_objects=navgen_context["other_visible_objects_str"],
            next_action=next_action,
        )

    def _parse_structured_response(
        self,
        raw_text: str,
        frame_data: dict[str, Any],
        current_subtask: str,
        next_action: str,
    ) -> tuple[Optional[NavGenStructuredTeacherTrace], Optional[str]]:
        """解析并校验 NavGen CoT 结构化输出。"""
        text = self._extract_json_object(raw_text)
        if not text:
            return None, "empty response"

        context = dict(frame_data.get("navgen_validation_context", {}))
        context["current_subtask"] = current_subtask
        context["next_action"] = next_action
        context["action_mentions"] = self._action_mentions(next_action)
        context["expected_action_label"] = next_action.lower()

        try:
            structured_trace = NavGenStructuredTeacherTrace.model_validate_json(
                text,
                context=context,
            )
        except ValidationError as exc:
            first_error = exc.errors(include_url=False)[0]
            return None, f"{first_error['loc']}: {first_error['msg']}"
        except json.JSONDecodeError as exc:
            return None, f"invalid json: {exc.msg}"

        return structured_trace, None

    def _build_template_result(
        self,
        frame_data: dict[str, Any],
        history_memory: Optional[str],
        global_instruction: str,
        global_subtasks: list[str],
        completed_index: int,
        next_action: str,
    ) -> FrameGenerationResult:
        """当模型多次返回不合格时，生成 NavGen 专用兜底答案。"""
        del history_memory
        current_subtask, _ = self._build_current_task_state(
            frame_data=frame_data,
            global_instruction=global_instruction,
            global_subtasks=global_subtasks,
            completed_index=completed_index,
        )
        navgen_context = frame_data["navgen_context"]
        total_steps = max(1, len(global_subtasks))
        completed_count = max(0, min(completed_index, total_steps))
        current_step_number = max(1, min(completed_index + 1, total_steps))
        short_subtask = self._truncate_template_text(current_subtask, 72)
        template_focus_statuses = navgen_context.get("template_focus_statuses", [])
        current_focus_labels = [
            str(status["label"])
            for status in template_focus_statuses
            if "current" in status.get("roles", [])
        ]
        next_focus_labels = [
            str(status["label"])
            for status in template_focus_statuses
            if "next" in status.get("roles", [])
        ]
        current_focus_text = self._template_focus_label_text(
            current_focus_labels,
            fallback_text="current target",
            max_chars=40,
        )
        next_focus_text = self._template_focus_label_text(
            next_focus_labels,
            fallback_text="next target",
            max_chars=40,
        )

        step1_candidates = [
            f"Done {completed_count}/{total_steps}. Now step {current_step_number}: '{short_subtask}'.",
            f"Done {completed_count}/{total_steps}. Now on step {current_step_number}.",
        ]
        if template_focus_statuses:
            step2_candidates = [
                " ".join(
                    self._format_template_focus_status(status)
                    for status in template_focus_statuses
                ),
                " ".join(
                    self._format_template_focus_status(status, include_distance=False)
                    for status in template_focus_statuses
                ),
                "; ".join(
                    self._format_template_focus_status_compact(status).rstrip(".")
                    for status in template_focus_statuses
                ) + ".",
                "; ".join(
                    self._format_template_focus_status_minimal(status).rstrip(".")
                    for status in template_focus_statuses
                ) + ".",
            ]
        else:
            step2_candidates = [
                "Current/next targets unseen, likely lower.",
                "Targets unseen, likely lower.",
            ]
        step3_candidates = [
            (
                f"The action '{next_action}' is correct because it keeps the agent aligned with "
                f"{current_focus_text}. It also keeps {next_focus_text} reachable."
            ),
            (
                f"The action '{next_action}' is correct because it follows the current target layout. "
                "It also preserves the next target cue."
            ),
            f"The action '{next_action}' is correct because it matches the current target layout.",
        ]
        step4_candidates = [
            f"Step {current_step_number} ongoing; action {next_action.lower()}.",
            f"On step {current_step_number}; action {next_action.lower()}.",
        ]

        validation_context = {
            **frame_data.get("navgen_validation_context", {}),
            "action_mentions": self._action_mentions(next_action),
        }
        last_error: Optional[ValidationError] = None
        structured_trace: Optional[NavGenStructuredTeacherTrace] = None
        for step1 in step1_candidates:
            if structured_trace is not None:
                break
            for step2 in step2_candidates:
                if len(collapse_whitespace(step2)) > MAX_STEP_CHARS:
                    continue
                for step3 in step3_candidates:
                    for step4 in step4_candidates:
                        try:
                            structured_trace = NavGenStructuredTeacherTrace.model_validate(
                                {
                                    "step1_task_progress": step1,
                                    "step2_spatial_perception": step2,
                                    "step3_decision_logic": step3,
                                    "step4_memory_update": step4,
                                },
                                context=validation_context,
                            )
                            break
                        except ValidationError as exc:
                            last_error = exc
                    if structured_trace is not None:
                        break
                if structured_trace is not None:
                    break

        if structured_trace is None:
            priority_statuses = self._select_template_priority_statuses(
                template_focus_statuses,
                max_items=2,
            )
            relaxed_visible_statuses = [
                status for status in priority_statuses if status.get("visible")
            ]
            relaxed_context = {
                **validation_context,
                "required_visible_labels": tuple(
                    collapse_whitespace(str(status.get("label") or "")).lower()
                    for status in relaxed_visible_statuses
                ),
                "required_visible_bbox_texts": tuple(
                    str(status.get("bbox_norm") or [])
                    for status in relaxed_visible_statuses
                ),
            }
            relaxed_step2_candidates = []
            if priority_statuses:
                relaxed_step2_candidates.extend(
                    [
                        " ".join(
                            self._format_template_focus_status(status, include_distance=False)
                            for status in priority_statuses
                        ),
                        "; ".join(
                            self._format_template_focus_status_minimal(status).rstrip(".")
                            for status in priority_statuses
                        ) + ".",
                    ]
                )
            else:
                relaxed_step2_candidates.append("Targets unseen, likely lower.")

            for step1 in step1_candidates:
                if structured_trace is not None:
                    break
                for step2 in relaxed_step2_candidates:
                    if len(collapse_whitespace(step2)) > MAX_STEP_CHARS:
                        continue
                    for step3 in step3_candidates:
                        for step4 in step4_candidates:
                            try:
                                structured_trace = NavGenStructuredTeacherTrace.model_validate(
                                    {
                                        "step1_task_progress": step1,
                                        "step2_spatial_perception": step2,
                                        "step3_decision_logic": step3,
                                        "step4_memory_update": step4,
                                    },
                                    context=relaxed_context,
                                )
                                logger.warning(
                                    "模板兜底已退化为最小 focus 版本，以避免超长输出导致丢帧。"
                                )
                                break
                            except ValidationError as exc:
                                last_error = exc
                        if structured_trace is not None:
                            break
                    if structured_trace is not None:
                        break

        if structured_trace is None:
            if last_error is not None:
                raise last_error
            raise RuntimeError("template fallback could not build a valid NavGen trace")
        return self._result_from_structured_trace(
            structured_trace=structured_trace,
            frame_data=frame_data,
            next_action=next_action,
        )

    def _build_retry_feedback_message(
        self,
        format_error: Optional[str],
        next_action: str,
    ) -> str:
        return (
            "Your previous JSON failed validation. "
            f"Validation error: {format_error}. "
            "Return one corrected compact JSON object with exactly these keys: "
            "step1_task_progress, step2_spatial_perception, "
            "step3_decision_logic, step4_memory_update. "
            "Do not include any extra keys. "
            "Do not repeat prompt instructions or placeholder text. "
            "Step 2 must focus on the current target first and the next target second, include distance in meters, "
            "and for invisible targets it must explicitly infer a likely screen direction such as left, right, upper-left, lower-right, or lower edge before mentioning any supporting objects. "
            "State the direction before the distance, and avoid vague phrases like ahead, to the side, or forward-left. "
            "Mention supporting objects only when no usable historical sighting exists. "
            f"Step 3 sentence 1 must start exactly with \"The action '{next_action}' is correct because ...\" and explicitly mention the action label '{next_action}'. "
            "Step 3 may use one optional second sentence for the next target or historical clue only. "
            "Step 4 must update the current active task progress together with the chosen action in one sentence."
        )

    @staticmethod
    def _encode_image_file(image_path: Path) -> str:
        return base64.b64encode(image_path.read_bytes()).decode("utf-8")

    def run_frame_info(
        self,
        frame_info: str,
        output_jsonl: str,
        step_task_json: Optional[str] = None,
        max_frames: Optional[int] = None,
    ) -> None:
        frame_info_path = Path(frame_info).expanduser().resolve()
        if not frame_info_path.exists():
            raise FileNotFoundError(f"frame_info.json not found: {frame_info_path}")

        step_task_json_path = self._resolve_step_task_json(frame_info_path, step_task_json)
        if not step_task_json_path.exists():
            raise FileNotFoundError(f"step_task json not found: {step_task_json_path}")

        with frame_info_path.open("r", encoding="utf-8") as f:
            frame_info_data = json.load(f)
        with step_task_json_path.open("r", encoding="utf-8") as f:
            step_task_data = json.load(f)

        meta = frame_info_data.get("meta", {})
        frames = frame_info_data.get("frames", [])
        subtasks = frame_info_data.get("subtasks", [])
        if not isinstance(frames, list) or not isinstance(subtasks, list):
            raise ValueError("frame_info.json must contain list fields `frames` and `subtasks`.")

        global_instruction = collapse_whitespace(
            str(
                meta.get("task_instruction")
                or step_task_data.get("Task instruction")
                or ""
            )
        )
        if not global_instruction:
            raise ValueError("task instruction is missing in frame_info/meta and step_task json.")

        global_subtasks = [
            collapse_whitespace(str(subtask.get("instruction") or ""))
            for subtask in subtasks
            if collapse_whitespace(str(subtask.get("instruction") or ""))
        ]
        if not global_subtasks:
            raise ValueError("no valid subtasks found in frame_info.json")

        image_width = int(meta.get("image_width") or 0)
        image_height = int(meta.get("image_height") or 0)
        if image_width <= 1 or image_height <= 1:
            raise ValueError("meta.image_width / meta.image_height are invalid")

        frame_info_parent = frame_info_path.parent
        output_path = Path(output_jsonl).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        item_id = self._infer_item_id_from_path(
            frame_info_path=frame_info_path,
            step_task_data=step_task_data,
            frame_info_meta=meta,
        )

        history_memory: Optional[str] = None
        history_source_frame: Optional[int] = None
        success_count = 0
        frame_limit = max_frames if max_frames is not None and max_frames >= 0 else None

        with output_path.open("w", encoding="utf-8") as out_f:
            for frame_pos, frame in enumerate(frames):
                if frame_limit is not None and frame_pos >= frame_limit:
                    break

                next_action_raw = str(frame.get("next_action") or "")
                next_action = self._normalize_navgen_action(next_action_raw)
                if next_action is None:
                    logger.warning(
                        "frame %s next_action 无法识别，跳过: %s",
                        frame.get("frame_index", frame_pos),
                        next_action_raw,
                    )
                    continue

                current_subtask_index = int(frame.get("subtask_index", 0))
                if current_subtask_index < 0 or current_subtask_index >= len(global_subtasks):
                    logger.warning(
                        "frame %s subtask_index 越界，跳过: %s",
                        frame.get("frame_index", frame_pos),
                        current_subtask_index,
                    )
                    continue

                current_subtask = collapse_whitespace(
                    str(frame.get("subtask_instruction") or global_subtasks[current_subtask_index])
                )
                navgen_context = self._build_navgen_context(
                    frames=frames,
                    current_frame_pos=frame_pos,
                    frame=frame,
                    subtasks=subtasks,
                    image_width=image_width,
                    image_height=image_height,
                )
                visible_focus_objects = navgen_context["visible_focus_objects"]
                frame_data: dict[str, Any] = {
                    "subtask": current_subtask,
                    "objects": [{"label": obj["label"]} for obj in visible_focus_objects],
                    "navgen_context": navgen_context,
                    "navgen_validation_context": navgen_context["validation_context"],
                }

                image_path = (frame_info_parent / str(frame.get("rgb_path") or "")).resolve()
                if not image_path.exists():
                    logger.error(
                        "frame %s 对应图片不存在，跳过: %s",
                        frame.get("frame_index", frame_pos),
                        image_path,
                    )
                    continue

                prompt = self.build_prompt(
                    frame_data=frame_data,
                    history_memory=history_memory,
                    global_instruction=global_instruction,
                    global_subtasks=global_subtasks,
                    completed_index=current_subtask_index,
                    next_action=next_action,
                )
                generation_result: Optional[FrameGenerationResult] = None

                try:
                    image_b64 = self._encode_image_file(image_path)
                    generation_result = self.call_vlm_api(
                        prompt=prompt,
                        image_b64=image_b64,
                        frame_data=frame_data,
                        history_memory=history_memory,
                        global_instruction=global_instruction,
                        global_subtasks=global_subtasks,
                        completed_index=current_subtask_index,
                        next_action=next_action,
                    )
                    if generation_result is None:
                        logger.error(
                            "frame %s VLM 调用失败，跳过",
                            frame.get("frame_index", frame_pos),
                        )
                        continue

                    record = {
                        "episode_id": item_id,
                        "frame": int(frame.get("frame_index", frame_pos)),
                        "input_state": {
                            "historical_trajectory_memory": self._format_history_memory_input(history_memory),
                            "history_source_frame": history_source_frame,
                            "item_id": item_id,
                            "global_step": frame.get("global_step"),
                            "source_trial_key": frame.get("source_trial_key"),
                            "current_subtask_index": current_subtask_index,
                            "next_subtask": navgen_context["next_subtask"],
                            "focus_targets": navgen_context["focus_targets_str"],
                            "frame_info_path": str(frame_info_path),
                            "step_task_json": str(step_task_json_path),
                        },
                        "messages": [
                            {
                                "role": "user",
                                "content": [
                                    {"type": "image_path", "image_path": str(image_path)},
                                    {"type": "text", "text": prompt},
                                ],
                            },
                            {
                                "role": "assistant",
                                "content": generation_result.assistant_content,
                            },
                        ],
                    }

                    out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    success_count += 1
                    logger.info(
                        "  => 成功写入 Frame %s",
                        frame.get("frame_index", frame_pos),
                    )

                except Exception as exc:  # noqa: BLE001
                    logger.error(
                        "frame %s 处理异常，跳过本帧: %s: %s",
                        frame.get("frame_index", frame_pos),
                        type(exc).__name__,
                        exc,
                    )
                finally:
                    if generation_result is not None:
                        history_memory = self._build_memory_anchor(
                            frame_data=frame_data,
                            next_action=next_action,
                            preferred_text=generation_result.memory_update,
                        )
                    else:
                        history_memory = self._build_memory_anchor(
                            frame_data=frame_data,
                            next_action=next_action,
                        )
                    history_source_frame = int(frame.get("frame_index", frame_pos))

        logger.info(
            "NavGen frame_info 处理完成！共生成 %d 条 CoT 记录，输出文件: %s",
            success_count,
            output_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT JSONL from NavGen step_task frame_info + RGB assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--frame-info",
        required=True,
        help="Input *.frame_info.json path.",
    )
    parser.add_argument(
        "--step-task-json",
        default=None,
        help="Optional sibling *.json path. If omitted, infer it from --frame-info.",
    )
    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT_DIR / "navgen_output_cot.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument(
        "--api-provider",
        choices=API_PROVIDER_CHOICES,
        default="dashscope",
        help="API provider type.",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="OpenAI compatible API base url.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit API key.",
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Environment variable name for API key.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=THINKING_MODE_CHOICES,
        default="off",
        help="Thinking mode.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API retry count.",
    )
    parser.add_argument(
        "--invalid-frame-policy",
        choices=("skip", "template"),
        default="skip",
        help="Policy when the model repeatedly returns invalid output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit on the number of frames to process.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    synthesizer = NavGenFrameInfoSynthesizer(
        max_retries=args.max_retries,
        api_provider=args.api_provider,
        api_base_url=args.api_base_url,
        model_name=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        thinking_mode=args.thinking_mode,
        invalid_frame_policy=args.invalid_frame_policy,
    )
    synthesizer.run_frame_info(
        frame_info=args.frame_info,
        step_task_json=args.step_task_json,
        output_jsonl=args.output,
        max_frames=args.max_frames,
    )


if __name__ == "__main__":
    main()
