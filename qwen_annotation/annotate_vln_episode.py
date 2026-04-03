#!/usr/bin/env python3

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2

from qwen_api import (
    DEFAULT_VL_MODEL,
    QwenClient,
    build_image_content_item,
    build_video_frames_content_item,
    build_video_url_content_item,
)


DEFAULT_VIDEO_PATH = Path(
    "/home/gs/my_project/data_generation_pipline/frames/1.mp4"
)
DEFAULT_FRAMES_DIR = Path(
    "/home/gs/my_project/data_generation_pipline/frames"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gs/my_project/data_generation_pipline/output/qwen_annotations"
)
DEFAULT_TRAIN_JSON = Path(
    "/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json"
)
DEFAULT_TRAIN_GT_JSON = Path(
    "/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json"
)

OBJECT_COLOR_BGR = (60, 179, 113)
REGION_COLOR_BGR = (0, 165, 255)
TEXT_COLOR = (255, 255, 255)
REGION_DEFINITION = (
    "A region must be a semantically self-contained and physically anchorable "
    "space. Valid region forms include absolute place names such as "
    "\"kitchen\", opening-like region phrases such as \"right side doorway\", "
    "and anchored spatial phrases such as "
    "\"the right side of the center unit\"."
)
REGION_SPATIAL_TERMS = (
    "left",
    "right",
    "left side",
    "right side",
    "front",
    "back",
    "rear",
    "top",
    "bottom",
    "middle",
    "center",
    "inside",
    "outside",
    "inner side",
    "outer side",
    "near side",
    "far side",
    "end",
    "corner",
)
ABSOLUTE_REGION_PLACE_NAMES = (
    "kitchen",
    "dining room",
    "living room",
    "bedroom",
    "bathroom",
    "pantry",
    "laundry room",
    "hall",
    "hallway",
    "corridor",
    "foyer",
    "entryway",
    "room",
)
ABSOLUTE_REGION_HEAD_NOUNS = (
    "doorway",
    "entrance",
    "entry",
    "exit",
    "opening",
    "hall",
    "hallway",
    "corridor",
    "foyer",
    "entryway",
    "staircase",
    "stairs",
    "landing",
)
RELATIONAL_REGION_CONTEXT_HINTS = (
    "doorway",
    "entrance",
    "entry",
    "exit",
    "opening",
    "room",
    "hall",
    "hallway",
    "corridor",
    "area",
    "space",
    "section",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Use Qwen only to segment a VLN episode and annotate per-frame "
            "object/region visibility, centers, and estimated distances."
        )
    )
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument(
        "--train-gt-json",
        type=Path,
        default=DEFAULT_TRAIN_GT_JSON,
    )
    parser.add_argument("--episode-id", type=int, default=1)
    parser.add_argument("--video-path", type=Path, default=DEFAULT_VIDEO_PATH)
    parser.add_argument("--frames-dir", type=Path, default=DEFAULT_FRAMES_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--reuse-run-dir",
        type=Path,
        default=None,
        help=(
            "Reuse a previous run directory and skip subtask segmentation by "
            "loading its saved subtasks."
        ),
    )
    parser.add_argument("--segment-model", default=DEFAULT_VL_MODEL)
    parser.add_argument("--localize-model", default=DEFAULT_VL_MODEL)
    parser.add_argument(
        "--qwen-temperature",
        type=float,
        default=0.0,
        help="Temperature applied to all Qwen requests in this pipeline.",
    )
    parser.add_argument(
        "--qwen-top-p",
        type=float,
        default=None,
        help="Optional top_p applied to all Qwen requests in this pipeline.",
    )
    parser.add_argument(
        "--qwen-result-format",
        default=None,
        help="Optional DashScope-compatible result_format value.",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable Qwen thinking mode.",
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=None,
        help="Optional thinking token budget used only when thinking is enabled.",
    )
    parser.add_argument(
        "--video-fps",
        type=float,
        default=5.0,
        help="Sampling FPS hint for the video and frame-list multimodal prompts.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-run Qwen and regenerate cached intermediate files.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(message, flush=True)


def build_qwen_request_settings(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "temperature": args.qwen_temperature,
        "top_p": args.qwen_top_p,
        "result_format": args.qwen_result_format,
        "enable_thinking": args.thinking,
        "thinking_budget": args.thinking_budget if args.thinking else None,
    }


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_episode_bundle(
    train_json_path: Path,
    train_gt_json_path: Path,
    episode_id: int,
) -> Tuple[dict, dict]:
    train_data = load_json(train_json_path)
    train_episodes = {
        str(episode["episode_id"]): episode for episode in train_data["episodes"]
    }
    gt_episodes = load_json(train_gt_json_path)
    episode_key = str(episode_id)
    if episode_key not in train_episodes:
        raise KeyError(f"Episode {episode_id} is missing from {train_json_path}")
    if episode_key not in gt_episodes:
        raise KeyError(f"Episode {episode_id} is missing from {train_gt_json_path}")
    return train_episodes[episode_key], gt_episodes[episode_key]


def discover_episode_frames(frames_dir: Path, episode_id: int) -> List[Path]:
    pattern = f"ep{episode_id:05d}_frame*.jpg"
    frame_paths = sorted(frames_dir.glob(pattern))
    if not frame_paths:
        raise FileNotFoundError(
            f"No episode frames matched {pattern} under {frames_dir}"
        )
    return frame_paths


def load_or_create_json(
    path: Path,
    overwrite: bool,
    factory,
) -> Dict[str, Any]:
    if path.exists() and not overwrite:
        return read_json(path)
    payload = factory()
    write_json(path, payload)
    return payload


def find_latest_matching_file(directory: Path, pattern: str) -> Optional[Path]:
    matches = sorted(directory.glob(pattern))
    if not matches:
        return None
    return matches[-1]


def load_subtasks_from_run_dir(run_dir: Path) -> List[Dict[str, Any]]:
    annotation_path = find_latest_matching_file(
        run_dir, "qwen_episode_annotations_*.json"
    )
    if annotation_path is not None:
        payload = read_json(annotation_path)
        subtasks = payload.get("subtasks")
        if isinstance(subtasks, list) and subtasks:
            return subtasks

    cache_dirs = sorted(path for path in run_dir.glob("cache_*") if path.is_dir())
    for cache_dir in reversed(cache_dirs):
        normalized_path = find_latest_matching_file(
            cache_dir, "frame_refinement_normalized_*.json"
        )
        if normalized_path is None:
            continue
        payload = read_json(normalized_path)
        subtasks = payload.get("subtasks")
        if isinstance(subtasks, list) and subtasks:
            return subtasks

    raise FileNotFoundError(
        f"Could not find reusable subtasks under {run_dir}. "
        "Expected qwen_episode_annotations_*.json or frame_refinement_normalized_*.json."
    )


def load_existing_frame_records(run_dir: Path) -> List[Dict[str, Any]]:
    frame_result_dirs = sorted(
        path for path in run_dir.glob("frame_results_*") if path.is_dir()
    )
    for frame_dir in reversed(frame_result_dirs):
        records: List[Dict[str, Any]] = []
        for frame_path in sorted(frame_dir.glob("frame_*.json")):
            payload = read_json(frame_path)
            if isinstance(payload, dict) and "frame_id" in payload:
                records.append(payload)
        if records:
            records.sort(key=lambda item: int(item["frame_id"]))
            return records
    return []


def summarize_subtasks(subtasks: Sequence[Dict[str, Any]]) -> str:
    parts = []
    for subtask in subtasks:
        parts.append(
            f"{subtask['subtask_id']}:{subtask['start_frame']}-{subtask['end_frame']} "
            f"{subtask['description']}"
        )
    return " | ".join(parts)


def split_instruction_clauses(instruction: str) -> List[str]:
    cleaned = " ".join(instruction.strip().split())
    if cleaned.endswith("."):
        cleaned = cleaned[:-1]
    if not cleaned:
        return []

    pattern = re.compile(
        r"\band\s+(?=(go|stop|turn|move|walk|head|enter|exit|approach|pass|continue|keep|make|proceed)\b)",
        flags=re.IGNORECASE,
    )
    clauses: List[str] = []
    last = 0
    for match in pattern.finditer(cleaned):
        candidate = " ".join(cleaned[last:match.start()].strip().split())
        if candidate:
            clauses.append(candidate)
        last = match.end()
    tail = " ".join(cleaned[last:].strip().split())
    if tail:
        clauses.append(tail)
    return clauses or [cleaned]


def normalize_text_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9\s]", " ", text.lower()).strip()


def content_tokens(text: str) -> List[str]:
    stopwords = {
        "the",
        "a",
        "an",
        "of",
        "to",
        "by",
        "with",
        "in",
        "on",
        "it",
        "and",
        "go",
        "move",
        "stop",
        "around",
        "near",
    }
    tokens = []
    for token in normalize_text_for_match(text).split():
        if len(token) <= 1 or token in stopwords:
            continue
        tokens.append(token)
    return tokens


def entity_allowed_by_instruction(name: str, instruction: str) -> bool:
    normalized_instruction = normalize_text_for_match(instruction)
    normalized_name = normalize_text_for_match(name)
    if not normalized_name:
        return False
    if normalized_name in normalized_instruction:
        return True

    tokens = content_tokens(name)
    if not tokens:
        return False
    return all(token in normalized_instruction.split() for token in tokens)


def canonicalize_entity_name_against_instruction(
    name: str,
    instruction: str,
) -> str:
    normalized_name = normalize_text_for_match(name)
    normalized_instruction = normalize_text_for_match(instruction)
    if normalized_name and normalized_name in normalized_instruction:
        return normalized_name

    target_tokens = content_tokens(name)
    if not target_tokens:
        return normalize_name(name)

    instruction_tokens = normalized_instruction.split()
    best_match = None
    for start in range(len(instruction_tokens)):
        for end in range(start + 1, len(instruction_tokens) + 1):
            span_tokens = instruction_tokens[start:end]
            if all(token in span_tokens for token in target_tokens):
                candidate = " ".join(span_tokens)
                if best_match is None or len(candidate) > len(best_match):
                    best_match = candidate
    return best_match or normalize_name(name)


def constrain_entities_to_instruction(
    subtasks: Sequence[Dict[str, Any]],
    instruction: str,
) -> List[Dict[str, Any]]:
    constrained: List[Dict[str, Any]] = []
    for subtask in subtasks:
        subtask_copy = dict(subtask)
        object_map: Dict[str, Dict[str, Any]] = {}
        for entity in subtask["objects"]:
            if not entity_allowed_by_instruction(entity["name"], instruction):
                continue
            canonical_name = canonicalize_entity_name_against_instruction(
                entity["name"], instruction
            )
            object_map.setdefault(
                canonical_name,
                dict(entity, name=canonical_name),
            )

        region_map: Dict[str, Dict[str, Any]] = {}
        for entity in subtask["regions"]:
            if not entity_allowed_by_instruction(entity["name"], instruction):
                continue
            canonical_name = canonicalize_entity_name_against_instruction(
                entity["name"], instruction
            )
            region_map.setdefault(
                canonical_name,
                dict(entity, name=canonical_name),
            )

        subtask_copy["objects"] = list(object_map.values())
        subtask_copy["regions"] = list(region_map.values())
        constrained.append(subtask_copy)
    return constrained


def align_subtasks_to_instruction(
    subtasks: Sequence[Dict[str, Any]],
    instruction: str,
) -> List[Dict[str, Any]]:
    clauses = split_instruction_clauses(instruction)
    if not clauses:
        return list(subtasks)

    aligned: List[Dict[str, Any]] = []
    count = len(subtasks)
    clause_count = len(clauses)
    for index, subtask in enumerate(subtasks):
        subtask_copy = dict(subtask)
        start_clause = int(index * clause_count / count)
        end_clause = int((index + 1) * clause_count / count)
        assigned = clauses[start_clause:max(start_clause + 1, end_clause)]
        instruction_span = " and ".join(assigned).strip()
        if instruction_span:
            subtask_copy["instruction_span"] = instruction_span
            subtask_copy["description"] = instruction_span
            subtask_copy["name"] = instruction_span[:80]
        else:
            subtask_copy["instruction_span"] = subtask_copy["description"]
        aligned.append(subtask_copy)
    return aligned


def refresh_entity_ids(
    subtasks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    refreshed: List[Dict[str, Any]] = []
    for subtask in subtasks:
        subtask_copy = dict(subtask)
        subtask_copy["objects"] = [dict(entity) for entity in subtask["objects"]]
        subtask_copy["regions"] = [dict(entity) for entity in subtask["regions"]]
        for entity_index, entity in enumerate(subtask_copy["objects"], start=1):
            entity["entity_id"] = f"{subtask_copy['subtask_id']}_obj_{entity_index}"
        for entity_index, entity in enumerate(subtask_copy["regions"], start=1):
            entity["entity_id"] = f"{subtask_copy['subtask_id']}_reg_{entity_index}"
        refreshed.append(subtask_copy)
    return refreshed


def build_result_payload(
    episode_id: int,
    scene_id: str,
    instruction: str,
    video_path: Path,
    frames_dir: Path,
    annotated_dir: Path,
    annotated_video_path: Path,
    segment_model: str,
    localize_model: str,
    qwen_request_settings: Dict[str, Any],
    total_frames: int,
    action_sequence: Sequence[int],
    subtasks: Sequence[Dict[str, Any]],
    frame_records: Sequence[Dict[str, Any]],
    phase: str,
    run_tag: str,
    error: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "episode_id": episode_id,
        "run_tag": run_tag,
        "scene_id": scene_id,
        "instruction": instruction,
        "video_path": str(video_path),
        "frames_dir": str(frames_dir),
        "annotated_frames_dir": str(annotated_dir),
        "annotated_video_path": str(annotated_video_path),
        "models": {
            "segment_model": segment_model,
            "localize_model": localize_model,
        },
        "qwen_request_settings": dict(qwen_request_settings),
        "status": {
            "phase": phase,
            "completed_frames": len(frame_records),
            "total_frames": total_frames,
            "error": error,
        },
        "total_frames": total_frames,
        "action_sequence": [int(action) for action in action_sequence],
        "distance_measurement": {
            "object_distance": "Model-estimated metric distance in meters from the camera to the object center.",
            "region_distance_range": "Model-estimated near-to-far visible distance range in meters for the region.",
            "source": "Pure Qwen visual estimation from RGB frames only.",
        },
        "subtasks": list(subtasks),
        "frames": list(frame_records),
    }


def save_checkpoint(
    episode_root: Path,
    payload: Dict[str, Any],
    run_tag: str,
) -> None:
    write_json(episode_root / f"qwen_episode_annotations_{run_tag}.json", payload)
    write_json(episode_root / f"progress_{run_tag}.json", payload["status"])


def normalize_name(value: Any) -> str:
    text = " ".join(str(value or "").strip().split())
    if not text:
        return ""
    lowered = text.lower()
    if lowered.startswith("the "):
        lowered = lowered[4:]
    return lowered


def infer_region_from_name(name: str) -> bool:
    normalized = normalize_name(name)
    if not normalized:
        return False
    if normalized in ABSOLUTE_REGION_PLACE_NAMES:
        return True
    if any(
        normalized.endswith(f" {place_name}")
        for place_name in ABSOLUTE_REGION_PLACE_NAMES
    ):
        return True
    if any(
        normalized == head_noun
        or normalized.startswith(f"{head_noun} ")
        or normalized.endswith(f" {head_noun}")
        for head_noun in ABSOLUTE_REGION_HEAD_NOUNS
    ):
        return True
    if any(
        normalized.startswith(f"{term} of ") and len(normalized.split(" of ", 1)[1].strip()) > 0
        for term in sorted(REGION_SPATIAL_TERMS, key=len, reverse=True)
    ):
        return True
    if any(
        re.search(rf"\b{re.escape(term)}\b", normalized)
        for term in sorted(REGION_SPATIAL_TERMS, key=len, reverse=True)
    ) and any(
        re.search(rf"\b{re.escape(head_noun)}\b", normalized)
        for head_noun in RELATIONAL_REGION_CONTEXT_HINTS
    ):
        return True
    return False


def choose_entity_type(name: str, proposed_type: Optional[str]) -> str:
    if proposed_type in {"object", "region"}:
        return proposed_type
    if infer_region_from_name(name):
        return "region"
    return "object"


def normalize_entity_entry(
    item: Any,
    proposed_type: Optional[str],
) -> Optional[Dict[str, Any]]:
    if isinstance(item, str):
        name = normalize_name(item)
        notes = ""
    elif isinstance(item, dict):
        name = normalize_name(
            item.get("name")
            or item.get("label")
            or item.get("entity")
            or item.get("phrase")
        )
        notes = " ".join(
            str(item.get(key, "")).strip()
            for key in ("reason", "notes", "salience", "evidence")
            if item.get(key)
        ).strip()
        proposed_type = item.get("type", proposed_type)
    else:
        return None

    if not name:
        return None

    return {
        "name": name,
        "type": choose_entity_type(name, proposed_type),
        "notes": notes,
    }


def collect_entities(
    raw_subtask: Dict[str, Any]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    merged: Dict[str, Dict[str, Any]] = {}
    for key, proposed_type in (
        ("objects", "object"),
        ("regions", "region"),
        ("entities", None),
    ):
        for item in raw_subtask.get(key, []) or []:
            normalized = normalize_entity_entry(item, proposed_type)
            if normalized is None:
                continue
            previous = merged.get(normalized["name"])
            if previous is None or normalized["type"] == "region":
                merged[normalized["name"]] = normalized
            elif previous.get("notes") and not normalized.get("notes"):
                continue
            else:
                merged[normalized["name"]] = normalized

    objects = []
    regions = []
    for entity in sorted(merged.values(), key=lambda value: value["name"]):
        if entity["type"] == "region":
            regions.append(entity)
        else:
            objects.append(entity)
    return objects, regions


def coerce_int(value: Any, default: int) -> int:
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        return default


def coerce_float(value: Any) -> Optional[float]:
    try:
        return round(float(value), 3)
    except (TypeError, ValueError):
        return None


def coerce_float_range(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    low = coerce_float(value[0])
    high = coerce_float(value[1])
    if low is None or high is None:
        return None
    if high < low:
        low, high = high, low
    return [low, high]


def normalize_subtasks(
    raw_payload: Dict[str, Any],
    total_frames: int,
) -> List[Dict[str, Any]]:
    raw_subtasks = raw_payload.get("subtasks")
    if not isinstance(raw_subtasks, list) or not raw_subtasks:
        raw_subtasks = [{"description": "Complete instruction"}]

    raw_subtasks = raw_subtasks[:total_frames]
    normalized: List[Dict[str, Any]] = []
    for index, item in enumerate(raw_subtasks, start=1):
        if not isinstance(item, dict):
            item = {"description": str(item)}
        objects, regions = collect_entities(item)
        raw_start = coerce_int(item.get("start_frame"), index)
        raw_end = coerce_int(item.get("end_frame"), raw_start)
        normalized.append(
            {
                "subtask_id": f"s{index}",
                "instruction_span": " ".join(
                    str(item.get("instruction_span") or "").split()
                ),
                "name": " ".join(
                    str(item.get("name") or item.get("title") or "").split()
                )
                or f"Subtask {index}",
                "description": " ".join(
                    str(item.get("description") or item.get("summary") or "").split()
                )
                or f"Subtask {index}",
                "evidence": " ".join(str(item.get("evidence") or "").split()),
                "raw_start_frame": raw_start,
                "raw_end_frame": raw_end,
                "objects": objects,
                "regions": regions,
            }
        )

    normalized.sort(
        key=lambda item: (
            item["raw_start_frame"],
            item["raw_end_frame"],
            item["subtask_id"],
        )
    )

    next_start = 1
    subtask_count = len(normalized)
    for index, subtask in enumerate(normalized, start=1):
        subtask["subtask_id"] = f"s{index}"
        remaining = subtask_count - index
        subtask["start_frame"] = next_start
        if index == subtask_count:
            subtask["end_frame"] = total_frames
        else:
            max_end = total_frames - remaining
            proposed_end = coerce_int(subtask.get("raw_end_frame"), next_start)
            subtask["end_frame"] = max(next_start, min(max_end, proposed_end))
        next_start = subtask["end_frame"] + 1

        for entity_index, entity in enumerate(subtask["objects"], start=1):
            entity["entity_id"] = f"{subtask['subtask_id']}_obj_{entity_index}"
        for entity_index, entity in enumerate(subtask["regions"], start=1):
            entity["entity_id"] = f"{subtask['subtask_id']}_reg_{entity_index}"

    return normalized


def merge_subtasks(
    primary: List[Dict[str, Any]],
    fallback: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if len(primary) != len(fallback):
        return primary

    merged: List[Dict[str, Any]] = []
    for primary_subtask, fallback_subtask in zip(primary, fallback):
        merged_entry = dict(primary_subtask)
        if not merged_entry["name"] or merged_entry["name"].startswith("Subtask "):
            merged_entry["name"] = fallback_subtask["name"]
        if (
            not merged_entry["description"]
            or merged_entry["description"].startswith("Subtask ")
        ):
            merged_entry["description"] = fallback_subtask["description"]
        if not merged_entry["evidence"]:
            merged_entry["evidence"] = fallback_subtask["evidence"]
        if not merged_entry["objects"]:
            merged_entry["objects"] = fallback_subtask["objects"]
        if not merged_entry["regions"]:
            merged_entry["regions"] = fallback_subtask["regions"]
        merged.append(merged_entry)
    return merged


def build_frame_lookup(
    subtasks: Sequence[Dict[str, Any]],
) -> Dict[int, Dict[str, Any]]:
    frame_to_subtask: Dict[int, Dict[str, Any]] = {}
    for subtask in subtasks:
        for frame_id in range(subtask["start_frame"], subtask["end_frame"] + 1):
            frame_to_subtask[frame_id] = subtask
    return frame_to_subtask


def build_video_segmentation_messages(
    video_path: Path,
    instruction: str,
    total_frames: int,
    fps: float,
) -> List[Dict[str, Any]]:
    prompt = f"""
You are annotating a navigation video for a VLN episode.

Instruction:
{instruction}

This episode has exactly {total_frames} extracted first-person RGB frames.
Watch the full video and split the instruction into an ordered list of subtasks.

Return JSON only with this schema:
{{
  "subtasks": [
    {{
      "subtask_id": "s1",
      "name": "short title",
      "instruction_span": "exact words copied from the instruction",
      "description": "same meaning as the copied instruction span",
      "start_frame": 1,
      "end_frame": {total_frames},
      "evidence": "brief visual evidence",
      "objects": [{{"name": "object name"}}],
      "regions": [{{"name": "region name"}}]
    }}
  ]
}}

Rules:
- Decompose the task according to the instruction, not according to extra scene details.
- Each subtask must correspond to a contiguous part of the instruction.
- The ordered concatenation of all instruction_span values should reconstruct the full instruction meaning.
- Copy each instruction_span from the instruction text as faithfully as possible.
- Do not introduce landmarks that are not explicitly mentioned in the instruction.
- Use short noun phrases for entity names.
- Countable things are objects.
- {REGION_DEFINITION}
- Absolute Spatial Grounding: a region must be a physically anchorable area with computable boundaries.
- Semantic Self-Containment: the region phrase must be understandable on its own.
- Direct Target of Action: if the navigation verb acts on a whole spatial area, use that full area as the region.
- Nested Entity Allowance: a region may include an object anchor, for example "the right side of the center unit".
- Single-word place names such as "kitchen" are valid regions because they denote self-contained spaces.
- Self-contained opening or area phrases such as "right side doorway" are also valid regions when they denote a navigable spatial target.
- Avoid incomplete relative phrases such as "right side" when they are not self-contained.
- Keep only entities that matter for supervising the current subtask.
- The subtasks must stay in chronological order.
- The frame ranges can be rough in this step, but they should be contiguous.
- Return JSON only.
""".strip()

    return [
        {
            "role": "system",
            "content": "You are a careful VLN annotation assistant. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                build_video_url_content_item(video_path, fps=fps),
                {"type": "text", "text": prompt},
            ],
        },
    ]


def build_frame_refinement_messages(
    frame_paths: Sequence[Path],
    instruction: str,
    total_frames: int,
    fps: float,
    rough_subtasks: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    rough_payload = json.dumps(
        {"subtasks": list(rough_subtasks)},
        ensure_ascii=False,
        indent=2,
    )
    prompt = f"""
You are refining the subtask segmentation for the same VLN episode.

Instruction:
{instruction}

The provided image list is the complete ordered sequence of all {total_frames} frames.
The first image is frame 1 and the last image is frame {total_frames}.

Use this rough segmentation as a starting point:
{rough_payload}

Return JSON only with this schema:
{{
  "subtasks": [
    {{
      "subtask_id": "s1",
      "name": "short title",
      "instruction_span": "exact words copied from the instruction",
      "description": "same meaning as the copied instruction span",
      "start_frame": 1,
      "end_frame": {total_frames},
      "evidence": "brief visual evidence",
      "objects": [{{"name": "object name"}}],
      "regions": [{{"name": "region name"}}]
    }}
  ]
}}

Rules:
- Keep the subtask semantics faithful to the instruction itself.
- Each subtask must map to a contiguous instruction_span copied from the instruction.
- The ordered set of instruction_span values should cover the whole instruction meaning.
- Do not introduce new landmarks that are not explicitly named in the instruction.
- Output exact frame boundaries.
- Every frame from 1 to {total_frames} must belong to exactly one subtask.
- No gaps and no overlaps.
- Preserve the chronological order.
- {REGION_DEFINITION}
- Absolute Spatial Grounding: a region must be a physically anchorable area with computable boundaries.
- Semantic Self-Containment: the region phrase must be understandable on its own.
- Direct Target of Action: if the navigation verb acts on a whole spatial area, use that full area as the region.
- Nested Entity Allowance: a region may include an object anchor, for example "the right side of the center unit".
- Single-word place names such as "kitchen" are valid regions because they denote self-contained spaces.
- Self-contained opening or area phrases such as "right side doorway" are also valid regions when they denote a navigable spatial target.
- Avoid incomplete relative phrases such as "right side" when they are not self-contained.
- Return JSON only.
""".strip()

    return [
        {
            "role": "system",
            "content": "You are a careful VLN annotation assistant. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                build_video_frames_content_item(frame_paths, fps=fps),
                {"type": "text", "text": prompt},
            ],
        },
    ]


def build_localization_messages(
    frame_path: Path,
    frame_id: int,
    total_frames: int,
    image_width: int,
    image_height: int,
    instruction: str,
    subtask: Dict[str, Any],
    entities: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    entity_payload = json.dumps(
        [{"name": item["name"], "type": item["type"]} for item in entities],
        ensure_ascii=False,
        indent=2,
    )
    prompt = f"""
You are labeling frame {frame_id} of {total_frames} for a VLN navigation episode.

Instruction:
{instruction}

Current subtask:
{json.dumps(
    {
        "subtask_id": subtask["subtask_id"],
        "instruction_span": subtask.get("instruction_span"),
        "name": subtask["name"],
        "description": subtask["description"],
        "start_frame": subtask["start_frame"],
        "end_frame": subtask["end_frame"],
    },
    ensure_ascii=False,
    indent=2,
)}

Entities that must be checked in this frame:
{entity_payload}

Image size: width={image_width}, height={image_height}

Return JSON only with this schema:
{{
  "frame_id": {frame_id},
  "entities": [
    {{
      "name": "entity name",
      "type": "object or region",
      "visible": true,
      "center": [x, y],
      "bbox": [x1, y1, x2, y2] or null,
      "distance_m": 1.8,
      "distance_range_m": [1.2, 2.5],
      "confidence": "high|medium|low",
      "notes": "short reason"
    }}
  ]
}}

Rules:
- Output exactly one entry for every requested entity.
- If an entity is missing or too uncertain in the current frame, set "visible": false.
- If "visible" is false, both "center" and "bbox" must be null.
- If "visible" is false, both "distance_m" and "distance_range_m" must be null.
- {REGION_DEFINITION}
- A region entry may be a self-contained place name such as "kitchen", an opening-like phrase such as "right side doorway", or an anchored spatial phrase such as "the right side of the center unit".
- For objects, provide a bbox, center, and "distance_m". Set "distance_range_m" to null.
- For regions, focus on "visible", "center", and "distance_range_m". Region bbox is optional and may be null.
- Do not spend effort outlining precise region boundaries.
- Be conservative. When unsure, prefer visible=false.
- Pixel coordinates must refer to the current image only.
- Return JSON only.
""".strip()

    return [
        {
            "role": "system",
            "content": "You are a careful visual grounding assistant. Return JSON only.",
        },
        {
            "role": "user",
            "content": [
                build_image_content_item(frame_path),
                {"type": "text", "text": prompt},
            ],
        },
    ]


def normalize_center(
    value: Any,
    width: int,
    height: int,
) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    x = coerce_int(value[0], -1)
    y = coerce_int(value[1], -1)
    if x < 0 or y < 0:
        return None
    return [min(max(0, x), width - 1), min(max(0, y), height - 1)]


def normalize_bbox(
    value: Any,
    width: int,
    height: int,
) -> Optional[List[int]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    coords = [coerce_int(item, -1) for item in value]
    if any(coord < 0 for coord in coords):
        return None
    x1, y1, x2, y2 = coords
    x1, x2 = sorted((x1, x2))
    y1, y2 = sorted((y1, y2))
    x1 = min(max(0, x1), width - 1)
    x2 = min(max(0, x2), width - 1)
    y1 = min(max(0, y1), height - 1)
    y2 = min(max(0, y2), height - 1)
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def bbox_center(bbox: Sequence[int]) -> List[int]:
    x1, y1, x2, y2 = bbox
    return [int(round((x1 + x2) / 2.0)), int(round((y1 + y2) / 2.0))]


def normalize_localization_payload(
    raw_payload: Dict[str, Any],
    entities: Sequence[Dict[str, Any]],
    image_width: int,
    image_height: int,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    raw_items = raw_payload.get("entities") or raw_payload.get("frame_observations") or []
    raw_lookup = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        name = normalize_name(
            item.get("name") or item.get("label") or item.get("entity")
        )
        if name:
            raw_lookup[name] = item

    objects: List[Dict[str, Any]] = []
    regions: List[Dict[str, Any]] = []
    for entity in entities:
        raw_item = raw_lookup.get(entity["name"], {})
        visible = bool(raw_item.get("visible", False))
        bbox = None
        center = None
        distance_m = None
        distance_range_m = None

        if visible:
            center = normalize_center(
                raw_item.get("center"), image_width, image_height
            )
            raw_bbox = normalize_bbox(raw_item.get("bbox"), image_width, image_height)
            if entity["type"] == "object":
                bbox = raw_bbox
                if center is None and bbox is not None:
                    center = bbox_center(bbox)
                if center is None:
                    visible = False
                    bbox = None
            else:
                if center is None and raw_bbox is not None:
                    center = bbox_center(raw_bbox)
                if center is None:
                    visible = False

        if visible:
            if entity["type"] == "object":
                distance_m = coerce_float(raw_item.get("distance_m"))
            else:
                distance_range_m = coerce_float_range(
                    raw_item.get("distance_range_m")
                )

        record: Dict[str, Any] = {
            "entity_id": entity["entity_id"],
            "name": entity["name"],
            "type": entity["type"],
            "visible": visible,
            "center": center if visible else None,
            "bbox": bbox if visible and entity["type"] == "object" else None,
            "confidence": raw_item.get("confidence") if visible else None,
            "notes": raw_item.get("notes") if visible else None,
        }

        if entity["type"] == "object":
            record["distance_m"] = distance_m if visible else None
            objects.append(record)
        else:
            record["distance_range_m"] = distance_range_m if visible else None
            regions.append(record)

    return objects, regions


def build_failed_visibility_records(
    entities: Sequence[Dict[str, Any]],
    error_message: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    objects: List[Dict[str, Any]] = []
    regions: List[Dict[str, Any]] = []
    for entity in entities:
        record: Dict[str, Any] = {
            "entity_id": entity["entity_id"],
            "name": entity["name"],
            "type": entity["type"],
            "visible": False,
            "center": None,
            "bbox": None,
            "confidence": None,
            "notes": f"Auto-filled after repeated localization failure: {error_message}",
        }
        if entity["type"] == "object":
            record["distance_m"] = None
            objects.append(record)
        else:
            record["distance_range_m"] = None
            regions.append(record)
    return objects, regions


def draw_label(
    image,
    text: str,
    origin: Tuple[int, int],
    color_bgr: Tuple[int, int, int],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    thickness = 1
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x = max(0, origin[0])
    y = max(text_size[1] + baseline + 4, origin[1])
    x2 = min(image.shape[1] - 1, x + text_size[0] + 8)
    y2 = min(image.shape[0] - 1, y + baseline + 4)
    cv2.rectangle(
        image,
        (x, y - text_size[1] - baseline - 4),
        (x2, y2),
        (0, 0, 0),
        -1,
    )
    cv2.rectangle(
        image,
        (x, y - text_size[1] - baseline - 4),
        (x2, y2),
        color_bgr,
        1,
    )
    cv2.putText(
        image,
        text,
        (x + 4, y - baseline - 2),
        font,
        font_scale,
        TEXT_COLOR,
        thickness,
        cv2.LINE_AA,
    )


def annotate_frame(
    image,
    objects: Sequence[Dict[str, Any]],
    regions: Sequence[Dict[str, Any]],
):
    annotated = image.copy()
    for entity in list(objects):
        if not entity["visible"]:
            continue
        color = OBJECT_COLOR_BGR
        bbox = entity.get("bbox")
        center = entity.get("center")
        if bbox is not None:
            x1, y1, x2, y2 = bbox
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 1)
        if center is not None:
            cv2.circle(annotated, tuple(center), 4, color, -1)
            cv2.circle(annotated, tuple(center), 8, color, 1)
            distance_text = (
                f"{entity['distance_m']:.2f}m"
                if entity.get("distance_m") is not None
                else "dist=?"
            )
            draw_label(
                annotated,
                f"{entity['name']} | {distance_text}",
                (center[0] + 6, center[1] - 6),
                color,
            )
    return annotated


def write_annotated_video(
    annotated_frame_paths: Sequence[Path],
    output_path: Path,
    fps: float,
) -> None:
    if not annotated_frame_paths:
        return

    first_frame = cv2.imread(str(annotated_frame_paths[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read {annotated_frame_paths[0]} for video export.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (first_frame.shape[1], first_frame.shape[0]),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_path}")

    try:
        for frame_path in annotated_frame_paths:
            frame = cv2.imread(str(frame_path))
            if frame is None:
                raise RuntimeError(f"Failed to read annotated frame {frame_path}")
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    run_tag = timestamp_tag()
    episode, gt_episode = load_episode_bundle(
        train_json_path=args.train_json,
        train_gt_json_path=args.train_gt_json,
        episode_id=args.episode_id,
    )
    instruction = episode["instruction"]["instruction_text"].strip()
    frame_paths = discover_episode_frames(args.frames_dir, args.episode_id)
    total_frames = len(frame_paths)
    if total_frames != len(gt_episode["actions"]):
        raise ValueError(
            f"Found {total_frames} frames, but train_gt.json contains "
            f"{len(gt_episode['actions'])} actions for episode {args.episode_id}."
        )

    episode_root = args.output_dir / f"episode_{args.episode_id}_{run_tag}"
    cache_dir = episode_root / f"cache_{run_tag}"
    raw_response_dir = cache_dir / f"raw_responses_{run_tag}"
    annotated_dir = episode_root / f"annotated_frames_{run_tag}"
    frame_results_dir = episode_root / f"frame_results_{run_tag}"
    localization_cache_dir = cache_dir / f"localization_{run_tag}"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    frame_results_dir.mkdir(parents=True, exist_ok=True)
    localization_cache_dir.mkdir(parents=True, exist_ok=True)
    raw_response_dir.mkdir(parents=True, exist_ok=True)

    log(
        f"[start] run={run_tag} episode={args.episode_id} frames={total_frames} "
        f"segment_model={args.segment_model} localize_model={args.localize_model}"
    )
    qwen_request_settings = build_qwen_request_settings(args)
    log(
        "[qwen] "
        f"temperature={qwen_request_settings['temperature']} "
        f"top_p={qwen_request_settings['top_p']} "
        f"result_format={qwen_request_settings['result_format']} "
        f"thinking={qwen_request_settings['enable_thinking']} "
        f"thinking_budget={qwen_request_settings['thinking_budget']}"
    )
    log(f"[instruction] clauses={split_instruction_clauses(instruction)}")
    log(f"[paths] video={args.video_path}")
    log(f"[paths] output={episode_root}")
    if args.reuse_run_dir is not None:
        log(f"[paths] reuse_run_dir={args.reuse_run_dir}")

    qwen_client = QwenClient(
        default_model=args.segment_model,
        default_temperature=args.qwen_temperature,
        default_top_p=args.qwen_top_p,
        default_result_format=args.qwen_result_format,
        default_enable_thinking=args.thinking,
        default_thinking_budget=args.thinking_budget,
    )

    if args.reuse_run_dir is not None:
        log("[reuse] Loading validated subtasks from the previous run directory...")
        subtasks = refresh_entity_ids(
            constrain_entities_to_instruction(
                align_subtasks_to_instruction(
                    load_subtasks_from_run_dir(args.reuse_run_dir),
                    instruction,
                ),
                instruction,
            )
        )
        write_json(
            cache_dir / f"reused_subtasks_{run_tag}.json",
            {"source_run_dir": str(args.reuse_run_dir), "subtasks": subtasks},
        )
        log(f"[reuse] Loaded subtasks: {summarize_subtasks(subtasks)}")
    else:
        log("[1/3] Segmenting the full video into rough subtasks...")
        rough_segmentation_raw = load_or_create_json(
            cache_dir / "video_segmentation_raw.json",
            overwrite=args.overwrite,
            factory=lambda: qwen_client.chat_json(
                messages=build_video_segmentation_messages(
                    video_path=args.video_path,
                    instruction=instruction,
                    total_frames=total_frames,
                    fps=args.video_fps,
                ),
                model=args.segment_model,
                max_tokens=4000,
                raw_text_path=raw_response_dir / f"video_segmentation_raw_text_{run_tag}.txt",
            ),
        )
        rough_subtasks = refresh_entity_ids(
            constrain_entities_to_instruction(
                align_subtasks_to_instruction(
                    normalize_subtasks(rough_segmentation_raw, total_frames),
                    instruction,
                ),
                instruction,
            )
        )
        write_json(
            cache_dir / f"video_segmentation_normalized_{run_tag}.json",
            {"subtasks": rough_subtasks},
        )
        log(f"[1/3] Rough subtasks saved: {summarize_subtasks(rough_subtasks)}")

        log("[2/3] Refining subtask boundaries with the ordered frame list...")
        refined_segmentation_raw = load_or_create_json(
            cache_dir / "frame_refinement_raw.json",
            overwrite=args.overwrite,
            factory=lambda: qwen_client.chat_json(
                messages=build_frame_refinement_messages(
                    frame_paths=frame_paths,
                    instruction=instruction,
                    total_frames=total_frames,
                    fps=args.video_fps,
                    rough_subtasks=rough_subtasks,
                ),
                model=args.segment_model,
                max_tokens=4000,
                raw_text_path=raw_response_dir / f"frame_refinement_raw_text_{run_tag}.txt",
            ),
        )
        refined_subtasks = refresh_entity_ids(
            constrain_entities_to_instruction(
                align_subtasks_to_instruction(
                    normalize_subtasks(refined_segmentation_raw, total_frames),
                    instruction,
                ),
                instruction,
            )
        )
        subtasks = refresh_entity_ids(
            constrain_entities_to_instruction(
                align_subtasks_to_instruction(
                    merge_subtasks(refined_subtasks, rough_subtasks),
                    instruction,
                ),
                instruction,
            )
        )
        write_json(
            cache_dir / f"frame_refinement_normalized_{run_tag}.json",
            {"subtasks": subtasks},
        )
        log(f"[2/3] Refined subtasks saved: {summarize_subtasks(subtasks)}")

    frame_to_subtask = build_frame_lookup(subtasks)
    frame_records: List[Dict[str, Any]] = []
    if args.reuse_run_dir is not None:
        frame_records = load_existing_frame_records(args.reuse_run_dir)
        if frame_records:
            log(
                f"[reuse] Loaded {len(frame_records)} existing frame results "
                f"from the previous run."
            )
    processed_frame_ids = {int(item["frame_id"]) for item in frame_records}
    annotated_frame_paths: List[Path] = [
        Path(item["annotated_frame_file"])
        for item in frame_records
        if item.get("annotated_frame_file")
        and Path(item["annotated_frame_file"]).exists()
    ]
    annotated_video_path = episode_root / f"annotated_episode_{run_tag}.mp4"

    save_checkpoint(
        episode_root,
        build_result_payload(
            episode_id=args.episode_id,
            scene_id=episode["scene_id"],
            instruction=instruction,
            video_path=args.video_path,
            frames_dir=args.frames_dir,
            annotated_dir=annotated_dir,
            annotated_video_path=annotated_video_path,
            segment_model=args.segment_model,
            localize_model=args.localize_model,
            qwen_request_settings=qwen_request_settings,
            total_frames=total_frames,
            action_sequence=gt_episode["actions"],
            subtasks=subtasks,
            frame_records=frame_records,
            phase="subtasks_ready",
            run_tag=run_tag,
        ),
        run_tag=run_tag,
    )
    log("[checkpoint] Saved initial subtask result bundle.")

    log("[3/3] Starting frame-by-frame localization and distance estimation...")
    for frame_id, frame_path in enumerate(frame_paths, start=1):
        if frame_id in processed_frame_ids:
            log(f"[frame {frame_id:02d}/{total_frames}] already available from reused run; skipping")
            continue
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Failed to read frame image {frame_path}")
        subtask = frame_to_subtask[frame_id]
        entities = list(subtask["objects"]) + list(subtask["regions"])
        log(
            f"[frame {frame_id:02d}/{total_frames}] subtask={subtask['subtask_id']} "
            f"entities={len(entities)} file={frame_path.name}"
        )

        attempt_errors: List[Dict[str, Any]] = []
        succeeded = False
        for attempt in range(1, 4):
            try:
                log(f"[frame {frame_id:02d}/{total_frames}] attempt={attempt}/3")
                localization_raw = load_or_create_json(
                    localization_cache_dir / f"frame_{frame_id:04d}_attempt_{attempt}_{run_tag}.json",
                    overwrite=args.overwrite,
                    factory=lambda frame_path=frame_path, frame_id=frame_id, subtask=subtask, entities=entities, image=image, attempt=attempt: qwen_client.chat_json(
                        messages=build_localization_messages(
                            frame_path=frame_path,
                            frame_id=frame_id,
                            total_frames=total_frames,
                            image_width=image.shape[1],
                            image_height=image.shape[0],
                            instruction=instruction,
                            subtask=subtask,
                            entities=entities,
                        ),
                        model=args.localize_model,
                        max_tokens=2000,
                        raw_text_path=raw_response_dir / f"frame_{frame_id:04d}_attempt_{attempt}_raw_text_{run_tag}.txt",
                    ),
                )

                objects, regions = normalize_localization_payload(
                    raw_payload=localization_raw,
                    entities=entities,
                    image_width=image.shape[1],
                    image_height=image.shape[0],
                )

                annotated_image = annotate_frame(image, objects, regions)
                annotated_path = annotated_dir / f"{frame_path.stem}_{run_tag}{frame_path.suffix}"
                cv2.imwrite(str(annotated_path), annotated_image)
                annotated_frame_paths.append(annotated_path)

                frame_records.append(
                    {
                        "frame_id": frame_id,
                        "frame_file": str(frame_path),
                        "annotated_frame_file": str(annotated_path),
                        "action": int(gt_episode["actions"][frame_id - 1]),
                        "subtask_id": subtask["subtask_id"],
                        "instruction_span": subtask.get("instruction_span"),
                        "objects": objects,
                        "regions": regions,
                    }
                )
                write_json(
                    frame_results_dir / f"frame_{frame_id:04d}_{run_tag}.json",
                    frame_records[-1],
                )
                save_checkpoint(
                    episode_root,
                    build_result_payload(
                        episode_id=args.episode_id,
                        scene_id=episode["scene_id"],
                        instruction=instruction,
                        video_path=args.video_path,
                        frames_dir=args.frames_dir,
                        annotated_dir=annotated_dir,
                        annotated_video_path=annotated_video_path,
                        segment_model=args.segment_model,
                        localize_model=args.localize_model,
                        qwen_request_settings=qwen_request_settings,
                        total_frames=total_frames,
                        action_sequence=gt_episode["actions"],
                        subtasks=subtasks,
                        frame_records=frame_records,
                        phase="localizing_frames",
                        run_tag=run_tag,
                    ),
                    run_tag=run_tag,
                )
                visible_objects = sum(1 for item in objects if item["visible"])
                visible_regions = sum(1 for item in regions if item["visible"])
                log(
                    f"[frame {frame_id:02d}/{total_frames}] saved "
                    f"objects_visible={visible_objects}/{len(objects)} "
                    f"regions_visible={visible_regions}/{len(regions)}"
                )
                succeeded = True
                break
            except Exception as exc:
                attempt_error = {
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                    "raw_response_path": str(
                        raw_response_dir
                        / f"frame_{frame_id:04d}_attempt_{attempt}_raw_text_{run_tag}.txt"
                    ),
                    "response_debug_path": str(
                        raw_response_dir
                        / f"frame_{frame_id:04d}_attempt_{attempt}_raw_text_{run_tag}_response.json"
                    ),
                }
                attempt_errors.append(attempt_error)
                log(
                    f"[frame {frame_id:02d}/{total_frames}] attempt={attempt}/3 "
                    f"failed type={type(exc).__name__}"
                )

        if succeeded:
            continue

        try:
            objects, regions = build_failed_visibility_records(
                entities=entities,
                error_message=attempt_errors[-1]["error_message"],
            )
            annotated_path = annotated_dir / f"{frame_path.stem}_{run_tag}{frame_path.suffix}"
            cv2.imwrite(str(annotated_path), image)
            annotated_frame_paths.append(annotated_path)
            frame_records.append(
                {
                    "frame_id": frame_id,
                    "frame_file": str(frame_path),
                    "annotated_frame_file": str(annotated_path),
                    "action": int(gt_episode["actions"][frame_id - 1]),
                    "subtask_id": subtask["subtask_id"],
                    "instruction_span": subtask.get("instruction_span"),
                    "objects": objects,
                    "regions": regions,
                    "frame_status": "failed_after_retries",
                }
            )
            write_json(
                frame_results_dir / f"frame_{frame_id:04d}_{run_tag}.json",
                frame_records[-1],
            )
            error_payload = {
                "frame_id": frame_id,
                "frame_file": str(frame_path),
                "error_type": attempt_errors[-1]["error_type"],
                "error_message": attempt_errors[-1]["error_message"],
                "attempts": attempt_errors,
            }
            write_json(
                episode_root / f"error_frame_{frame_id:04d}_{run_tag}.json",
                error_payload,
            )
            save_checkpoint(
                episode_root,
                build_result_payload(
                    episode_id=args.episode_id,
                    scene_id=episode["scene_id"],
                    instruction=instruction,
                    video_path=args.video_path,
                    frames_dir=args.frames_dir,
                    annotated_dir=annotated_dir,
                    annotated_video_path=annotated_video_path,
                    segment_model=args.segment_model,
                    localize_model=args.localize_model,
                    qwen_request_settings=qwen_request_settings,
                    total_frames=total_frames,
                    action_sequence=gt_episode["actions"],
                    subtasks=subtasks,
                    frame_records=frame_records,
                    phase="localizing_frames",
                    run_tag=run_tag,
                    error=None,
                ),
                run_tag=run_tag,
            )
            log(
                f"[error] frame={frame_id} failed after 3 attempts; "
                f"details saved to {episode_root / f'error_frame_{frame_id:04d}_{run_tag}.json'}; continuing"
            )
        except Exception:
            raise

    log("[finalize] Building annotated video from saved frames...")
    write_annotated_video(
        annotated_frame_paths=annotated_frame_paths,
        output_path=annotated_video_path,
        fps=args.video_fps,
    )
    save_checkpoint(
        episode_root,
        build_result_payload(
            episode_id=args.episode_id,
            scene_id=episode["scene_id"],
            instruction=instruction,
            video_path=args.video_path,
            frames_dir=args.frames_dir,
            annotated_dir=annotated_dir,
            annotated_video_path=annotated_video_path,
            segment_model=args.segment_model,
            localize_model=args.localize_model,
            qwen_request_settings=qwen_request_settings,
            total_frames=total_frames,
            action_sequence=gt_episode["actions"],
            subtasks=subtasks,
            frame_records=frame_records,
            phase="completed",
            run_tag=run_tag,
        ),
        run_tag=run_tag,
    )
    log(
        f"[done] Saved final results to "
        f"{episode_root / f'qwen_episode_annotations_{run_tag}.json'}"
    )
    log(f"[done] Saved per-frame normalized results to {frame_results_dir}")
    log(f"[done] Saved annotated video to {annotated_video_path}")


if __name__ == "__main__":
    main()
