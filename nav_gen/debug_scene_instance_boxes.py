import argparse
import json
import os
import re
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np

from export_trajectory_rgb_video import (
    _build_args,
    _capture_frame,
    _coerce_list,
    _collect_scene_candidates,
    _collect_target_candidates,
    _draw_target_boxes,
    _extract_target_boxes,
    _flatten_entries,
    _load_json,
)
from habitat_base.simulation import SceneSimulator


BASIC_IGNORE_CATEGORY_SUBSTRINGS = ("ceiling", "floor")

# Stronger structural/background filter for semantic instances.
# This intentionally removes scene envelope and other non-actionable clutter
# that otherwise dominates the per-frame frequency counts.
STRONG_IGNORE_CATEGORY_SUBSTRINGS = (
    "unknown",
    "wall",
    "frame",
    "floor",
    "ceiling",
    "window",
    "curtain",
    "sheet",
    "stair",
    "stairs",
    "stairwell",
    "beam",
    "decoration",
    "door frame",
    "doorway",
    "door",
    "pillar",
    "column",
    "arch",
    "railing",
    "rail",
    "balustrade",
    "banister",
    "trim",
    "baseboard",
    "vent",
    "outlet",
    "switch",
    "socket",
    "ceiling lamp",
    "light fixture",
    "pillow",
)


def _safe_name(text, max_length=80):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text).strip())
    safe = safe.strip("._-")
    if not safe:
        safe = "item"
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("._-")
    return safe


def _safe_target_dir_name(text):
    return str(text or "").replace("/", "")


def _should_skip_category(category, ignore_category_substrings):
    category = str(category or "").lower()
    return any(token in category for token in ignore_category_substrings)


def _resolve_ignore_category_substrings(filter_profile, extra_ignore_substrings):
    if filter_profile == "basic":
        values = list(BASIC_IGNORE_CATEGORY_SUBSTRINGS)
    elif filter_profile == "strong":
        values = list(STRONG_IGNORE_CATEGORY_SUBSTRINGS)
    else:
        raise ValueError(f"Unsupported filter profile: {filter_profile}")

    for item in extra_ignore_substrings:
        text = str(item).strip().lower()
        if text and text not in values:
            values.append(text)
    return tuple(values)


def _find_debug_jsons(debug_jsons, debug_root):
    paths = []
    for path in debug_jsons:
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"debug json not found: {resolved}")
        paths.append(resolved)

    if debug_root:
        root = Path(debug_root).resolve()
        if not root.exists():
            raise FileNotFoundError(f"debug root not found: {root}")
        paths.extend(sorted(root.rglob("result3_step_tags_and_images.json")))

    unique = []
    seen = set()
    for path in paths:
        if path not in seen:
            seen.add(path)
            unique.append(path)
    if not unique:
        raise ValueError("No result3_step_tags_and_images.json files were provided or found.")
    return unique


def _find_ancestor_with_name(path, name):
    current = Path(path).resolve()
    for candidate in (current, *current.parents):
        if candidate.name == name:
            return candidate
    return None


def _candidate_task_json_from_debug_path(debug_json_path):
    split_root = _find_ancestor_with_name(debug_json_path, "split_traj_debug")
    if split_root is None:
        return []

    candidates = []
    rel_parts = debug_json_path.relative_to(split_root).parts
    if len(rel_parts) >= 2:
        task_rel = Path(rel_parts[0]) / rel_parts[1]
        base_dir = split_root.parent.parent
        candidates.append(base_dir / "task" / task_rel / "success" / "trial_1" / "task.json")
    return candidates


def _resolve_task_json(debug_json_path, payload):
    candidates = []

    source_task_path = payload.get("source_task_path")
    if source_task_path:
        source_task_path = Path(source_task_path)
        candidates.append(source_task_path / "success" / "trial_1" / "task.json")
        candidates.append(source_task_path / "task.json")

    candidates.extend(_candidate_task_json_from_debug_path(debug_json_path))

    seen = set()
    for candidate in candidates:
        candidate = Path(candidate).resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        "Could not resolve the source task.json for debug file "
        f"`{debug_json_path}`. Checked: {[str(path) for path in seen]}"
    )


def _resolve_replay_settings(payload, cli_args):
    debug_settings = payload.get("debug_render_settings") or {}

    width = int(debug_settings.get("render_width", cli_args.render_width))
    height = int(debug_settings.get("render_height", cli_args.render_height))

    sensor_height = cli_args.render_sensor_height
    sensor_height_source = "cli"
    if sensor_height is None and debug_settings.get("render_sensor_height_m") is not None:
        sensor_height = float(debug_settings["render_sensor_height_m"])
        sensor_height_source = "debug_json"
    elif sensor_height is None:
        raise ValueError(
            "Missing replay camera height for a legacy split_traj debug sample. "
            "This result3_step_tags_and_images.json does not contain "
            "`debug_render_settings.render_sensor_height_m`, so falling back to the "
            "robot-default camera height can shift boxes and masks vertically. "
            "Please rerun with `--render-sensor-height <meters>`."
        )

    hfov = float(debug_settings.get("sensor_hfov_deg", cli_args.sensor_hfov))
    hfov_source = "debug_json" if debug_settings.get("sensor_hfov_deg") is not None else "cli"

    return {
        "width": width,
        "height": height,
        "sensor_height": sensor_height,
        "hfov": hfov,
        "debug_render_settings": debug_settings,
        "effective_replay_settings": {
            "render_width": width,
            "render_height": height,
            "render_sensor_height_m": float(sensor_height),
            "sensor_hfov_deg": hfov,
            "sources": {
                "render_width": "debug_json" if debug_settings.get("render_width") is not None else "cli",
                "render_height": "debug_json" if debug_settings.get("render_height") is not None else "cli",
                "render_sensor_height_m": sensor_height_source,
                "sensor_hfov_deg": hfov_source,
            },
        },
    }


def _collect_scene_instance_index(simulator):
    index = {}
    for obj in simulator.sim.semantic_scene.objects:
        try:
            category_name = obj.category.name()
        except Exception:
            continue
        semantic_id = getattr(obj, "semantic_id", None)
        center = getattr(getattr(obj, "aabb", None), "center", None)
        if semantic_id is None or center is None or not category_name:
            continue
        index[int(semantic_id)] = {
            "semantic_id": int(semantic_id),
            "object_id": getattr(obj, "id", None),
            "category": category_name,
            "center": [float(value) for value in center],
        }
    return index


def _build_saved_observation_lookup(task_config):
    lookup = {}
    for entry in _flatten_entries(task_config):
        global_step = entry.get("global_step")
        action = entry.get("action")
        target = _safe_target_dir_name(entry.get("target"))
        if global_step is None or action is None:
            continue
        dirname = f"{global_step}_{action}_for_{target}"
        lookup[dirname] = entry
    return lookup


def _semantic_to_target_shape(semantic_obs, target_height, target_width):
    if semantic_obs is None:
        return None
    if semantic_obs.shape[0] == target_height and semantic_obs.shape[1] == target_width:
        return semantic_obs
    return cv2.resize(
        semantic_obs.astype(np.int32),
        (target_width, target_height),
        interpolation=cv2.INTER_NEAREST,
    )


def _resolve_saved_debug_image_path(debug_json_path, step_record, image_index, image_record):
    candidates = []

    saved_image = image_record.get("saved_image")
    if saved_image:
        candidates.append(Path(saved_image))

    step_dir_name = f"step_{step_record['step_index']:03d}_{_safe_name(step_record['label'])}"
    image_name = Path(saved_image).name if saved_image else f"front_{image_index:03d}.png"
    candidates.append(
        debug_json_path.parent / "result3_step_images" / step_dir_name / image_name
    )
    candidates.append(
        debug_json_path.parent / "result3_step_images" / step_dir_name / f"front_{image_index:03d}.png"
    )

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Could not resolve an existing debug image for step {step_record['step_index']} "
        f"image {image_index}. Checked: {[str(path) for path in seen]}"
    )


class TaskReplayContext:
    def __init__(self, task_json_path, width, height, sensor_height, hfov, sim_gpu_device):
        self.task_json_path = Path(task_json_path).resolve()
        self.task_config = _load_json(str(self.task_json_path))
        project_root = self.task_json_path.parent
        while project_root.name != "nav_gen":
            if project_root.parent == project_root:
                raise RuntimeError(
                    f"Could not locate nav_gen root for task json: {self.task_json_path}"
                )
            project_root = project_root.parent
        project_root = project_root.parent

        args = _build_args(
            str(project_root),
            width=width,
            height=height,
            sensor_height=sensor_height,
            hfov=hfov,
            sim_gpu_device=sim_gpu_device,
            enable_front_semantic=True,
        )
        args.front_rgb_only = True
        args.front_semantic_sensor = True
        args.render_width = width
        args.render_height = height
        args.render_sensor_height = sensor_height
        args.sensor_hfov = hfov

        self.simulator = SceneSimulator(args, self.task_config)
        self.scene_index = _collect_scene_instance_index(self.simulator)
        self.observation_lookup = _build_saved_observation_lookup(self.task_config)
        self.semantic_cache = {}

    def get_semantic_for_source_image(self, source_image_path, target_height, target_width):
        dir_name = Path(source_image_path).parent.name
        cache_key = (dir_name, target_height, target_width)
        if cache_key in self.semantic_cache:
            return self.semantic_cache[cache_key]

        entry = self.observation_lookup.get(dir_name)
        if entry is None:
            raise KeyError(
                f"Could not map saved observation directory `{dir_name}` back to task.json entries."
            )

        _, semantic_obs = _capture_frame(
            self.simulator,
            entry["pos"],
            entry["yaw"],
        )
        resized = _semantic_to_target_shape(semantic_obs, target_height, target_width)
        self.semantic_cache[cache_key] = resized
        return resized

    def close(self):
        if getattr(self, "simulator", None) is not None:
            self.simulator.sim.close()
            self.simulator = None


def _build_instance_label(instance_record):
    object_id = str(instance_record.get("object_id") or "").strip()
    if object_id:
        return object_id
    category = str(instance_record.get("category") or "object").strip().replace(" ", "_")
    semantic_id = instance_record.get("semantic_id")
    return f"{category}_{semantic_id}"


def _serialize_ranked_instance(instance_record):
    return {
        **instance_record,
        "tag": _build_instance_label(instance_record),
        "count": int(instance_record.get("frame_count", 0)),
        "label": _build_instance_label(instance_record),
    }


def _parse_subtraj_index(debug_json_path):
    match = re.match(r"subtraj_(\d+)_", debug_json_path.parent.name)
    if match is None:
        return None
    return int(match.group(1))


def _normalize_text(value):
    return str(value or "").strip().lower()


def _resolve_declared_target_from_global_start(context, payload):
    global_start = payload.get("subtrajectory_global_start")
    try:
        global_start = int(global_start)
    except (TypeError, ValueError):
        return None

    for entry in _flatten_entries(context.task_config):
        if entry.get("global_step") != global_start:
            continue
        return {
            "target_index": entry.get("trial_index"),
            "category": entry.get("target"),
            "region_id": entry.get("target_region_id"),
            "region_name": entry.get("target_region_name"),
            "match_source": "global_step_exact",
            "matched_global_step": global_start,
        }
    return None


def _resolve_declared_target(context, debug_json_path, payload):
    payload_target = payload.get("target")
    task_objects = _coerce_list(context.task_config.get("Object"))
    task_regions = _coerce_list(context.task_config.get("Region"))
    target_index = None
    category = payload_target
    region_id = None
    region_name = None
    match_source = "payload_target_only"
    matched_global_step = None
    subtraj_index = _parse_subtraj_index(debug_json_path)

    resolved_from_start = _resolve_declared_target_from_global_start(context, payload)
    if resolved_from_start is not None:
        target_index = resolved_from_start.get("target_index")
        category = resolved_from_start.get("category") or category
        region_id = resolved_from_start.get("region_id")
        region_name = resolved_from_start.get("region_name")
        match_source = resolved_from_start.get("match_source") or match_source
        matched_global_step = resolved_from_start.get("matched_global_step")
    else:
        normalized_payload_target = _normalize_text(payload_target)
        matching_indices = [
            index
            for index, obj in enumerate(task_objects)
            if _normalize_text(obj) == normalized_payload_target
        ]
        if len(matching_indices) == 1:
            target_index = matching_indices[0]
            category = task_objects[target_index] or category
            if target_index < len(task_regions):
                region_id = task_regions[target_index]
            match_source = "payload_target_unique_match"
        elif subtraj_index is not None and subtraj_index < len(task_objects):
            target_index = subtraj_index
            category = task_objects[target_index] or category
            if target_index < len(task_regions):
                region_id = task_regions[target_index]
            match_source = "subtraj_index_fallback"
        elif matching_indices:
            target_index = matching_indices[0]
            category = task_objects[target_index] or category
            if target_index < len(task_regions):
                region_id = task_regions[target_index]
            match_source = "payload_target_first_match"

    return {
        "target_index": target_index,
        "category": category,
        "region_id": region_id,
        "region_name": region_name,
        "match_source": match_source,
        "matched_global_step": matched_global_step,
        "payload_target": payload_target,
        "subtraj_index": subtraj_index,
    }


def _resolve_declared_target_instance(context, debug_json_path, payload):
    target_info = _resolve_declared_target(context, debug_json_path, payload)
    category = target_info["category"]
    region_id = target_info["region_id"]

    if not category:
        return {
            **target_info,
            "resolved": False,
            "reason": "missing_declared_target_category",
        }

    candidates = _collect_target_candidates(
        context.simulator,
        category=category,
        region_id=region_id,
    )
    candidate_source = "region"
    if not candidates:
        candidates = _collect_scene_candidates(context.simulator, category=category)
        candidate_source = "scene"

    if not candidates:
        return {
            **target_info,
            "resolved": False,
            "reason": "no_matching_semantic_object",
            "candidate_source": candidate_source,
            "candidate_count": 0,
        }

    stats_by_instance = {
        candidate["semantic_id"]: {
            "semantic_id": candidate["semantic_id"],
            "object_id": candidate["object_id"],
            "category": candidate["category"],
            "center": [float(value) for value in candidate["center"]],
            "frame_count": 0,
            "total_area": 0,
            "max_area": 0,
        }
        for candidate in candidates
    }

    for step_record in payload.get("steps", []):
        for image_index, image_record in enumerate(step_record.get("images", [])):
            saved_image_path = _resolve_saved_debug_image_path(
                debug_json_path,
                step_record,
                image_index,
                image_record,
            )
            frame = cv2.imread(str(saved_image_path))
            if frame is None:
                continue

            height, width = frame.shape[:2]
            semantic_obs = context.get_semantic_for_source_image(
                image_record["source_image"],
                target_height=height,
                target_width=width,
            )
            visible_ids, visible_counts = np.unique(semantic_obs, return_counts=True)
            for semantic_id, pixel_count in zip(visible_ids, visible_counts):
                semantic_id = int(semantic_id)
                pixel_count = int(pixel_count)
                instance_stats = stats_by_instance.get(semantic_id)
                if instance_stats is None:
                    continue
                instance_stats["frame_count"] += 1
                instance_stats["total_area"] += pixel_count
                instance_stats["max_area"] = max(instance_stats["max_area"], pixel_count)

    ranked_target_candidates = sorted(
        stats_by_instance.values(),
        key=lambda item: (
            -item["frame_count"],
            -item["total_area"],
            -item["max_area"],
            item["semantic_id"],
        ),
    )

    selected_target = ranked_target_candidates[0]
    if selected_target["frame_count"] > 0:
        selection_strategy = "most_visible_in_subtrajectory"
    else:
        selection_strategy = "first_matching_object"

    target_label = f"TARGET:{_build_instance_label(selected_target)}"
    return {
        **target_info,
        "resolved": True,
        "candidate_source": candidate_source,
        "candidate_count": len(candidates),
        "selection_strategy": selection_strategy,
        "selected_instance": {
            **selected_target,
            "label": target_label,
        },
        "target_spec": {
            "label": target_label,
            "semantic_ids": [selected_target["semantic_id"]],
            "semantic_labels": {
                selected_target["semantic_id"]: target_label,
            },
        },
        "all_candidate_rankings": [
            _serialize_ranked_instance(item)
            for item in ranked_target_candidates
        ],
    }


def _rank_instances_for_step(context, debug_json_path, step_record, ignore_category_substrings):
    stats_by_instance = {}
    per_image_data = []

    for image_index, image_record in enumerate(step_record.get("images", [])):
        saved_image_path = _resolve_saved_debug_image_path(
            debug_json_path,
            step_record,
            image_index,
            image_record,
        )
        frame = cv2.imread(str(saved_image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read debug image: {saved_image_path}")

        height, width = frame.shape[:2]
        semantic_obs = context.get_semantic_for_source_image(
            image_record["source_image"],
            target_height=height,
            target_width=width,
        )

        visible_instances = []
        visible_ids, visible_counts = np.unique(semantic_obs, return_counts=True)
        for semantic_id, pixel_count in zip(visible_ids, visible_counts):
            semantic_id = int(semantic_id)
            pixel_count = int(pixel_count)
            instance_info = context.scene_index.get(semantic_id)
            if instance_info is None or _should_skip_category(
                instance_info["category"],
                ignore_category_substrings,
            ):
                continue

            instance_stats = stats_by_instance.setdefault(
                semantic_id,
                {
                    "semantic_id": semantic_id,
                    "object_id": instance_info["object_id"],
                    "category": instance_info["category"],
                    "center": instance_info["center"],
                    "frame_count": 0,
                    "total_area": 0,
                    "max_area": 0,
                },
            )
            instance_stats["frame_count"] += 1
            instance_stats["total_area"] += pixel_count
            instance_stats["max_area"] = max(instance_stats["max_area"], pixel_count)

            visible_instances.append(
                {
                    "semantic_id": semantic_id,
                    "category": instance_info["category"],
                    "object_id": instance_info["object_id"],
                    "pixel_count": pixel_count,
                }
            )

        per_image_data.append(
            {
                "image_index": image_index,
                "image_record": {
                    **image_record,
                    "saved_image": str(saved_image_path),
                },
                "frame": frame,
                "semantic_obs": semantic_obs,
                "visible_instances": visible_instances,
            }
        )

    ranked_instances = sorted(
        stats_by_instance.values(),
        key=lambda item: (
            -item["frame_count"],
            -item["total_area"],
            -item["max_area"],
            item["category"],
            item["semantic_id"],
        ),
    )
    return ranked_instances, per_image_data


def _annotate_step(
    step_record,
    ranked_instances,
    per_image_data,
    output_dir,
    min_pixels,
    top_k,
    declared_target_resolution,
):
    os.makedirs(output_dir, exist_ok=True)

    selected_instances = ranked_instances[:top_k]
    selected_by_id = {item["semantic_id"]: item for item in selected_instances}
    declared_target_spec = (
        declared_target_resolution.get("target_spec")
        if declared_target_resolution and declared_target_resolution.get("resolved")
        else None
    )
    declared_target_semantic_ids = set()
    if declared_target_spec:
        declared_target_semantic_ids = set(declared_target_spec.get("semantic_ids", []))

    top_k_target_spec = [
        {
            "label": _build_instance_label(instance_record),
            "semantic_ids": [instance_record["semantic_id"]],
            "semantic_labels": {
                instance_record["semantic_id"]: _build_instance_label(instance_record)
            },
        }
        for instance_record in selected_instances
        if instance_record["semantic_id"] not in declared_target_semantic_ids
    ]

    draw_target_spec = []
    if declared_target_spec:
        draw_target_spec.append(declared_target_spec)
    draw_target_spec.extend(top_k_target_spec)

    image_records = []
    for image_data in per_image_data:
        frame = image_data["frame"].copy()
        semantic_obs = image_data["semantic_obs"]
        boxes = _extract_target_boxes(
            semantic_obs,
            draw_target_spec,
            min_pixels=min_pixels,
        )
        annotated = _draw_target_boxes(frame, boxes, semantic_obs=semantic_obs)

        annotated_path = Path(output_dir) / f"front_{image_data['image_index']:03d}.png"
        cv2.imwrite(str(annotated_path), annotated)

        visible_selected = []
        for visible in image_data["visible_instances"]:
            semantic_id = visible["semantic_id"]
            if semantic_id not in selected_by_id:
                continue
            record = dict(visible)
            record["label"] = _build_instance_label(selected_by_id[semantic_id])
            visible_selected.append(record)

        visible_declared_target = []
        for visible in image_data["visible_instances"]:
            semantic_id = visible["semantic_id"]
            if semantic_id not in declared_target_semantic_ids:
                continue
            record = dict(visible)
            if declared_target_resolution and declared_target_resolution.get("selected_instance"):
                record["label"] = declared_target_resolution["selected_instance"]["label"]
            visible_declared_target.append(record)

        image_records.append(
            {
                "image_index": image_data["image_index"],
                "source_image": image_data["image_record"]["source_image"],
                "saved_image": image_data["image_record"]["saved_image"],
                "annotated_image": str(annotated_path),
                "boxes_drawn": len(boxes),
                "declared_target_boxes_drawn": sum(
                    1 for box in boxes if box["semantic_id"] in declared_target_semantic_ids
                ),
                "visible_selected_instances": visible_selected,
                "visible_declared_target_instances": visible_declared_target,
            }
        )

    return selected_instances, image_records


def _process_debug_json(
    debug_json_path,
    context,
    replay_settings,
    top_k,
    min_pixels,
    filter_profile,
    ignore_category_substrings,
):
    payload = _load_json(str(debug_json_path))
    output_dir = debug_json_path.parent / "result4_scene_instance_boxes"
    os.makedirs(output_dir, exist_ok=True)
    declared_target_resolution = _resolve_declared_target_instance(
        context,
        debug_json_path,
        payload,
    )

    step_summaries = []
    for step_record in payload.get("steps", []):
        ranked_instances, per_image_data = _rank_instances_for_step(
            context,
            debug_json_path,
            step_record,
            ignore_category_substrings,
        )
        step_output_dir = output_dir / (
            f"step_{step_record['step_index']:03d}_{_safe_name(step_record['label'])}"
        )
        serialized_ranked_instances = [
            _serialize_ranked_instance(instance_record)
            for instance_record in ranked_instances
        ]
        serialized_selected_instances = serialized_ranked_instances[:top_k]
        selected_instances, image_records = _annotate_step(
            step_record,
            ranked_instances,
            per_image_data,
            step_output_dir,
            min_pixels=min_pixels,
            top_k=top_k,
            declared_target_resolution=declared_target_resolution,
        )

        declared_target_semantic_id = None
        if declared_target_resolution.get("resolved"):
            declared_target_semantic_id = declared_target_resolution["selected_instance"]["semantic_id"]
        declared_target_rank = None
        if declared_target_semantic_id is not None:
            for rank_index, instance_record in enumerate(ranked_instances, start=1):
                if instance_record["semantic_id"] == declared_target_semantic_id:
                    declared_target_rank = rank_index
                    break

        step_summaries.append(
            {
                "step_index": step_record["step_index"],
                "label": step_record["label"],
                "target": step_record["target"],
                "start": step_record["start"],
                "end": step_record["end"],
                "top_k": top_k,
                "selection_rule": (
                    "count one vote for each visible semantic instance per frame, "
                    "ignore configured background categories, sort by frame_count then area"
                ),
                "top_5_scene_tags": [
                    item["tag"]
                    for item in serialized_ranked_instances[:top_k]
                ],
                "all_tag_rankings": serialized_ranked_instances,
                "declared_target_rank_in_step": declared_target_rank,
                "declared_target_in_top_k": (
                    declared_target_rank is not None and declared_target_rank <= top_k
                ),
                "selected_instances": serialized_selected_instances,
                "all_ranked_instances": serialized_ranked_instances,
                "annotated_image_dir": str(step_output_dir),
                "images": image_records,
            }
        )

    result = {
        "debug_json": str(debug_json_path),
        "task_json": str(context.task_json_path),
        "source_task_path": payload.get("source_task_path"),
        "target": payload.get("target"),
        "subtrajectory_global_start": payload.get("subtrajectory_global_start"),
        "debug_render_settings": payload.get("debug_render_settings"),
        "effective_replay_settings": replay_settings.get("effective_replay_settings"),
        "top_k": top_k,
        "min_pixels": min_pixels,
        "filter_profile": filter_profile,
        "ignored_category_substrings": list(ignore_category_substrings),
        "declared_target_resolution": declared_target_resolution,
        "steps": step_summaries,
    }

    output_json = debug_json_path.parent / "result4_scene_instance_boxes.json"
    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(result, file, indent=2, ensure_ascii=False)
    return output_json


def generate_scene_instance_box_debug(
    debug_json_path,
    *,
    top_k=5,
    min_pixels=25,
    filter_profile="strong",
    extra_ignore_substrings=(),
    render_width=512,
    render_height=512,
    render_sensor_height=None,
    sensor_hfov=90.0,
    sim_gpu_device=0,
):
    debug_json_path = Path(debug_json_path).resolve()
    payload = _load_json(str(debug_json_path))
    task_json_path = _resolve_task_json(debug_json_path, payload)
    cli_args = SimpleNamespace(
        render_width=render_width,
        render_height=render_height,
        render_sensor_height=render_sensor_height,
        sensor_hfov=sensor_hfov,
    )
    replay_settings = _resolve_replay_settings(payload, cli_args)
    ignore_category_substrings = _resolve_ignore_category_substrings(
        filter_profile,
        list(extra_ignore_substrings),
    )
    context = TaskReplayContext(
        task_json_path=task_json_path,
        width=replay_settings["width"],
        height=replay_settings["height"],
        sensor_height=replay_settings["sensor_height"],
        hfov=replay_settings["hfov"],
        sim_gpu_device=sim_gpu_device,
    )
    try:
        return _process_debug_json(
            debug_json_path=debug_json_path,
            context=context,
            replay_settings=replay_settings,
            top_k=top_k,
            min_pixels=min_pixels,
            filter_profile=filter_profile,
            ignore_category_substrings=ignore_category_substrings,
        )
    finally:
        context.close()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Replace RAM tag counting with semantic-scene instance counting, "
            "then draw the selected instances on split_traj debug front images."
        )
    )
    parser.add_argument(
        "--debug-json",
        action="append",
        default=[],
        help="Path to a result3_step_tags_and_images.json file. Can be passed multiple times.",
    )
    parser.add_argument(
        "--debug-root",
        type=str,
        default=None,
        help="Recursively find all result3_step_tags_and_images.json files under this directory.",
    )
    parser.add_argument("--top-k", type=int, default=5, help="Number of instances to keep per step.")
    parser.add_argument(
        "--filter-profile",
        choices=["basic", "strong"],
        default="strong",
        help="basic: only filter ceiling/floor; strong: also filter structural/background categories.",
    )
    parser.add_argument(
        "--extra-ignore-substring",
        action="append",
        default=[],
        help="Extra lowercase/substring category filters to append. Can be passed multiple times.",
    )
    parser.add_argument(
        "--min-pixels",
        type=int,
        default=25,
        help="Minimum connected-component pixel area to draw a box.",
    )
    parser.add_argument("--render-width", type=int, default=512, help="Replay render width before resize.")
    parser.add_argument("--render-height", type=int, default=512, help="Replay render height before resize.")
    parser.add_argument(
        "--render-sensor-height",
        type=float,
        default=None,
        help="Replay camera / semantic sensor height in meters. Defaults to the robot-specific height.",
    )
    parser.add_argument(
        "--sensor-hfov",
        type=float,
        default=90.0,
        help="Replay horizontal FOV in degrees.",
    )
    parser.add_argument(
        "--sim-gpu-device",
        type=int,
        default=int(os.environ.get("NAVGEN_SIM_GPU_DEVICE", 0)),
        help="GPU device id for Habitat-Sim replay.",
    )
    args = parser.parse_args()

    debug_json_paths = _find_debug_jsons(args.debug_json, args.debug_root)
    ignore_category_substrings = _resolve_ignore_category_substrings(
        args.filter_profile,
        args.extra_ignore_substring,
    )

    context_cache = {}
    try:
        for debug_json_path in debug_json_paths:
            payload = _load_json(str(debug_json_path))
            task_json_path = _resolve_task_json(debug_json_path, payload)
            replay_settings = _resolve_replay_settings(payload, args)
            context_key = (
                task_json_path,
                replay_settings["width"],
                replay_settings["height"],
                replay_settings["sensor_height"],
                replay_settings["hfov"],
            )
            context = context_cache.get(context_key)
            if context is None:
                context = TaskReplayContext(
                    task_json_path=task_json_path,
                    width=replay_settings["width"],
                    height=replay_settings["height"],
                    sensor_height=replay_settings["sensor_height"],
                    hfov=replay_settings["hfov"],
                    sim_gpu_device=args.sim_gpu_device,
                )
                context_cache[context_key] = context

            output_json = _process_debug_json(
                debug_json_path=debug_json_path,
                context=context,
                replay_settings=replay_settings,
                top_k=args.top_k,
                min_pixels=args.min_pixels,
                filter_profile=args.filter_profile,
                ignore_category_substrings=ignore_category_substrings,
            )
            print(f"[ok] wrote {output_json}")
    finally:
        for context in context_cache.values():
            context.close()


if __name__ == "__main__":
    main()
