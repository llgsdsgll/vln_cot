import argparse
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
from types import SimpleNamespace

import cv2
import numpy as np
import habitat_sim
from habitat_sim.utils.common import quat_from_angle_axis

from habitat_base.simulation import SceneSimulator


DEFAULT_WIDTH = 1280
DEFAULT_HEIGHT = 744
DEFAULT_SENSOR_HEIGHT = 1.0
DEFAULT_HFOV = 86.0
DEFAULT_FPS = 1
DEFAULT_OUTPUT_CODEC = "h264"
DEFAULT_DEPTH_VIS_MAX_METERS = 10.0


def _compute_vfov(hfov_deg, width, height):
    return math.degrees(
        2.0 * math.atan(math.tan(math.radians(hfov_deg / 2.0)) * (height / width))
    )


def _resolve_ffmpeg_bin():
    ffmpeg_bin = os.environ.get("NAVGEN_FFMPEG_BIN", "ffmpeg")
    resolved = shutil.which(ffmpeg_bin)
    return resolved or ffmpeg_bin


def _build_output_paths(output_video, output_codec):
    if output_codec == "mp4v":
        return output_video, None
    if output_codec == "h264":
        raw_video = os.path.splitext(output_video)[0] + ".raw_mp4v.mp4"
        return raw_video, output_video
    raise ValueError(f"Unsupported output codec: {output_codec}")


def _transcode_video_to_h264(raw_video, output_video):
    ffmpeg_bin = _resolve_ffmpeg_bin()
    command = [
        ffmpeg_bin,
        "-y",
        "-i",
        raw_video,
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        output_video,
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "ffmpeg H.264 transcode failed for "
            f"{raw_video} -> {output_video}\n{result.stderr.strip()}"
        )
    if not os.path.exists(output_video):
        raise RuntimeError(f"ffmpeg did not create output video: {output_video}")
    os.remove(raw_video)
    return ffmpeg_bin


def _build_args(
    project_root,
    width,
    height,
    sensor_height,
    hfov,
    sim_gpu_device,
    enable_front_semantic,
    enable_front_depth,
):
    temp_task_root = os.path.join(project_root, "nav_gen", "_video_export_tmp") + "/"
    return SimpleNamespace(
        scene=os.path.join(project_root, "data", "hm3d") + "/",
        scene_dataset=os.path.join(project_root, "data", "hm3d", "hm3d_annotated_basis.scene_dataset_config.json"),
        sim_gpu_device=sim_gpu_device,
        task_path=temp_task_root,
        front_rgb_only=True,
        front_semantic_sensor=enable_front_semantic,
        front_depth_sensor=enable_front_depth,
        render_width=width,
        render_height=height,
        render_sensor_height=sensor_height,
        sensor_hfov=hfov,
    )


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return list(value)
    return [value]


def _parse_region_text(value):
    if value is None:
        return None, None
    text = str(value).strip()
    if not text:
        return None, None

    match = re.match(r"^Region\s+(\d+)\s*:\s*(.*)$", text)
    if match:
        return match.group(1), match.group(2).strip() or None
    if text.isdigit():
        return text, None
    return None, text


def _normalize_region_id(value):
    if value is None:
        return None
    parsed_id, _ = _parse_region_text(value)
    if parsed_id:
        return parsed_id

    text = str(value).strip()
    if not text or text == "_-1":
        return None
    match = re.search(r"(\d+)$", text)
    if match:
        return match.group(1)
    return text


def _normalize_trial_targets(task_config):
    objects = task_config.get("Object", [])
    regions = task_config.get("Region", [])
    region_names = task_config.get("Region Name", [])
    target_count = max(
        len(objects),
        len(regions),
        len(region_names),
        len(task_config.get("trial", {})),
    )

    targets = []
    for index in range(target_count):
        category = None
        region_id = None
        region_name = None

        if index < len(objects):
            obj_entry = objects[index]
            if isinstance(obj_entry, str):
                category = obj_entry.strip() or None
            elif isinstance(obj_entry, (list, tuple)):
                if obj_entry:
                    category = str(obj_entry[0]).strip() or None
                if len(obj_entry) > 1:
                    parsed_id, parsed_name = _parse_region_text(obj_entry[1])
                    region_id = parsed_id or region_id
                    region_name = parsed_name or region_name
            elif obj_entry is not None:
                category = str(obj_entry).strip() or None

        if index < len(regions):
            parsed_id, parsed_name = _parse_region_text(regions[index])
            region_id = parsed_id or _normalize_region_id(regions[index])
            region_name = parsed_name or region_name

        if index < len(region_names) and region_names[index]:
            region_name = str(region_names[index]).strip()

        target_ref = category
        if category and region_id:
            target_ref = f"{category}_{region_id}"

        targets.append(
            {
                "trial_index": index,
                "category": category,
                "region_id": region_id,
                "region_name": region_name,
                "target_ref": target_ref,
            }
        )
    return targets


def _sorted_trial_keys(task_config):
    return sorted(
        task_config["trial"].keys(),
        key=lambda key: int(key.split("_")[-1]),
    )


def _saved_yaw_to_world_angle(yaw):
    # NavGen stores yaw in a simulator-local convention whose initial facing is 180 deg.
    return float(yaw) - 180.0


def _capture_frame(simulator, pos, yaw):
    world_angle = _saved_yaw_to_world_angle(yaw)
    agent_state = habitat_sim.AgentState()
    agent_state.position = np.array(pos, dtype=np.float32)
    agent_state.rotation = quat_from_angle_axis(
        math.radians(world_angle),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
    )
    simulator.agent.set_state(agent_state)
    simulator.yaw = yaw
    observations = simulator.sim.get_sensor_observations()
    rgb = observations["color_sensor_f"][..., :3]
    frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    semantic = observations.get("semantic_sensor")
    depth = observations.get("depth_sensor_f")
    if depth is not None:
        depth = np.array(depth, dtype=np.float32, copy=True)
    return frame, semantic, depth


def _flatten_entries(task_config):
    entries = []
    targets = _normalize_trial_targets(task_config)
    trial_keys = _sorted_trial_keys(task_config)
    global_step = -1
    for trial_index, trial_key in enumerate(trial_keys):
        trial = task_config["trial"][trial_key]
        pos_list = trial.get("pos", [])
        yaw_list = trial.get("yaw", [])
        action_list = trial.get("action", [])
        target_info = targets[trial_index] if trial_index < len(targets) else {}
        entry_count = min(len(pos_list), len(yaw_list))
        if action_list:
            entry_count = min(entry_count, len(action_list))
        for state_index in range(entry_count):
            action = action_list[state_index] if action_list else None
            is_initial_stop = (
                trial_index == 0
                and state_index == 0
                and action == "stop"
            )
            is_terminal_stop = (
                action == "stop"
                and state_index == entry_count - 1
                and entry_count > 1
            )
            if is_initial_stop:
                entry_global_step = -1
            elif is_terminal_stop:
                entry_global_step = global_step
            else:
                global_step += 1
                entry_global_step = global_step
            entries.append(
                {
                    "trial_index": trial_index,
                    "trial_key": trial_key,
                    "state_index": state_index,
                    "pos": pos_list[state_index],
                    "yaw": yaw_list[state_index],
                    "action": action,
                    "target": target_info.get("category"),
                    "target_region_id": target_info.get("region_id"),
                    "target_region_name": target_info.get("region_name"),
                    "target_ref": target_info.get("target_ref"),
                    "global_step": entry_global_step,
                    "is_initial_stop": is_initial_stop,
                    "is_terminal_stop": is_terminal_stop,
                }
            )
    return entries


def _apply_frame_mode(entries, frame_mode):
    if frame_mode == "state":
        return entries
    if frame_mode == "action":
        if entries and entries[0].get("action") == "stop":
            return entries[1:]
        return entries
    raise ValueError(f"Unsupported frame_mode: {frame_mode}")


def _build_timeline(task_config, frame_mode):
    return _apply_frame_mode(_flatten_entries(task_config), frame_mode)


def _overlay_lines(task_config, entry, frame_index, total_frames, subtask_state=None):
    trial_total = len(task_config.get("trial", {}))
    lines = [
        f"Frame {frame_index + 1}/{total_frames} | Trial {entry['trial_index'] + 1}/{trial_total}",
    ]
    if entry.get("action"):
        lines.append(f"Action: {entry['action']}")
    if entry.get("target_ref"):
        lines.append(f"Target: {entry['target_ref']}")
    elif entry.get("target"):
        lines.append(f"Target: {entry['target']}")
    if subtask_state:
        lines.append(
            "Subtask: "
            f"{subtask_state['subtask_index'] + 1}/{subtask_state['subtask_total']} "
            f"| Compressed steps {subtask_state['step_index_start']}-{subtask_state['step_index_end']}"
        )
        if subtask_state.get("instruction"):
            lines.append(f"Subtask text: {subtask_state['instruction']}")
    instruction = task_config.get("Task instruction")
    if instruction:
        lines.append(f"Task: {instruction}")
    return lines


def _wrap_overlay_lines(lines, frame_width, wrap_width=None):
    wrapped_lines = []
    if wrap_width is None:
        wrap_width = 70 if frame_width >= 1000 else 45
    for line in lines:
        wrapped = textwrap.wrap(line, width=wrap_width) or [line]
        wrapped_lines.extend(wrapped)
    return wrapped_lines


def _draw_overlay_panel(
    frame,
    lines,
    anchor="top_left",
    wrap_width=None,
    max_width_fraction=0.82,
    alpha=0.62,
):
    if not lines:
        return frame

    wrapped_lines = _wrap_overlay_lines(
        lines,
        frame.shape[1],
        wrap_width=wrap_width,
    )
    if not wrapped_lines:
        return frame

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 if frame.shape[1] >= 1000 else 0.55
    thickness = 2
    padding = 12
    line_height = 28 if frame.shape[1] >= 1000 else 22
    box_width = min(frame.shape[1] - 2 * padding, int(frame.shape[1] * max_width_fraction))
    box_height = padding * 2 + line_height * len(wrapped_lines)

    if anchor not in {"top_left", "top_right", "bottom_left", "bottom_right"}:
        raise ValueError(f"Unsupported overlay anchor: {anchor}")

    if "left" in anchor:
        x1 = padding
        x2 = padding + box_width
        text_x = x1 + 10
    else:
        x2 = frame.shape[1] - padding
        x1 = max(padding, x2 - box_width)
        text_x = x1 + 10

    if "top" in anchor:
        y1 = padding
        y2 = padding + box_height
        y = y1 + 22
    else:
        y2 = frame.shape[0] - padding
        y1 = max(padding, y2 - box_height)
        y = y1 + 22

    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, alpha, frame, 1.0 - alpha, 0)
    for line in wrapped_lines:
        cv2.putText(
            frame,
            line,
            (text_x, y),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
        y += line_height
    return frame


def _annotate_frame(frame, lines):
    return _draw_overlay_panel(frame, lines, anchor="top_left")


def _semantic_region_id(region):
    return _normalize_region_id(getattr(region, "id", None))


def _as_float_list(values):
    return [float(value) for value in values]


def _normalize_match_text(text):
    text = str(text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _pluralize_word(word):
    if not word:
        return word
    if word.endswith("y") and len(word) > 1 and word[-2] not in "aeiou":
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


def _singularize_word(word):
    if not word:
        return word
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    for suffix in ("ches", "shes", "sses", "xes", "zes"):
        if word.endswith(suffix) and len(word) > len(suffix):
            return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def _build_category_aliases(category):
    normalized = _normalize_match_text(category)
    if not normalized:
        return set()

    words = normalized.split()
    last_word = words[-1]
    aliases = {normalized}
    for variant in {
        last_word,
        _singularize_word(last_word),
        _pluralize_word(last_word),
    }:
        if not variant:
            continue
        aliases.add(" ".join(words[:-1] + [variant]))
    return {alias for alias in aliases if alias}


def _collect_scene_categories(simulator):
    categories = set()
    for obj in simulator.sim.semantic_scene.objects:
        try:
            category_name = obj.category.name()
        except Exception:
            continue
        semantic_id = getattr(obj, "semantic_id", None)
        center = getattr(getattr(obj, "aabb", None), "center", None)
        if not category_name or semantic_id is None or center is None:
            continue
        categories.add(category_name)
    return sorted(categories)


def _extract_instruction_object_categories(instruction, categories):
    normalized_instruction = f" {_normalize_match_text(instruction)} "
    alias_records = []
    for category in categories:
        for alias in _build_category_aliases(category):
            alias_records.append(
                (
                    len(alias.split()),
                    len(alias),
                    alias,
                    category,
                )
            )
    alias_records.sort(reverse=True)

    matched_categories = []
    seen_categories = set()
    occupied = [False] * len(normalized_instruction)

    for _, _, alias, category in alias_records:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        for match in pattern.finditer(normalized_instruction):
            start, end = match.span()
            if any(occupied[start:end]):
                continue
            if category not in seen_categories:
                matched_categories.append(category)
                seen_categories.add(category)
            for idx in range(start, end):
                occupied[idx] = True
            break
    return matched_categories


def _collect_target_candidates(simulator, category, region_id=None):
    wanted_region_id = _normalize_region_id(region_id)
    candidates = []
    for region in simulator.sim.semantic_scene.regions:
        semantic_region_id = _semantic_region_id(region)
        if wanted_region_id and semantic_region_id != wanted_region_id:
            continue
        for obj in region.objects:
            try:
                category_name = obj.category.name()
            except Exception:
                continue
            if category and category_name != category:
                continue

            semantic_id = getattr(obj, "semantic_id", None)
            center = getattr(getattr(obj, "aabb", None), "center", None)
            if semantic_id is None or center is None:
                continue

            candidates.append(
                {
                    "semantic_id": int(semantic_id),
                    "category": category_name,
                    "region_id": semantic_region_id,
                    "object_id": getattr(obj, "id", None),
                    "center": np.array(center, dtype=np.float32),
                }
            )
    return candidates


def _collect_scene_candidates(simulator, category):
    candidates = []
    for obj in simulator.sim.semantic_scene.objects:
        try:
            category_name = obj.category.name()
        except Exception:
            continue
        if category and category_name != category:
            continue

        semantic_id = getattr(obj, "semantic_id", None)
        center = getattr(getattr(obj, "aabb", None), "center", None)
        if semantic_id is None or center is None:
            continue

        candidates.append(
            {
                "semantic_id": int(semantic_id),
                "category": category_name,
                "region_id": None,
                "object_id": getattr(obj, "id", None),
                "center": np.array(center, dtype=np.float32),
            }
        )
    return candidates


def _build_scene_object_indices(simulator):
    by_semantic_id = {}
    by_object_id = {}
    for obj in simulator.sim.semantic_scene.objects:
        try:
            category_name = obj.category.name()
        except Exception:
            continue

        semantic_id = getattr(obj, "semantic_id", None)
        center = getattr(getattr(obj, "aabb", None), "center", None)
        if semantic_id is None or center is None:
            continue

        record = {
            "semantic_id": int(semantic_id),
            "category": category_name,
            "object_id": getattr(obj, "id", None),
            "center": np.array(center, dtype=np.float32),
        }
        by_semantic_id[int(semantic_id)] = record

        object_id = str(record["object_id"] or "").strip()
        if object_id:
            by_object_id[object_id] = record

    return by_semantic_id, by_object_id


def _resolve_explicit_instance_record(explicit_record, by_semantic_id, by_object_id):
    semantic_id = explicit_record.get("semantic_id")
    if semantic_id is not None:
        try:
            semantic_id = int(semantic_id)
        except (TypeError, ValueError):
            semantic_id = None
    if semantic_id is not None and semantic_id in by_semantic_id:
        return by_semantic_id[semantic_id], "semantic_id"

    object_id = str(explicit_record.get("object_id") or "").strip()
    if object_id and object_id in by_object_id:
        return by_object_id[object_id], "object_id"

    return None, None


def _snap_position(pathfinder, position):
    snapped = pathfinder.snap_point(np.array(position, dtype=np.float32))
    if snapped is None:
        return np.array(position, dtype=np.float32)
    snapped = np.array(snapped, dtype=np.float32).reshape(-1)
    if snapped.size != 3 or not np.all(np.isfinite(snapped)):
        return np.array(position, dtype=np.float32)
    return snapped


def _candidate_distance_key(pathfinder, start_pos, end_pos):
    start_nav = _snap_position(pathfinder, start_pos)
    end_nav = _snap_position(pathfinder, end_pos)
    path = habitat_sim.nav.ShortestPath()
    path.requested_start = start_nav
    path.requested_end = end_nav

    if pathfinder.find_path(path):
        return {
            "rank": (0, float(path.geodesic_distance)),
            "distance": float(path.geodesic_distance),
            "strategy": "closest_geodesic",
        }

    planar_distance = math.dist(
        [float(start_nav[0]), float(start_nav[2])],
        [float(end_nav[0]), float(end_nav[2])],
    )
    return {
        "rank": (1, float(planar_distance)),
        "distance": float(planar_distance),
        "strategy": "closest_planar_fallback",
    }


def _select_candidate_by_visibility(simulator, trial, candidates, min_pixels=25):
    if not candidates:
        return None

    semantic_index = {candidate["semantic_id"]: candidate for candidate in candidates}
    visibility_stats = {
        semantic_id: {
            "tail_area": 0,
            "weighted_area": 0.0,
            "max_area": 0,
            "last_seen_index": -1,
            "seen_frames": 0,
        }
        for semantic_id in semantic_index
    }

    pos_list = trial.get("pos", [])
    yaw_list = trial.get("yaw", [])
    frame_count = min(len(pos_list), len(yaw_list))
    if frame_count == 0:
        return None

    tail_size = max(3, frame_count // 5)
    tail_start = max(0, frame_count - tail_size)

    for frame_index in range(frame_count):
        _, semantic_obs, _ = _capture_frame(
            simulator,
            pos_list[frame_index],
            yaw_list[frame_index],
        )
        if semantic_obs is None:
            continue

        visible_ids, visible_counts = np.unique(semantic_obs, return_counts=True)
        for semantic_id, pixel_count in zip(visible_ids, visible_counts):
            semantic_id = int(semantic_id)
            pixel_count = int(pixel_count)
            if semantic_id not in visibility_stats or pixel_count < min_pixels:
                continue

            stats = visibility_stats[semantic_id]
            stats["weighted_area"] += float(pixel_count) * (
                1.0 + frame_index / max(1, frame_count - 1)
            )
            if frame_index >= tail_start:
                stats["tail_area"] += pixel_count
            stats["max_area"] = max(stats["max_area"], pixel_count)
            stats["last_seen_index"] = frame_index
            stats["seen_frames"] += 1

    ranked = []
    for semantic_id, stats in visibility_stats.items():
        if stats["seen_frames"] == 0:
            continue
        ranked.append(
            (
                (
                    stats["tail_area"],
                    stats["weighted_area"],
                    stats["max_area"],
                    stats["last_seen_index"],
                    stats["seen_frames"],
                ),
                semantic_index[semantic_id],
                stats,
            )
        )

    if not ranked:
        return None

    ranked.sort(key=lambda item: item[0], reverse=True)
    _, candidate, stats = ranked[0]
    return {
        "candidate": candidate,
        "stats": {
            "tail_area": int(stats["tail_area"]),
            "weighted_area": float(stats["weighted_area"]),
            "max_area": int(stats["max_area"]),
            "last_seen_index": int(stats["last_seen_index"]),
            "seen_frames": int(stats["seen_frames"]),
        },
    }


def _build_explicit_step_task_target_lookup(simulator, replay_context):
    raw_step_task_config = replay_context["raw_task_config"]
    target_instance = raw_step_task_config.get("instruction_target_instance") or {}
    anchor_instances = raw_step_task_config.get("instruction_anchor_instances") or []
    if not target_instance and not anchor_instances:
        return None

    by_semantic_id, by_object_id = _build_scene_object_indices(simulator)
    explicit_records = []
    if target_instance.get("semantic_id") is not None or target_instance.get("object_id"):
        explicit_records.append(
            {
                "role": "target",
                "step_index": None,
                "action": None,
                **target_instance,
            }
        )
    for anchor in sorted(
        anchor_instances,
        key=lambda item: int(item.get("step_index", 0))
        if str(item.get("step_index", "")).strip().lstrip("-").isdigit()
        else 0,
    ):
        if anchor.get("semantic_id") is None and not anchor.get("object_id"):
            continue
        explicit_records.append(
            {
                "role": "anchor",
                **anchor,
            }
        )

    if not explicit_records:
        return None

    target_specs = []
    seen_semantic_ids = set()
    resolution_info = []
    instruction_object_categories = []

    for explicit in explicit_records:
        resolved_record, resolved_by = _resolve_explicit_instance_record(
            explicit,
            by_semantic_id,
            by_object_id,
        )
        category = explicit.get("category") or (
            resolved_record.get("category") if resolved_record else None
        )
        if category:
            instruction_object_categories.append(category)

        requested_object_id = str(explicit.get("object_id") or "").strip() or None
        requested_semantic_id = explicit.get("semantic_id")
        label = (
            requested_object_id
            or (resolved_record.get("object_id") if resolved_record else None)
            or category
            or "object"
        )
        resolution_record = {
            "trial_key": "trial_0",
            "trial_index": 0,
            "role": explicit.get("role"),
            "step_index": explicit.get("step_index"),
            "action": explicit.get("action"),
            "label": label,
            "category": category,
            "scope": "explicit_instance",
            "requested_object_id": requested_object_id,
            "requested_semantic_id": requested_semantic_id,
        }

        if resolved_record is None:
            resolution_record["resolved"] = False
            resolution_record["reason"] = "explicit_instance_not_found"
            resolution_info.append(resolution_record)
            continue

        semantic_id = int(resolved_record["semantic_id"])
        resolution_record.update(
            {
                "resolved": True,
                "selected_semantic_ids": [semantic_id],
                "selected_object_id": resolved_record.get("object_id"),
                "selected_center": _as_float_list(resolved_record["center"]),
                "selection_strategy": f"step_task_explicit_instance_by_{resolved_by}",
            }
        )
        resolution_info.append(resolution_record)

        if semantic_id in seen_semantic_ids:
            continue
        seen_semantic_ids.add(semantic_id)
        target_specs.append(
            {
                "label": label,
                "scope": "explicit_instance",
                "semantic_ids": [semantic_id],
                "semantic_labels": {
                    semantic_id: label,
                },
            }
        )

    trial_lookup = {0: target_specs} if target_specs else {}
    return trial_lookup, resolution_info, instruction_object_categories


def _build_trial_target_lookup(simulator, task_config, scope):
    trial_targets = _normalize_trial_targets(task_config)
    trial_lookup = {}
    resolution_info = []
    trial_keys = _sorted_trial_keys(task_config)

    for trial_index, trial_key in enumerate(trial_keys):
        target_info = trial_targets[trial_index] if trial_index < len(trial_targets) else {}
        category = target_info.get("category")
        region_id = target_info.get("region_id")
        label = target_info.get("target_ref") or category
        trial = task_config["trial"][trial_key]
        pos_list = trial.get("pos", [])
        start_pos = pos_list[0] if pos_list else None

        if not category:
            resolution_info.append(
                {
                    "trial_key": trial_key,
                    "trial_index": trial_index,
                    "label": None,
                    "scope": scope,
                    "resolved": False,
                    "reason": "missing_category",
                }
            )
            continue

        candidates = _collect_target_candidates(simulator, category=category, region_id=region_id)
        candidate_source = "region"
        if not candidates:
            candidates = _collect_scene_candidates(simulator, category=category)
            candidate_source = "scene"

        if not candidates:
            resolution_info.append(
                {
                    "trial_key": trial_key,
                    "trial_index": trial_index,
                    "label": label,
                    "scope": scope,
                    "resolved": False,
                    "reason": "no_matching_semantic_object",
                    "category": category,
                    "region_id": region_id,
                }
            )
            continue

        if scope == "class":
            semantic_ids = sorted({candidate["semantic_id"] for candidate in candidates})
            semantic_labels = {semantic_id: label for semantic_id in semantic_ids}
            selected_candidate = candidates[0]
            strategy = "all_matching_objects_in_target_region"
        else:
            visibility_choice = _select_candidate_by_visibility(simulator, trial, candidates)
            if visibility_choice is not None:
                selected_candidate = visibility_choice["candidate"]
                strategy = "most_visible_along_trial"
                distance_value = None
                visibility_stats = visibility_choice["stats"]
            else:
                visibility_stats = None
                if start_pos is not None:
                    ranked_candidates = []
                    for candidate in candidates:
                        distance_info = _candidate_distance_key(simulator.pathfinder, start_pos, candidate["center"])
                        ranked_candidates.append(
                            (
                                distance_info["rank"],
                                candidate,
                                distance_info,
                            )
                        )
                    ranked_candidates.sort(key=lambda item: item[0])
                    _, selected_candidate, selected_distance = ranked_candidates[0]
                    strategy = selected_distance["strategy"]
                    distance_value = selected_distance["distance"]
                else:
                    selected_candidate = candidates[0]
                    strategy = "first_matching_object"
                    distance_value = None

            semantic_ids = [selected_candidate["semantic_id"]]
            semantic_labels = {selected_candidate["semantic_id"]: label}

        trial_lookup[trial_index] = {
            "label": label,
            "scope": scope,
            "semantic_ids": semantic_ids,
            "semantic_labels": semantic_labels,
        }

        resolution_record = {
            "trial_key": trial_key,
            "trial_index": trial_index,
            "label": label,
            "category": category,
            "region_id": region_id,
            "scope": scope,
            "resolved": True,
            "candidate_count": len(candidates),
            "candidate_source": candidate_source,
            "selected_semantic_ids": semantic_ids,
            "selected_object_id": selected_candidate.get("object_id"),
            "selected_center": _as_float_list(selected_candidate["center"]),
            "selection_strategy": strategy,
        }
        if scope == "instance" and start_pos is not None:
            resolution_record["trial_start_pos"] = _as_float_list(start_pos)
            if distance_value is not None:
                resolution_record["selected_distance"] = float(distance_value)
        if scope == "instance" and visibility_stats is not None:
            resolution_record["visibility_stats"] = visibility_stats
        resolution_info.append(resolution_record)

    return trial_lookup, resolution_info


def _is_step_task_config(task_config):
    return (
        "trajectory path" in task_config
        and "start" in task_config
        and "end" in task_config
    )


def _resolve_step_task_source_trial_dir(task_json_path, step_task_config):
    trajectory_path = str(step_task_config.get("trajectory path", "")).strip()
    if not trajectory_path:
        raise ValueError("step_task json is missing `trajectory path`.")

    candidate_paths = []
    seen_paths = set()

    def add_candidate(path):
        normalized = os.path.normpath(path)
        if normalized not in seen_paths:
            seen_paths.add(normalized)
            candidate_paths.append(normalized)

    if os.path.isabs(trajectory_path):
        add_candidate(trajectory_path)

    current_dir = os.path.abspath(os.path.dirname(task_json_path))
    while True:
        add_candidate(os.path.join(current_dir, trajectory_path))
        parent_dir = os.path.dirname(current_dir)
        if parent_dir == current_dir:
            break
        current_dir = parent_dir

    for candidate_dir in candidate_paths:
        task_json_candidate = candidate_dir
        if os.path.isdir(task_json_candidate):
            task_json_candidate = os.path.join(task_json_candidate, "task.json")
        if os.path.isfile(task_json_candidate):
            return os.path.dirname(task_json_candidate), task_json_candidate

    raise FileNotFoundError(
        "Could not resolve source task.json from step_task trajectory path "
        f"`{trajectory_path}` starting at `{task_json_path}`."
    )


def _load_step_task_instruction_subtasks(task_json_path, step_task_config):
    raw_subtasks = step_task_config.get("instruction_subtasks") or step_task_config.get("subtasks")
    if raw_subtasks:
        return raw_subtasks

    sidecar_name = str(step_task_config.get("instruction_subtask_alignment_json") or "").strip()
    if not sidecar_name:
        return []

    sidecar_path = os.path.join(os.path.dirname(task_json_path), sidecar_name)
    if not os.path.isfile(sidecar_path):
        return []

    try:
        payload = _load_json(sidecar_path)
    except Exception:
        return []
    return payload.get("subtasks") or []


def _normalize_step_task_instruction_subtasks(task_json_path, step_task_config):
    normalized = []
    raw_subtasks = _load_step_task_instruction_subtasks(task_json_path, step_task_config)
    for fallback_index, item in enumerate(raw_subtasks):
        if not isinstance(item, dict):
            continue
        try:
            subtask_index = int(item.get("subtask_index", fallback_index))
            source_start = int(item["source_start"])
            source_end = int(item["source_end"])
            step_index_start = int(item["step_index_start"])
            step_index_end = int(item["step_index_end"])
        except (KeyError, TypeError, ValueError):
            continue
        if source_end < source_start or step_index_end < step_index_start:
            continue

        normalized.append(
            {
                "subtask_index": subtask_index,
                "instruction": str(
                    item.get("instruction")
                    or item.get("subtask_instruction")
                    or ""
                ).strip(),
                "source_start": source_start,
                "source_end": source_end,
                "step_index_start": step_index_start,
                "step_index_end": step_index_end,
                "actions": list(item.get("actions") or []),
                "anchor_instances": list(item.get("anchor_instances") or []),
            }
        )

    normalized.sort(
        key=lambda item: (
            item["source_start"],
            item["source_end"],
            item["subtask_index"],
        )
    )
    total = len(normalized)
    for item in normalized:
        item["subtask_total"] = total
    return normalized


def _match_step_task_instruction_subtask(instruction_subtasks, global_step):
    if global_step is None:
        return None
    try:
        global_step = int(global_step)
    except (TypeError, ValueError):
        return None

    for item in instruction_subtasks:
        if item["source_start"] <= global_step <= item["source_end"]:
            return item
    return None


def _build_step_task_context(task_json_path, step_task_config, frame_mode):
    source_trial_dir, source_task_json = _resolve_step_task_source_trial_dir(
        task_json_path,
        step_task_config,
    )
    source_task_config = _load_json(source_task_json)
    source_entries = _flatten_entries(source_task_config)

    start_step = int(step_task_config["start"])
    end_step = int(step_task_config["end"])
    if end_step < start_step:
        raise ValueError(
            f"Invalid step_task range: start={start_step}, end={end_step}"
        )

    selected_entries = [
        entry
        for entry in source_entries
        if start_step <= entry["global_step"] <= end_step
    ]
    if not selected_entries:
        raise ValueError(
            "No source trajectory entries matched the requested step_task range "
            f"{start_step}..{end_step} from `{source_task_json}`."
        )

    source_trial_indices = sorted(
        {entry["trial_index"] for entry in selected_entries}
    )
    source_trial_keys = list(
        dict.fromkeys(entry["trial_key"] for entry in selected_entries)
    )

    target_objects = _coerce_list(
        step_task_config.get("target") or step_task_config.get("Object")
    )
    region_values = _coerce_list(step_task_config.get("Region"))
    region_names = _coerce_list(step_task_config.get("Region Name"))

    source_targets = _normalize_trial_targets(source_task_config)
    if not target_objects and source_trial_indices:
        source_target = source_targets[source_trial_indices[0]]
        if source_target.get("category"):
            target_objects = [source_target["category"]]
        if not region_values and source_target.get("region_id"):
            region_values = [source_target["region_id"]]
        if not region_names and source_target.get("region_name"):
            region_names = [source_target["region_name"]]

    target_label = target_objects[0] if target_objects else None
    target_region_id = region_values[0] if region_values else None
    target_ref = target_label
    if target_label and target_region_id:
        target_ref = f"{target_label}_{_normalize_region_id(target_region_id)}"

    timeline = []
    for state_index, entry in enumerate(selected_entries):
        replay_entry = dict(entry)
        replay_entry["source_trial_index"] = entry["trial_index"]
        replay_entry["source_trial_key"] = entry["trial_key"]
        replay_entry["source_state_index"] = entry["state_index"]
        replay_entry["trial_index"] = 0
        replay_entry["trial_key"] = "trial_0"
        replay_entry["state_index"] = state_index
        replay_entry["target"] = target_label or entry.get("target")
        replay_entry["target_region_id"] = target_region_id or entry.get(
            "target_region_id"
        )
        replay_entry["target_region_name"] = (
            region_names[0] if region_names else entry.get("target_region_name")
        )
        replay_entry["target_ref"] = target_ref or entry.get("target_ref")
        timeline.append(replay_entry)

    replay_timeline = _apply_frame_mode(timeline, frame_mode)
    if not replay_timeline:
        raise ValueError(
            f"Requested step_task range {start_step}..{end_step} produced no frames."
        )

    replay_config = {
        "Task instruction": step_task_config.get("Task instruction")
        or source_task_config.get("Task instruction"),
        "Robot": step_task_config.get("Robot") or source_task_config.get("Robot"),
        "Scene": step_task_config.get("Scene") or source_task_config.get("Scene"),
        "Object": target_objects,
        "Region": region_values,
        "Region Name": region_names,
        "trial": {
            "trial_0": {
                "pos": [entry["pos"] for entry in replay_timeline],
                "yaw": [entry["yaw"] for entry in replay_timeline],
                "action": [entry["action"] for entry in replay_timeline],
            }
        },
    }
    instruction_subtasks = _normalize_step_task_instruction_subtasks(
        task_json_path,
        step_task_config,
    )

    return {
        "kind": "step_task",
        "display_task_config": replay_config,
        "simulator_config": replay_config,
        "timeline": replay_timeline,
        "trial_targets": _normalize_trial_targets(replay_config),
        "source_task_json": source_task_json,
        "source_trial_dir": source_trial_dir,
        "source_task_config": source_task_config,
        "source_trial_indices": source_trial_indices,
        "source_trial_keys": source_trial_keys,
        "source_step_range": {
            "start": start_step,
            "end": end_step,
        },
        "instruction_subtasks": instruction_subtasks,
        "raw_task_config": step_task_config,
    }


def _build_full_task_context(task_config, frame_mode):
    return {
        "kind": "task",
        "display_task_config": task_config,
        "simulator_config": task_config,
        "timeline": _build_timeline(task_config, frame_mode),
        "trial_targets": _normalize_trial_targets(task_config),
        "raw_task_config": task_config,
    }


def _build_replay_context(task_json_path, task_config, frame_mode):
    if _is_step_task_config(task_config):
        return _build_step_task_context(task_json_path, task_config, frame_mode)
    return _build_full_task_context(task_config, frame_mode)


def _build_step_task_target_lookup(simulator, replay_context, scope):
    explicit_target_lookup = _build_explicit_step_task_target_lookup(
        simulator,
        replay_context,
    )
    if explicit_target_lookup is not None:
        return explicit_target_lookup

    replay_config = replay_context["display_task_config"]
    raw_step_task_config = replay_context["raw_task_config"]
    instruction = (
        raw_step_task_config.get("Task instruction")
        or replay_config.get("Task instruction")
    )
    scene_categories = _collect_scene_categories(simulator)
    mentioned_categories = _extract_instruction_object_categories(
        instruction,
        scene_categories,
    )

    target_objects = _coerce_list(
        raw_step_task_config.get("target") or replay_config.get("Object")
    )
    ordered_categories = list(mentioned_categories)
    for category in target_objects:
        if category and category not in ordered_categories:
            ordered_categories.append(category)

    region_values = _coerce_list(raw_step_task_config.get("Region") or replay_config.get("Region"))
    target_region_lookup = {}
    for index, category in enumerate(target_objects):
        normalized_category = _normalize_match_text(category)
        region_value = region_values[index] if index < len(region_values) else None
        target_region_lookup[normalized_category] = region_value

    trial = replay_config["trial"]["trial_0"]
    start_pos = trial["pos"][0] if trial.get("pos") else None

    target_specs = []
    resolution_info = []
    mentioned_set = set(mentioned_categories)

    for category in ordered_categories:
        region_id = target_region_lookup.get(_normalize_match_text(category))
        label = category

        candidates = _collect_target_candidates(
            simulator,
            category=category,
            region_id=region_id,
        )
        candidate_source = "region"
        if not candidates:
            candidates = _collect_scene_candidates(simulator, category=category)
            candidate_source = "scene"

        if not candidates:
            resolution_info.append(
                {
                    "trial_key": "trial_0",
                    "trial_index": 0,
                    "label": label,
                    "category": category,
                    "region_id": region_id,
                    "scope": scope,
                    "resolved": False,
                    "reason": "no_matching_semantic_object",
                    "mentioned_in_instruction": category in mentioned_set,
                }
            )
            continue

        if scope == "class":
            semantic_ids = sorted(
                {candidate["semantic_id"] for candidate in candidates}
            )
            semantic_labels = {
                semantic_id: label for semantic_id in semantic_ids
            }
            selected_candidate = candidates[0]
            strategy = (
                "all_matching_objects_in_target_region"
                if candidate_source == "region"
                else "all_matching_objects_in_scene"
            )
            visibility_stats = None
            distance_value = None
        else:
            visibility_choice = _select_candidate_by_visibility(
                simulator,
                trial,
                candidates,
            )
            if visibility_choice is not None:
                selected_candidate = visibility_choice["candidate"]
                strategy = "most_visible_along_step_segment"
                visibility_stats = visibility_choice["stats"]
                distance_value = None
            else:
                visibility_stats = None
                if start_pos is not None:
                    ranked_candidates = []
                    for candidate in candidates:
                        distance_info = _candidate_distance_key(
                            simulator.pathfinder,
                            start_pos,
                            candidate["center"],
                        )
                        ranked_candidates.append(
                            (
                                distance_info["rank"],
                                candidate,
                                distance_info,
                            )
                        )
                    ranked_candidates.sort(key=lambda item: item[0])
                    _, selected_candidate, selected_distance = ranked_candidates[0]
                    strategy = selected_distance["strategy"]
                    distance_value = selected_distance["distance"]
                else:
                    selected_candidate = candidates[0]
                    strategy = "first_matching_object"
                    distance_value = None

            semantic_ids = [selected_candidate["semantic_id"]]
            semantic_labels = {
                selected_candidate["semantic_id"]: label
            }

        target_specs.append(
            {
                "label": label,
                "scope": scope,
                "semantic_ids": semantic_ids,
                "semantic_labels": semantic_labels,
            }
        )

        resolution_record = {
            "trial_key": "trial_0",
            "trial_index": 0,
            "label": label,
            "category": category,
            "region_id": region_id,
            "scope": scope,
            "resolved": True,
            "candidate_count": len(candidates),
            "candidate_source": candidate_source,
            "selected_semantic_ids": semantic_ids,
            "selected_object_id": selected_candidate.get("object_id"),
            "selected_center": _as_float_list(selected_candidate["center"]),
            "selection_strategy": strategy,
            "mentioned_in_instruction": category in mentioned_set,
        }
        if scope == "instance" and start_pos is not None:
            resolution_record["trial_start_pos"] = _as_float_list(start_pos)
            if distance_value is not None:
                resolution_record["selected_distance"] = float(distance_value)
        if scope == "instance" and visibility_stats is not None:
            resolution_record["visibility_stats"] = visibility_stats
        resolution_info.append(resolution_record)

    trial_lookup = {0: target_specs} if target_specs else {}
    return trial_lookup, resolution_info, mentioned_categories


def _extract_target_boxes(semantic_obs, target_spec, min_pixels=25):
    if semantic_obs is None or not target_spec:
        return []

    boxes = []
    target_specs = target_spec if isinstance(target_spec, list) else [target_spec]
    processed_pairs = set()

    for spec in target_specs:
        semantic_ids = set(spec.get("semantic_ids", []))
        semantic_labels = spec.get("semantic_labels", {})
        for semantic_id in np.unique(semantic_obs):
            semantic_id = int(semantic_id)
            if semantic_id not in semantic_ids:
                continue
            label_str = semantic_labels.get(
                semantic_id,
                spec.get("label", str(semantic_id)),
            )
            if (semantic_id, label_str) in processed_pairs:
                continue
            processed_pairs.add((semantic_id, label_str))

            mask = (semantic_obs == semantic_id).astype(np.uint8)
            n, labels = cv2.connectedComponents(mask)
            for comp_id in range(1, n):
                ys, xs = np.where(labels == comp_id)
                if xs.size < min_pixels:
                    continue
                boxes.append(
                    {
                        "label": label_str,
                        "semantic_id": semantic_id,
                        "object_id": spec.get("object_id"),
                        "category": spec.get("category"),
                        "subtask_numbers": list(spec.get("subtask_numbers") or []),
                        "is_current_subtask_target": bool(
                            spec.get("is_current_subtask_target")
                        ),
                        "bbox": (
                            int(xs.min()),
                            int(ys.min()),
                            int(xs.max()),
                            int(ys.max()),
                        ),
                        "area": int(xs.size),
                    }
                )
    boxes.sort(key=lambda item: item["area"], reverse=True)
    return boxes


def _color_for_label(label):
    palette = [
        (66, 133, 244),
        (52, 168, 83),
        (251, 188, 5),
        (234, 67, 53),
        (171, 71, 188),
        (0, 172, 193),
    ]
    return palette[sum(ord(ch) for ch in label) % len(palette)]


def _draw_target_boxes(frame, boxes, semantic_obs=None):
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        if box.get("is_current_subtask_target"):
            color = (255, 191, 0)
        else:
            color = _color_for_label(box["label"])
        if semantic_obs is not None:
            mask = (semantic_obs == box["semantic_id"]).astype(np.uint8)
            overlay = frame.copy()
            overlay_strength = 0.72 if box.get("is_current_subtask_target") else 0.5
            overlay[mask == 1] = (
                overlay[mask == 1] * (1.0 - overlay_strength)
                + np.array(color[::-1], dtype=np.float32) * overlay_strength
            ).astype(np.uint8)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        thickness = 4 if box.get("is_current_subtask_target") else 2
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

        subtask_numbers = [
            int(item)
            for item in (box.get("subtask_numbers") or [])
            if item is not None
        ]
        label_tokens = []
        if box.get("is_current_subtask_target"):
            label_tokens.append("cur")
        if subtask_numbers:
            label_tokens.extend(f"sub{number}" for number in subtask_numbers)
        if label_tokens:
            label_text = f"[{'|'.join(label_tokens)}] {box['label']}"
        else:
            label_text = box["label"]
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )
        text_y = y1 - 8 if y1 - 8 - text_h >= 0 else y1 + text_h + 8
        box_top = text_y - text_h - baseline - 4
        box_bottom = text_y + baseline + 4
        box_right = min(frame.shape[1] - 1, x1 + text_w + 12)
        cv2.rectangle(frame, (x1, max(0, box_top)), (box_right, min(frame.shape[0] - 1, box_bottom)), color, -1)
        cv2.putText(
            frame,
            label_text,
            (x1 + 6, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


def _compute_bbox_center(bbox):
    x1, y1, x2, y2 = bbox
    return [
        float((x1 + x2) / 2.0),
        float((y1 + y2) / 2.0),
    ]


def _grid_position_from_center(center_xy, frame_width, frame_height):
    if center_xy is None or frame_width <= 0 or frame_height <= 0:
        return None

    center_x, center_y = center_xy
    col = min(2, max(0, int(center_x * 3 / float(frame_width))))
    row = min(2, max(0, int(center_y * 3 / float(frame_height))))
    mapping = {
        (0, 0): "top_left",
        (0, 1): "top",
        (0, 2): "top_right",
        (1, 0): "left",
        (1, 1): "center",
        (1, 2): "right",
        (2, 0): "bottom_left",
        (2, 1): "bottom",
        (2, 2): "bottom_right",
    }
    return mapping[(row, col)]


def _extract_visible_instance_bbox(semantic_obs, semantic_id, min_pixels=25):
    if semantic_obs is None or semantic_id is None:
        return None

    mask = (semantic_obs == int(semantic_id)).astype(np.uint8)
    visible_pixel_count = int(mask.sum())
    if visible_pixel_count < min_pixels:
        return {
            "visible": False,
            "visible_pixel_count": visible_pixel_count,
            "bbox": None,
            "bbox_center_xy": None,
            "view_position_3x3": None,
        }

    ys, xs = np.where(mask == 1)
    if xs.size == 0:
        return {
            "visible": False,
            "visible_pixel_count": visible_pixel_count,
            "bbox": None,
            "bbox_center_xy": None,
            "view_position_3x3": None,
        }

    bbox = [
        int(xs.min()),
        int(ys.min()),
        int(xs.max()),
        int(ys.max()),
    ]
    bbox_center_xy = _compute_bbox_center(bbox)
    view_position = _grid_position_from_center(
        bbox_center_xy,
        semantic_obs.shape[1],
        semantic_obs.shape[0],
    )
    return {
        "visible": True,
        "visible_pixel_count": visible_pixel_count,
        "bbox": bbox,
        "bbox_center_xy": bbox_center_xy,
        "view_position_3x3": view_position,
    }


def _coerce_semantic_id(value):
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _make_explicit_instance_key(record):
    object_id = str(record.get("object_id") or "").strip()
    semantic_id = _coerce_semantic_id(record.get("semantic_id"))
    if object_id:
        return f"object_id::{object_id}"
    if semantic_id is not None:
        return f"semantic_id::{semantic_id}"
    return None


def _instruction_mentions_category(instruction, category):
    normalized_instruction = f" {_normalize_match_text(instruction)} "
    if not normalized_instruction.strip() or not category:
        return False

    alias_records = sorted(
        _build_category_aliases(category),
        key=lambda alias: (len(alias.split()), len(alias)),
        reverse=True,
    )
    for alias in alias_records:
        pattern = re.compile(rf"(?<!\w){re.escape(alias)}(?!\w)")
        if pattern.search(normalized_instruction):
            return True
    return False


def _build_step_task_explicit_instance_catalog(simulator, replay_context):
    raw_step_task_config = replay_context["raw_task_config"]
    by_semantic_id, by_object_id = _build_scene_object_indices(simulator)

    raw_explicit_records = []
    target_instance = raw_step_task_config.get("instruction_target_instance") or {}
    if (
        isinstance(target_instance, dict)
        and (
            target_instance.get("semantic_id") is not None
            or target_instance.get("object_id")
        )
    ):
        raw_explicit_records.append(
            {
                "role": "target",
                "step_index": None,
                **target_instance,
            }
        )

    for anchor in raw_step_task_config.get("instruction_anchor_instances") or []:
        if not isinstance(anchor, dict):
            continue
        if anchor.get("semantic_id") is None and not anchor.get("object_id"):
            continue
        raw_explicit_records.append(
            {
                "role": "anchor",
                **anchor,
            }
        )

    catalog = {}
    target_instance_key = None
    for explicit_record in raw_explicit_records:
        resolved_record, resolved_by = _resolve_explicit_instance_record(
            explicit_record,
            by_semantic_id,
            by_object_id,
        )
        object_id = str(
            (
                resolved_record.get("object_id")
                if resolved_record is not None
                else explicit_record.get("object_id")
            )
            or ""
        ).strip() or None
        semantic_id = _coerce_semantic_id(
            resolved_record.get("semantic_id")
            if resolved_record is not None
            else explicit_record.get("semantic_id")
        )
        category = (
            resolved_record.get("category")
            if resolved_record is not None
            else explicit_record.get("category")
        ) or explicit_record.get("category")
        instance_key = _make_explicit_instance_key(
            {
                "object_id": object_id,
                "semantic_id": semantic_id,
            }
        )
        if instance_key is None:
            continue

        step_index = _coerce_semantic_id(explicit_record.get("step_index"))
        existing = catalog.get(instance_key)
        if existing is None:
            existing = {
                "instance_key": instance_key,
                "object_id": object_id,
                "semantic_id": semantic_id,
                "category": category,
                "center": (
                    np.array(resolved_record["center"], dtype=np.float32)
                    if resolved_record is not None
                    else None
                ),
                "resolved_by": resolved_by,
                "resolved": resolved_record is not None,
                "roles": [],
                "step_indices": [],
            }
            catalog[instance_key] = existing

        role = str(explicit_record.get("role") or "").strip() or None
        if role and role not in existing["roles"]:
            existing["roles"].append(role)
        if step_index is not None and step_index not in existing["step_indices"]:
            existing["step_indices"].append(step_index)
        if existing["category"] is None and category is not None:
            existing["category"] = category
        if existing["center"] is None and resolved_record is not None:
            existing["center"] = np.array(resolved_record["center"], dtype=np.float32)
            existing["resolved"] = True
            existing["resolved_by"] = resolved_by

        if role == "target" and target_instance_key is None:
            target_instance_key = instance_key

    explicit_instances = list(catalog.values())
    explicit_instances.sort(
        key=lambda item: (
            min(item["step_indices"]) if item["step_indices"] else 10**9,
            item.get("object_id") or "",
            item.get("semantic_id") or -1,
        )
    )
    explicit_lookup = {
        item["instance_key"]: item for item in explicit_instances
    }
    return explicit_instances, explicit_lookup, target_instance_key


def _pick_subtask_explicit_instance_for_category(
    candidates,
    subtask_state,
    target_instance_key=None,
):
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    step_start = _coerce_semantic_id(subtask_state.get("step_index_start"))
    step_end = _coerce_semantic_id(subtask_state.get("step_index_end"))
    if step_start is not None and step_end is not None:
        in_step_candidates = [
            item
            for item in candidates
            if any(step_start <= idx <= step_end for idx in item.get("step_indices", []))
        ]
        if len(in_step_candidates) == 1:
            return in_step_candidates[0]
        if in_step_candidates:
            candidates = in_step_candidates

    if target_instance_key:
        destination_hint = _normalize_match_text(subtask_state.get("instruction") or "")
        hinted_target = next(
            (
                item
                for item in candidates
                if item.get("instance_key") == target_instance_key
            ),
            None,
        )
        if hinted_target is not None and (
            subtask_state.get("subtask_index") == subtask_state.get("subtask_total", 0) - 1
            or " reach " in f" {destination_hint} "
            or " stop " in f" {destination_hint} "
            or " toward " in f" {destination_hint} "
            or " to the " in f" {destination_hint} "
        ):
            return hinted_target

    if step_start is not None:
        candidates = sorted(
            candidates,
            key=lambda item: (
                min(
                    [abs(step_start - idx) for idx in item.get("step_indices", [])]
                    or [10**9]
                ),
                item.get("object_id") or "",
                item.get("semantic_id") or -1,
            ),
        )
    return candidates[0]


def _resolve_current_subtask_explicit_instances(
    subtask_state,
    explicit_instances,
    explicit_lookup,
    target_instance_key=None,
):
    if not subtask_state:
        return []

    selected = {}
    for anchor_instance in subtask_state.get("anchor_instances") or []:
        if not isinstance(anchor_instance, dict):
            continue
        instance_key = _make_explicit_instance_key(anchor_instance)
        if instance_key and instance_key in explicit_lookup:
            selected[instance_key] = explicit_lookup[instance_key]

    instruction = str(subtask_state.get("instruction") or "").strip()
    category_matches = {}
    for explicit_instance in explicit_instances:
        category = explicit_instance.get("category")
        if not category:
            continue
        if _instruction_mentions_category(instruction, category):
            category_matches.setdefault(category, []).append(explicit_instance)

    for category, candidates in category_matches.items():
        if any(
            item.get("category") == category for item in selected.values()
        ):
            continue
        chosen = _pick_subtask_explicit_instance_for_category(
            candidates,
            subtask_state,
            target_instance_key=target_instance_key,
        )
        if chosen is not None:
            selected[chosen["instance_key"]] = chosen

    ordered_instances = []
    for explicit_instance in explicit_instances:
        instance_key = explicit_instance.get("instance_key")
        if instance_key in selected:
            ordered_instances.append(selected[instance_key])
    return ordered_instances


def _annotate_explicit_instances_with_subtask_memberships(
    explicit_instances,
    explicit_lookup,
    instruction_subtasks,
    target_instance_key=None,
):
    membership_map = {
        item.get("instance_key"): []
        for item in explicit_instances
        if item.get("instance_key")
    }

    for subtask_state in instruction_subtasks or []:
        resolved_instances = _resolve_current_subtask_explicit_instances(
            subtask_state,
            explicit_instances,
            explicit_lookup,
            target_instance_key=target_instance_key,
        )
        membership_record = {
            "subtask_index": int(subtask_state.get("subtask_index", 0)),
            "subtask_number": int(subtask_state.get("subtask_index", 0)) + 1,
            "instruction": subtask_state.get("instruction"),
            "step_index_start": (
                int(subtask_state["step_index_start"])
                if subtask_state.get("step_index_start") is not None
                else None
            ),
            "step_index_end": (
                int(subtask_state["step_index_end"])
                if subtask_state.get("step_index_end") is not None
                else None
            ),
            "source_start": (
                int(subtask_state["source_start"])
                if subtask_state.get("source_start") is not None
                else None
            ),
            "source_end": (
                int(subtask_state["source_end"])
                if subtask_state.get("source_end") is not None
                else None
            ),
        }
        for explicit_instance in resolved_instances:
            instance_key = explicit_instance.get("instance_key")
            if not instance_key or instance_key not in membership_map:
                continue
            if any(
                item.get("subtask_index") == membership_record["subtask_index"]
                for item in membership_map[instance_key]
            ):
                continue
            membership_map[instance_key].append(dict(membership_record))

    for explicit_instance in explicit_instances:
        instance_key = explicit_instance.get("instance_key")
        memberships = sorted(
            membership_map.get(instance_key, []),
            key=lambda item: item.get("subtask_index", 10**9),
        )
        explicit_instance["subtask_memberships"] = memberships
        explicit_instance["subtask_indices"] = [
            int(item["subtask_index"]) for item in memberships
        ]
        explicit_instance["subtask_numbers"] = [
            int(item["subtask_number"]) for item in memberships
        ]
    return membership_map


def _build_target_specs_from_explicit_instances(
    explicit_instances,
    current_subtask_instance_keys=None,
):
    highlighted_keys = set(current_subtask_instance_keys or [])
    target_specs = []
    for explicit_instance in explicit_instances:
        semantic_id = _coerce_semantic_id(explicit_instance.get("semantic_id"))
        if semantic_id is None:
            continue
        label = (
            explicit_instance.get("object_id")
            or explicit_instance.get("category")
            or str(semantic_id)
        )
        target_specs.append(
            {
                "label": label,
                "object_id": explicit_instance.get("object_id"),
                "category": explicit_instance.get("category"),
                "subtask_numbers": [
                    int(item)
                    for item in (explicit_instance.get("subtask_numbers") or [])
                    if item is not None
                ],
                "scope": "explicit_instance",
                "semantic_ids": [semantic_id],
                "semantic_labels": {
                    semantic_id: label,
                },
                "is_current_subtask_target": (
                    explicit_instance.get("instance_key") in highlighted_keys
                ),
            }
        )
    return target_specs


def _build_step_task_frame_target_state(
    target_instance_record,
    semantic_obs,
    entry,
    pathfinder,
    min_pixels=25,
):
    if not target_instance_record:
        return None

    state = {
        "object_id": target_instance_record.get("object_id"),
        "semantic_id": target_instance_record.get("semantic_id"),
        "category": target_instance_record.get("category"),
        "subtask_indices": [
            int(item) for item in (target_instance_record.get("subtask_indices") or [])
        ],
        "subtask_numbers": [
            int(item) for item in (target_instance_record.get("subtask_numbers") or [])
        ],
        "subtask_memberships": [
            {
                "subtask_index": int(item["subtask_index"]),
                "subtask_number": int(item["subtask_number"]),
                "instruction": item.get("instruction"),
                "step_index_start": item.get("step_index_start"),
                "step_index_end": item.get("step_index_end"),
                "source_start": item.get("source_start"),
                "source_end": item.get("source_end"),
            }
            for item in (target_instance_record.get("subtask_memberships") or [])
            if isinstance(item, dict) and item.get("subtask_index") is not None
        ],
        "visible_in_current_view": False,
        "visible_pixel_count": 0,
        "bbox_xyxy": None,
        "bbox_center_xy": None,
        "view_position_3x3": None,
        "distance_m": None,
        "distance_strategy": None,
    }

    semantic_id = target_instance_record.get("semantic_id")
    center = target_instance_record.get("center")
    if semantic_id is not None and semantic_obs is not None:
        visible_info = _extract_visible_instance_bbox(
            semantic_obs,
            semantic_id,
            min_pixels=min_pixels,
        )
        if visible_info is not None:
            state["visible_in_current_view"] = bool(visible_info["visible"])
            state["visible_pixel_count"] = int(visible_info["visible_pixel_count"])
            state["bbox_xyxy"] = visible_info["bbox"]
            state["bbox_center_xy"] = visible_info["bbox_center_xy"]
            state["view_position_3x3"] = visible_info["view_position_3x3"]

    if center is not None:
        distance_info = _candidate_distance_key(
            pathfinder,
            entry["pos"],
            center,
        )
        state["distance_m"] = float(distance_info["distance"])
        state["distance_strategy"] = distance_info["strategy"]

    return state


def _build_step_task_frame_target_states(
    target_instance_records,
    semantic_obs,
    entry,
    pathfinder,
    min_pixels=25,
):
    target_states = []
    for target_instance_record in target_instance_records or []:
        target_state = _build_step_task_frame_target_state(
            target_instance_record,
            semantic_obs,
            entry,
            pathfinder,
            min_pixels=min_pixels,
        )
        if target_state is not None:
            target_states.append(target_state)
    return target_states


def _filter_visible_step_task_frame_target_states(target_states):
    return [
        target_state
        for target_state in target_states or []
        if target_state.get("visible_in_current_view")
    ]


def _build_step_task_frame_record(
    entry,
    frame_index,
    subtask_state,
    target_object_states,
    current_subtask_target_object_states=None,
    all_subtask_target_object_states=None,
    frame_asset_record=None,
):
    frame_record = {
        "frame_index": int(frame_index),
        "global_step": int(entry["global_step"]) if entry.get("global_step") is not None else None,
        "source_trial_index": int(entry["source_trial_index"]) if entry.get("source_trial_index") is not None else None,
        "source_trial_key": entry.get("source_trial_key"),
        "source_state_index": int(entry["source_state_index"]) if entry.get("source_state_index") is not None else None,
        "next_action": entry.get("action"),
        "agent_position": _as_float_list(entry["pos"]),
        "agent_yaw": float(entry["yaw"]),
        "target_object_state": target_object_states[0] if target_object_states else None,
        "target_object_states": target_object_states,
        "current_subtask_target_object_state": (
            current_subtask_target_object_states[0]
            if current_subtask_target_object_states
            else None
        ),
        "current_subtask_target_object_states": current_subtask_target_object_states or [],
        "all_subtask_target_object_state": (
            all_subtask_target_object_states[0]
            if all_subtask_target_object_states
            else None
        ),
        "all_subtask_target_object_states": all_subtask_target_object_states or [],
        "rgb_path": frame_asset_record.get("rgb_path") if frame_asset_record else None,
        "depth_vis_path": frame_asset_record.get("depth_vis_path") if frame_asset_record else None,
        "depth_raw_path": frame_asset_record.get("depth_raw_path") if frame_asset_record else None,
        "depth_min_m": frame_asset_record.get("depth_min_m") if frame_asset_record else None,
        "depth_max_m": frame_asset_record.get("depth_max_m") if frame_asset_record else None,
    }
    if subtask_state:
        frame_record.update(
            {
                "subtask_index": int(subtask_state["subtask_index"]),
                "subtask_instruction": subtask_state.get("instruction"),
                "step_index_start": int(subtask_state["step_index_start"]),
                "step_index_end": int(subtask_state["step_index_end"]),
                "source_start": int(subtask_state["source_start"]),
                "source_end": int(subtask_state["source_end"]),
            }
        )
    else:
        frame_record.update(
            {
                "subtask_index": None,
                "subtask_instruction": None,
                "step_index_start": None,
                "step_index_end": None,
                "source_start": None,
                "source_end": None,
            }
        )
    return frame_record


def _build_step_task_frame_overlay_lines(frame_record):
    target_states = list(frame_record.get("current_subtask_target_object_states") or [])
    if not target_states:
        target_states = list(frame_record.get("target_object_states") or [])
    lines = [
        "Frame debug",
        f"Next action: {frame_record.get('next_action') or 'n/a'}",
    ]

    subtask_index = frame_record.get("subtask_index")
    if subtask_index is not None:
        lines.append(
            "Subtask: "
            f"{subtask_index + 1} | Steps "
            f"{frame_record.get('step_index_start')}-{frame_record.get('step_index_end')}"
        )
        if frame_record.get("subtask_instruction"):
            lines.append(f"Subtask text: {frame_record['subtask_instruction']}")
    else:
        lines.append("Subtask: unmatched")

    if not target_states:
        lines.append("Subtask objects: none")
        return lines

    object_labels = [
        item.get("object_id") or item.get("category") or "unknown"
        for item in target_states
    ]
    lines.append(f"Subtask objects: {', '.join(object_labels[:3])}")
    visible_count = sum(
        1 for item in target_states if item.get("visible_in_current_view")
    )
    lines.append(f"Visible now: {visible_count}/{len(target_states)}")

    for target_state in target_states[:3]:
        target_label = (
            target_state.get("object_id")
            or target_state.get("category")
            or "unresolved"
        )
        line = (
            f"{target_label} | "
            + ("visible" if target_state.get("visible_in_current_view") else "hidden")
        )
        if target_state.get("view_position_3x3"):
            line += f" | {target_state['view_position_3x3']}"
        if target_state.get("distance_m") is not None:
            line += f" | {target_state['distance_m']:.2f} m"
        lines.append(line)
        if target_state.get("bbox_xyxy") is not None:
            lines.append(f"BBox: {target_state['bbox_xyxy']}")
    return lines


def _to_relative_output_path(path, task_json_path):
    if not path:
        return path
    try:
        return os.path.relpath(path, os.path.dirname(task_json_path))
    except Exception:
        return path


def _reset_generated_dir(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def _prepare_step_task_frame_asset_dirs(task_json_path):
    asset_root = os.path.splitext(task_json_path)[0] + "_frame_assets"
    _reset_generated_dir(asset_root)

    dirs = {
        "root": asset_root,
        "rgb": os.path.join(asset_root, "frames_rgb"),
        "depth_vis": os.path.join(asset_root, "frames_depth_vis"),
        "depth_raw": os.path.join(asset_root, "frames_depth_raw"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def _compute_depth_stats(depth_obs):
    if depth_obs is None:
        return {
            "depth_min_m": None,
            "depth_max_m": None,
        }

    finite = depth_obs[np.isfinite(depth_obs)]
    if finite.size == 0:
        return {
            "depth_min_m": None,
            "depth_max_m": None,
        }

    return {
        "depth_min_m": float(np.min(finite)),
        "depth_max_m": float(np.max(finite)),
    }


def _depth_to_vis_uint8(depth_obs, clip_max_m=DEFAULT_DEPTH_VIS_MAX_METERS):
    if depth_obs is None:
        return None

    depth = np.array(depth_obs, dtype=np.float32, copy=True)
    depth[~np.isfinite(depth)] = clip_max_m
    depth = np.clip(depth, 0.0, clip_max_m)
    if clip_max_m <= 0:
        raise ValueError("clip_max_m must be positive.")
    return np.round(depth / clip_max_m * 255.0).astype(np.uint8)


def _save_step_task_frame_assets(
    task_json_path,
    frame_index,
    rgb_frame,
    depth_obs,
    frame_asset_dirs,
):
    stem = f"frame_{frame_index:06d}"
    rgb_path = os.path.join(frame_asset_dirs["rgb"], stem + ".png")
    depth_vis_path = os.path.join(frame_asset_dirs["depth_vis"], stem + ".png")
    depth_raw_path = os.path.join(frame_asset_dirs["depth_raw"], stem + ".npy")

    if not cv2.imwrite(rgb_path, rgb_frame):
        raise RuntimeError(f"Failed to save RGB frame to {rgb_path}")

    saved_depth_vis_path = None
    saved_depth_raw_path = None
    if depth_obs is not None:
        depth_vis = _depth_to_vis_uint8(depth_obs)
        if depth_vis is not None:
            if not cv2.imwrite(depth_vis_path, depth_vis):
                raise RuntimeError(
                    f"Failed to save depth visualization frame to {depth_vis_path}"
                )
            saved_depth_vis_path = depth_vis_path

        np.save(depth_raw_path, np.array(depth_obs, dtype=np.float32, copy=True))
        saved_depth_raw_path = depth_raw_path

    return {
        "rgb_path": _to_relative_output_path(rgb_path, task_json_path),
        "depth_vis_path": _to_relative_output_path(
            saved_depth_vis_path,
            task_json_path,
        ),
        "depth_raw_path": _to_relative_output_path(
            saved_depth_raw_path,
            task_json_path,
        ),
        **_compute_depth_stats(depth_obs),
    }


def _write_step_task_frame_info(
    task_json_path,
    replay_context,
    frame_mode,
    width,
    height,
    frames,
    frame_asset_dirs=None,
):
    frame_info_path = os.path.splitext(task_json_path)[0] + ".frame_info.json"
    raw_task_config = replay_context["raw_task_config"]
    payload = {
        "meta": {
            "step_task_json": task_json_path,
            "task_instruction": raw_task_config.get("Task instruction"),
            "trajectory_path": raw_task_config.get("trajectory path"),
            "source_task_json": replay_context.get("source_task_json"),
            "scene": raw_task_config.get("Scene"),
            "robot": raw_task_config.get("Robot"),
            "frame_mode": frame_mode,
            "image_width": int(width),
            "image_height": int(height),
            "step_task_source_range": replay_context.get("source_step_range"),
            "instruction_target_instance": raw_task_config.get("instruction_target_instance"),
            "instruction_anchor_instances": raw_task_config.get("instruction_anchor_instances") or [],
            "instruction_subtask_alignment_json": raw_task_config.get("instruction_subtask_alignment_json"),
            "source_compressed_steps_json": raw_task_config.get("source_compressed_steps_json"),
            "frame_target_source": "visible_step_task_explicit_objects",
            "current_subtask_target_source": "current_subtask_explicit_objects",
            "all_subtask_target_source": "all_step_task_explicit_objects",
            "frame_assets_dir": _to_relative_output_path(
                frame_asset_dirs["root"],
                task_json_path,
            ) if frame_asset_dirs else None,
            "frames_rgb_dir": _to_relative_output_path(
                frame_asset_dirs["rgb"],
                task_json_path,
            ) if frame_asset_dirs else None,
            "frames_depth_vis_dir": _to_relative_output_path(
                frame_asset_dirs["depth_vis"],
                task_json_path,
            ) if frame_asset_dirs else None,
            "frames_depth_raw_dir": _to_relative_output_path(
                frame_asset_dirs["depth_raw"],
                task_json_path,
            ) if frame_asset_dirs else None,
            "depth_visualization_clip_max_m": DEFAULT_DEPTH_VIS_MAX_METERS,
            "depth_visualization_format": "uint8_png_grayscale",
            "depth_raw_format": "float32_npy_meters",
        },
        "subtasks": replay_context.get("instruction_subtasks", []),
        "frames": frames,
    }
    with open(frame_info_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return frame_info_path


def export_video(
    task_json_path,
    output_video,
    fps,
    width,
    height,
    sensor_height,
    hfov,
    sim_gpu_device,
    frame_mode="state",
    annotate=False,
    target_boxes=False,
    target_box_scope="instance",
    output_codec=DEFAULT_OUTPUT_CODEC,
    skip_video_output=False,
):
    task_json_path = os.path.abspath(task_json_path)
    output_video = os.path.abspath(output_video) if output_video is not None else None
    if not os.path.exists(task_json_path):
        raise FileNotFoundError(f"task.json not found: {task_json_path}")
    if "/nav_gen/" not in task_json_path:
        raise RuntimeError("task_json path must be inside the LH-VLN nav_gen directory.")
    nav_gen_root = task_json_path.split("/nav_gen/")[0] + "/nav_gen"
    project_root = os.path.dirname(nav_gen_root)

    task_config = _load_json(task_json_path)
    replay_context = _build_replay_context(task_json_path, task_config, frame_mode)
    if skip_video_output and replay_context["kind"] != "step_task":
        raise RuntimeError("--skip-video-output is only supported for step_task json inputs.")
    if not skip_video_output and output_video is None:
        raise ValueError("output_video is required unless --skip-video-output is enabled.")
    need_semantic_sensor = target_boxes or replay_context["kind"] == "step_task"
    need_depth_sensor = replay_context["kind"] == "step_task"

    args = _build_args(
        project_root,
        width,
        height,
        sensor_height,
        hfov,
        sim_gpu_device,
        enable_front_semantic=need_semantic_sensor,
        enable_front_depth=need_depth_sensor,
    )
    writer_output_video = None
    final_output_video = None
    video_writer = None
    if not skip_video_output:
        os.makedirs(os.path.dirname(output_video), exist_ok=True)
        writer_output_video, final_output_video = _build_output_paths(output_video, output_codec)

        video_writer = cv2.VideoWriter(
            writer_output_video,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not video_writer.isOpened():
            raise RuntimeError(f"Failed to open video writer for {writer_output_video}")

    display_task_config = replay_context["display_task_config"]
    simulator = SceneSimulator(args=args, config=replay_context["simulator_config"])
    timeline = replay_context["timeline"]
    trial_target_lookup = {}
    target_resolution = []
    instruction_object_categories = []
    if target_boxes:
        if replay_context["kind"] == "step_task":
            (
                trial_target_lookup,
                target_resolution,
                instruction_object_categories,
            ) = _build_step_task_target_lookup(
                simulator,
                replay_context,
                scope=target_box_scope,
            )
        else:
            trial_target_lookup, target_resolution = _build_trial_target_lookup(
                simulator,
                display_task_config,
                scope=target_box_scope,
            )
    frame_count = 0
    labeled_frame_count = 0
    frame_info_records = []
    frame_info_path = None
    frame_asset_dirs = None
    try:
        explicit_step_task_instances = []
        explicit_step_task_lookup = {}
        target_instance_key = None
        if replay_context["kind"] == "step_task":
            frame_asset_dirs = _prepare_step_task_frame_asset_dirs(task_json_path)
            (
                explicit_step_task_instances,
                explicit_step_task_lookup,
                target_instance_key,
            ) = _build_step_task_explicit_instance_catalog(
                simulator,
                replay_context,
            )
            _annotate_explicit_instances_with_subtask_memberships(
                explicit_step_task_instances,
                explicit_step_task_lookup,
                replay_context.get("instruction_subtasks", []),
                target_instance_key=target_instance_key,
            )
        for frame_index, entry in enumerate(timeline):
            frame, semantic_obs, depth_obs = _capture_frame(
                simulator,
                entry["pos"],
                entry["yaw"],
            )
            current_subtask = None
            if replay_context["kind"] == "step_task":
                current_subtask = _match_step_task_instruction_subtask(
                    replay_context.get("instruction_subtasks", []),
                    entry.get("global_step"),
                )
            current_target_instances = []
            if replay_context["kind"] == "step_task":
                current_target_instances = _resolve_current_subtask_explicit_instances(
                    current_subtask,
                    explicit_step_task_instances,
                    explicit_step_task_lookup,
                    target_instance_key=target_instance_key,
                )
            if replay_context["kind"] == "step_task":
                current_subtask_instance_keys = [
                    item.get("instance_key")
                    for item in current_target_instances
                    if item.get("instance_key")
                ]
                target_spec = _build_target_specs_from_explicit_instances(
                    explicit_step_task_instances,
                    current_subtask_instance_keys=current_subtask_instance_keys,
                ) if target_boxes else None
            else:
                target_spec = trial_target_lookup.get(entry["trial_index"])
            boxes = _extract_target_boxes(semantic_obs, target_spec) if target_boxes else []
            current_target_states = []
            if replay_context["kind"] == "step_task":
                frame_asset_record = _save_step_task_frame_assets(
                    task_json_path,
                    frame_index,
                    frame,
                    depth_obs,
                    frame_asset_dirs,
                )
                all_subtask_target_states = _build_step_task_frame_target_states(
                    explicit_step_task_instances,
                    semantic_obs,
                    entry,
                    simulator.pathfinder,
                )
                visible_subtask_target_states = _filter_visible_step_task_frame_target_states(
                    all_subtask_target_states,
                )
                current_target_states = _build_step_task_frame_target_states(
                    current_target_instances,
                    semantic_obs,
                    entry,
                    simulator.pathfinder,
                )
                frame_info_record = _build_step_task_frame_record(
                    entry,
                    frame_index,
                    current_subtask,
                    visible_subtask_target_states,
                    current_subtask_target_object_states=current_target_states,
                    all_subtask_target_object_states=all_subtask_target_states,
                    frame_asset_record=frame_asset_record,
                )
                frame_info_records.append(frame_info_record)
            if annotate:
                frame = _annotate_frame(
                    frame,
                    _overlay_lines(
                        display_task_config,
                        entry,
                        frame_index,
                        len(timeline),
                        subtask_state=current_subtask,
                    ),
                )
                if replay_context["kind"] == "step_task" and frame_info_records:
                    frame = _draw_overlay_panel(
                        frame,
                        _build_step_task_frame_overlay_lines(frame_info_records[-1]),
                        anchor="top_right",
                        wrap_width=36 if frame.shape[1] >= 1000 else 24,
                        max_width_fraction=0.42,
                    )
            if boxes:
                frame = _draw_target_boxes(frame, boxes, semantic_obs=semantic_obs)
                labeled_frame_count += 1
            if video_writer is not None:
                video_writer.write(frame)
            frame_count += 1
    finally:
        if video_writer is not None:
            video_writer.release()
        simulator.sim.close()

    ffmpeg_bin = None
    if final_output_video is not None:
        ffmpeg_bin = _transcode_video_to_h264(writer_output_video, final_output_video)

    metadata = {
        "task_json": task_json_path,
        "output_video": output_video,
        "writer_output_video": writer_output_video,
        "frame_count": frame_count,
        "fps": fps,
        "width": width,
        "height": height,
        "sensor_height_m": sensor_height,
        "horizontal_fov_deg": hfov,
        "vertical_fov_deg": _compute_vfov(hfov, width, height),
        "yaw_mapping": "world_angle_deg = saved_yaw_deg - 180",
        "frame_mode": frame_mode,
        "annotate": annotate,
        "target_boxes": target_boxes,
        "target_box_scope": target_box_scope,
        "output_codec": output_codec,
        "video_output_enabled": not skip_video_output,
        "task_json_type": replay_context["kind"],
        "trial_targets": replay_context["trial_targets"],
        "resolved_target_boxes": target_resolution,
        "frames_with_target_boxes": labeled_frame_count,
    }
    if replay_context["kind"] == "step_task":
        frame_info_path = _write_step_task_frame_info(
            task_json_path,
            replay_context,
            frame_mode,
            width,
            height,
            frame_info_records,
            frame_asset_dirs=frame_asset_dirs,
        )
        metadata["source_task_json"] = replay_context["source_task_json"]
        metadata["source_trial_dir"] = replay_context["source_trial_dir"]
        metadata["source_trial_indices"] = replay_context["source_trial_indices"]
        metadata["source_trial_keys"] = replay_context["source_trial_keys"]
        metadata["source_step_range"] = replay_context["source_step_range"]
        metadata["instruction_object_categories"] = instruction_object_categories
        metadata["instruction_target_instance"] = replay_context["raw_task_config"].get("instruction_target_instance")
        metadata["instruction_anchor_instances"] = replay_context["raw_task_config"].get("instruction_anchor_instances")
        metadata["instruction_subtasks"] = replay_context.get("instruction_subtasks", [])
        metadata["frame_info_json"] = _to_relative_output_path(frame_info_path, task_json_path)
        metadata["frame_assets_dir"] = _to_relative_output_path(
            frame_asset_dirs["root"],
            task_json_path,
        ) if frame_asset_dirs else None
    if ffmpeg_bin is not None:
        metadata["ffmpeg_bin"] = ffmpeg_bin
        metadata["video_codec_details"] = "libx264 + yuv420p + faststart"
    if not skip_video_output and output_video is not None:
        metadata_path = os.path.splitext(output_video)[0] + ".json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Replay a NavGen trajectory and export a first-person RGB video.")
    parser.add_argument("--task_json", required=True, help="path to success/fail trial task.json")
    parser.add_argument("--output_video", required=True, help="output mp4 path")
    parser.add_argument("--fps", type=int, default=DEFAULT_FPS, help="video fps")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="video width")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="video height")
    parser.add_argument("--sensor_height", type=float, default=DEFAULT_SENSOR_HEIGHT, help="camera height in meters")
    parser.add_argument("--hfov", type=float, default=DEFAULT_HFOV, help="horizontal FOV in degrees")
    parser.add_argument("--sim_gpu_device", type=int, default=int(os.environ.get("NAVGEN_SIM_GPU_DEVICE", 0)), help="Habitat-Sim GPU id")
    parser.add_argument("--frame_mode", choices=["state", "action"], default="state", help="state: one frame per saved state, action: one frame per executed action")
    parser.add_argument("--annotate", action="store_true", help="overlay frame, action, target, and task text")
    parser.add_argument("--target_boxes", action="store_true", help="draw 2D bounding boxes for visible target objects mentioned in the instruction")
    parser.add_argument("--target_box_scope", choices=["instance", "class"], default="instance", help="instance: only the current trial's selected target instance, class: all matching objects for the current trial")
    parser.add_argument("--output_codec", choices=["h264", "mp4v"], default=DEFAULT_OUTPUT_CODEC, help="h264: write a broadly compatible MP4 via ffmpeg, mp4v: keep the raw OpenCV mp4v output")
    parser.add_argument("--skip-video-output", action="store_true", help="replay the trajectory and export step-task frame assets / frame_info only, without writing an mp4")
    cli_args = parser.parse_args()

    metadata = export_video(
        task_json_path=cli_args.task_json,
        output_video=cli_args.output_video,
        fps=cli_args.fps,
        width=cli_args.width,
        height=cli_args.height,
        sensor_height=cli_args.sensor_height,
        hfov=cli_args.hfov,
        sim_gpu_device=cli_args.sim_gpu_device,
        frame_mode=cli_args.frame_mode,
        annotate=cli_args.annotate,
        target_boxes=cli_args.target_boxes,
        target_box_scope=cli_args.target_box_scope,
        output_codec=cli_args.output_codec,
        skip_video_output=cli_args.skip_video_output,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
