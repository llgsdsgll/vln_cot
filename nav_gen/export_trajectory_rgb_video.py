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
DEFAULT_FPS = 8
DEFAULT_OUTPUT_CODEC = "h264"


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


def _build_args(project_root, width, height, sensor_height, hfov, sim_gpu_device, enable_front_semantic):
    temp_task_root = os.path.join(project_root, "nav_gen", "_video_export_tmp") + "/"
    return SimpleNamespace(
        scene=os.path.join(project_root, "data", "hm3d") + "/",
        scene_dataset=os.path.join(project_root, "data", "hm3d", "hm3d_annotated_basis.scene_dataset_config.json"),
        sim_gpu_device=sim_gpu_device,
        task_path=temp_task_root,
        front_rgb_only=True,
        front_semantic_sensor=enable_front_semantic,
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
    return frame, semantic


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


def _overlay_lines(task_config, entry, frame_index, total_frames):
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
    instruction = task_config.get("Task instruction")
    if instruction:
        lines.append(f"Task: {instruction}")
    return lines


def _annotate_frame(frame, lines):
    wrapped_lines = []
    wrap_width = 70 if frame.shape[1] >= 1000 else 45
    for line in lines:
        wrapped = textwrap.wrap(line, width=wrap_width) or [line]
        wrapped_lines.extend(wrapped)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7 if frame.shape[1] >= 1000 else 0.55
    thickness = 2
    padding = 12
    line_height = 28 if frame.shape[1] >= 1000 else 22
    box_width = min(frame.shape[1] - 2 * padding, int(frame.shape[1] * 0.82))
    box_height = padding * 2 + line_height * len(wrapped_lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (padding, padding), (padding + box_width, padding + box_height), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.62, frame, 0.38, 0)

    y = padding + 22
    for line in wrapped_lines:
        cv2.putText(frame, line, (padding + 10, y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        y += line_height
    return frame


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
        _, semantic_obs = _capture_frame(simulator, pos_list[frame_index], yaw_list[frame_index])
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
        color = _color_for_label(box["label"])
        if semantic_obs is not None:
            mask = (semantic_obs == box["semantic_id"]).astype(np.uint8)
            overlay = frame.copy()
            overlay[mask == 1] = (
                overlay[mask == 1] * 0.5 + np.array(color[::-1], dtype=np.float32) * 0.5
            ).astype(np.uint8)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        label_text = box["label"]
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
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
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return frame


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
):
    task_json_path = os.path.abspath(task_json_path)
    output_video = os.path.abspath(output_video)
    if not os.path.exists(task_json_path):
        raise FileNotFoundError(f"task.json not found: {task_json_path}")
    if "/nav_gen/" not in task_json_path:
        raise RuntimeError("task_json path must be inside the LH-VLN nav_gen directory.")
    nav_gen_root = task_json_path.split("/nav_gen/")[0] + "/nav_gen"
    project_root = os.path.dirname(nav_gen_root)

    task_config = _load_json(task_json_path)
    replay_context = _build_replay_context(task_json_path, task_config, frame_mode)

    args = _build_args(
        project_root,
        width,
        height,
        sensor_height,
        hfov,
        sim_gpu_device,
        enable_front_semantic=target_boxes,
    )
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
    try:
        for frame_index, entry in enumerate(timeline):
            frame, semantic_obs = _capture_frame(simulator, entry["pos"], entry["yaw"])
            target_spec = trial_target_lookup.get(entry["trial_index"])
            boxes = _extract_target_boxes(semantic_obs, target_spec) if target_boxes else []
            if annotate:
                frame = _annotate_frame(
                    frame,
                    _overlay_lines(display_task_config, entry, frame_index, len(timeline)),
                )
            if boxes:
                frame = _draw_target_boxes(frame, boxes, semantic_obs=semantic_obs)
                labeled_frame_count += 1
            video_writer.write(frame)
            frame_count += 1
    finally:
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
        "task_json_type": replay_context["kind"],
        "trial_targets": replay_context["trial_targets"],
        "resolved_target_boxes": target_resolution,
        "frames_with_target_boxes": labeled_frame_count,
    }
    if replay_context["kind"] == "step_task":
        metadata["source_task_json"] = replay_context["source_task_json"]
        metadata["source_trial_dir"] = replay_context["source_trial_dir"]
        metadata["source_trial_indices"] = replay_context["source_trial_indices"]
        metadata["source_trial_keys"] = replay_context["source_trial_keys"]
        metadata["source_step_range"] = replay_context["source_step_range"]
        metadata["instruction_object_categories"] = instruction_object_categories
        metadata["instruction_target_instance"] = replay_context["raw_task_config"].get("instruction_target_instance")
        metadata["instruction_anchor_instances"] = replay_context["raw_task_config"].get("instruction_anchor_instances")
    if ffmpeg_bin is not None:
        metadata["ffmpeg_bin"] = ffmpeg_bin
        metadata["video_codec_details"] = "libx264 + yuv420p + faststart"
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
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
