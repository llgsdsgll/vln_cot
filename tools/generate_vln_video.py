#!/usr/bin/env python3

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

os.environ.setdefault("GLOG_minloglevel", "2")

import cv2
import habitat_sim
import numpy as np


DEFAULT_TRAIN_JSON = Path(
    "/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json"
)
DEFAULT_TRAIN_GT_JSON = Path(
    "/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json"
)
DEFAULT_SCENE_ROOT = Path(
    "/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat/mp3d"
)
DEFAULT_OUTPUT_DIR = Path("/home/gs/my_project/data_generation_pipline/output")

STOP = 0
MOVE_FORWARD = 1
TURN_LEFT = 2
TURN_RIGHT = 3


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render first-person RGB and semantic videos from R2R ground-truth "
            "actions. One action corresponds to exactly one frame."
        )
    )
    parser.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    parser.add_argument(
        "--train-gt-json", type=Path, default=DEFAULT_TRAIN_GT_JSON
    )
    parser.add_argument(
        "--scene-root",
        type=Path,
        default=DEFAULT_SCENE_ROOT,
        help=(
            "Path to either .../mp3d_habitat or .../mp3d_habitat/mp3d. "
            "The script resolves both layouts automatically."
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--episode-ids",
        type=int,
        nargs="+",
        default=None,
        help="Only render the specified episode ids.",
    )
    parser.add_argument(
        "--max-episodes",
        type=int,
        default=None,
        help="Only render the first N selected episodes.",
    )
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--sensor-height", type=float, default=1.25)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--forward-step-size", type=float, default=0.25)
    parser.add_argument("--turn-angle", type=float, default=15.0)
    parser.add_argument(
        "--semantic-label-min-area-ratio",
        type=float,
        default=0.01,
        help="Minimum connected-component area ratio for semantic labels.",
    )
    parser.add_argument(
        "--max-semantic-labels-per-frame",
        type=int,
        default=12,
        help="Maximum number of semantic region labels to draw per frame.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite already generated videos.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_call(value: Any) -> Any:
    return value() if callable(value) else value


def select_episode_ids(
    train_episodes: Dict[str, dict],
    gt_episodes: Dict[str, dict],
    requested_ids: Sequence[int] = None,
    max_episodes: int = None,
) -> List[str]:
    if requested_ids is not None:
        episode_ids = [str(episode_id) for episode_id in requested_ids]
    else:
        episode_ids = sorted(gt_episodes.keys(), key=lambda item: int(item))

    filtered_ids = []
    missing_from_train = 0
    missing_from_gt = 0

    for episode_id in episode_ids:
        if episode_id not in train_episodes:
            missing_from_train += 1
            continue
        if episode_id not in gt_episodes:
            missing_from_gt += 1
            continue
        filtered_ids.append(episode_id)

    if max_episodes is not None:
        filtered_ids = filtered_ids[:max_episodes]

    if missing_from_train:
        print(f"Skipped {missing_from_train} episode ids missing from train.json")
    if missing_from_gt:
        print(f"Skipped {missing_from_gt} episode ids missing from train_gt.json")

    return filtered_ids


def resolve_scene_path(scene_root: Path, scene_id: str) -> Path:
    scene_id_path = Path(scene_id)
    candidates: List[Path] = []

    if scene_id_path.is_absolute():
        candidates.append(scene_id_path)
    else:
        candidates.append(scene_root / scene_id_path)
        candidates.append(scene_root.parent / scene_id_path)
        if scene_id_path.parts and scene_id_path.parts[0] == "mp3d":
            candidates.append(scene_root / Path(*scene_id_path.parts[1:]))
        else:
            candidates.append(scene_root / "mp3d" / scene_id_path)

    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.exists():
            return candidate

    candidate_str = "\n".join(str(path) for path in seen)
    raise FileNotFoundError(
        f"Could not resolve scene '{scene_id}'. Checked:\n{candidate_str}"
    )


def build_simulator(
    scene_path: Path,
    width: int,
    height: int,
    sensor_height: float,
    forward_step_size: float,
    turn_angle: float,
    include_depth: bool = False,
    normalize_depth: bool = False,
) -> habitat_sim.Simulator:
    backend_cfg = habitat_sim.SimulatorConfiguration()
    backend_cfg.scene_id = str(scene_path)
    backend_cfg.enable_physics = False

    rgb_sensor = habitat_sim.SensorSpec()
    rgb_sensor.uuid = "rgb"
    rgb_sensor.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor.resolution = [height, width]
    rgb_sensor.position = [0.0, sensor_height, 0.0]

    semantic_sensor = habitat_sim.SensorSpec()
    semantic_sensor.uuid = "semantic"
    semantic_sensor.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor.resolution = [height, width]
    semantic_sensor.position = [0.0, sensor_height, 0.0]

    sensor_specs = [rgb_sensor, semantic_sensor]
    if include_depth:
        depth_sensor = habitat_sim.SensorSpec()
        depth_sensor.uuid = "depth"
        depth_sensor.sensor_type = habitat_sim.SensorType.DEPTH
        depth_sensor.resolution = [height, width]
        depth_sensor.position = [0.0, sensor_height, 0.0]
        depth_sensor.normalize_depth = normalize_depth
        sensor_specs.append(depth_sensor)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        MOVE_FORWARD: habitat_sim.ActionSpec(
            "move_forward",
            habitat_sim.ActuationSpec(amount=forward_step_size),
        ),
        TURN_LEFT: habitat_sim.ActionSpec(
            "turn_left", habitat_sim.ActuationSpec(amount=turn_angle)
        ),
        TURN_RIGHT: habitat_sim.ActionSpec(
            "turn_right", habitat_sim.ActuationSpec(amount=turn_angle)
        ),
    }

    return habitat_sim.Simulator(
        habitat_sim.Configuration(backend_cfg, [agent_cfg])
    )


def set_agent_state(
    sim: habitat_sim.Simulator,
    start_position: Sequence[float],
    start_rotation: Sequence[float],
) -> None:
    state = habitat_sim.AgentState()
    state.position = np.asarray(start_position, dtype=np.float32)
    state.rotation = np.asarray(start_rotation, dtype=np.float32)
    sim.get_agent(0).set_state(state)


def rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
    if frame.shape[-1] == 4:
        return cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)


def semantic_id_colors(semantic_id: int) -> Tuple[List[int], List[int]]:
    semantic_id = int(semantic_id)
    color_bgr = [
        (semantic_id * 37) % 255,
        (semantic_id * 17) % 255,
        (semantic_id * 29) % 255,
    ]
    if semantic_id == 0:
        color_bgr = [0, 0, 0]
    color_rgb = list(reversed(color_bgr))
    return color_rgb, color_bgr


def semantic_entry_from_scene(
    semantic_scene: Any, semantic_id: int
) -> Dict[str, Any]:
    object_index = semantic_scene.semantic_index_to_object_index(int(semantic_id))
    instance_id = None
    name = None

    if (
        object_index is not None
        and object_index >= 0
        and object_index < len(semantic_scene.objects)
        and semantic_scene.objects[object_index] is not None
    ):
        obj = semantic_scene.objects[object_index]
        instance_id = obj.id
        if obj.category is not None:
            name = obj.category.name()
    else:
        object_index = None

    color_rgb, color_bgr = semantic_id_colors(semantic_id)

    return {
        "semantic_id": int(semantic_id),
        "object_index": object_index,
        "instance_id": instance_id,
        "name": name,
        "color_rgb": color_rgb,
        "color_bgr": color_bgr,
        "color_hex": "#{:02X}{:02X}{:02X}".format(*color_rgb),
    }


def colorize_semantic(
    semantic_frame: np.ndarray,
    semantic_scene: Any,
    semantic_entries: Dict[int, Dict[str, Any]],
) -> np.ndarray:
    semantic_ids = semantic_frame.astype(np.uint32, copy=False)
    unique_ids, inverse = np.unique(semantic_ids, return_inverse=True)
    colors_bgr = np.empty((len(unique_ids), 3), dtype=np.uint8)

    for index, semantic_id in enumerate(unique_ids.tolist()):
        semantic_id = int(semantic_id)
        entry = semantic_entries.get(semantic_id)
        if entry is None:
            entry = semantic_entry_from_scene(semantic_scene, semantic_id)
            semantic_entries[semantic_id] = entry
        colors_bgr[index] = np.asarray(entry["color_bgr"], dtype=np.uint8)

    return colors_bgr[inverse].reshape(semantic_ids.shape + (3,))


def semantic_label_text(entry: Dict[str, Any]) -> str:
    if entry["name"]:
        return str(entry["name"])
    if entry["instance_id"]:
        return str(entry["instance_id"])
    return f"semantic_{entry['semantic_id']}"


def draw_text_box(
    image: np.ndarray,
    text: str,
    anchor: Tuple[int, int],
    color_bgr: Sequence[int],
    occupied_boxes: List[Tuple[int, int, int, int]],
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    thickness = 1
    padding = 4
    img_h, img_w = image.shape[:2]
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    box_w = text_size[0] + padding * 2
    box_h = text_size[1] + padding * 2 + baseline

    x = max(0, min(int(anchor[0]) - box_w // 2, img_w - box_w))
    y = max(box_h, min(int(anchor[1]), img_h - 2))

    candidate_positions = [(x, y)]
    step = box_h + 4
    for offset in range(1, 6):
        candidate_positions.append((x, max(box_h, y - offset * step)))
        candidate_positions.append((x, min(img_h - 2, y + offset * step)))

    selected = candidate_positions[0]
    for cand_x, cand_y in candidate_positions:
        x1 = cand_x
        y1 = cand_y - box_h
        x2 = cand_x + box_w
        y2 = cand_y
        overlaps = any(
            not (x2 < ox1 or x1 > ox2 or y2 < oy1 or y1 > oy2)
            for ox1, oy1, ox2, oy2 in occupied_boxes
        )
        if not overlaps:
            selected = (cand_x, cand_y)
            break

    x, y = selected
    x1 = x
    y1 = y - box_h
    x2 = x + box_w
    y2 = y
    occupied_boxes.append((x1, y1, x2, y2))

    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), tuple(color_bgr), 1)
    text_org = (x + padding, y - baseline - padding)
    cv2.putText(
        image, text, text_org, font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
    )


def annotate_semantic_frame(
    semantic_vis_bgr: np.ndarray,
    semantic_frame: np.ndarray,
    semantic_entries: Dict[int, Dict[str, Any]],
    min_area_ratio: float,
    max_labels: int,
) -> np.ndarray:
    annotated = semantic_vis_bgr.copy()
    frame_h, frame_w = semantic_frame.shape[:2]
    min_area = max(80, int(frame_h * frame_w * min_area_ratio))

    candidate_components: List[Tuple[int, int, int, int, Dict[str, Any]]] = []
    for semantic_id in np.unique(semantic_frame).tolist():
        mask = (semantic_frame == int(semantic_id)).astype(np.uint8)
        num_labels, label_map, stats, centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=8
        )
        for component_id in range(1, num_labels):
            area = int(stats[component_id, cv2.CC_STAT_AREA])
            if area < min_area:
                continue
            x = int(stats[component_id, cv2.CC_STAT_LEFT])
            y = int(stats[component_id, cv2.CC_STAT_TOP])
            w = int(stats[component_id, cv2.CC_STAT_WIDTH])
            entry = semantic_entries[int(semantic_id)]
            candidate_components.append((area, x, y, x + w, entry))

    candidate_components.sort(reverse=True, key=lambda item: item[0])
    occupied_boxes: List[Tuple[int, int, int, int]] = []
    for area, x1, y1, x2, entry in candidate_components[:max_labels]:
        del area
        anchor_x = (x1 + x2) // 2
        anchor_y = max(18, y1 + 18)
        draw_text_box(
            annotated,
            semantic_label_text(entry),
            (anchor_x, anchor_y),
            entry["color_bgr"],
            occupied_boxes,
        )

    return annotated


def create_video_writer(path: Path, fps: float, size: Tuple[int, int]) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {path}")
    return writer


def output_paths(output_dir: Path, episode_id: str) -> Tuple[Path, Path]:
    rgb_path = output_dir / "rgb" / f"episode_{episode_id}.mp4"
    semantic_path = output_dir / "semantic" / f"episode_{episode_id}.mp4"
    return rgb_path, semantic_path


def scene_regions_path(output_dir: Path, scene_id: str) -> Path:
    scene_key = scene_id.replace("/", "__").replace("\\", "__")
    if scene_key.endswith(".glb"):
        scene_key = scene_key[:-4]
    return output_dir / "scene_regions" / f"{scene_key}.json"


def semantic_annotated_path(output_dir: Path, episode_id: str) -> Path:
    return output_dir / "semantic_annotated" / f"episode_{episode_id}.mp4"


def semantic_meta_path(output_dir: Path, episode_id: str) -> Path:
    return output_dir / "semantic_meta" / f"episode_{episode_id}.json"


def build_instance_entries(
    semantic_entries: Dict[int, Dict[str, Any]]
) -> List[Dict[str, Any]]:
    grouped: Dict[str, Dict[str, Any]] = {}

    for entry in semantic_entries.values():
        if entry["object_index"] is not None:
            group_key = f"instance:{entry['object_index']}"
        else:
            group_key = f"unmapped:{entry['semantic_id']}"

        if group_key not in grouped:
            grouped[group_key] = {
                "object_index": entry["object_index"],
                "instance_id": entry["instance_id"],
                "name": entry["name"],
                "semantic_items": [],
            }

        grouped[group_key]["semantic_items"].append(
            {
                "semantic_id": entry["semantic_id"],
                "color_rgb": entry["color_rgb"],
                "color_bgr": entry["color_bgr"],
                "color_hex": entry["color_hex"],
            }
        )

    instance_entries = list(grouped.values())
    for entry in instance_entries:
        entry["semantic_items"].sort(key=lambda item: item["semantic_id"])

    instance_entries.sort(
        key=lambda item: (
            item["object_index"] is None,
            item["object_index"] if item["object_index"] is not None else 10**12,
            item["semantic_items"][0]["semantic_id"],
        )
    )
    return instance_entries


def write_semantic_mapping_json(
    path: Path,
    episode: dict,
    gt_episode: dict,
    semantic_entries: Dict[int, Dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sorted_entries = sorted(
        semantic_entries.values(),
        key=lambda item: (
            item["object_index"] is None,
            item["object_index"] if item["object_index"] is not None else 10**12,
            item["semantic_id"],
        ),
    )
    payload = {
        "episode_id": str(episode["episode_id"]),
        "scene_id": episode["scene_id"],
        "instruction": episode["instruction"]["instruction_text"],
        "num_actions": len(gt_episode["actions"]),
        "color_space": "rgb",
        "color_rule": (
            "Semantic-id-colored. Each semantic_id has a fixed color, and "
            "large regions are labeled with the mapped category name."
        ),
        "entries": sorted_entries,
        "instances": build_instance_entries(semantic_entries),
    }

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def serialize_aabb(aabb: Any) -> Dict[str, List[float]]:
    return {
        "center": [float(value) for value in aabb.center],
        "sizes": [float(value) for value in aabb.sizes],
    }


def extract_scene_regions_payload(
    sim: habitat_sim.Simulator,
    scene_id: str,
    scene_path: Path,
    episode_ids: Sequence[str],
) -> Dict[str, Any]:
    semantic_scene = sim.semantic_scene
    object_index_by_id: Dict[str, int] = {}
    for object_index, obj in enumerate(semantic_scene.objects):
        if obj is None:
            continue
        object_index_by_id[obj.id] = object_index

    regions = []
    for region in semantic_scene.regions:
        if region is None:
            continue

        region_objects = []
        for obj in region.objects:
            if obj is None:
                continue
            region_objects.append(
                {
                    "object_id": obj.id,
                    "object_index": object_index_by_id.get(obj.id),
                    "name": maybe_call(obj.category.name)
                    if getattr(obj, "category", None) is not None
                    else None,
                    "category_index": maybe_call(obj.category.index)
                    if getattr(obj, "category", None) is not None
                    else None,
                }
            )

        region_objects.sort(
            key=lambda item: (
                item["object_index"] is None,
                item["object_index"] if item["object_index"] is not None else 10**12,
                item["object_id"],
            )
        )

        regions.append(
            {
                "region_id": region.id,
                "level_id": region.level.id if region.level is not None else None,
                "category": maybe_call(region.category.name)
                if region.category is not None
                else None,
                "category_index": maybe_call(region.category.index)
                if region.category is not None
                else None,
                "aabb": serialize_aabb(region.aabb) if region.aabb is not None else None,
                "object_count": len(region_objects),
                "objects": region_objects,
            }
        )

    regions.sort(key=lambda item: item["region_id"])

    return {
        "scene_id": scene_id,
        "scene_path": str(scene_path),
        "episode_ids": list(episode_ids),
        "num_regions": len(regions),
        "num_scene_objects": len(object_index_by_id),
        "regions": regions,
    }


def write_scene_regions_json(
    path: Path,
    sim: habitat_sim.Simulator,
    scene_id: str,
    scene_path: Path,
    episode_ids: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = extract_scene_regions_payload(
        sim=sim,
        scene_id=scene_id,
        scene_path=scene_path,
        episode_ids=episode_ids,
    )
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def iter_episode_observations(
    sim: habitat_sim.Simulator,
    episode: dict,
    gt_episode: dict,
) -> Iterable[Tuple[int, Dict[str, Any]]]:
    set_agent_state(sim, episode["start_position"], episode["start_rotation"])
    for action in gt_episode["actions"]:
        if action == STOP:
            observations = sim.get_sensor_observations()
        else:
            observations = sim.step(action)
        yield int(action), observations


def render_episode(
    sim: habitat_sim.Simulator,
    episode: dict,
    gt_episode: dict,
    rgb_path: Path,
    semantic_path: Path,
    semantic_annotated_path_value: Path,
    fps: float,
    semantic_label_min_area_ratio: float,
    max_semantic_labels_per_frame: int,
) -> Tuple[int, Dict[int, Dict[str, Any]]]:
    set_agent_state(sim, episode["start_position"], episode["start_rotation"])
    initial_observations = sim.get_sensor_observations()
    frame_size = (
        initial_observations["rgb"].shape[1],
        initial_observations["rgb"].shape[0],
    )
    rgb_writer = create_video_writer(rgb_path, fps, frame_size)
    semantic_writer = create_video_writer(semantic_path, fps, frame_size)
    semantic_annotated_writer = create_video_writer(
        semantic_annotated_path_value, fps, frame_size
    )

    frame_count = 0
    semantic_entries: Dict[int, Dict[str, Any]] = {}
    try:
        for action, observations in iter_episode_observations(
            sim=sim,
            episode=episode,
            gt_episode=gt_episode,
        ):
            del action
            rgb_writer.write(rgb_to_bgr(np.asarray(observations["rgb"])))
            semantic_frame = np.asarray(observations["semantic"])
            semantic_vis = colorize_semantic(
                semantic_frame,
                sim.semantic_scene,
                semantic_entries,
            )
            semantic_writer.write(semantic_vis)
            semantic_annotated_writer.write(
                annotate_semantic_frame(
                    semantic_vis,
                    semantic_frame,
                    semantic_entries,
                    min_area_ratio=semantic_label_min_area_ratio,
                    max_labels=max_semantic_labels_per_frame,
                )
            )
            frame_count += 1
    finally:
        rgb_writer.release()
        semantic_writer.release()
        semantic_annotated_writer.release()

    return frame_count, semantic_entries


def group_episode_ids_by_scene(
    episode_ids: Iterable[str], train_episodes: Dict[str, dict]
) -> Dict[str, List[str]]:
    grouped: Dict[str, List[str]] = {}
    for episode_id in episode_ids:
        scene_id = train_episodes[episode_id]["scene_id"]
        grouped.setdefault(scene_id, []).append(episode_id)
    return grouped


def write_manifest(output_dir: Path, entries: List[dict]) -> None:
    manifest_path = output_dir / "manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()

    train_data = load_json(args.train_json)
    gt_data = load_json(args.train_gt_json)
    train_episodes = {
        str(episode["episode_id"]): episode for episode in train_data["episodes"]
    }

    selected_episode_ids = select_episode_ids(
        train_episodes=train_episodes,
        gt_episodes=gt_data,
        requested_ids=args.episode_ids,
        max_episodes=args.max_episodes,
    )
    if not selected_episode_ids:
        raise ValueError("No episodes selected for rendering.")

    scene_groups = group_episode_ids_by_scene(selected_episode_ids, train_episodes)
    total_episodes = len(selected_episode_ids)
    print(
        f"Preparing to render {total_episodes} episodes across "
        f"{len(scene_groups)} scenes."
    )

    manifest_entries: List[dict] = []
    processed = 0

    for scene_index, scene_id in enumerate(sorted(scene_groups.keys()), start=1):
        scene_path = resolve_scene_path(args.scene_root, scene_id)
        scene_region_json_path = scene_regions_path(args.output_dir, scene_id)
        print(
            f"[scene {scene_index}/{len(scene_groups)}] Loading {scene_id} "
            f"from {scene_path}"
        )
        sim = build_simulator(
            scene_path=scene_path,
            width=args.width,
            height=args.height,
            sensor_height=args.sensor_height,
            forward_step_size=args.forward_step_size,
            turn_angle=args.turn_angle,
        )

        try:
            if args.overwrite or not scene_region_json_path.exists():
                write_scene_regions_json(
                    path=scene_region_json_path,
                    sim=sim,
                    scene_id=scene_id,
                    scene_path=scene_path,
                    episode_ids=scene_groups[scene_id],
                )

            for episode_id in scene_groups[scene_id]:
                episode = train_episodes[episode_id]
                gt_episode = gt_data[episode_id]
                rgb_path, semantic_path = output_paths(args.output_dir, episode_id)
                semantic_annotated_video_path = semantic_annotated_path(
                    args.output_dir, episode_id
                )
                semantic_json_path = semantic_meta_path(args.output_dir, episode_id)

                if (
                    not args.overwrite
                    and rgb_path.exists()
                    and semantic_path.exists()
                    and semantic_annotated_video_path.exists()
                    and semantic_json_path.exists()
                ):
                    processed += 1
                    print(
                        f"[{processed}/{total_episodes}] episode {episode_id}: "
                        "skip existing outputs"
                    )
                    manifest_entries.append(
                        {
                            "episode_id": episode_id,
                            "scene_id": scene_id,
                            "instruction": episode["instruction"][
                                "instruction_text"
                            ],
                            "num_actions": len(gt_episode["actions"]),
                            "rgb_video": str(rgb_path),
                            "semantic_video": str(semantic_path),
                            "semantic_annotated_video": str(
                                semantic_annotated_video_path
                            ),
                            "semantic_mapping_json": str(semantic_json_path),
                            "scene_region_json": str(scene_region_json_path),
                            "status": "skipped_existing",
                        }
                    )
                    continue

                frame_count, semantic_entries = render_episode(
                    sim=sim,
                    episode=episode,
                    gt_episode=gt_episode,
                    rgb_path=rgb_path,
                    semantic_path=semantic_path,
                    semantic_annotated_path_value=semantic_annotated_video_path,
                    fps=args.fps,
                    semantic_label_min_area_ratio=args.semantic_label_min_area_ratio,
                    max_semantic_labels_per_frame=args.max_semantic_labels_per_frame,
                )
                write_semantic_mapping_json(
                    path=semantic_json_path,
                    episode=episode,
                    gt_episode=gt_episode,
                    semantic_entries=semantic_entries,
                )
                processed += 1
                print(
                    f"[{processed}/{total_episodes}] episode {episode_id}: "
                    f"{frame_count} frames"
                )

                manifest_entries.append(
                    {
                        "episode_id": episode_id,
                        "scene_id": scene_id,
                        "instruction": episode["instruction"][
                            "instruction_text"
                        ],
                        "num_actions": len(gt_episode["actions"]),
                        "rgb_video": str(rgb_path),
                        "semantic_video": str(semantic_path),
                        "semantic_annotated_video": str(
                            semantic_annotated_video_path
                        ),
                        "semantic_mapping_json": str(semantic_json_path),
                        "scene_region_json": str(scene_region_json_path),
                        "status": "generated",
                    }
                )
        finally:
            sim.close()

    write_manifest(args.output_dir, manifest_entries)
    print(f"Finished. Manifest written to {args.output_dir / 'manifest.jsonl'}")


if __name__ == "__main__":
    main()
