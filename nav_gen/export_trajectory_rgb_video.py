import argparse
import json
import math
import os
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


def _compute_vfov(hfov_deg, width, height):
    return math.degrees(
        2.0 * math.atan(math.tan(math.radians(hfov_deg / 2.0)) * (height / width))
    )


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
    targets = task_config.get("Object", [])
    trial_keys = _sorted_trial_keys(task_config)
    for trial_index, trial_key in enumerate(trial_keys):
        trial = task_config["trial"][trial_key]
        pos_list = trial.get("pos", [])
        yaw_list = trial.get("yaw", [])
        action_list = trial.get("action", [])
        entry_count = min(len(pos_list), len(yaw_list))
        if action_list:
            entry_count = min(entry_count, len(action_list))
        for state_index in range(entry_count):
            entries.append(
                {
                    "trial_index": trial_index,
                    "trial_key": trial_key,
                    "state_index": state_index,
                    "pos": pos_list[state_index],
                    "yaw": yaw_list[state_index],
                    "action": action_list[state_index] if action_list else None,
                    "target": targets[trial_index] if trial_index < len(targets) else None,
                }
            )
    return entries


def _build_timeline(task_config, frame_mode):
    entries = _flatten_entries(task_config)
    if frame_mode == "state":
        return entries
    if frame_mode == "action":
        if entries and entries[0].get("action") == "stop":
            return entries[1:]
        return entries
    raise ValueError(f"Unsupported frame_mode: {frame_mode}")


def _overlay_lines(task_config, entry, frame_index, total_frames):
    trial_total = len(task_config.get("trial", {}))
    lines = [
        f"Frame {frame_index + 1}/{total_frames} | Trial {entry['trial_index'] + 1}/{trial_total}",
    ]
    if entry.get("action"):
        lines.append(f"Action: {entry['action']}")
    if entry.get("target"):
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


def _build_target_semantic_lookup(simulator, task_config):
    target_names = set(task_config.get("Object", []))
    semantic_lookup = {}
    for obj in simulator.sim.semantic_scene.objects:
        try:
            category_name = obj.category.name()
        except Exception:
            continue
        if category_name not in target_names:
            continue
        semantic_id = getattr(obj, "semantic_id", None)
        if semantic_id is None:
            continue
        semantic_lookup[int(semantic_id)] = category_name
    return semantic_lookup


def _extract_target_boxes(semantic_obs, semantic_lookup, min_pixels=25):
    if semantic_obs is None or not semantic_lookup:
        return []

    boxes = []
    unique_ids = np.unique(semantic_obs)
    for semantic_id in unique_ids:
        semantic_id = int(semantic_id)
        if semantic_id not in semantic_lookup:
            continue
        ys, xs = np.where(semantic_obs == semantic_id)
        if xs.size < min_pixels:
            continue
        boxes.append(
            {
                "label": semantic_lookup[semantic_id],
                "semantic_id": semantic_id,
                "bbox": (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())),
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


def _draw_target_boxes(frame, boxes):
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        color = _color_for_label(box["label"])
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
):
    task_json_path = os.path.abspath(task_json_path)
    output_video = os.path.abspath(output_video)
    if not os.path.exists(task_json_path):
        raise FileNotFoundError(f"task.json not found: {task_json_path}")
    if "/nav_gen/" not in task_json_path:
        raise RuntimeError("task_json path must be inside the LH-VLN nav_gen directory.")
    nav_gen_root = task_json_path.split("/nav_gen/")[0] + "/nav_gen"
    project_root = os.path.dirname(nav_gen_root)

    with open(task_json_path, "r", encoding="utf-8") as f:
        task_config = json.load(f)

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

    video_writer = cv2.VideoWriter(
        output_video,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not video_writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_video}")

    simulator = SceneSimulator(args=args, config=task_config)
    timeline = _build_timeline(task_config, frame_mode)
    semantic_lookup = _build_target_semantic_lookup(simulator, task_config) if target_boxes else {}
    frame_count = 0
    labeled_frame_count = 0
    try:
        for frame_index, entry in enumerate(timeline):
            frame, semantic_obs = _capture_frame(simulator, entry["pos"], entry["yaw"])
            boxes = _extract_target_boxes(semantic_obs, semantic_lookup) if target_boxes else []
            if annotate:
                frame = _annotate_frame(
                    frame,
                    _overlay_lines(task_config, entry, frame_index, len(timeline)),
                )
            if boxes:
                frame = _draw_target_boxes(frame, boxes)
                labeled_frame_count += 1
            video_writer.write(frame)
            frame_count += 1
    finally:
        video_writer.release()
        simulator.sim.close()

    metadata = {
        "task_json": task_json_path,
        "output_video": output_video,
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
        "target_box_classes": sorted(set(semantic_lookup.values())),
        "frames_with_target_boxes": labeled_frame_count,
    }
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
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
