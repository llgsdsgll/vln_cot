import argparse
import json
import math
import textwrap
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm, colors

from habitat_base.simulation import SceneSimulator


DEFAULT_SENSOR_HEIGHT = 1.0
DEFAULT_VISIBLE_PIXELS = 25
DEFAULT_METERS_PER_PIXEL = 0.03


def parse_args():
    parser = argparse.ArgumentParser(
        description="Render a top-down view of NavGen goal-viewpoint sampling."
    )
    parser.add_argument("--task-json", required=True, help="path to success/fail trial task.json")
    parser.add_argument(
        "--trajectory-meta",
        default=None,
        help="optional trajectory_rgb.json for extra target metadata",
    )
    parser.add_argument("--output", required=True, help="output PNG path")
    parser.add_argument(
        "--pixel-count-output",
        default=None,
        help="optional output PNG path for the pixel-count-colored top-down figure",
    )
    parser.add_argument(
        "--probe-view-dir",
        default=None,
        help="optional output directory for highlighted probe-view RGB images",
    )
    parser.add_argument(
        "--probe-overview-output",
        default=None,
        help="optional output PNG path for the combined probe-view overview",
    )
    parser.add_argument(
        "--summary-output",
        default=None,
        help="optional JSON summary path; defaults next to the output PNG",
    )
    parser.add_argument(
        "--render-sensor-height",
        type=float,
        default=DEFAULT_SENSOR_HEIGHT,
        help="camera / semantic sensor height in meters",
    )
    parser.add_argument(
        "--success-visible-pixels",
        type=int,
        default=DEFAULT_VISIBLE_PIXELS,
        help="minimum semantic pixels required for a sampled viewpoint",
    )
    parser.add_argument(
        "--allow-occluded-goal-fallback",
        action="store_true",
        help="allow snapped object-center fallback when no visible viewpoint exists",
    )
    parser.add_argument(
        "--meters-per-pixel",
        type=float,
        default=DEFAULT_METERS_PER_PIXEL,
        help="top-down map resolution",
    )
    parser.add_argument(
        "--sim-gpu-device",
        type=int,
        default=0,
        help="Habitat-Sim GPU device id",
    )
    parser.add_argument(
        "--target-index",
        type=int,
        default=None,
        help="optional target index to visualize; useful for config-only failed tasks",
    )
    return parser.parse_args()


def project_root():
    return Path(__file__).resolve().parents[1]


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def maybe_normalize_task_config(task_config):
    objects = task_config.get("Object", [])
    if not objects or not isinstance(objects[0], list):
        return task_config

    config = json.loads(json.dumps(task_config))
    obj = []
    region_id = []
    for item in config["Subtask list"]:
        if "Move_to" not in item:
            continue
        obj_id = item[9:-2].split("_")
        obj.append(obj_id[0])
        region_id.append(obj_id[1])

    rooms = []
    for room in config["Object"]:
        if len(room) > 1 and ": " in room[1]:
            rooms.append(room[1].split(": ", 1)[1])
        else:
            rooms.append(str(room[1]))

    config["Object"] = obj
    config["Region Name"] = rooms
    config["Region"] = region_id
    return config


def build_sim_args(cli_args):
    root = project_root()
    tmp_root = root / "nav_gen" / "_goal_viewpoint_viz_tmp"
    (tmp_root / "task").mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        scene=str(root / "data" / "hm3d") + "/",
        scene_dataset=str(root / "data" / "hm3d" / "hm3d_annotated_basis.scene_dataset_config.json"),
        sim_gpu_device=cli_args.sim_gpu_device,
        task_path=str(tmp_root / "task") + "/",
        front_rgb_only=True,
        front_semantic_sensor=True,
        render_width=640,
        render_height=480,
        render_sensor_height=cli_args.render_sensor_height,
        sensor_hfov=86.0,
        max_step=500,
        success_dis=1.0,
        success_visible_pixels=max(1, int(cli_args.success_visible_pixels)),
        allow_stop_without_visibility=False,
        allow_occluded_goal_fallback=cli_args.allow_occluded_goal_fallback,
        success_view_radius_min=0.75,
        success_view_radius_max=2.5,
        success_view_radius_step=0.5,
        success_view_angle_step=30.0,
    )


def sorted_trial_keys(task_config):
    return sorted(task_config["trial"].keys(), key=lambda key: int(key.split("_")[-1]))


def trial_index_from_key(trial_key):
    return int(str(trial_key).split("_")[-1])


def ensure_trial_data(task_config, simulator, cli_args):
    requested_index = cli_args.target_index

    if task_config.get("trial"):
        if requested_index is None:
            return task_config, "trial_data"

        trial_key = f"trial_{requested_index}"
        if trial_key not in task_config["trial"]:
            raise ValueError(
                f"requested --target-index={requested_index}, but {trial_key} is not present"
            )
        config = json.loads(json.dumps(task_config))
        config["trial"] = {trial_key: config["trial"][trial_key]}
        return config, "trial_data_selected_target"

    config = json.loads(json.dumps(task_config))
    object_count = len(config.get("Object", []))
    target_index = 0 if requested_index is None else int(requested_index)
    if target_index < 0 or target_index >= object_count:
        raise ValueError(
            f"requested --target-index={target_index}, but valid range is [0, {object_count - 1}]"
        )
    start_pos, start_yaw = simulator.return_state()
    config["trial"] = {
        f"trial_{target_index}": {
            "pos": [np.array(start_pos, dtype=np.float32).tolist()],
            "yaw": [float(start_yaw)],
            "action": ["stop"],
        }
    }
    return config, "config_only_selected_target"


def set_pose(simulator, pos, yaw):
    simulator._set_agent_pose_direct(np.array(pos, dtype=np.float32), float(yaw))
    simulator.observations = simulator.sim.get_sensor_observations()


def set_pose_looking_at(simulator, pos, target_point):
    position = np.array(pos, dtype=np.float32)
    target = np.array(target_point, dtype=np.float32)
    target_saved_yaw = simulator._goal_center_to_saved_yaw(position, target)
    set_pose(simulator, position, target_saved_yaw)


def capture_front_rgb(simulator, pos, yaw=None, target_point=None):
    pose = simulator._capture_pose()
    try:
        if target_point is not None:
            set_pose_looking_at(simulator, pos, target_point)
        elif yaw is not None:
            set_pose(simulator, pos, yaw)
        else:
            raise ValueError("capture_front_rgb requires either yaw or target_point")
        observations = simulator.sim.get_sensor_observations()
        return np.array(observations["color_sensor_f"][..., :3], copy=True)
    finally:
        simulator._restore_pose(pose)


def selected_target_lookup(trajectory_meta):
    if not trajectory_meta:
        return {}
    lookup = {}
    for item in trajectory_meta.get("resolved_target_boxes", []):
        trial_key = item.get("trial_key")
        if trial_key:
            lookup[trial_key] = item
    return lookup


def world_to_grid(bounds, meters_per_pixel, point):
    x = (float(point[0]) - float(bounds[0][0])) / meters_per_pixel
    y = (float(point[2]) - float(bounds[0][2])) / meters_per_pixel
    return np.array([x, y], dtype=np.float32)


def topdown_map(pathfinder, height, meters_per_pixel):
    raw_map = pathfinder.get_topdown_view(meters_per_pixel, height)
    image = np.zeros((raw_map.shape[0], raw_map.shape[1], 3), dtype=np.uint8)
    image[:] = np.array([28, 31, 36], dtype=np.uint8)
    image[raw_map] = np.array([244, 243, 239], dtype=np.uint8)
    return image


def viewpoint_sort_key(viewpoint):
    center = viewpoint["center"]
    position = viewpoint["position"]
    return (
        -int(viewpoint["pixel_count"]),
        float(
            math.dist(
                [float(position[0]), float(position[2])],
                [float(center[0]), float(center[2])],
            )
        ),
    )


def sample_candidate_probe_points(simulator, candidate):
    semantic_id = candidate.get("semantic_id")
    goal_center = candidate.get("center")
    if semantic_id is None or goal_center is None:
        return []

    radius_min = float(getattr(simulator.args, "success_view_radius_min", 0.75))
    radius_max = float(getattr(simulator.args, "success_view_radius_max", 2.5))
    radius_step = float(getattr(simulator.args, "success_view_radius_step", 0.5))
    angle_step = float(getattr(simulator.args, "success_view_angle_step", 30.0))
    if radius_step <= 0:
        radius_step = 0.5
    if angle_step <= 0:
        angle_step = 30.0

    probe_points = []
    seen_keys = set()
    radius = radius_min
    while radius <= radius_max + 1e-6:
        angle_deg = 0.0
        while angle_deg < 360.0:
            rad = math.radians(angle_deg)
            raw_point = np.array(
                [
                    float(goal_center[0] + radius * math.sin(rad)),
                    float(goal_center[1]),
                    float(goal_center[2] + radius * math.cos(rad)),
                ],
                dtype=np.float32,
            )
            nav_point = simulator.pathfinder.snap_point(raw_point)
            if simulator._is_valid_nav_point(nav_point):
                nav_point = np.array(nav_point, dtype=np.float32)
                key = tuple(np.round(nav_point, 3).tolist())
                if key not in seen_keys:
                    seen_keys.add(key)
                    saved_yaw = simulator._goal_center_to_saved_yaw(nav_point, goal_center)
                    pixel_count = simulator._semantic_pixels_for_pose(
                        nav_point,
                        saved_yaw,
                        semantic_id,
                    )
                    probe_points.append(
                        {
                            "position": nav_point,
                            "yaw": saved_yaw,
                            "pixel_count": int(pixel_count),
                            "semantic_id": semantic_id,
                            "object_id": candidate.get("object_id"),
                            "center": np.array(goal_center, dtype=np.float32),
                            "radius_m": float(radius),
                            "angle_deg": float(angle_deg),
                        }
                    )
            angle_deg += angle_step
        radius += radius_step

    probe_points.sort(key=viewpoint_sort_key)
    return probe_points


def geodesic_distance_between(simulator, start_pos, end_pos):
    try:
        from habitat_sim import nav

        path = nav.ShortestPath()
    except Exception:
        return None

    path.requested_start = np.array(start_pos, dtype=np.float32)
    path.requested_end = np.array(end_pos, dtype=np.float32)
    if not simulator.pathfinder.find_path(path):
        return None
    return float(path.geodesic_distance)


def collect_trial_data(simulator, task_config, trajectory_meta, cli_args):
    selected_lookup = selected_target_lookup(trajectory_meta)
    trial_rows = []
    required_pixels = max(1, int(cli_args.success_visible_pixels))

    for trial_key in sorted_trial_keys(task_config):
        trial_index = trial_index_from_key(trial_key)
        trial = task_config["trial"][trial_key]
        set_pose(simulator, trial["pos"][0], trial["yaw"][0])
        trial_start_pos = np.array(trial["pos"][0], dtype=np.float32)

        candidates = simulator.get_target_candidates(trial_index)
        candidate_rows = []
        visible_viewpoints = []
        reachable_visible_viewpoints = []

        for candidate in candidates:
            probe_points = sample_candidate_probe_points(simulator, candidate)
            for point in probe_points:
                path_distance = geodesic_distance_between(
                    simulator,
                    trial_start_pos,
                    point["position"],
                )
                point["path_reachable"] = path_distance is not None
                point["path_distance"] = path_distance
            strong_viewpoints = [
                point for point in probe_points if int(point["pixel_count"]) >= required_pixels
            ]
            relaxed_viewpoints = [point for point in probe_points if int(point["pixel_count"]) >= 1]
            displayed_viewpoints = strong_viewpoints
            sampling_mode = "visible_viewpoint"
            if not displayed_viewpoints and required_pixels > 1:
                displayed_viewpoints = relaxed_viewpoints
                if displayed_viewpoints:
                    sampling_mode = "visible_viewpoint_relaxed"

            visible_viewpoints.extend(displayed_viewpoints)
            reachable_visible_viewpoints.extend(
                point for point in displayed_viewpoints if point.get("path_reachable")
            )
            candidate_rows.append(
                {
                    "center": np.array(candidate["center"], dtype=np.float32),
                    "semantic_id": candidate.get("semantic_id"),
                    "object_id": candidate.get("object_id"),
                    "probe_points": probe_points,
                    "strong_viewpoints": strong_viewpoints,
                    "relaxed_viewpoints": relaxed_viewpoints,
                    "displayed_viewpoints": displayed_viewpoints,
                    "reachable_displayed_viewpoints": [
                        point for point in displayed_viewpoints if point.get("path_reachable")
                    ],
                    "sampling_mode": sampling_mode,
                }
            )

        geo_dis, goal_coord, goal_meta = simulator.get_goal_info(trial_index)
        trial_rows.append(
            {
                "trial_index": trial_index,
                "trial_key": trial_key,
                "target_name": task_config["Object"][trial_index],
                "region_id": task_config["Region"][trial_index],
                "region_name": task_config.get("Region Name", [None] * len(task_config["Object"]))[trial_index],
                "positions": [np.array(pos, dtype=np.float32) for pos in trial["pos"]],
                "yaws": [float(yaw) for yaw in trial["yaw"]],
                "actions": list(trial["action"]),
                "candidates": candidate_rows,
                "visible_viewpoints": visible_viewpoints,
                "reachable_visible_viewpoints": reachable_visible_viewpoints,
                "goal_coord": None if goal_coord is None else np.array(goal_coord, dtype=np.float32),
                "goal_meta": goal_meta,
                "goal_distance": None if geo_dis is None or math.isinf(geo_dis) else float(geo_dis),
                "selected_target_meta": selected_lookup.get(trial_key),
            }
        )
        trial_rows[-1]["highlight_probe"] = pick_highlight_probe(trial_rows[-1])

    return trial_rows


def unique_label_handles(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle
    ax.legend(
        unique.values(),
        unique.keys(),
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        fontsize=8,
        framealpha=0.95,
    )


def annotate_steps(ax, grid_points, color):
    seen = defaultdict(int)
    for step_index, point in enumerate(grid_points):
        key = tuple(np.round(point, 1).tolist())
        repeat_index = seen[key]
        seen[key] += 1
        angle = repeat_index * (math.pi / 3.0)
        radius = 3.0 + repeat_index * 2.0
        dx = math.cos(angle) * radius
        dy = math.sin(angle) * radius
        ax.text(
            float(point[0] + dx),
            float(point[1] + dy),
            str(step_index),
            fontsize=6,
            color=color,
            ha="center",
            va="center",
        )


def choose_target_center(trial_row):
    goal_meta = trial_row.get("goal_meta") or {}
    if goal_meta.get("center") is not None:
        return np.array(goal_meta["center"], dtype=np.float32)
    selected = trial_row.get("selected_target_meta") or {}
    if selected.get("selected_center") is not None:
        return np.array(selected["selected_center"], dtype=np.float32)
    if trial_row["candidates"]:
        return np.array(trial_row["candidates"][0]["center"], dtype=np.float32)
    return None


def pick_reference_object_id(trial_row):
    goal_meta = trial_row.get("goal_meta") or {}
    if goal_meta.get("object_id"):
        return goal_meta["object_id"]
    selected = trial_row.get("selected_target_meta") or {}
    if selected.get("selected_object_id"):
        return selected["selected_object_id"]
    return None


def pick_highlight_probe(trial_row):
    reference_object_id = pick_reference_object_id(trial_row)

    def collect_probes(filter_object_id):
        probe_rows = []
        for candidate_index, candidate in enumerate(trial_row["candidates"]):
            if filter_object_id and candidate.get("object_id") != filter_object_id:
                continue
            for probe_index, probe in enumerate(candidate["probe_points"]):
                probe_rows.append(
                    {
                        "candidate_index": candidate_index,
                        "probe_index": probe_index,
                        "probe": probe,
                    }
                )
        return probe_rows

    probe_rows = collect_probes(reference_object_id)
    if not probe_rows:
        probe_rows = collect_probes(None)
    if not probe_rows:
        return None

    best_row = sorted(probe_rows, key=lambda item: viewpoint_sort_key(item["probe"]))[0]
    probe = best_row["probe"]
    return {
        "label": f"P{trial_row['trial_index']}",
        "candidate_index": best_row["candidate_index"],
        "probe_index": best_row["probe_index"],
        "position": np.array(probe["position"], dtype=np.float32),
        "yaw": float(probe["yaw"]),
        "pixel_count": int(probe["pixel_count"]),
        "semantic_id": probe.get("semantic_id"),
        "object_id": probe.get("object_id"),
        "center": np.array(probe["center"], dtype=np.float32),
        "radius_m": float(probe["radius_m"]),
        "angle_deg": float(probe["angle_deg"]),
        "reference_object_id": reference_object_id,
        "path_reachable": bool(probe.get("path_reachable")),
        "path_distance": probe.get("path_distance"),
    }


def to_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {key: to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def build_summary(task_config, trial_rows, cli_args, diagnostic_mode):
    return {
        "task_instruction": task_config["Task instruction"],
        "scene": task_config["Scene"],
        "robot": task_config["Robot"],
        "diagnostic_mode": diagnostic_mode,
        "sampling_params": {
            "success_visible_pixels": int(cli_args.success_visible_pixels),
            "allow_occluded_goal_fallback": bool(cli_args.allow_occluded_goal_fallback),
            "target_index": cli_args.target_index,
            "success_view_radius_min": 0.75,
            "success_view_radius_max": 2.5,
            "success_view_radius_step": 0.5,
            "success_view_angle_step": 30.0,
            "render_sensor_height": float(cli_args.render_sensor_height),
            "meters_per_pixel": float(cli_args.meters_per_pixel),
        },
        "trials": [
            {
                "trial_key": row["trial_key"],
                "target_name": row["target_name"],
                "region_id": row["region_id"],
                "region_name": row["region_name"],
                "step_count": len(row["positions"]),
                "candidate_count": len(row["candidates"]),
                "visible_viewpoint_count": len(row["visible_viewpoints"]),
                "path_reachable_visible_viewpoint_count": len(row["reachable_visible_viewpoints"]),
                "goal_coord": to_jsonable(row["goal_coord"]),
                "goal_meta": to_jsonable(row["goal_meta"]),
                "goal_distance": row["goal_distance"],
                "highlight_probe": to_jsonable(row["highlight_probe"]),
                "selected_target_meta": to_jsonable(row["selected_target_meta"]),
                "candidates": [
                    {
                        "center": to_jsonable(candidate["center"]),
                        "semantic_id": candidate["semantic_id"],
                        "object_id": candidate["object_id"],
                        "probe_point_count": len(candidate["probe_points"]),
                        "strong_viewpoint_count": len(candidate["strong_viewpoints"]),
                        "relaxed_viewpoint_count": len(candidate["relaxed_viewpoints"]),
                        "path_reachable_viewpoint_count": len(candidate["reachable_displayed_viewpoints"]),
                        "sampling_mode": candidate["sampling_mode"],
                    }
                    for candidate in row["candidates"]
                ],
            }
            for row in trial_rows
        ],
    }


def render_figure(task_config, trial_rows, simulator, cli_args, output_path):
    colors = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
    bounds = simulator.pathfinder.get_bounds()

    all_robot_positions = [pos for row in trial_rows for pos in row["positions"]]
    floor_height = float(np.median([float(pos[1]) for pos in all_robot_positions]))
    map_image = topdown_map(simulator.pathfinder, floor_height, cli_args.meters_per_pixel)

    fig, ax = plt.subplots(figsize=(15, 12), dpi=200)
    ax.imshow(map_image)
    ax.set_axis_off()

    try:
        from matplotlib.collections import LineCollection
        verts = np.array(simulator.pathfinder.build_navmesh_vertices(), dtype=np.float32)
        if verts.size > 0:
            grid_tris = np.array(
                [[world_to_grid(bounds, cli_args.meters_per_pixel, v) for v in tri]
                 for tri in verts.reshape(-1, 3, 3)],
                dtype=np.float32,
            )
            edges = np.concatenate(
                [grid_tris[:, [0, 1]], grid_tris[:, [1, 2]], grid_tris[:, [2, 0]]],
                axis=0,
            )
            ax.add_collection(LineCollection(edges, linewidths=0.3, colors="#aaaaaa", alpha=0.3, zorder=1))
    except Exception:
        pass

    all_grid_points = []
    for trial_index, row in enumerate(trial_rows):
        color = colors[trial_index % len(colors)]
        target_name = row["target_name"]

        path_points = np.array(
            [world_to_grid(bounds, cli_args.meters_per_pixel, pos) for pos in row["positions"]],
            dtype=np.float32,
        )
        all_grid_points.extend(path_points.tolist())

        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            color=color,
            linewidth=2.2,
            alpha=0.95,
            label=f"{row['trial_key']} path ({target_name})",
        )
        ax.scatter(
            path_points[:, 0],
            path_points[:, 1],
            s=18,
            color=color,
            alpha=0.85,
            zorder=4,
            label=f"{row['trial_key']} steps",
        )
        annotate_steps(ax, path_points, color)
        ax.scatter(
            path_points[0, 0],
            path_points[0, 1],
            s=90,
            marker="^",
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label=f"{row['trial_key']} start",
        )
        ax.scatter(
            path_points[-1, 0],
            path_points[-1, 1],
            s=90,
            marker="X",
            color=color,
            edgecolors="black",
            linewidths=0.8,
            zorder=6,
            label=f"{row['trial_key']} stop",
        )

        for candidate_index, candidate in enumerate(row["candidates"]):
            center_grid = world_to_grid(bounds, cli_args.meters_per_pixel, candidate["center"])
            all_grid_points.append(center_grid.tolist())
            ax.scatter(
                center_grid[0],
                center_grid[1],
                s=90,
                marker="s",
                facecolors="none",
                edgecolors=color,
                linewidths=1.2,
                zorder=6,
                label=f"{row['trial_key']} candidate centers" if candidate_index == 0 else None,
            )

            probe_points = candidate["probe_points"]
            if probe_points:
                probe_grid = np.array(
                    [
                        world_to_grid(bounds, cli_args.meters_per_pixel, point["position"])
                        for point in probe_points
                    ],
                    dtype=np.float32,
                )
                all_grid_points.extend(probe_grid.tolist())
                ax.scatter(
                    probe_grid[:, 0],
                    probe_grid[:, 1],
                    s=12,
                    facecolors="none",
                    edgecolors=color,
                    alpha=0.3,
                    linewidths=0.6,
                    zorder=2,
                    label=f"{row['trial_key']} tested samples" if candidate_index == 0 else None,
                )
                for pt, gp in zip(probe_points, probe_grid):
                    theta = math.radians(float(pt["yaw"]) - 180.0)
                    arrow_len = 8.0
                    ax.annotate(
                        "",
                        xy=(float(gp[0]) + arrow_len * math.sin(theta),
                            float(gp[1]) - arrow_len * math.cos(theta)),
                        xytext=(float(gp[0]), float(gp[1])),
                        arrowprops={"arrowstyle": "->", "color": color,
                                    "lw": 0.8, "alpha": 0.5},
                        zorder=3,
                    )

            displayed_viewpoints = candidate["displayed_viewpoints"]
            if displayed_viewpoints:
                viewpoint_grid = np.array(
                    [
                        world_to_grid(bounds, cli_args.meters_per_pixel, vp["position"])
                        for vp in displayed_viewpoints
                    ],
                    dtype=np.float32,
                )
                all_grid_points.extend(viewpoint_grid.tolist())
                ax.scatter(
                    viewpoint_grid[:, 0],
                    viewpoint_grid[:, 1],
                    s=16,
                    color=color,
                    alpha=0.24,
                    linewidths=0,
                    zorder=3,
                    label=f"{row['trial_key']} sampled viewpoints" if candidate_index == 0 else None,
                )
                for vp, gp in zip(displayed_viewpoints, viewpoint_grid):
                    theta = math.radians(float(vp["yaw"]) - 180.0)
                    ax.quiver(
                        float(gp[0]), float(gp[1]),
                        math.sin(theta), math.cos(theta),
                        angles="xy", scale=25, scale_units="xy",
                        width=0.003, color=color, alpha=0.7, zorder=4,
                    )

        target_center = choose_target_center(row)
        if target_center is not None:
            target_grid = world_to_grid(bounds, cli_args.meters_per_pixel, target_center)
            all_grid_points.append(target_grid.tolist())
            ax.scatter(
                target_grid[0],
                target_grid[1],
                s=150,
                marker="*",
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
                label=f"{row['trial_key']} target center",
            )
            ax.text(
                float(target_grid[0] + 6),
                float(target_grid[1] - 6),
                f"{row['trial_key']} target",
                fontsize=8,
                color=color,
                weight="bold",
            )

        if row["goal_coord"] is not None:
            goal_grid = world_to_grid(bounds, cli_args.meters_per_pixel, row["goal_coord"])
            all_grid_points.append(goal_grid.tolist())
            ax.scatter(
                goal_grid[0],
                goal_grid[1],
                s=120,
                marker="D",
                color=color,
                edgecolors="black",
                linewidths=0.8,
                zorder=7,
                label=f"{row['trial_key']} chosen goal",
            )
            if target_center is not None:
                target_grid = world_to_grid(bounds, cli_args.meters_per_pixel, target_center)
                ax.plot(
                    [goal_grid[0], target_grid[0]],
                    [goal_grid[1], target_grid[1]],
                    linestyle="--",
                    linewidth=1.0,
                    color=color,
                    alpha=0.75,
                    zorder=5,
                )

        highlight = row.get("highlight_probe")
        if highlight is not None:
            probe_grid = world_to_grid(bounds, cli_args.meters_per_pixel, highlight["position"])
            all_grid_points.append(probe_grid.tolist())
            reachability_text = "" if highlight.get("path_reachable") else ", no path"
            ax.scatter(
                probe_grid[0],
                probe_grid[1],
                s=180,
                marker="P",
                color=color,
                edgecolors="black",
                linewidths=1.0,
                zorder=9,
                label=f"{highlight['label']} probe used for first-person view",
            )
            ax.annotate(
                f"{highlight['label']} ({highlight['pixel_count']} px{reachability_text})",
                xy=(float(probe_grid[0]), float(probe_grid[1])),
                xytext=(float(probe_grid[0] + 16), float(probe_grid[1] - 10)),
                fontsize=8,
                color=color,
                weight="bold",
                arrowprops={"arrowstyle": "->", "color": color, "lw": 1.0},
            )

    points = np.array(all_grid_points, dtype=np.float32)
    padding = 45.0
    ax.set_xlim(float(points[:, 0].min() - padding), float(points[:, 0].max() + padding))
    ax.set_ylim(float(points[:, 1].max() + padding), float(points[:, 1].min() - padding))

    title = textwrap.fill(task_config["Task instruction"], width=70)
    ax.set_title(
        title
        + "\n"
        + (
            f"Scene {task_config['Scene']} | "
            f"sampling radii 0.75-2.5m step 0.5m | "
            f"angle step 30 deg | "
            f"visible threshold {cli_args.success_visible_pixels} px"
        ),
        fontsize=11,
        pad=16,
    )

    unique_label_handles(ax)
    fig.tight_layout(rect=(0.0, 0.0, 0.84, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def render_pixel_count_figure(task_config, trial_rows, simulator, cli_args, output_path):
    trial_colors = ["#1f77b4", "#d95f02", "#2ca02c", "#9467bd"]
    bounds = simulator.pathfinder.get_bounds()

    all_robot_positions = [pos for row in trial_rows for pos in row["positions"]]
    floor_height = float(np.median([float(pos[1]) for pos in all_robot_positions]))
    map_image = topdown_map(simulator.pathfinder, floor_height, cli_args.meters_per_pixel)

    global_max_pixel_count = 0
    for row in trial_rows:
        for candidate in row["candidates"]:
            for probe in candidate["probe_points"]:
                global_max_pixel_count = max(global_max_pixel_count, int(probe["pixel_count"]))

    norm = colors.Normalize(vmin=0, vmax=max(1, global_max_pixel_count))
    cmap = matplotlib.colormaps["viridis"]

    fig, ax = plt.subplots(figsize=(15, 12), dpi=200)
    ax.imshow(map_image)
    ax.set_axis_off()

    all_grid_points = []
    for trial_index, row in enumerate(trial_rows):
        trial_color = trial_colors[trial_index % len(trial_colors)]
        path_points = np.array(
            [world_to_grid(bounds, cli_args.meters_per_pixel, pos) for pos in row["positions"]],
            dtype=np.float32,
        )
        all_grid_points.extend(path_points.tolist())

        ax.plot(
            path_points[:, 0],
            path_points[:, 1],
            color=trial_color,
            linewidth=2.2,
            alpha=0.92,
            label=f"{row['trial_key']} path ({row['target_name']})",
        )
        ax.scatter(
            path_points[0, 0],
            path_points[0, 1],
            s=88,
            marker="^",
            color=trial_color,
            edgecolors="black",
            linewidths=0.8,
            zorder=7,
            label=f"{row['trial_key']} start",
        )
        ax.scatter(
            path_points[-1, 0],
            path_points[-1, 1],
            s=88,
            marker="X",
            color=trial_color,
            edgecolors="black",
            linewidths=0.8,
            zorder=7,
            label=f"{row['trial_key']} stop",
        )

        for candidate_index, candidate in enumerate(row["candidates"]):
            center_grid = world_to_grid(bounds, cli_args.meters_per_pixel, candidate["center"])
            all_grid_points.append(center_grid.tolist())
            ax.scatter(
                center_grid[0],
                center_grid[1],
                s=90,
                marker="s",
                facecolors="none",
                edgecolors=trial_color,
                linewidths=1.1,
                zorder=7,
                label=f"{row['trial_key']} candidate centers" if candidate_index == 0 else None,
            )

            probe_points = candidate["probe_points"]
            if probe_points:
                probe_grid = np.array(
                    [
                        world_to_grid(bounds, cli_args.meters_per_pixel, point["position"])
                        for point in probe_points
                    ],
                    dtype=np.float32,
                )
                probe_pixel_counts = np.array(
                    [int(point["pixel_count"]) for point in probe_points],
                    dtype=np.float32,
                )
                all_grid_points.extend(probe_grid.tolist())
                ax.scatter(
                    probe_grid[:, 0],
                    probe_grid[:, 1],
                    c=probe_pixel_counts,
                    cmap=cmap,
                    norm=norm,
                    s=26,
                    alpha=0.95,
                    edgecolors="white",
                    linewidths=0.25,
                    zorder=3,
                )

        target_center = choose_target_center(row)
        if target_center is not None:
            target_grid = world_to_grid(bounds, cli_args.meters_per_pixel, target_center)
            all_grid_points.append(target_grid.tolist())
            ax.scatter(
                target_grid[0],
                target_grid[1],
                s=150,
                marker="*",
                color=trial_color,
                edgecolors="black",
                linewidths=0.8,
                zorder=8,
                label=f"{row['trial_key']} target center",
            )
            ax.text(
                float(target_grid[0] + 6),
                float(target_grid[1] - 6),
                f"{row['trial_key']} target",
                fontsize=8,
                color=trial_color,
                weight="bold",
            )

        if row["goal_coord"] is not None:
            goal_grid = world_to_grid(bounds, cli_args.meters_per_pixel, row["goal_coord"])
            all_grid_points.append(goal_grid.tolist())
            ax.scatter(
                goal_grid[0],
                goal_grid[1],
                s=120,
                marker="D",
                color=trial_color,
                edgecolors="black",
                linewidths=0.8,
                zorder=8,
                label=f"{row['trial_key']} chosen goal",
            )
            if target_center is not None:
                target_grid = world_to_grid(bounds, cli_args.meters_per_pixel, target_center)
                ax.plot(
                    [goal_grid[0], target_grid[0]],
                    [goal_grid[1], target_grid[1]],
                    linestyle="--",
                    linewidth=1.0,
                    color=trial_color,
                    alpha=0.75,
                    zorder=5,
                )

        highlight = row.get("highlight_probe")
        if highlight is not None:
            probe_grid = world_to_grid(bounds, cli_args.meters_per_pixel, highlight["position"])
            all_grid_points.append(probe_grid.tolist())
            reachability_text = "" if highlight.get("path_reachable") else ", no path"
            ax.scatter(
                probe_grid[0],
                probe_grid[1],
                s=180,
                marker="P",
                color=cmap(norm(highlight["pixel_count"])),
                edgecolors="black",
                linewidths=1.0,
                zorder=9,
                label=f"{highlight['label']} best sampled probe",
            )
            ax.annotate(
                f"{highlight['label']} ({highlight['pixel_count']} px{reachability_text})",
                xy=(float(probe_grid[0]), float(probe_grid[1])),
                xytext=(float(probe_grid[0] + 16), float(probe_grid[1] - 10)),
                fontsize=8,
                color=trial_color,
                weight="bold",
                arrowprops={"arrowstyle": "->", "color": trial_color, "lw": 1.0},
            )

    points = np.array(all_grid_points, dtype=np.float32)
    padding = 45.0
    ax.set_xlim(float(points[:, 0].min() - padding), float(points[:, 0].max() + padding))
    ax.set_ylim(float(points[:, 1].max() + padding), float(points[:, 1].min() - padding))

    max_text = (
        "all sampled probe points are 0 px"
        if global_max_pixel_count == 0
        else f"global max sampled visibility = {global_max_pixel_count} px"
    )
    ax.set_title(
        textwrap.fill(task_config["Task instruction"], width=70)
        + "\n"
        + (
            f"Scene {task_config['Scene']} | probe points colored by target semantic pixel_count | "
            f"{max_text}"
        ),
        fontsize=11,
        pad=16,
    )

    unique_label_handles(ax)
    scalar_mappable = cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])
    colorbar = fig.colorbar(scalar_mappable, ax=ax, fraction=0.035, pad=0.02)
    colorbar.set_label("target semantic pixel_count at sampled view", fontsize=9)
    fig.tight_layout(rect=(0.0, 0.0, 0.82, 1.0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def save_probe_views(simulator, trial_rows, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    view_records = []

    for row in trial_rows:
        highlight = row.get("highlight_probe")
        if highlight is None:
            continue

        rgb = capture_front_rgb(
            simulator,
            highlight["position"],
            yaw=highlight["yaw"],
        )
        output_path = output_dir / f"{row['trial_key']}_{highlight['label']}_best_probe_rgb.png"
        visible_text = (
            f"{highlight['pixel_count']} px visible"
            if highlight["pixel_count"] > 0
            else "0 px visible (target not seen from sampled probe)"
        )
        path_text = (
            "path reachable"
            if highlight.get("path_reachable")
            else "no path from trial start"
        )

        fig, ax = plt.subplots(figsize=(8.5, 5.2), dpi=180)
        ax.imshow(rgb)
        ax.set_axis_off()
        ax.set_title(
            "\n".join(
                [
                    f"{highlight['label']} | {row['trial_key']} | target={row['target_name']}",
                    (
                    f"object_id={highlight['object_id']} | "
                    f"radius={highlight['radius_m']:.2f}m | "
                    f"angle={highlight['angle_deg']:.0f} deg | "
                    f"render=look_at(target_center) | {visible_text} | {path_text}"
                ),
            ]
        ),
            fontsize=10,
            pad=10,
        )
        fig.tight_layout()
        fig.savefig(output_path, bbox_inches="tight")
        plt.close(fig)

        view_records.append(
            {
                "trial_key": row["trial_key"],
                "trial_index": row["trial_index"],
                "label": highlight["label"],
                "target_name": row["target_name"],
                "pixel_count": highlight["pixel_count"],
                "object_id": highlight["object_id"],
                "radius_m": highlight["radius_m"],
                "angle_deg": highlight["angle_deg"],
                "render_mode": "look_at_target_center",
                "path_reachable": bool(highlight.get("path_reachable")),
                "path_distance": highlight.get("path_distance"),
                "output": str(output_path),
                "image": rgb,
            }
        )

    return view_records


def render_probe_overview(view_records, output_path):
    if not view_records:
        return

    fig, axes = plt.subplots(
        1,
        len(view_records),
        figsize=(7.0 * len(view_records), 5.6),
        dpi=180,
    )
    if len(view_records) == 1:
        axes = [axes]

    for ax, record in zip(axes, view_records):
        ax.imshow(record["image"])
        ax.set_axis_off()
        ax.set_title(
            "\n".join(
                [
                    f"{record['label']} | {record['trial_key']} | {record['target_name']}",
                    (
                        f"object_id={record['object_id']} | "
                        f"probe radius={record['radius_m']:.2f}m | "
                        f"pixel_count={record['pixel_count']} px"
                    ),
                ]
            ),
            fontsize=10,
            pad=10,
        )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main():
    cli_args = parse_args()
    task_json_path = Path(cli_args.task_json).resolve()
    trajectory_meta_path = (
        None if not cli_args.trajectory_meta else Path(cli_args.trajectory_meta).resolve()
    )
    output_path = Path(cli_args.output).resolve()
    pixel_count_output_path = (
        Path(cli_args.pixel_count_output).resolve()
        if cli_args.pixel_count_output
        else output_path.with_name(output_path.stem + "_pixel_count" + output_path.suffix)
    )
    probe_view_dir = (
        Path(cli_args.probe_view_dir).resolve()
        if cli_args.probe_view_dir
        else output_path.parent / (output_path.stem + "_probe_views")
    )
    probe_overview_output_path = (
        Path(cli_args.probe_overview_output).resolve()
        if cli_args.probe_overview_output
        else output_path.with_name(output_path.stem + "_probe_overview" + output_path.suffix)
    )
    summary_path = (
        Path(cli_args.summary_output).resolve()
        if cli_args.summary_output
        else output_path.with_suffix(".summary.json")
    )

    task_config = maybe_normalize_task_config(load_json(task_json_path))
    trajectory_meta = load_json(trajectory_meta_path) if trajectory_meta_path else None
    sim_args = build_sim_args(cli_args)
    simulator = SceneSimulator(args=sim_args, config=task_config)

    try:
        task_config, diagnostic_mode = ensure_trial_data(task_config, simulator, cli_args)
        trial_rows = collect_trial_data(simulator, task_config, trajectory_meta, cli_args)
        render_figure(task_config, trial_rows, simulator, cli_args, output_path)
        render_pixel_count_figure(
            task_config,
            trial_rows,
            simulator,
            cli_args,
            pixel_count_output_path,
        )
        view_records = save_probe_views(simulator, trial_rows, probe_view_dir)
        render_probe_overview(view_records, probe_overview_output_path)
        summary = build_summary(task_config, trial_rows, cli_args, diagnostic_mode)
        summary["outputs"] = {
            "topdown": str(output_path),
            "pixel_count_topdown": str(pixel_count_output_path),
            "probe_view_dir": str(probe_view_dir),
            "probe_overview": str(probe_overview_output_path),
            "probe_views": [
                {
                    "trial_key": record["trial_key"],
                    "label": record["label"],
                    "output": record["output"],
                    "pixel_count": record["pixel_count"],
                    "render_mode": record["render_mode"],
                }
                for record in view_records
            ],
        }
        summary_path.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "pixel_count_output": str(pixel_count_output_path),
                    "probe_view_dir": str(probe_view_dir),
                    "probe_overview_output": str(probe_overview_output_path),
                    "summary": str(summary_path),
                },
                ensure_ascii=False,
            )
        )
    finally:
        simulator.sim.close()


if __name__ == "__main__":
    main()
