import argparse
import copy
import json
import math
import os
from pathlib import Path

from dataset_gen import eval_for_one_task
from habitat_base.simulation import SceneSimulator
from run_smoke_test import build_args, normalize_task_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Inspect and rerun a single generated NavGen task."
    )
    parser.add_argument("--config_path", required=True, help="path to task config.json")
    parser.add_argument(
        "--output_root",
        default=None,
        help="NavGen output root; defaults to the smoke/batch root inferred from config_path",
    )
    parser.add_argument("--max_attempts", type=int, default=1, help="rerun attempts")
    parser.add_argument("--max_step", type=int, default=500, help="max nav steps")
    parser.add_argument(
        "--render_sensor_height",
        type=float,
        default=1.0,
        help="camera / semantic sensor height in meters",
    )
    parser.add_argument(
        "--success_dis", type=float, default=1.0, help="success distance threshold"
    )
    parser.add_argument(
        "--success_visible_pixels",
        type=int,
        default=25,
        help="minimum front-view semantic pixels required for visibility success",
    )
    parser.add_argument(
        "--allow_stop_without_visibility",
        action="store_true",
        help="fall back to the old distance-only stop rule",
    )
    parser.add_argument(
        "--allow_occluded_goal_fallback",
        action="store_true",
        help="deprecated compatibility flag; ignored because strict visible-viewpoint mode now marks no-visible-viewpoint targets unreachable",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="only inspect target goal metadata without rerunning the trajectory",
    )
    return parser.parse_args()


def infer_output_root(config_path: Path) -> Path:
    # .../<output_root>/task/<length>/<instruction>/config.json
    return config_path.parents[3]


def to_jsonable(value):
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isinf(value) or math.isnan(value)):
        return str(value)
    return value


def inspect_targets(args, config):
    sim = SceneSimulator(args=args, config=copy.deepcopy(config))
    try:
        print(
            json.dumps(
                {
                    "scene": config["Scene"],
                    "robot": config["Robot"],
                    "instruction": config["Task instruction"],
                    "targets": config["Object"],
                    "regions": config["Region"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

        for index, target_name in enumerate(sim.target):
            obj_target, coord, goal_meta, position, yaw, geo_dis = sim.get_info(index)
            print(
                json.dumps(
                    {
                        "target_index": index,
                        "target_name": target_name,
                        "reported_target": obj_target,
                        "goal_coord": to_jsonable(coord),
                        "goal_position": to_jsonable(position),
                        "goal_yaw": yaw,
                        "geo_dis": geo_dis,
                        "goal_meta": to_jsonable(goal_meta),
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )
    finally:
        sim.sim.close()


def count_trials(task_dir: Path, name: str) -> int:
    trial_dir = task_dir / name
    if not trial_dir.is_dir():
        return 0
    return len([p for p in trial_dir.iterdir() if p.is_dir()])


def main():
    cli_args = parse_args()
    config_path = Path(cli_args.config_path).resolve()
    output_root = (
        Path(cli_args.output_root).resolve()
        if cli_args.output_root
        else infer_output_root(config_path)
    )

    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    config = normalize_task_config(raw_config)

    args = build_args(str(output_root), cli_args.max_step, scene_id=None)
    args.render_sensor_height = cli_args.render_sensor_height
    args.success_dis = cli_args.success_dis
    args.success_visible_pixels = cli_args.success_visible_pixels
    args.allow_stop_without_visibility = cli_args.allow_stop_without_visibility
    args.allow_occluded_goal_fallback = False

    task_dir = config_path.parent
    print(f"config_path={config_path}", flush=True)
    print(f"output_root={output_root}", flush=True)
    print(
        "rerun_args="
        + json.dumps(
            {
                "max_attempts": cli_args.max_attempts,
                "max_step": cli_args.max_step,
                "render_sensor_height": args.render_sensor_height,
                "success_dis": args.success_dis,
                "success_visible_pixels": args.success_visible_pixels,
                "allow_stop_without_visibility": args.allow_stop_without_visibility,
                "requested_allow_occluded_goal_fallback": cli_args.allow_occluded_goal_fallback,
                "strict_visible_viewpoints": True,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    inspect_targets(args, config)
    if cli_args.inspect_only:
        return

    for attempt in range(1, cli_args.max_attempts + 1):
        print(f"=== DIRECT RETRY {attempt}/{cli_args.max_attempts} ===", flush=True)
        out = eval_for_one_task(args, copy.deepcopy(config))
        print(f"result={out}", flush=True)
        print(
            "trial_counts="
            + json.dumps(
                {
                    "success": count_trials(task_dir, "success"),
                    "fail": count_trials(task_dir, "fail"),
                }
            ),
            flush=True,
        )
        if isinstance(out, list) and out and out[0]:
            print("DIRECT_SUCCESS=1", flush=True)
            return

    print("DIRECT_SUCCESS=0", flush=True)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
