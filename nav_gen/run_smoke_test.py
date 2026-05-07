import argparse
import copy
import gc
import json
import os
import time
import traceback
from types import SimpleNamespace

from dataset_gen import eval_for_one_task
from task_gen import gen_task


def normalize_task_config(task):
    config = copy.deepcopy(task)
    subtask = config["Subtask list"]
    obj = []
    region_id = []
    for item in subtask:
        if "Move_to" in item:
            obj_id = item[9:-2].split("_")
            obj.append(obj_id[0])
            region_id.append(obj_id[1])
    rooms = []
    for room in config["Object"]:
        rooms.append(room[1].split(": ")[1])
    config["Object"] = obj
    config["Region Name"] = rooms
    config["Region"] = region_id
    return config


def build_args(output_root, max_step, scene_id):
    nav_gen_path = os.getcwd()
    project_path = os.path.dirname(nav_gen_path)
    for subdir in ("task", "step_task", "logs"):
        os.makedirs(os.path.join(output_root, subdir), exist_ok=True)

    return SimpleNamespace(
        API_KEY=os.getenv("DASHSCOPE_API_KEY"),
        llm_model=os.getenv("NAVGEN_LLM_MODEL", "qwen3.6-plus"),
        vlm_model=os.getenv("NAVGEN_VLM_MODEL", os.getenv("NAVGEN_LLM_MODEL", "qwen3.6-plus")),
        llm_base_url=os.getenv("DASHSCOPE_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        scene_path=nav_gen_path + "/scene/",
        prompt_path=nav_gen_path + "/prompt/",
        task_path=os.path.join(output_root, "task") + "/",
        region_file=nav_gen_path + "/scene/Per_Scene_Region_Weighted_Votes.csv",
        loop=1,
        scene_id=scene_id,
        sample_region=False,
        sample_obj=True,
        scene=project_path + "/data/hm3d/",
        scene_dataset=project_path + "/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json",
        sim_gpu_device=int(os.getenv("NAVGEN_SIM_GPU_DEVICE", "0")),
        render_sensor_height=1.0,
        max_step=max_step,
        success_dis=1,
        success_visible_pixels=25,
        allow_stop_without_visibility=False,
        allow_occluded_goal_fallback=False,
        success_view_radius_min=0.75,
        success_view_radius_max=2.5,
        success_view_radius_step=0.5,
        success_view_angle_step=30.0,
        step_task_path=os.path.join(output_root, "step_task") + "/",
        ram_model=project_path + "/data/models/ram_plus_swin_large_14m.pth",
        ram_device=os.getenv("NAVGEN_RAM_DEVICE", "auto"),
        ram_logs=os.path.join(output_root, "logs", "step_task_logs.txt"),
        split_save_path=os.path.join(output_root, "task", "trail_list.txt"),
    )


def run_smoke(output_root, target_successes, max_attempts, max_step, scene_id):
    args = build_args(output_root, max_step, scene_id)
    records = []
    successes = 0
    attempt = 0

    while successes < target_successes and attempt < max_attempts:
        attempt += 1
        print(f"=== ATTEMPT {attempt} / {max_attempts} ===", flush=True)
        try:
            raw_task = gen_task(args)
        except Exception as exc:
            traceback.print_exc()
            records.append(
                {
                    "attempt": attempt,
                    "status": "task_generation_error",
                    "error": repr(exc),
                }
            )
            continue

        if raw_task is False:
            records.append(
                {
                    "attempt": attempt,
                    "status": "task_generation_invalid",
                }
            )
            continue

        task = normalize_task_config(raw_task)
        task_len = len(task["Object"])
        task_dir = os.path.join(args.task_path, str(task_len), task["Task instruction"])
        config_path = os.path.join(task_dir, "config.json")

        time_start = time.time()
        try:
            out = eval_for_one_task(args, task)
        except Exception as exc:
            traceback.print_exc()
            records.append(
                {
                    "attempt": attempt,
                    "status": "trajectory_error",
                    "task_instruction": task["Task instruction"],
                    "scene": task["Scene"],
                    "robot": task["Robot"],
                    "config_path": config_path,
                    "error": repr(exc),
                }
            )
            gc.collect()
            continue

        duration_s = time.time() - time_start
        record = {
            "attempt": attempt,
            "task_instruction": task["Task instruction"],
            "scene": task["Scene"],
            "robot": task["Robot"],
            "target_count": len(task["Object"]),
            "targets": task["Object"],
            "regions": task["Region"],
            "config_path": config_path,
            "duration_s": round(duration_s, 3),
        }

        if isinstance(out, str):
            record["status"] = "unreachable_target"
            record["unreachable_task_instruction"] = out
        elif out[0]:
            successes += 1
            success_dir = os.path.join(task_dir, "success")
            trials = sorted(os.listdir(success_dir)) if os.path.isdir(success_dir) else []
            record["status"] = "success"
            record["task_step_sum"] = out[1]
            record["nav_steps"] = out[2]
            record["success_trial_dir"] = os.path.join(success_dir, trials[-1]) if trials else None
        else:
            fail_dir = os.path.join(task_dir, "fail")
            trials = sorted(os.listdir(fail_dir)) if os.path.isdir(fail_dir) else []
            record["status"] = "max_step_fail"
            record["nav_steps"] = out[2]
            record["fail_trial_dir"] = os.path.join(fail_dir, trials[-1]) if trials else None

        records.append(record)
        print(
            json.dumps(
                {
                    "attempt": attempt,
                    "status": record["status"],
                    "successes": successes,
                    "task_instruction": record["task_instruction"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        gc.collect()

    summary = {
        "smoke_root": output_root,
        "attempts": attempt,
        "target_successes": target_successes,
        "successful_trajectories": successes,
        "records": records,
    }
    summary_path = os.path.join(output_root, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(
        json.dumps(
            {
                "smoke_root": output_root,
                "summary_path": summary_path,
                "attempts": attempt,
                "successful_trajectories": successes,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", required=True, help="directory for isolated smoke-test outputs")
    parser.add_argument("--target_successes", type=int, default=5, help="stop after this many successful trajectories")
    parser.add_argument("--max_attempts", type=int, default=12, help="maximum generation attempts")
    parser.add_argument("--max_step", type=int, default=500, help="max navigation steps per task")
    parser.add_argument("--scene_id", type=str, default=None, help="optional fixed scene id")
    cli_args = parser.parse_args()
    run_smoke(
        output_root=os.path.abspath(cli_args.output_root),
        target_successes=cli_args.target_successes,
        max_attempts=cli_args.max_attempts,
        max_step=cli_args.max_step,
        scene_id=cli_args.scene_id,
    )


if __name__ == "__main__":
    main()
