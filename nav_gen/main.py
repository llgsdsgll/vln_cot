import argparse
import os
from task_gen import gen_task
from dataset_gen import gen_traj
from split_task import split_traj, gen_step_task


nav_gen_path = os.getcwd()
project_path = os.path.dirname(nav_gen_path)

if not os.path.exists(nav_gen_path + '/task'):    # for LH-VLN task
    os.makedirs(nav_gen_path + '/task')
if not os.path.exists(nav_gen_path + '/step_task'):    # for step-by-step task
    os.makedirs(nav_gen_path + '/step_task')
if not os.path.exists(nav_gen_path + '/logs'):    # for logs
    os.makedirs(nav_gen_path + '/logs')

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--API_KEY', type=str, default=os.environ.get('DASHSCOPE_API_KEY'), help="API key for the DashScope-compatible chat endpoint")
    parser.add_argument('--llm_model', type=str, default=os.environ.get('NAVGEN_LLM_MODEL', 'qwen3.6-plus'), help="chat model name, defaulting to qwen3.6-plus")
    parser.add_argument('--vlm_model', type=str, default=os.environ.get('NAVGEN_VLM_MODEL', os.environ.get('NAVGEN_LLM_MODEL', 'qwen3.6-plus')), help="vision-capable model name when image input is used")
    parser.add_argument('--llm_base_url', type=str, default=os.environ.get('DASHSCOPE_BASE_URL', 'https://dashscope.aliyuncs.com/compatible-mode/v1'), help="base URL for the DashScope-compatible endpoint")
    # generate task
    parser.add_argument('--scene_path', type=str, default=nav_gen_path + '/scene/', help="root scene path")
    parser.add_argument('--prompt_path', type=str, default=nav_gen_path + '/prompt/', help="root prompt path")
    parser.add_argument('--task_path', type=str, default=nav_gen_path + '/task/', help="root task path")
    parser.add_argument('--region_file', type=str, default=nav_gen_path + '/scene/Per_Scene_Region_Weighted_Votes.csv', help="region semantic file")
    parser.add_argument('--loop', type=int, default=100, help="num of tasks to generate")
    
    parser.add_argument('--scene_id', type=str, default=None, help="Whether or not to select a specific scene")
    parser.add_argument('--sample_region', type=bool, default=False, help="Whether or not to sample room in a scene")
    parser.add_argument('--sample_obj', type=bool, default=True, help="Whether or not to sample obj in a room")

    # simulation
    parser.add_argument('--scene', type=str, default=project_path + '/data/hm3d/', help="scene path")
    parser.add_argument('--scene_dataset', type=str, default=project_path + '/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json', help="scene dataset path")
    parser.add_argument('--sim_gpu_device', type=int, default=int(os.environ.get('NAVGEN_SIM_GPU_DEVICE', 0)), help="GPU device id for Habitat-Sim")
    parser.add_argument('--render_sensor_height', type=float, default=None, help="camera / semantic sensor height in meters; defaults to the robot-specific height")
    parser.add_argument('--max_step', type=int, default=500, help="max step for record")
    parser.add_argument('--success_dis', type=float, default=1, help="distance to be considered as success")
    parser.add_argument('--success_visible_pixels', type=int, default=25, help="minimum front semantic pixels required before a target can be treated as visible for success")
    parser.add_argument('--allow_stop_without_visibility', action='store_true', help="fall back to the old distance-only success rule")
    parser.add_argument('--allow_occluded_goal_fallback', action='store_true', help="fall back to snapped object-center goals when no visible target viewpoint can be found")
    parser.add_argument('--success_view_radius_min', type=float, default=0.75, help="minimum radius when sampling visible target viewpoints")
    parser.add_argument('--success_view_radius_max', type=float, default=2.5, help="maximum radius when sampling visible target viewpoints")
    parser.add_argument('--success_view_radius_step', type=float, default=0.5, help="radius step when sampling visible target viewpoints")
    parser.add_argument('--success_view_angle_step', type=float, default=30.0, help="angle step in degrees when sampling visible target viewpoints")

    # step
    parser.add_argument('--step_task_path', type=str, default=nav_gen_path + '/step_task/', help="root task path")
    parser.add_argument('--ram_model', type=str, default=project_path + '/data/models/ram_plus_swin_large_14m.pth', help="path for ram model")
    parser.add_argument('--ram_device', type=str, default=os.environ.get('NAVGEN_RAM_DEVICE', 'auto'), help="device for RAM inference: auto, cpu, cuda, or cuda:<id>")
    parser.add_argument('--ram_logs', type=str, default=nav_gen_path + '/logs/step_task_logs.txt', help="path to save logs")
    parser.add_argument('--split_save_path', type=str, default=nav_gen_path + '/task/trail_list.txt', help="path to save the split trajectory")


    args = parser.parse_args()
    return args


def main():
    args = read_args()

    # step 1: Generate task
    for i in range(args.loop):
        gen_task(args)

    # step 2: Generate trajectory
    gen_traj(args)

    # step 3: Split trajectory into segments
    split_traj(args)

    # step 4: Generate step-by-step task
    gen_step_task(args)

if __name__ == '__main__':
    main()
