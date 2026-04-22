#!/usr/bin/env python3
"""
从 VLN episode 轨迹生成带 3D bounding box 标注的视频。

使用方法：
  /home/gs/anaconda3/envs/vlnce/bin/python scripts/ply_data_processing/render_episode_video.py \
      --episode-id 1 \
      --label-json data/processed/labeled_episode_ply/ep00001_7y3sRwLe3Va.json \
      --output-dir data/processed/episode_videos
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import quaternion
import habitat.sims
from habitat.config.default import get_config

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_JSON = Path("/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json")
DEFAULT_GT_JSON = Path("/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json")
DEFAULT_SCENE_DIR = Path("/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat/mp3d")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "episode_videos"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episode-id", type=int, required=True)
    p.add_argument("--label-json", type=Path, required=True)
    p.add_argument("--train-json", type=Path, default=DEFAULT_TRAIN_JSON)
    p.add_argument("--gt-json", type=Path, default=DEFAULT_GT_JSON)
    p.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--fps", type=int, default=1)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=426)  # 640 * 57/86 ≈ 424, 取426
    p.add_argument("--hfov", type=float, default=86.0)
    p.add_argument("--vfov", type=float, default=57.0)
    p.add_argument("--sensor-height", type=float, default=1.0)
    return p.parse_args()


def ply_to_habitat(pts):
    """PLY(x,y,z) -> Habitat(X,Y,Z): (x, z, -y)"""
    pts = np.asarray(pts)
    return np.stack([pts[..., 0], pts[..., 2], -pts[..., 1]], axis=-1)


def load_episode_data(train_json, gt_json, episode_id):
    with open(train_json) as f:
        train = json.load(f)
    with open(gt_json) as f:
        gt = json.load(f)

    ep = next((e for e in train["episodes"] if e["episode_id"] == episode_id), None)
    if not ep:
        raise ValueError(f"episode {episode_id} not found")

    actions = gt[str(episode_id)]["actions"]
    return ep, actions


def load_labels(label_json):
    with open(label_json) as f:
        data = json.load(f)
    objects = []
    for obj in data["objects"]:
        verts_ply = np.array(obj["vertices"])
        verts_habitat = ply_to_habitat(verts_ply)
        objects.append({"name": obj["name"], "vertices": verts_habitat})
    return objects


def project_3d_to_2d(pts_world, agent_pos, agent_rot_q, hfov, vfov, width, height):
    """
    世界坐标 -> 图像坐标。
    Habitat 坐标系：+Y上，+X右，+Z后（-Z前）。
    agent_rot_q: np.quaternion，表示 agent 朝向（local->world）。
    """
    # world -> agent-local: R^T @ (p - pos)
    R = quaternion.as_rotation_matrix(agent_rot_q)  # local->world
    pts_local = (pts_world - agent_pos) @ R  # R^T @ v = v @ R

    # agent-local: X右，Y上，-Z前 -> 相机坐标（Z为深度，向前为正）
    # 翻转 Z：cam_z = -local_z
    cam_x = pts_local[:, 0]
    cam_y = pts_local[:, 1]
    cam_z = -pts_local[:, 2]  # 深度，正值表示在相机前方

    fx = width / (2 * np.tan(np.radians(hfov) / 2))
    fy = height / (2 * np.tan(np.radians(vfov) / 2))
    cx, cy = width / 2.0, height / 2.0

    x_img = fx * cam_x / cam_z + cx
    y_img = -fy * cam_y / cam_z + cy  # Y轴向下翻转

    return np.stack([x_img, y_img], axis=1), cam_z


def draw_bbox_3d(img, vertices, agent_pos, agent_rot_q, hfov, vfov, width, height):
    pts_2d, depths = project_3d_to_2d(vertices, agent_pos, agent_rot_q, hfov, vfov, width, height)
    # 只有全部顶点都在相机前方且投影在图像内才绘制
    if not np.all(depths > 0):
        return
    x1, y1 = pts_2d.min(axis=0).astype(int)
    x2, y2 = pts_2d.max(axis=0).astype(int)
    # 裁剪到图像范围内
    x1, x2 = np.clip([x1, x2], 0, width - 1)
    y1, y2 = np.clip([y1, y2], 0, height - 1)
    if x2 > x1 and y2 > y1:
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading episode {args.episode_id}...")
    ep, actions = load_episode_data(args.train_json, args.gt_json, args.episode_id)
    scene_id = ep["scene_id"]
    scene_name = Path(scene_id).stem
    scene_path = args.scene_dir / scene_name / f"{scene_name}.glb"

    print(f"Loading labels from {args.label_json}...")
    objects = load_labels(args.label_json)
    print(f"  {len(objects)} objects loaded")

    print(f"Setting up Habitat simulator...")
    config = get_config()
    config.defrost()
    config.SIMULATOR.SCENE = str(scene_path)
    config.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
    config.SIMULATOR.RGB_SENSOR.WIDTH = args.width
    config.SIMULATOR.RGB_SENSOR.HEIGHT = args.height
    config.SIMULATOR.RGB_SENSOR.HFOV = args.hfov
    config.SIMULATOR.RGB_SENSOR.POSITION = [0, args.sensor_height, 0]
    config.SIMULATOR.TURN_ANGLE = 15
    config.freeze()

    sim = habitat.sims.make_sim(config.SIMULATOR.TYPE, config=config.SIMULATOR)

    # 设置初始 agent 状态
    start_pos = np.array(ep["start_position"], dtype=np.float32)
    sr = ep["start_rotation"]  # [x, y, z, w]
    start_rot = np.quaternion(sr[3], sr[0], sr[1], sr[2])  # np.quaternion(w,x,y,z)
    sim.set_agent_state(start_pos, start_rot)

    def render_current(sim):
        state = sim.get_agent_state()
        obs = sim.get_observations_at(state.position, state.rotation)
        img = obs["rgb"].copy()
        sensor_state = state.sensor_states['rgb']
        sensor_pos = np.array(sensor_state.position, dtype=np.float64)
        sensor_rot_q = sensor_state.rotation
        for obj in objects:
            draw_bbox_3d(img, obj["vertices"], sensor_pos, sensor_rot_q,
                        args.hfov, args.vfov, args.width, args.height)
        return img

    print(f"Rendering {len(actions)+1} frames (initial + {len(actions)} actions)...")
    frames = [render_current(sim)]
    print(f"  Frame 0/initial")

    for i, action in enumerate(actions):
        if action == 0:
            pass  # STOP: 不移动，直接渲染当前位置
        else:
            sim.step(action)
        img = render_current(sim)
        frames.append(img)
        print(f"  Frame {i+1}/{len(actions)} action={action}")

    sim.close()

    # 生成视频
    video_path = args.output_dir / f"ep{args.episode_id:05d}_{scene_name}.mp4"
    print(f"Writing video to {video_path}...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(video_path), fourcc, args.fps,
                         (args.width, args.height))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()

    print(f"Done. Video saved to {video_path}")


if __name__ == "__main__":
    main()
