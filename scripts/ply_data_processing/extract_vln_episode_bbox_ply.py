#!/usr/bin/env python3
"""
批量从 MP3D 语义 PLY 文件中裁剪 VLN episode 轨迹对应的 bounding box 区域。

使用方法：
  python3 scripts/ply_data_processing/extract_vln_episode_bbox_ply.py \
      [--train-json /path/to/train.json] \
      [--gt-json    /path/to/train_gt.json] \
      [--ply-dir    /path/to/mp3d] \
      [--output-dir data/processed/episode_ply] \
      [--padding 3.0] \
      [--episode-ids 1 2 3]
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import open3d as o3d

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_TRAIN_JSON = Path("/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json")
DEFAULT_GT_JSON    = Path("/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json")
DEFAULT_PLY_DIR    = Path("/home/gs/my_test/vln_dataset/data/scene_datasets/mp3d/mp3d_habitat/mp3d")
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "episode_ply"

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-json",  type=Path, default=DEFAULT_TRAIN_JSON)
    p.add_argument("--gt-json",     type=Path, default=DEFAULT_GT_JSON)
    p.add_argument("--ply-dir",     type=Path, default=DEFAULT_PLY_DIR)
    p.add_argument("--output-dir",  type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--padding",     type=float, default=3.0,
                   help="各方向统一膨胀（被各方向独立参数覆盖时忽略）")
    p.add_argument("--pad-x-pos",   type=float, default=None, help="Habitat +X 方向膨胀(m)")
    p.add_argument("--pad-x-neg",   type=float, default=None, help="Habitat -X 方向膨胀(m)")
    p.add_argument("--pad-y-pos",   type=float, default=None, help="Habitat +Y 方向膨胀(m)")
    p.add_argument("--pad-y-neg",   type=float, default=None, help="Habitat -Y 方向膨胀(m)")
    p.add_argument("--pad-z-pos",   type=float, default=None, help="Habitat +Z 方向膨胀(m)")
    p.add_argument("--pad-z-neg",   type=float, default=None, help="Habitat -Z 方向膨胀(m)")
    p.add_argument("--episode-ids", type=int, nargs="+", default=None,
                   help="只处理指定 episode_id，不指定则处理全部")
    return p.parse_args()


def load_episode_data(train_json: Path, gt_json: Path, episode_ids: list | None) -> dict:
    """读取两个 JSON，按 scene_name 分组返回 {scene_name: [episode_dict, ...]}"""
    with open(train_json) as f:
        train = json.load(f)
    with open(gt_json) as f:
        gt = json.load(f)

    ep_set = set(episode_ids) if episode_ids else None
    result = {}
    for ep in train["episodes"]:
        if ep_set and ep["episode_id"] not in ep_set:
            continue
        eid = str(ep["episode_id"])
        if eid not in gt:
            logger.warning("episode %s not in gt, skip", eid)
            continue
        scene_name = Path(ep["scene_id"]).stem
        result.setdefault(scene_name, []).append({
            "episode_id":    ep["episode_id"],
            "trajectory_id": ep["trajectory_id"],
            "scene_id":      ep["scene_id"],
            "locations":     gt[eid]["locations"],
        })
    return result


def habitat_to_ply(locations: list) -> np.ndarray:
    """Habitat (X, Y, Z) -> PLY (X, -Z, Y)"""
    pts = np.array(locations, dtype=np.float64)
    return np.stack([pts[:, 0], -pts[:, 2], pts[:, 1]], axis=1)


def compute_padded_bbox(locations: list, pad: np.ndarray) -> tuple:
    """pad: shape (6,) 顺序为 [+X,-X,+Y,-Y,+Z,-Z]（Habitat 坐标系），转换后应用到 PLY 坐标系"""
    pts = habitat_to_ply(locations)
    # Habitat(+X,-X,+Y,-Y,+Z,-Z) -> PLY(+x,-x,+z,-z,-y,+y) 对应 PLY(+x,-x,+y,-y,+z,-z)
    # PLY x = Habitat X  => pad_ply_x+ = pad[0], pad_ply_x- = pad[1]
    # PLY y = Habitat Z  => pad_ply_y+ = pad[4], pad_ply_y- = pad[5]  (注意 PLY y = -Habitat Z，所以 +PLY_y 对应 -Habitat_Z)
    # PLY z = Habitat Y  => pad_ply_z+ = pad[2], pad_ply_z- = pad[3]  (PLY z = Habitat Y)
    # 实际变换: PLY(x,y,z) = Habitat(X,-Z,Y)
    # +PLY_x = +Habitat_X => pad[0]; -PLY_x = -Habitat_X => pad[1]
    # +PLY_y = -Habitat_Z => pad[5]; -PLY_y = +Habitat_Z => pad[4]
    # +PLY_z = +Habitat_Y => pad[2]; -PLY_z = -Habitat_Y => pad[3]
    pad_min = np.array([pad[1], pad[4], pad[3]], dtype=np.float64)  # PLY -x,-y,-z
    pad_max = np.array([pad[0], pad[5], pad[2]], dtype=np.float64)  # PLY +x,+y,+z
    return pts.min(axis=0) - pad_min, pts.max(axis=0) + pad_max


def process_scene(scene_name: str, episodes: list, ply_dir: Path, output_dir: Path, pad: np.ndarray) -> int:
    ply_path = ply_dir / scene_name / f"{scene_name}_semantic.ply"
    if not ply_path.exists():
        logger.error("PLY not found: %s", ply_path)
        return 0

    logger.info("Loading %s (%.0f MB)...", scene_name, ply_path.stat().st_size / 1e6)
    mesh = o3d.io.read_triangle_mesh(str(ply_path))

    count = 0
    for ep in episodes:
        min_b, max_b = compute_padded_bbox(ep["locations"], pad)
        bbox    = o3d.geometry.AxisAlignedBoundingBox(min_bound=min_b, max_bound=max_b)
        cropped = mesh.crop(bbox)

        if len(cropped.vertices) == 0:
            logger.warning("ep%d: empty crop, skip", ep["episode_id"])
            continue

        stem     = f"ep{ep['episode_id']:05d}_{scene_name}"
        ply_out  = output_dir / f"{stem}.ply"
        json_out = output_dir / f"{stem}.json"

        o3d.io.write_triangle_mesh(str(ply_out), cropped, write_ascii=False)
        json_out.write_text(json.dumps({
            "episode_id":    ep["episode_id"],
            "trajectory_id": ep["trajectory_id"],
            "scene_id":      ep["scene_id"],
            "bbox_min":      min_b.tolist(),
            "bbox_max":      max_b.tolist(),
            "padding_m":     pad.tolist(),
            "num_vertices":  len(cropped.vertices),
            "num_faces":     len(cropped.triangles),
            "locations":     ep["locations"],
        }, indent=2, ensure_ascii=False))
        logger.info("  ep%d: %d vertices, %d faces -> %s",
                    ep["episode_id"], len(cropped.vertices), len(cropped.triangles), ply_out.name)
        count += 1

    del mesh
    return count


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # 构建 6 方向膨胀数组 [+X,-X,+Y,-Y,+Z,-Z]（Habitat 坐标系）
    def _p(val, default):
        return val if val is not None else default
    pad = np.array([
        _p(args.pad_x_pos, args.padding),
        _p(args.pad_x_neg, args.padding),
        _p(args.pad_y_pos, args.padding),
        _p(args.pad_y_neg, args.padding),
        _p(args.pad_z_pos, args.padding),
        _p(args.pad_z_neg, args.padding),
    ])

    logger.info("Loading episode data...")
    scenes = load_episode_data(args.train_json, args.gt_json, args.episode_ids)
    total_eps = sum(len(v) for v in scenes.values())
    logger.info("%d scenes, %d episodes to process", len(scenes), total_eps)

    total = 0
    for i, (scene_name, episodes) in enumerate(scenes.items(), 1):
        logger.info("[%d/%d] scene: %s (%d episodes)", i, len(scenes), scene_name, len(episodes))
        total += process_scene(scene_name, episodes, args.ply_dir, args.output_dir, pad)

    logger.info("Done. %d episodes saved to %s", total, args.output_dir)


if __name__ == "__main__":
    main()
