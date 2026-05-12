"""
Run in conda lhvln environment:
  conda run -n lhvln python extract_scene_data.py --task_json <path> --output_dir ./extracted

Extracts: trajectory positions, target object positions, navmesh vertices/faces,
          and scene point cloud (cropped to trajectory bounding box).
"""
import argparse
import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from pcd_io import write_pcd


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--task_json', required=True,
                   help='Path to success/trial_1/task.json')
    p.add_argument('--output_dir', default='./extracted')
    p.add_argument('--scene_root', default='/home/gs/my_project/LH-VLN/data/hm3d/',
                   help='Root dir of hm3d scenes')
    p.add_argument('--scene_dataset', default='/home/gs/my_project/LH-VLN/data/hm3d/hm3d_annotated_basis.scene_dataset_config.json')
    p.add_argument('--trajectory_meta', default=None,
                   help='Optional trajectory_rgb.json with resolved_target_boxes metadata')
    p.add_argument('--pointcloud_source', choices=['depth', 'mesh'], default='depth',
                   help='Use trajectory depth unprojection or basis.glb mesh sampling')
    p.add_argument('--mesh_points', type=int, default=300000,
                   help='Number of mesh surface points to sample when --pointcloud_source mesh')
    p.add_argument('--sensor_height', type=float, default=1.5,
                   help='Camera sensor height above agent for depth point cloud')
    p.add_argument('--padding', type=float, default=3.0,
                   help='Bounding box padding (m) for point cloud crop')
    p.add_argument('--pointcloud_format', choices=['pcd', 'npy', 'both'], default='both',
                   help='Output format for extracted point cloud')
    return p.parse_args()


def scene_glb_path(scene_root, scene_id):
    split = 'train' if int(scene_id[2]) < 8 else 'val'
    short = scene_id.split('-')[1]
    return os.path.join(scene_root, split, scene_id, f'{short}.basis.glb')


def scene_navmesh_path(scene_root, scene_id):
    split = 'train' if int(scene_id[2]) < 8 else 'val'
    short = scene_id.split('-')[1]
    return os.path.join(scene_root, split, scene_id, f'{short}.basis.navmesh')


def build_sim(scene_glb, scene_dataset, sensor_height=1.5):
    import habitat_sim
    import magnum as mn

    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = scene_glb
    sim_cfg.scene_dataset_config_file = scene_dataset
    sim_cfg.enable_physics = False

    specs = []
    for uuid, stype, orient in [
        ('color', habitat_sim.SensorType.COLOR,  [0., 0., 0.]),
        ('depth', habitat_sim.SensorType.DEPTH,  [0., 0., 0.]),
    ]:
        s = habitat_sim.CameraSensorSpec()
        s.uuid = uuid
        s.sensor_type = stype
        s.resolution = [256, 256]
        s.position = mn.Vector3(0, sensor_height, 0)
        s.orientation = mn.Vector3(*orient)
        specs.append(s)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = specs
    agent_cfg.action_space = {
        'move_forward': habitat_sim.agent.ActionSpec('move_forward', habitat_sim.agent.ActuationSpec(amount=0.25)),
        'turn_left':    habitat_sim.agent.ActionSpec('turn_left',    habitat_sim.agent.ActuationSpec(amount=30.0)),
        'turn_right':   habitat_sim.agent.ActionSpec('turn_right',   habitat_sim.agent.ActuationSpec(amount=30.0)),
    }

    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    sim = habitat_sim.Simulator(cfg)
    return sim


def extract_trajectory(task_data):
    """Returns list of (x,y,z) for all trial positions in order."""
    positions = []
    trial = task_data.get('trial', {})
    for i in range(len(task_data.get('Object', []))):
        key = f'trial_{i}'
        if key in trial:
            positions.extend(trial[key]['pos'])
    return np.array(positions, dtype=np.float32)


def load_trajectory_meta(path):
    if not path:
        return None
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def normalize_region_id(region_id):
    if region_id is None:
        return None
    text = str(region_id).strip()
    if not text:
        return None
    if text.startswith('Region '):
        text = text.split(':', 1)[0].replace('Region ', '').strip()
    elif text[-1:].isdigit():
        digits = []
        for ch in reversed(text):
            if not ch.isdigit():
                break
            digits.append(ch)
        if digits:
            text = ''.join(reversed(digits))
    return text


def trial_end_position(task_data, target_index):
    trial = task_data.get('trial', {}).get(f'trial_{target_index}')
    if not trial or not trial.get('pos'):
        return None
    return np.array(trial['pos'][-1], dtype=np.float32)


def extract_selected_targets_from_meta(trajectory_meta):
    if not trajectory_meta:
        return {}
    selected = {}
    for item in trajectory_meta.get('resolved_target_boxes', []):
        if not item.get('resolved', False) or item.get('selected_center') is None:
            continue
        trial_index = item.get('trial_index')
        if trial_index is None:
            trial_key = item.get('trial_key')
            if isinstance(trial_key, str) and trial_key.startswith('trial_'):
                trial_index = int(trial_key.split('_', 1)[1])
        if trial_index is None:
            continue
        selected[int(trial_index)] = item
    return selected


def target_candidates(sim, task_data, target_index):
    obj_name = task_data.get('Object', [])[target_index]
    region_ids = task_data.get('Region', [])
    region_id = normalize_region_id(region_ids[target_index]) if target_index < len(region_ids) else None
    candidates = []

    for region in sim.semantic_scene.regions:
        if region_id is not None and normalize_region_id(region.id) != region_id:
            continue
        for obj in region.objects:
            if obj is None or obj.category is None:
                continue
            if obj.category.name().lower() != obj_name.lower():
                continue
            center = getattr(getattr(obj, 'aabb', None), 'center', None)
            if center is None:
                continue
            candidates.append({
                'center': np.array(center, dtype=np.float32),
                'semantic_id': getattr(obj, 'semantic_id', None),
                'object_id': getattr(obj, 'id', None),
                'source': 'region',
            })

    if candidates or region_id is None:
        return candidates

    for obj in sim.semantic_scene.objects:
        if obj is None or obj.category is None:
            continue
        if obj.category.name().lower() != obj_name.lower():
            continue
        center = getattr(getattr(obj, 'aabb', None), 'center', None)
        if center is None:
            continue
        candidates.append({
            'center': np.array(center, dtype=np.float32),
            'semantic_id': getattr(obj, 'semantic_id', None),
            'object_id': getattr(obj, 'id', None),
            'source': 'scene_fallback',
        })
    return candidates


def choose_candidate(candidates, reference_position):
    if not candidates:
        return None
    if reference_position is None:
        return candidates[0]
    reference_xz = np.array([reference_position[0], reference_position[2]], dtype=np.float32)
    return min(
        candidates,
        key=lambda item: float(np.linalg.norm(
            np.array([item['center'][0], item['center'][2]], dtype=np.float32) - reference_xz
        )),
    )


def extract_target_positions(sim, task_data, trajectory_meta=None):
    """Find target instance centers matching generation-time target resolution."""
    selected_from_meta = extract_selected_targets_from_meta(trajectory_meta)
    targets = []
    for target_index, obj_name in enumerate(task_data.get('Object', [])):
        selected_meta = selected_from_meta.get(target_index)
        if selected_meta is not None:
            selected_semantic_ids = selected_meta.get('selected_semantic_ids', [])
            target = {
                'name': obj_name,
                'position': selected_meta['selected_center'],
                'source': 'trajectory_meta',
                'trial_key': selected_meta.get('trial_key', f'trial_{target_index}'),
                'semantic_id': selected_semantic_ids[0] if selected_semantic_ids else None,
                'object_id': selected_meta.get('selected_object_id'),
            }
            targets.append(target)
            continue

        candidates = target_candidates(sim, task_data, target_index)
        chosen = choose_candidate(candidates, trial_end_position(task_data, target_index))
        if chosen is None:
            print(f'[warn] object "{obj_name}" not found in target region or semantic scene')
            continue
        targets.append({
            'name': obj_name,
            'position': chosen['center'].tolist(),
            'source': chosen['source'],
            'trial_key': f'trial_{target_index}',
            'semantic_id': chosen.get('semantic_id'),
            'object_id': chosen.get('object_id'),
            'candidate_count': len(candidates),
        })
    return targets


def extract_navmesh(sim, traj_positions, padding):
    """Extract navmesh vertices/faces cropped to trajectory bounding box."""
    pf = sim.pathfinder
    if not pf.is_loaded:
        print('[warn] navmesh not loaded')
        return None, None

    mn_bounds = pf.get_bounds()
    # get all navmesh vertices via island sampling
    # habitat_sim exposes navmesh via get_topdown_view or direct vertex access
    # Use get_random_navigable_point sampling as fallback if direct access unavailable
    raw_verts = pf.build_navmesh_vertices()
    raw_idx   = pf.build_navmesh_vertex_indices()
    verts = np.array(raw_verts, dtype=np.float32)          # already (N,3) or list of arrays
    if verts.ndim == 1:
        verts = verts.reshape(-1, 3)
    faces = np.array(raw_idx, dtype=np.int32).reshape(-1, 3)

    # Crop to trajectory bounding box + padding
    if len(traj_positions) > 0:
        mn_pos = traj_positions.min(axis=0) - padding
        mx_pos = traj_positions.max(axis=0) + padding
        mask = np.all((verts >= mn_pos) & (verts <= mx_pos), axis=1)
        verts = verts[mask]
        if faces is not None:
            # keep faces where all 3 verts are in mask
            orig_indices = np.where(mask)[0]
            idx_map = {old: new for new, old in enumerate(orig_indices)}
            valid_faces = []
            for f in faces:
                if all(v in idx_map for v in f):
                    valid_faces.append([idx_map[v] for v in f])
            faces = np.array(valid_faces, dtype=np.int32) if valid_faces else None

    return verts, faces


def crop_pointcloud(verts, colors, traj_positions, padding):
    if len(traj_positions) == 0 or len(verts) == 0:
        return verts, colors
    mn_pos = traj_positions.min(axis=0) - padding
    mx_pos = traj_positions.max(axis=0) + padding
    mask = np.all((verts >= mn_pos) & (verts <= mx_pos), axis=1)
    verts = verts[mask]
    if colors is not None:
        colors = colors[mask]
    return verts, colors


def sample_mesh_pointcloud(scene_glb, traj_positions, padding, n_points):
    """Sample a dense point cloud directly from the HM3D basis.glb mesh."""
    try:
        import trimesh
    except ImportError as exc:
        raise RuntimeError('Mesh point cloud mode requires trimesh: pip install trimesh') from exc

    mesh = trimesh.load(scene_glb, force='mesh', process=False)
    if mesh.is_empty:
        return np.zeros((0, 3), dtype=np.float32), None

    points, face_indices = trimesh.sample.sample_surface(mesh, int(n_points))
    points = points.astype(np.float32)
    colors = None
    visual = getattr(mesh, 'visual', None)
    face_colors = getattr(visual, 'face_colors', None)
    if face_colors is not None and len(face_colors) > 0:
        colors = np.asarray(face_colors[face_indices, :3], dtype=np.float32) / 255.0
    return crop_pointcloud(points, colors, traj_positions, padding)


def extract_pointcloud(sim, traj_positions, padding, sensor_height=1.5, n_views=24):
    """Render depth from trajectory positions and unproject to world coords."""
    import habitat_sim
    import quaternion as qt

    W, H = 256, 256
    hfov = math.radians(90)
    fx = fy = (W / 2) / math.tan(hfov / 2)
    cx, cy = W / 2.0, H / 2.0

    indices = np.linspace(0, len(traj_positions) - 1, min(n_views, len(traj_positions)), dtype=int)
    all_pts, all_cols = [], []

    for idx in indices:
        pos = traj_positions[idx]
        for yaw_deg in [0, 90, 180, 270]:
            yaw = math.radians(yaw_deg)
            state = habitat_sim.AgentState()
            state.position = pos
            agent_rotation = qt.quaternion(math.cos(yaw / 2), 0, math.sin(yaw / 2), 0)
            state.rotation = agent_rotation
            sim.agents[0].set_state(state)
            obs = sim.get_sensor_observations()

            depth = obs.get('depth')
            color = obs.get('color')
            if depth is None:
                continue

            d = depth.astype(np.float32)
            valid = (d > 0.1) & (d < 8.0)
            if not valid.any():
                continue

            v_idx, u_idx = np.where(valid)
            z = d[valid]
            x_cam = (u_idx - cx) * z / fx
            y_cam = -(v_idx - cy) * z / fy
            z_cam = -z

            pts_cam = np.stack([x_cam, y_cam, z_cam], axis=1)
            sensor_origin = np.array(pos, dtype=np.float32) + np.array([0.0, sensor_height, 0.0], dtype=np.float32)
            pts = qt.rotate_vectors(agent_rotation, pts_cam) + sensor_origin
            all_pts.append(pts.astype(np.float32))
            if color is not None:
                all_cols.append(color[valid, :3].astype(np.float32) / 255.0)

    if not all_pts:
        return np.zeros((0, 3), dtype=np.float32), None

    verts = np.concatenate(all_pts, axis=0)
    colors = np.concatenate(all_cols, axis=0) if all_cols else None

    if len(traj_positions) > 0:
        verts, colors = crop_pointcloud(verts, colors, traj_positions, padding)

    return verts, colors


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    with open(args.task_json) as f:
        task_data = json.load(f)

    scene_id = task_data['Scene']
    glb = scene_glb_path(args.scene_root, scene_id)
    print(f'Scene: {glb}')

    sim = build_sim(glb, args.scene_dataset, sensor_height=args.sensor_height)
    trajectory_meta = load_trajectory_meta(args.trajectory_meta)

    # 1. Trajectory
    traj = extract_trajectory(task_data)
    np.save(os.path.join(args.output_dir, 'trajectory.npy'), traj)
    print(f'Trajectory: {len(traj)} points')

    # 2. Target objects
    targets = extract_target_positions(sim, task_data, trajectory_meta=trajectory_meta)
    with open(os.path.join(args.output_dir, 'targets.json'), 'w') as f:
        json.dump(targets, f, indent=2)
    print(f'Targets: {targets}')

    # 3. Navmesh
    nav_verts, nav_faces = extract_navmesh(sim, traj, args.padding)
    if nav_verts is not None:
        np.save(os.path.join(args.output_dir, 'navmesh_verts.npy'), nav_verts)
        if nav_faces is not None:
            np.save(os.path.join(args.output_dir, 'navmesh_faces.npy'), nav_faces)
        print(f'Navmesh: {len(nav_verts)} verts')

    # 4. Point cloud
    if args.pointcloud_source == 'mesh':
        pc_verts, pc_colors = sample_mesh_pointcloud(glb, traj, args.padding, args.mesh_points)
    else:
        pc_verts, pc_colors = extract_pointcloud(sim, traj, args.padding, sensor_height=args.sensor_height)
    if args.pointcloud_format in ('pcd', 'both'):
        write_pcd(os.path.join(args.output_dir, 'pointcloud.pcd'), pc_verts, pc_colors)
    if args.pointcloud_format in ('npy', 'both'):
        np.save(os.path.join(args.output_dir, 'pointcloud.npy'), pc_verts)
        if pc_colors is not None:
            np.save(os.path.join(args.output_dir, 'pointcloud_colors.npy'), pc_colors)
    print(f'Point cloud: {len(pc_verts)} points')

    # Save task metadata
    with open(os.path.join(args.output_dir, 'meta.json'), 'w') as f:
        json.dump({
            'task_instruction': task_data['Task instruction'],
            'scene': scene_id,
            'robot': task_data['Robot'],
            'object': task_data['Object'],
            'region_name': task_data.get('Region Name', []),
            'trajectory_meta': args.trajectory_meta,
            'pointcloud_source': args.pointcloud_source,
            'pointcloud_format': args.pointcloud_format,
            'mesh_points': args.mesh_points if args.pointcloud_source == 'mesh' else None,
            'sensor_height': args.sensor_height,
        }, f, indent=2)

    sim.close()
    print(f'Done. Data saved to {args.output_dir}')


if __name__ == '__main__':
    main()
