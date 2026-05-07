import habitat_sim
import magnum as mn
import numpy as np
import math
import operator
import random
import os
import re
from typing import List, Optional, Tuple, Union
from habitat_sim.utils.common import quat_from_angle_axis
from .visualization import display_env
from .config import make_setting, make_cfg


class SceneSimulator:
    def __init__(self, args, config):
        self.args = args
        self.scene = config['Scene']            # scene file
        self.robot = config['Robot']            # robot type
        self.target = config['Object']          # target object
        self.region = config['Region']          # region of target object
        self.ins = config['Task instruction']   # task instruction
        # self.task = config['Subtask list']      # subtask list

        self.save_path = args.task_path + str(len(self.target)) + '/' + self.ins
        self.target_candidate_cache = {}
        self.target_viewpoint_cache = {}

        # init simulator
        self.sim_settings = make_setting(self.args, self.scene, self.robot)
        self.cfg = make_cfg(self.sim_settings)
        self.sim = habitat_sim.Simulator(self.cfg)

        # Managers of various Attributes templates
        self.obj_attr_mgr = self.sim.get_object_template_manager()
        self.prim_attr_mgr = self.sim.get_asset_template_manager()
        self.stage_attr_mgr = self.sim.get_stage_template_manager()
        # Manager providing access to rigid objects
        self.rigid_obj_mgr = self.sim.get_rigid_object_manager()
        # Pathfinder
        self.pathfinder = self.sim.pathfinder
        # init the agent with navigable point
        self.agent = self.sim.initialize_agent(self.sim_settings["default_agent"])
        # Set agent state
        agent_state = habitat_sim.AgentState()
        # sample the navigable point as agent's initial position
        sample_navigable_point = self.pathfinder.get_random_navigable_point()
        agent_state.position = sample_navigable_point - np.array([0, 0, -0.25])  # in world space
        self.agent.set_state(agent_state)
        # init yaw
        self.yaw = 180
        # init path follower
        self.action_space = self.cfg.agents[self.sim_settings["default_agent"]].action_space
        self.follower = habitat_sim.nav.GreedyGeodesicFollower(
            pathfinder = self.pathfinder,
            agent = self.agent,
            goal_radius = 1.0,
            stop_key="stop",
            forward_key="move_forward",
            left_key="turn_left",
            right_key="turn_right",
        )

        # init obs
        self.observations = self.sim.step("move_forward")

    def actor(self, action, step, success):
        """
        Perform base action
        Args:
            action: the action to be taken
            step: the step number
            success: the number of finished subtargets
        Returns:
            images: the visual observations
        """
        if action == "stop":
            pass
        else:
            self.observations = self.sim.step(action)    # obtain visual observations
        # adjust the angle based on the action
        if action == "move_forward" or action == "stop":
            pass
        elif action == "turn_left":
            self.yaw += 30
        elif action == "turn_right":
            self.yaw -= 30
        if self.yaw > 180:
            self.yaw -= 360
        elif self.yaw <= -180:
            self.yaw += 360
        if step is not None or step != -1:
            print("action: %s, step: %d" % (action, step))
        obj_target = self.target[success]   # get the target object
        images = display_env(self.observations, action, self.save_path, step, obj_target)
        return images
    
    def set_state(self, pos, yaw):
        """
        Args:
            pos: the position of the agent
            yaw: the yaw of the agent
        Returns:
            None
        """
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(pos)  # in world space
        self.agent.set_state(agent_state)

        d_yaw = (yaw - self.yaw)/30
        if d_yaw > 0:
            for i in range(int(d_yaw)):
                self.observations = self.sim.step("turn_left")
        else:
            for i in range(int(-d_yaw)):
                self.observations = self.sim.step("turn_right")

    def obj_count(self):
        obj_num = 0
        useless = ["wall", "frame", "floor", "sheet", "Unknown", "stairs", "unknown",
                    "ceiling", "window", "curtain", "pillow", "beam", "decoration"]
        scene = self.sim.semantic_scene
        for region in scene.regions:
            for obj in region.objects:
                obj_id = obj.id.split("_")[1]
                if any(c in obj.category.name() for c in useless):
                    continue
                else:
                    obj_num += 1
        return obj_num

    def print_scene_recur(self, file):
        """
        Args:
            file: the file name
        Returns:
            None
        """
        out_path = "nav_gen/data/gen_data/scene/" + file
        useless = ["wall", "frame", "floor", "sheet", "Unknown", "stairs", "unknown",
                    "ceiling", "window", "curtain", "pillow", "beam", "decoration"]
        # if not os.path.exists(out_path):
        #     os.makedirs(out_path)
        scene = self.sim.semantic_scene
        print(
            f"House has {len(scene.levels)} levels, {len(scene.regions)} regions and {len(scene.objects)} objects"
        )
        print(f"House center:{scene.aabb.center} dims:{scene.aabb.sizes}")
        for region in scene.regions:
            print(
                f"Region id:{region.id},"
                f" center:{region.aabb.center}, dims:{region.aabb.sizes}"
            )
            with open(out_path + ".txt",'a') as f:
                f.write(
                    f"Region id:{region.id},"
                    f" position:{region.aabb.center}"
                    f"\n"
                )
            for obj in region.objects:
                obj_id = obj.id.split("_")[1]
                print(
                    f"Object id:{obj_id}, category:{obj.category.name()},"
                    f" center:{obj.aabb.center}, dims:{obj.aabb.sizes}"
                )
                if any(c in obj.category.name() for c in useless):
                    continue
                else:
                    with open(out_path + ".txt",'a') as f:
                        f.write(
                            f"Id:{obj_id}, name:{obj.category.name()},"
                            f" position:{[obj.aabb.center[0], obj.aabb.center[1], obj.aabb.center[2]]}"
                            f"\n"
                        )

    def get_coord(self, obj_target):
        """
        Return the coord of the target object
        Args:
            obj_target: the target object
        Returns:
            coord_list: the list of the coordinates of the target object
        """
        scene = self.sim.semantic_scene
        coord_list = []
        index = self.target.index(obj_target)
        region_id = self.region[index]
        for region in scene.regions:
            if region.id[1:] != region_id:
                continue
            for obj in region.objects:
                if obj.category.name() == obj_target:
                    coord_list.append(obj.aabb.center)
        
        if coord_list == []:
            print("wrong target")
            return 0
        else:
            return coord_list

    def _normalize_region_id(self, region_id):
        if region_id is None:
            return None
        region_text = str(region_id).strip()
        if not region_text:
            return None
        match = re.match(r"^Region\s+(\d+)\s*:", region_text)
        if match:
            return match.group(1)
        match = re.search(r"(\d+)$", region_text)
        if match:
            return match.group(1)
        return region_text

    def get_target_candidates(self, target_index):
        if target_index in self.target_candidate_cache:
            return self.target_candidate_cache[target_index]

        obj_target = self.target[target_index]
        region_id = self._normalize_region_id(self.region[target_index])
        candidates = []
        scene = self.sim.semantic_scene
        for region in scene.regions:
            if region_id is not None and self._normalize_region_id(region.id) != region_id:
                continue
            for obj in region.objects:
                try:
                    category_name = obj.category.name()
                except Exception:
                    continue
                if category_name != obj_target:
                    continue
                center = getattr(getattr(obj, "aabb", None), "center", None)
                if center is None:
                    continue
                semantic_id = getattr(obj, "semantic_id", None)
                candidates.append(
                    {
                        "center": np.array(center, dtype=np.float32),
                        "semantic_id": int(semantic_id) if semantic_id is not None else None,
                        "object_id": getattr(obj, "id", None),
                    }
                )

        self.target_candidate_cache[target_index] = candidates
        return candidates

    def _normalize_saved_yaw(self, yaw):
        yaw = float(yaw)
        while yaw > 180:
            yaw -= 360
        while yaw <= -180:
            yaw += 360
        return yaw

    def _saved_yaw_to_world_angle(self, saved_yaw):
        return float(saved_yaw) - 180.0

    def _world_angle_to_saved_yaw(self, world_angle):
        return self._normalize_saved_yaw(float(world_angle) + 180.0)

    def _goal_center_to_saved_yaw(self, position, goal_center):
        delta_x = float(goal_center[0] - position[0])
        delta_z = float(goal_center[2] - position[2])
        if abs(delta_x) < 1e-6 and abs(delta_z) < 1e-6:
            return self.yaw
        world_angle = math.degrees(math.atan2(delta_x, -delta_z))
        return self._world_angle_to_saved_yaw(world_angle)

    def _capture_pose(self):
        agent_state = self.agent.get_state()
        return {
            "position": np.array(agent_state.position, dtype=np.float32),
            "rotation": agent_state.rotation,
            "yaw": self.yaw,
            "observations": self.observations,
        }

    def _restore_pose(self, pose):
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(pose["position"], dtype=np.float32)
        agent_state.rotation = pose["rotation"]
        self.agent.set_state(agent_state)
        self.yaw = pose["yaw"]
        self.observations = pose["observations"]

    def _set_agent_pose_direct(self, position, saved_yaw):
        world_angle = self._saved_yaw_to_world_angle(saved_yaw)
        agent_state = habitat_sim.AgentState()
        agent_state.position = np.array(position, dtype=np.float32)
        agent_state.rotation = quat_from_angle_axis(
            math.radians(world_angle),
            np.array([0.0, 1.0, 0.0], dtype=np.float32),
        )
        self.agent.set_state(agent_state)
        self.yaw = self._normalize_saved_yaw(saved_yaw)

    def _semantic_pixels_for_pose(self, position, saved_yaw, semantic_id):
        pose = self._capture_pose()
        try:
            self._set_agent_pose_direct(position, saved_yaw)
            observations = self.sim.get_sensor_observations()
            semantic_obs = observations.get("semantic_sensor")
            if semantic_obs is None:
                return 0
            return int(np.count_nonzero(semantic_obs == int(semantic_id)))
        finally:
            self._restore_pose(pose)

    def _is_valid_nav_point(self, point):
        if point is None:
            return False
        array = np.array(point, dtype=np.float32).reshape(-1)
        return array.size == 3 and bool(np.all(np.isfinite(array)))

    def _sample_visible_viewpoints_for_candidate(self, candidate, min_pixels):
        semantic_id = candidate.get("semantic_id")
        goal_center = candidate.get("center")
        if semantic_id is None or goal_center is None:
            return []

        radius_min = float(getattr(self.args, "success_view_radius_min", 0.75))
        radius_max = float(getattr(self.args, "success_view_radius_max", 2.5))
        radius_step = float(getattr(self.args, "success_view_radius_step", 0.5))
        angle_step = float(getattr(self.args, "success_view_angle_step", 30.0))
        required_pixels = max(1, int(min_pixels))

        if radius_step <= 0:
            radius_step = 0.5
        if angle_step <= 0:
            angle_step = 30.0

        viewpoints = []
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
                nav_point = self.pathfinder.snap_point(raw_point)
                if self._is_valid_nav_point(nav_point):
                    nav_point = np.array(nav_point, dtype=np.float32)
                    key = tuple(np.round(nav_point, 3).tolist())
                    if key not in seen_keys:
                        seen_keys.add(key)
                        saved_yaw = self._goal_center_to_saved_yaw(nav_point, goal_center)
                        pixel_count = self._semantic_pixels_for_pose(nav_point, saved_yaw, semantic_id)
                        if pixel_count >= required_pixels:
                            viewpoints.append(
                                {
                                    "position": nav_point,
                                    "yaw": saved_yaw,
                                    "pixel_count": pixel_count,
                                    "semantic_id": semantic_id,
                                    "object_id": candidate.get("object_id"),
                                    "center": np.array(goal_center, dtype=np.float32),
                                }
                            )
                angle_deg += angle_step
            radius += radius_step

        viewpoints.sort(
            key=lambda item: (
                -int(item["pixel_count"]),
                float(math.dist(
                    [float(item["position"][0]), float(item["position"][2])],
                    [float(goal_center[0]), float(goal_center[2])],
                )),
            )
        )
        return viewpoints

    def get_visible_viewpoints(self, target_index, min_pixels=25):
        cache_key = (int(target_index), int(max(1, min_pixels)))
        if cache_key in self.target_viewpoint_cache:
            return self.target_viewpoint_cache[cache_key]

        viewpoints = []
        for candidate in self.get_target_candidates(target_index):
            viewpoints.extend(
                self._sample_visible_viewpoints_for_candidate(
                    candidate,
                    min_pixels=min_pixels,
                )
            )
        self.target_viewpoint_cache[cache_key] = viewpoints
        return viewpoints

    def get_goal_info(self, target_index):
        required_pixels = max(1, int(getattr(self.args, "success_visible_pixels", 25)))
        visible_viewpoints = self.get_visible_viewpoints(
            target_index,
            min_pixels=required_pixels,
        )
        goal_type = "visible_viewpoint"
        if not visible_viewpoints and required_pixels > 1:
            relaxed_viewpoints = self.get_visible_viewpoints(
                target_index,
                min_pixels=1,
            )
            if relaxed_viewpoints:
                visible_viewpoints = relaxed_viewpoints
                goal_type = "visible_viewpoint_relaxed"
        if visible_viewpoints:
            agent_state = self.agent.get_state()
            position_a = agent_state.position
            geo_dis = math.inf
            best_viewpoint = None

            for viewpoint in visible_viewpoints:
                path = habitat_sim.nav.ShortestPath()
                path.requested_end = np.array(viewpoint["position"], dtype=np.float32)
                path.requested_start = np.array(position_a, dtype=np.float32)
                if not self.pathfinder.find_path(path):
                    continue
                if path.geodesic_distance < geo_dis:
                    geo_dis = path.geodesic_distance
                    best_viewpoint = viewpoint

            if best_viewpoint is not None:
                return (
                    geo_dis,
                    np.array(best_viewpoint["position"], dtype=np.float32),
                    {
                        "center": np.array(best_viewpoint["center"], dtype=np.float32),
                        "semantic_id": best_viewpoint.get("semantic_id"),
                        "object_id": best_viewpoint.get("object_id"),
                        "goal_yaw": best_viewpoint.get("yaw"),
                        "goal_pixel_count": best_viewpoint.get("pixel_count"),
                        "goal_type": goal_type,
                    },
                )

        candidates = self.get_target_candidates(target_index)
        if not candidates:
            return math.inf, None, None

        if not getattr(self.args, "allow_occluded_goal_fallback", False):
            return (
                math.inf,
                None,
                {
                    "goal_type": "no_visible_viewpoint",
                    "candidate_count": len(candidates),
                    "object_ids": [candidate.get("object_id") for candidate in candidates],
                },
            )

        agent_state = self.agent.get_state()
        position_a = agent_state.position
        geo_dis = math.inf
        best_snap_coord = None
        best_candidate = None

        for candidate in candidates:
            snap_coord = self.pathfinder.snap_point(candidate["center"])
            if not self._is_valid_nav_point(snap_coord):
                continue
            snap_coord = np.array(snap_coord, dtype=np.float32)

            path = habitat_sim.nav.ShortestPath()
            path.requested_end = snap_coord
            path.requested_start = np.array(position_a, dtype=np.float32)

            if not self.pathfinder.find_path(path):
                continue
            if path.geodesic_distance < geo_dis:
                geo_dis = path.geodesic_distance
                best_snap_coord = snap_coord
                best_candidate = dict(candidate)
                best_candidate["goal_type"] = "object_center_snap"

        return geo_dis, best_snap_coord, best_candidate

    def get_target_visibility(self, target_index, min_pixels=25):
        required_pixels = max(1, int(min_pixels))
        candidates = self.get_target_candidates(target_index)
        semantic_ids = {
            candidate["semantic_id"]
            for candidate in candidates
            if candidate.get("semantic_id") is not None
        }
        if not semantic_ids:
            return {
                "visible": False,
                "reason": "no_target_semantic_ids",
                "candidate_count": len(candidates),
                "required_pixels": required_pixels,
                "max_pixels": 0,
                "visible_ids": [],
                "pixel_counts": {},
            }

        observations = self.sim.get_sensor_observations()
        semantic_obs = observations.get("semantic_sensor")
        if semantic_obs is None:
            return {
                "visible": False,
                "reason": "semantic_sensor_unavailable",
                "candidate_count": len(candidates),
                "required_pixels": required_pixels,
                "max_pixels": 0,
                "visible_ids": [],
                "pixel_counts": {},
            }

        visible_ids, visible_counts = np.unique(semantic_obs, return_counts=True)
        pixel_counts = {}
        max_pixels = 0
        matched_ids = []
        for semantic_id, pixel_count in zip(visible_ids, visible_counts):
            semantic_id = int(semantic_id)
            pixel_count = int(pixel_count)
            if semantic_id not in semantic_ids:
                continue
            pixel_counts[semantic_id] = pixel_count
            max_pixels = max(max_pixels, pixel_count)
            if pixel_count >= required_pixels:
                matched_ids.append(semantic_id)

        return {
            "visible": bool(matched_ids),
            "reason": "visible" if matched_ids else "below_pixel_threshold",
            "candidate_count": len(candidates),
            "required_pixels": required_pixels,
            "max_pixels": max_pixels,
            "visible_ids": matched_ids,
            "pixel_counts": pixel_counts,
        }

    def get_visibility_search_action(self, goal_center, angle_threshold=15.0):
        if goal_center is None:
            return "turn_left"

        agent_state = self.agent.get_state()
        delta_x = float(goal_center[0] - agent_state.position[0])
        delta_z = float(goal_center[2] - agent_state.position[2])
        if abs(delta_x) < 1e-6 and abs(delta_z) < 1e-6:
            return "turn_left"

        desired_yaw = math.degrees(math.atan2(delta_x, delta_z))
        delta_yaw = desired_yaw - self.yaw
        while delta_yaw <= -180:
            delta_yaw += 360
        while delta_yaw > 180:
            delta_yaw -= 360

        if delta_yaw > angle_threshold:
            return "turn_left"
        if delta_yaw < -angle_threshold:
            return "turn_right"
        return "move_forward"

    def get_goal_pose_alignment_action(self, goal_center, goal_yaw, angle_threshold=15.0):
        if goal_center is None or goal_yaw is None:
            return self.get_visibility_search_action(goal_center, angle_threshold=angle_threshold)

        agent_state = self.agent.get_state()
        planar_distance = math.dist(
            [float(agent_state.position[0]), float(agent_state.position[2])],
            [float(goal_center[0]), float(goal_center[2])],
        )
        if planar_distance > 0.6:
            return self.get_visibility_search_action(goal_center, angle_threshold=angle_threshold)

        delta_yaw = self._normalize_saved_yaw(goal_yaw - self.yaw)
        if delta_yaw > angle_threshold:
            return "turn_left"
        if delta_yaw < -angle_threshold:
            return "turn_right"
        return None

    def target_dis(self, coord_list):
        """
        Return the cloest object and the distance(Euler distance)
        Args:
            coord_list: the list of the coordinates of the target object
        Returns:
            min_dis: the distance between agent and target
            coord: the coordinate of the target object
        """
        agent_state = self.agent.get_state()
        agent_position = agent_state.position
        # compute the distance between agent and target
        dis = [math.dist(np.roll(agent_position, 1)[:-1], np.roll(coord, 1)[:-1]) for coord in coord_list]
        index, min_dis = min(enumerate(dis), key=operator.itemgetter(1))
        return min_dis, coord_list[index]

    def geodesic_distance(
        self,
        position_b_list
    ) -> float:
        agent_state = self.agent.get_state()
        position_a = agent_state.position
        # print(self.pathfinder.is_navigable(position_a))
        # print(self.pathfinder.is_navigable(position_b))
        geo_dis = math.inf
        coord = position_b_list[0]
        
        for position_b in position_b_list:
            path = habitat_sim.nav.ShortestPath()
            path.requested_end = np.array(
                np.array(position_b, dtype=np.float32)
            )

            path.requested_start = np.array(position_a, dtype=np.float32)

            if self.pathfinder.find_path(path):
                if path.geodesic_distance < geo_dis:
                    geo_dis = path.geodesic_distance
                    coord = position_b

        return geo_dis, coord
        # return path
    
    def get_info(self, success):
        """
        Return the info of the current state
        Args:
            success: the number of finished subtargets
        Returns:
            obj_target: the target object
            coord: the coordinate of the current navigation goal
            goal_meta: metadata of the current goal/viewpoint
            agent_state: the position of the agent
            yaw: the yaw of the agent
            geo_dis: the geodesic distance between agent and target goal
        """
        obj_target = self.target[success]
        geo_dis, snap_coord, best_candidate = self.get_goal_info(success)
        if snap_coord is None:
            if best_candidate is not None and best_candidate.get("goal_type") == "no_visible_viewpoint":
                print(
                    "No visible goal viewpoint found for target %s; skipping as unreachable. "
                    "Candidates: %s"
                    % (obj_target, best_candidate.get("object_ids", []))
                )
            return None, None, None, None, None, math.inf

        agent_state = self.agent.get_state()

        print("target coord: ", snap_coord)       
        print("agent_state: position ", agent_state.position, ", yaw ", self.yaw)
        print("%f meters from the %s" % (geo_dis, obj_target))

        return obj_target, snap_coord, best_candidate, agent_state.position, self.yaw, geo_dis
    
    def get_next_action(self, goal_pos) -> Optional[Union[int, np.ndarray]]:
        """Returns the next action along the shortest path."""
        assert self.follower is not None
        next_action = self.follower.next_action_along(goal_pos)
        return next_action
    
    def return_state(self):
        agent_state = self.agent.get_state()
        return agent_state.position, self.yaw


def simulate(sim, dt=1.0, get_frames=True):
    # simulate dt seconds at 60Hz to the nearest fixed timestep
    print("Simulating " + str(dt) + " world seconds.")
    observations = []
    start_time = sim.get_world_time()
    while sim.get_world_time() < start_time + dt:
        sim.step_physics(1.0 / 60.0)
        if get_frames:
            observations.append(sim.get_sensor_observations())
    return observations

# Set an object transform relative to the agent state
def set_object_state_from_agent(
    sim,
    obj,
    offset=np.array([0, 2.0, -1.5]),
    orientation=mn.Quaternion(((0, 0, 0), 1)),
):
    agent_transform = sim.agents[0].scene_node.transformation_matrix()
    ob_translation = agent_transform.transform_point(offset)
    obj.translation = ob_translation
    obj.rotation = orientation


# sample a random valid state for the object from the scene bounding box or navmesh
def sample_object_state(
    sim, obj, from_navmesh=True, maintain_object_up=True, max_tries=100, bb=None
):
    # check that the object is not STATIC
    if obj.motion_type is habitat_sim.physics.MotionType.STATIC:
        print("sample_object_state : Object is STATIC, aborting.")
    if from_navmesh:
        if not sim.pathfinder.is_loaded:
            print("sample_object_state : No pathfinder, aborting.")
            return False
    elif not bb:
        print(
            "sample_object_state : from_navmesh not specified and no bounding box provided, aborting."
        )
        return False
    tries = 0
    valid_placement = False
    # Note: following assumes sim was not reconfigured without close
    scene_collision_margin = stage_attr_mgr.get_template_by_id(0).margin
    while not valid_placement and tries < max_tries:
        tries += 1
        # initialize sample location to random point in scene bounding box
        sample_location = np.array([0, 0, 0])
        if from_navmesh:
            # query random navigable point
            sample_location = sim.pathfinder.get_random_navigable_point()
        else:
            sample_location = np.random.uniform(bb.min, bb.max)
        # set the test state
        obj.translation = sample_location
        if maintain_object_up:
            # random rotation only on the Y axis
            y_rotation = mn.Quaternion.rotation(
                mn.Rad(random.random() * 2 * math.pi), mn.Vector3(0, 1.0, 0)
            )
            obj.rotation = y_rotation * obj.rotation
        else:
            # unconstrained random rotation
            obj.rotation = ut.random_quaternion()

        # raise object such that lowest bounding box corner is above the navmesh sample point.
        if from_navmesh:
            obj_node = obj.root_scene_node
            xform_bb = habitat_sim.geo.get_transformed_bb(
                obj_node.cumulative_bb, obj_node.transformation
            )
            # also account for collision margin of the scene
            obj.translation += mn.Vector3(
                0, xform_bb.size_y() / 2.0 + scene_collision_margin, 0
            )

        # test for penetration with the environment
        if not sim.contact_test(obj.object_id):
            valid_placement = True

    if not valid_placement:
        return False
    return True
