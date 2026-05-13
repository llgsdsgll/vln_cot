import habitat_sim
import math
import magnum as mn

def make_setting(args, scene_file, robot):
    # test_scene = "102343992"
    if int(scene_file[2]) < 8:
        split = 'train/'
    else:
        split = 'val/'
    test_scene = args.scene + split + scene_file #+ '/' + scene_file.split('-')[-1] + '.semantic.glb'
    scene_dataset = args.scene_dataset

    rgb_sensor = True  # @param {type:"boolean"}
    front_rgb_only = bool(getattr(args, "front_rgb_only", False))
    front_semantic_sensor = bool(getattr(args, "front_semantic_sensor", False))
    front_depth_sensor = bool(getattr(args, "front_depth_sensor", False))
    depth_sensor = not front_rgb_only  # @param {type:"boolean"}
    semantic_sensor = front_semantic_sensor or not front_rgb_only  # @param {type:"boolean"}
    front_depth_enabled = front_depth_sensor or depth_sensor

    if robot == 'spot':
        default_sensor_height = 0.5
    else:
        default_sensor_height = 1.0
    width = int(getattr(args, "render_width", 512))
    height = int(getattr(args, "render_height", 512))
    sensor_height_arg = getattr(args, "render_sensor_height", None)
    sensor_height = float(default_sensor_height if sensor_height_arg is None else sensor_height_arg)
    sensor_hfov = float(getattr(args, "sensor_hfov", 90.0))
    sim_settings = {
        "width": width,  # Spatial resolution of the observations
        "height": height,
        "scene": test_scene,  # Scene path
        "scene_dataset": scene_dataset,
        "sim_gpu_device": getattr(args, "sim_gpu_device", 0),
        "default_agent": 0,
        "sensor_height": sensor_height,  # Height of sensors in meters
        "sensor_hfov": sensor_hfov,  # Horizontal FOV in degrees
        "color_sensor_f": rgb_sensor,  # RGB sensor
        "color_sensor_l": rgb_sensor and not front_rgb_only,  # RGB sensor
        "color_sensor_r": rgb_sensor and not front_rgb_only,  # RGB sensor
        "color_sensor_3rd": rgb_sensor and not front_rgb_only,  # RGB sensor
        "depth_sensor_f": front_depth_enabled,  # depth sensor
        "depth_sensor_l": depth_sensor,  # depth sensor
        "depth_sensor_r": depth_sensor,  # depth sensor
        "semantic_sensor": semantic_sensor,  # Semantic sensor
        "seed": 1,  # used in the random navigation
        "enable_physics": False,  # kinematics only
    }
    return sim_settings

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = settings.get("sim_gpu_device", 0)
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = settings["enable_physics"]

    if "scene_dataset" in settings:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset"]

    # Note: all sensors must have the same resolution
    sensors = {
        "color_sensor_f": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, 0.0, 0.0],
        },
        "color_sensor_l": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, math.pi / 3.0, 0.0],
        },
        "color_sensor_r": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, -math.pi / 3.0, 0.0],
        },
        "color_sensor_3rd": {
            "sensor_type": habitat_sim.SensorType.COLOR,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(
                0.0,
                settings["sensor_height"] + 0.5,
                1.0,
            ),
            "orientation": [-math.pi / 4, 0.0, 0.0],
        },
        "depth_sensor_l": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, math.pi / 3.0, 0.0],
        },
        "depth_sensor_f": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, 0.0, 0.0],
        },
        "depth_sensor_r": {
            "sensor_type": habitat_sim.SensorType.DEPTH,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, -math.pi / 3.0, 0.0],
        },
        "semantic_sensor": {
            "sensor_type": habitat_sim.SensorType.SEMANTIC,
            "resolution": [settings["height"], settings["width"]],
            "position": mn.Vector3(0.0, settings["sensor_height"], 0.0),
            "orientation": [0.0, 0.0, 0.0],
        },
    }
    sensor_specs = []
    for sensor_uuid, sensor_params in sensors.items():
        if settings[sensor_uuid]:
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = sensor_uuid
            sensor_spec.sensor_type = sensor_params["sensor_type"]
            sensor_spec.resolution = sensor_params["resolution"]
            sensor_spec.position = sensor_params["position"]
            sensor_spec.orientation = sensor_params["orientation"]
            if "sensor_hfov" in settings:
                sensor_spec.hfov = mn.Deg(settings["sensor_hfov"])
            if sensor_uuid == "color_sensor_3rd":
                sensor_spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            sensor_specs.append(sensor_spec)
    # Here you can specify the amount of displacement in a forward action and the turn angle
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        # "stop": habitat_sim.agent.ActionSpec("stop"),
    }
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])
