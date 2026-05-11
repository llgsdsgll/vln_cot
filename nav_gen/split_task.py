import numpy as np
import random
import os 
import sys
import torch
import json
import logging
import shutil
import re
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from recognize_anything.ram.models import ram_plus
from recognize_anything.ram import inference_ram as inference
from recognize_anything.ram import get_transform
from gpt import gpt4o_mini, gpt4o
from tqdm import tqdm
from debug_scene_instance_boxes import generate_scene_instance_box_debug


def _cuda_runtime_supported():
    if not torch.cuda.is_available():
        return False, "CUDA is not available"

    try:
        capability = torch.cuda.get_device_capability(0)
        arch_list = torch.cuda.get_arch_list()
        sm_tag = f"sm_{capability[0]}{capability[1]}"
        if arch_list and sm_tag not in arch_list:
            return False, f"PyTorch does not support GPU arch {sm_tag}"

        x = torch.randn(8, 8, device='cuda')
        _ = x @ x
        return True, None
    except Exception as err:
        return False, str(err)


def _resolve_ram_device(args):
    requested = getattr(args, 'ram_device', 'auto')
    if requested != 'auto':
        device = torch.device(requested)
        if device.type == 'cuda':
            ok, reason = _cuda_runtime_supported()
            if not ok:
                raise RuntimeError(f"Requested RAM device `{requested}` is unavailable: {reason}")
        return device

    ok, reason = _cuda_runtime_supported()
    if ok:
        return torch.device('cuda')

    print(f"Falling back to CPU for RAM inference: {reason}")
    return torch.device('cpu')


def init_model(args):
    global transform 
    global ram_model 
    global device

    pretrained = args.ram_model
    device = _resolve_ram_device(args)
    
    transform = get_transform(image_size=384)

    ram_model = ram_plus(pretrained=pretrained,
                                image_size=384,
                                vit='swin_l')
    ram_model.eval()
    ram_model = ram_model.to(device)

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    # create file handler which logs even debug messages
    fh = logging.FileHandler(args.ram_logs)
    fh.setLevel(logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(fh)

    # redirect stdout and stderr to logging module
    sys.stdout = LoggerWriter(logger.info)
    sys.stderr = LoggerWriter(logger.error)


def _debug_safe_name(text, max_length=80):
    safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(text).strip())
    safe = safe.strip("._-")
    if not safe:
        safe = "item"
    if len(safe) > max_length:
        safe = safe[:max_length].rstrip("._-")
    return safe


def _save_json(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2, ensure_ascii=False)


def _get_split_debug_root(args):
    explicit = getattr(args, "split_debug_path", None)
    if explicit:
        return explicit

    ram_logs = getattr(args, "ram_logs", "")
    if ram_logs:
        return os.path.join(os.path.dirname(ram_logs), "split_traj_debug")

    return os.path.join(os.getcwd(), "logs", "split_traj_debug")


def _get_task_debug_dir(args, task_path):
    task_root = os.path.abspath(os.path.normpath(args.task_path))
    task_path = os.path.abspath(os.path.normpath(task_path))
    task_rel = os.path.relpath(task_path, task_root)
    return os.path.join(_get_split_debug_root(args), task_rel)


def _copy_debug_images(img_list, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    copied = []
    for index, src_path in enumerate(img_list):
        ext = os.path.splitext(src_path)[1] or ".png"
        dst_path = os.path.join(output_dir, f"front_{index:03d}{ext}")
        shutil.copy2(src_path, dst_path)
        copied.append(
            {
                "index": index,
                "source_image": src_path,
                "saved_image": dst_path,
            }
        )
    return copied


def _effective_render_sensor_height(args, task_path):
    if getattr(args, "render_sensor_height", None) is not None:
        return float(args.render_sensor_height)

    task_json = os.path.join(task_path, "success", "trial_1", "task.json")
    robot = None
    try:
        with open(task_json, "r", encoding="utf-8") as file:
            robot = str(json.load(file).get("Robot", "")).strip().lower()
    except Exception:
        robot = None

    return 0.5 if robot == "spot" else 1.0


def _build_debug_render_settings(args, task_path):
    return {
        "render_sensor_height_m": _effective_render_sensor_height(args, task_path),
        "sensor_hfov_deg": float(getattr(args, "sensor_hfov", 90.0)),
        "render_width": int(getattr(args, "render_width", 512)),
        "render_height": int(getattr(args, "render_height", 512)),
    }


def _build_scene_instance_options(args, debug_render_settings):
    return {
        "top_k": 5,
        "min_pixels": 25,
        "filter_profile": "strong",
        "extra_ignore_substrings": [],
        "render_width": int(debug_render_settings["render_width"]),
        "render_height": int(debug_render_settings["render_height"]),
        "render_sensor_height": float(debug_render_settings["render_sensor_height_m"]),
        "sensor_hfov": float(debug_render_settings["sensor_hfov_deg"]),
        "sim_gpu_device": int(getattr(args, "sim_gpu_device", 0)),
    }


def _read_json(path):
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def _step_top_instance_tags(step_record, top_k=5):
    return [
        item["tag"]
        for item in step_record.get("all_tag_rankings", [])[:top_k]
        if item.get("tag")
    ]


def _build_instruction_tags_from_scene_instances(trail, scene_instance_payload=None):
    target_category = trail[0]["target"]
    target_object_id = None
    step_lookup = {}

    if scene_instance_payload:
        selected_instance = (
            scene_instance_payload.get("declared_target_resolution", {})
            .get("selected_instance", {})
        )
        target_object_id = selected_instance.get("object_id")
        step_lookup = {
            int(step["step_index"]): step
            for step in scene_instance_payload.get("steps", [])
        }

    tags = {
        "target": {
            "category": target_category,
            "object_id": target_object_id or target_category,
        }
    }

    for local_step_index, seg in enumerate(trail):
        step_record = step_lookup.get(seg.get("step_index"), {})
        rankings = step_record.get("all_tag_rankings") or seg.get("all_tag_rankings", [])
        tags[f"step_{local_step_index}"] = {
            "action": seg["label"],
            "all_tag_rankings": [
                {
                    "tag": item.get("tag") or item.get("object_id"),
                    "count": int(item.get("count", item.get("frame_count", 0))),
                    "category": item.get("category"),
                }
                for item in rankings
                if item.get("tag") or item.get("object_id")
            ],
        }

    return tags


def _extract_json_payload_from_text(text):
    text = str(text or "").strip()
    if not text:
        raise ValueError("Empty model response.")

    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise
        return json.loads(text[start:end + 1])


def _pick_fallback_anchor_instance(rankings, target_object_id, action_label):
    if not rankings:
        return None, None

    deprioritized_categories = {"carpet", "rug", "lamp", "ornament", "vase"}
    normalized_action = str(action_label or "").lower()
    if "turn" not in normalized_action and "zigzag" not in normalized_action:
        deprioritized_categories = {"lamp", "ornament", "vase"}

    for rank_index, item in enumerate(rankings, start=1):
        tag = item.get("tag") or item.get("object_id")
        category = str(item.get("category") or "").strip().lower()
        if not tag or tag == target_object_id:
            continue
        if category in deprioritized_categories:
            continue
        return item, rank_index

    for rank_index, item in enumerate(rankings, start=1):
        tag = item.get("tag") or item.get("object_id")
        if tag and tag != target_object_id:
            return item, rank_index

    return rankings[0], 1


def _resolve_target_instance_metadata(tags, scene_instance_payload, response_payload):
    target_category = tags.get("target", {}).get("category")
    target_object_id = (
        response_payload.get("target_object_id")
        or tags.get("target", {}).get("object_id")
        or target_category
    )
    selected_instance = (
        (scene_instance_payload or {})
        .get("declared_target_resolution", {})
        .get("selected_instance", {})
    )

    if selected_instance:
        if target_object_id and selected_instance.get("object_id") != target_object_id:
            candidate_rankings = (
                (scene_instance_payload or {})
                .get("declared_target_resolution", {})
                .get("all_candidate_rankings", [])
            )
            for candidate in candidate_rankings:
                if candidate.get("object_id") == target_object_id or candidate.get("tag") == target_object_id:
                    selected_instance = candidate
                    break

        return {
            "object_id": selected_instance.get("object_id") or target_object_id,
            "semantic_id": selected_instance.get("semantic_id"),
            "category": selected_instance.get("category") or target_category,
            "count": int(selected_instance.get("count", selected_instance.get("frame_count", 0))),
        }

    return {
        "object_id": target_object_id if target_object_id and target_object_id != target_category else None,
        "semantic_id": None,
        "category": target_category,
        "count": 0,
    }


def _resolve_step_anchor_instances(trail, response_payload, target_instance):
    raw_step_anchors = response_payload.get("step_anchors") or []
    step_anchor_lookup = {}
    for item in raw_step_anchors:
        try:
            step_index = int(item.get("step_index"))
        except (TypeError, ValueError):
            continue
        step_anchor_lookup[step_index] = item

    resolved_anchors = []
    for local_step_index, seg in enumerate(trail):
        rankings = seg.get("all_tag_rankings", [])
        raw_anchor = step_anchor_lookup.get(local_step_index, {})
        requested_tag = str(raw_anchor.get("anchor_object_id") or "").strip()

        selected_item = None
        selected_rank = None
        if requested_tag:
            for rank_index, item in enumerate(rankings, start=1):
                tag = str(item.get("tag") or item.get("object_id") or "").strip()
                if tag == requested_tag:
                    selected_item = item
                    selected_rank = rank_index
                    break

        source = "llm_selected_instance"
        if selected_item is None:
            selected_item, selected_rank = _pick_fallback_anchor_instance(
                rankings,
                target_instance.get("object_id"),
                seg.get("label"),
            )
            source = "fallback_ranked_instance"

        if selected_item is None:
            continue

        resolved_anchors.append(
            {
                "step_index": local_step_index,
                "action": seg["label"],
                "object_id": selected_item.get("object_id") or selected_item.get("tag"),
                "semantic_id": selected_item.get("semantic_id"),
                "category": selected_item.get("category"),
                "rank": int(selected_rank),
                "count": int(selected_item.get("count", selected_item.get("frame_count", 0))),
                "source": source,
            }
        )

    return resolved_anchors


def _generate_step_instruction_bundle(args, trail, tags, scene_instance_payload):
    response_text = gpt4o_mini(
        args,
        args.prompt_path + "gen_step_task_from_instances_structured.txt",
        json.dumps(tags, ensure_ascii=False),
    )
    try:
        response_payload = _extract_json_payload_from_text(response_text)
        instruction = str(response_payload.get("instruction", "")).strip()
        if not instruction:
            raise ValueError("Model response missing `instruction`.")
    except Exception as err:
        print(f"Structured step-task response parsing failed, falling back to legacy instruction prompt: {err}")
        instruction = gpt4o_mini(
            args,
            args.prompt_path + "gen_step_task_from_instances.txt",
            json.dumps(tags, ensure_ascii=False),
        ).strip()
        response_payload = {}

    target_instance = _resolve_target_instance_metadata(tags, scene_instance_payload, response_payload)
    step_anchor_instances = _resolve_step_anchor_instances(trail, response_payload, target_instance)

    return {
        "instruction": instruction,
        "target_instance": target_instance,
        "step_anchor_instances": step_anchor_instances,
        "raw_model_response": response_text,
    }

class TrainDataset(Dataset):
    def __init__(self, task_data, max_len=None):
        self.data_path = task_data
        self.max_len = max_len

        self.task_list = self.make_task_list()

    def make_task_list(self):
        task_list = []
        task_type = os.listdir(self.data_path)
        for type_ in task_type:
            type_dir = self.data_path + type_
            if not os.path.isdir(type_dir):
                continue
            type_task_list = os.listdir(type_dir)
            for type_task in type_task_list:
                task_dir = os.path.join(type_dir, type_task)
                if not os.path.isdir(task_dir):
                    continue
                task_list.append(type_ + '/' + type_task)
        
        if self.max_len:
            task_list = task_list[:self.max_len]

        return task_list

    def make_task_dic(self):
        task_dic = {}
        task_type = os.listdir(self.data_path)
        for type_ in task_type:
            type_dir = self.data_path + type_
            if not os.path.isdir(type_dir):
                continue
            task_dic[type_] = [
                type_task
                for type_task in os.listdir(type_dir)
                if os.path.isdir(os.path.join(type_dir, type_task))
            ]

        return task_dic
    
    def __getitem__(self, index):
        # return the path of the task
        task_path = self.data_path + self.task_list[index]
        return task_path


def get_trail(path):
    action_path = path + '/success/trial_1'
    if not os.path.isdir(action_path):
        return None, None
    actions = os.listdir(action_path)
    action_dic = {}
    for action in actions:
        if 'task' not in action:
            action_dic[int(action.split('_')[0])] = ["_".join(action.split('_')[1:-2]), action.split('_')[-1]]  
    obs_dic = {}

    for key, value in action_dic.items():
        if key == -1:
            continue

        img_list = os.listdir(action_path + '/' + str(key-1) + "_" + action_dic[key-1][0] + "_for_" + action_dic[key-1][1])
        obs = {}
        for img in img_list:
            img_name = img.split('.')[0]
            img_path = action_path + '/' + str(key-1) + "_" + action_dic[key-1][0] + "_for_" + action_dic[key-1][1] + '/' + img
            obs[img_name] = img_path

        obs_dic[key] = obs

    action_ = action_dic[key][0]
    obj_ = action_dic[key][1]

    removed_value = action_dic.pop(-1, None)

    sorted_action_dic = sorted(action_dic.items(), key=lambda item: item[0])
    sorted_obs_dic = sorted(obs_dic.items(), key=lambda item: item[0])

    return dict(sorted_action_dic), dict(sorted_obs_dic)

def batch_ram(img_list, return_details=False):
    tag_dic = {}
    for img in img_list:
        image = transform(Image.open(img)).unsqueeze(0).to(device)
        res = inference(image, ram_model)
        tags = res[0].split(' | ')
        for tag in tags:
            if any(sub in tag for sub in ['ceiling', 'floor']):
                continue
            if tag not in tag_dic:
                tag_dic[tag] = 1
            else:
                tag_dic[tag] += 1

    sorted_tags = sorted(tag_dic.items(), key=lambda item: item[1], reverse=True)
    top_five_tags = [key for key, value in sorted_tags[:5]]

    if return_details:
        return top_five_tags, [{"tag": key, "count": value} for key, value in sorted_tags]
    return top_five_tags

def segment_trajectory(
    trajectory,
    obs_dic,
    task,
    key,
    subtraj_debug_dir=None,
    debug_render_settings=None,
    scene_instance_options=None,
):
    target = trajectory[0][1]
    trajectory = [tjc[0] for tjc in trajectory]
    n = len(trajectory)
    segments = []
    
    # Step 1: Identify same direction segments
    i = 0
    while i <= n - 3:
        window = trajectory[i:i + 3]
        if window.count('turn_left') >= 2:
            left_indices = [idx for idx, action in enumerate(window) if action == 'turn_left']
            start = i + left_indices[0]
            end = i + left_indices[-1]
            segments.append((start, end, 'turn_left'))
            i = end + 1  # Move past this segment
        else:
            i += 1
    i = 0
    while i <= n - 3:
        window = trajectory[i:i + 3]
        if window.count('turn_right') >= 2:
            right_indices = [idx for idx, action in enumerate(window) if action == 'turn_right']
            start = i + right_indices[0]
            end = i + right_indices[-1]
            segments.append((start, end, 'turn_right'))
            i = end + 1  # Move past this segment
        else:
            i += 1
            
    # Step 2: Merge overlapping or adjacent same direction segments
    merged_segments = []
    if segments:
        segments.sort()
        current_start, current_end, current_label = segments[0]
        for segment in segments[1:]:
            start, end, label = segment
            if start <= current_end + 3 and label == current_label:
                # merge segments
                current_end = max(current_end, end)
            else:
                merged_segments.append((current_start, current_end, current_label))
                current_start, current_end, current_label = segment
                
        merged_segments.append((current_start, current_end, current_label))
                
    # Step 3: Handle overlapping different direction segments
    result_segments = []
    temp = merged_segments[:]
            
    while len(temp) > 1:
        curr_start, curr_end, curr_label = temp.pop(0)
        next_start, next_end, next_label = temp.pop(0)
        if curr_end + 1 >= next_start:
            # overlapping segments
            new_start = min(curr_start, next_start)
            new_end = max(curr_end, next_end)
            result_segments.append((new_start, new_end, "zigzag"))
        else:
            result_segments.append((curr_start, curr_end, curr_label))
            temp.insert(0, (next_start, next_end, next_label))
    if temp:
        result_segments += temp

    # Step 4: Mark `move_forward` segment 
    final_segments = []
    last_end = -1    
    for seg_start, seg_end, label in result_segments:
        if last_end + 2 < seg_start:
            final_segments.append({
                "trajectory": task,
                "start": last_end + 1 + key,
                "end": seg_start - 2 + key,
                "obs": [obs_dic[i]['front'] for i in range(last_end + 1 + key, seg_start - 1 + key)],
                "label": "move_forward",
                "target": target   
                })
        final_segments.append({
            "trajectory": task,
            "start": seg_start - 1 + key,
            "end": seg_end + key,
            "obs": [obs_dic[i]['front'] for i in range(seg_start - 1 + key, seg_end + key + 1)],
            "label": label if trajectory[seg_start - 1 : seg_end + 1].count(label) <=3 else "make a " + label.split("_")[-1] + " turn",
            "target": target
                })
        last_end = seg_end

    # Append last `move_forward` segment if needed
    if last_end < n - 1:
        final_segments.append({
            "trajectory": task,
            "start": last_end + 1 + key,
            "end": n - 1 + key,
            "obs": [obs_dic[i]['front'] for i in range(last_end + 1 + key, n + key)],
            "label": "move_forward",
            "target": target
            })

    result2_steps = []
    result3_steps = []
    for step_index, seg in enumerate(final_segments):
        seg["step_index"] = step_index
        seg["debug_subtraj_dir"] = subtraj_debug_dir
        seg["scene tags"] = []
        seg["all_tag_rankings"] = []
        result2_steps.append(
            {
                "step_index": step_index,
                "label": seg["label"],
                "target": seg["target"],
                "start": seg["start"],
                "end": seg["end"],
                "front_image_count": len(seg["obs"]),
            }
        )
        if subtraj_debug_dir:
            step_debug_dir = os.path.join(
                subtraj_debug_dir,
                "result3_step_images",
                f"step_{step_index:03d}_{_debug_safe_name(seg['label'])}",
            )
            copied_images = _copy_debug_images(seg["obs"], step_debug_dir)
            result3_steps.append(
                {
                    "step_index": step_index,
                    "label": seg["label"],
                    "target": seg["target"],
                    "start": seg["start"],
                    "end": seg["end"],
                    "top_5_scene_tags": [],
                    "all_tag_rankings": [],
                    "saved_image_dir": step_debug_dir,
                    "images": copied_images,
                }
            )
        del seg['obs']

    if subtraj_debug_dir:
        result2_json_path = os.path.join(subtraj_debug_dir, "result2_segmented_steps.json")
        result3_json_path = os.path.join(subtraj_debug_dir, "result3_step_tags_and_images.json")
        _save_json(
            result2_json_path,
            {
                "source_task_path": task,
                "target": target,
                "subtrajectory_global_start": key,
                "debug_render_settings": debug_render_settings,
                "compressed_steps": result2_steps,
            },
        )
        _save_json(
            result3_json_path,
            {
                "source_task_path": task,
                "target": target,
                "subtrajectory_global_start": key,
                "debug_render_settings": debug_render_settings,
                "tag_source": "scene_instance_object_id_rankings",
                "steps": result3_steps,
            },
        )
        if scene_instance_options is not None:
            result4_json_path = generate_scene_instance_box_debug(
                result3_json_path,
                **scene_instance_options,
            )
            result4_payload = _read_json(result4_json_path)
            step_lookup = {
                int(step["step_index"]): step
                for step in result4_payload.get("steps", [])
            }
            for seg in final_segments:
                step_record = step_lookup.get(seg["step_index"], {})
                seg["scene tags"] = _step_top_instance_tags(step_record)
                seg["all_tag_rankings"] = step_record.get("all_tag_rankings", [])
                seg["scene_instance_result_json"] = str(result4_json_path)
            for step_record in result3_steps:
                ranked_step = step_lookup.get(step_record["step_index"], {})
                step_record["top_5_scene_tags"] = ranked_step.get("top_5_scene_tags", [])
                step_record["all_tag_rankings"] = ranked_step.get("all_tag_rankings", [])
            _save_json(
                result3_json_path,
                {
                    "source_task_path": task,
                    "target": target,
                    "subtrajectory_global_start": key,
                    "debug_render_settings": debug_render_settings,
                    "tag_source": "scene_instance_object_id_rankings",
                    "scene_instance_result_json": str(result4_json_path),
                    "steps": result3_steps,
                },
            )

    return final_segments

def make_task(args, trail_list):
    for trail in tqdm(trail_list, desc="Make Task"):
        scene_instance_payload = None
        scene_instance_result_json = trail[0].get("scene_instance_result_json")
        if scene_instance_result_json and os.path.exists(scene_instance_result_json):
            scene_instance_payload = _read_json(scene_instance_result_json)
        tags = _build_instruction_tags_from_scene_instances(trail, scene_instance_payload)
        task = {}
        with open(trail[0]['trajectory'] + "/success/trial_1/task.json", "r", encoding='utf-8') as r:
            config = json.load(r)
        task["trajectory path"] = '/'.join(trail[0]['trajectory'].split('/')[-3:]) + '/success/trial_1'
        task["start"] = trail[0]['start']
        task["end"] = trail[-1]['end']
        task["Robot"] = config['Robot']
        task["Scene"] = config['Scene']
        task["target"] = trail[0]['target']
        index = config["Object"].index(task["target"])
        task["target"] = [config["Object"][index]]
        task['Region'] = [config["Region"][index]]
        step_former = -1
        for i in range(0, index):
            trail_name = 'trial_' + str(i)
            step_former += (len(config["trial"][trail_name]['pos'])-1)
        task['start_pos'] = config["trial"]['trial_' + str(index)]['pos'][task["start"]-step_former-1]
        task['start_yaw'] = config["trial"]['trial_' + str(index)]['yaw'][task["start"]-step_former-1]
        instruction_bundle = _generate_step_instruction_bundle(
            args,
            trail,
            tags,
            scene_instance_payload,
        )
        task["Task instruction"] = instruction_bundle["instruction"]
        task["instruction_target_instance"] = instruction_bundle["target_instance"]
        task["instruction_anchor_instances"] = instruction_bundle["step_anchor_instances"]
        task["instruction_object_ids"] = list(
            dict.fromkeys(
                [
                    item["object_id"]
                    for item in (
                        instruction_bundle["step_anchor_instances"]
                        + [instruction_bundle["target_instance"]]
                    )
                    if item.get("object_id")
                ]
            )
        )
        task["instruction_instance_source"] = "scene_instance_object_id_rankings"
        task["scene_instance_result_json"] = scene_instance_result_json
        print(task)

        task_root_path = args.step_task_path
        if len(task["Task instruction"]) > 100:
            ins_path = task["Task instruction"][:100]
        else:
            ins_path = task["Task instruction"]
        task_path = task_root_path + ins_path + '.json' if ins_path[-1] != ' ' else task_root_path + ins_path[:-1] + '.json'
        _save_json(task_path, task)

class LoggerWriter:
    def __init__(self, level):
        # self.level is really like using log.debug(message)
        # at least in my case
        self.level = level

    def write(self, message):
        # if statement reduces the amount of newlines that are
        # printed to the logger
        if message != '\n':
            self.level(message)

    def flush(self):
        # create a flush method so things can be flushed when
        # the system wants to. Not sure if simply 'printing'
        # sys.stderr is the correct way to do it, but it seemed
        # to work properly for me.
        self.level(sys.stderr)
         

def split_traj(args):
    task_data = args.task_path
    task_dataset = TrainDataset(task_data)
    debug_root = _get_split_debug_root(args)
    os.makedirs(debug_root, exist_ok=True)

    trail_list = []
    for task in tqdm(task_dataset, desc="Segment Trajectory"):
        print(task)
        action_dic, obs_dic = get_trail(task)
        if not action_dic or not obs_dic:
            print(f"Skip task without successful trajectory: {task}")
            continue
        debug_render_settings = _build_debug_render_settings(args, task)
        scene_instance_options = _build_scene_instance_options(args, debug_render_settings)
        task_debug_dir = _get_task_debug_dir(args, task)
        full_action_sequence = []
        for step_index in sorted(action_dic.keys()):
            full_action_sequence.append(
                {
                    "step_index": step_index,
                    "action": action_dic[step_index][0],
                    "target": action_dic[step_index][1],
                    "front_image": obs_dic.get(step_index, {}).get("front"),
                }
            )
        debug_subtrajectories = []
        # print(obs_dic)
        # print(action_dic)
        start = 0
        end = start + 1
        target = action_dic[start][1]
        while end < len(obs_dic):
            # 完成一个轨迹
            if action_dic[end][1] != target or end == len(obs_dic) - 1:
                if end - start > 5:
                    trail = [action_dic[i] for i in range(start, end)]
                    subtraj_index = len(debug_subtrajectories)
                    subtraj_debug_dir = os.path.join(
                        task_debug_dir,
                        f"subtraj_{subtraj_index:03d}_{_debug_safe_name(target)}",
                    )
                    raw_actions = []
                    for action_step in range(start, end):
                        raw_actions.append(
                            {
                                "step_index": action_step,
                                "action": action_dic[action_step][0],
                                "target": action_dic[action_step][1],
                                "front_image": obs_dic.get(action_step, {}).get("front"),
                            }
                        )
                    patch_trail = segment_trajectory(
                        trail,
                        obs_dic,
                        task,
                        start,
                        subtraj_debug_dir=subtraj_debug_dir,
                        debug_render_settings=debug_render_settings,
                        scene_instance_options=scene_instance_options,
                    )
                    trail_list.append(patch_trail)
                    debug_subtrajectories.append(
                        {
                            "subtrajectory_index": subtraj_index,
                            "target": target,
                            "global_start_step": start,
                            "global_end_step": end - 1,
                            "raw_action_count": len(raw_actions),
                            "raw_actions": raw_actions,
                            "compressed_step_count": len(patch_trail),
                            "debug_dir": subtraj_debug_dir,
                        }
                    )
                start = end
                end += 1
                target = action_dic[start][1]
            if end - start == 1:
                if  start < len(obs_dic) - 3:
                    if [action_dic[i][0] for i in range(start, start+3)].count('move_forward') == 3:
                        start -= 1
                start += 1

            end += 1  

        _save_json(
            os.path.join(task_debug_dir, "result1_target_split.json"),
            {
                "source_task_path": task,
                "success_trial_dir": os.path.join(task, "success", "trial_1"),
                "debug_render_settings": debug_render_settings,
                "full_action_sequence": full_action_sequence,
                "subtrajectories": debug_subtrajectories,
            },
        )
              
    with open(args.split_save_path, 'w') as file:
        for item in trail_list:
            file.write(f'{item}\n')

def gen_step_task(args):
    import ast
    if not os.path.exists(args.split_save_path):
        print(f"Split trajectory file not found, skip step-task generation: {args.split_save_path}")
        return

    with open(args.split_save_path, 'r') as file:
        trail_list = [ast.literal_eval(line.strip()) for line in file]
    if not trail_list:
        print(f"No segmented trajectories found in {args.split_save_path}, skip step-task generation")
        return

    # print(trail_list)
    make_task(args, trail_list)
