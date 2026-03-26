#!/usr/bin/env python3
import json
import re
from collections import defaultdict

def extract_episode_id(url):
    match = re.search(r'ep(\d+)', url)
    return int(match.group(1)) if match else None

def load_instructions(train_json_path):
    with open(train_json_path, 'r') as f:
        data = json.load(f)
    return {ep['episode_id']: ep['instruction']['instruction_text']
            for ep in data['episodes']}

def process_video(video_data, instructions, errors):
    video_key = list(video_data['data']['meta'].keys())[0]
    video_url = video_data['data'][video_key]
    video_meta = video_data['data']['meta'][video_key]

    episode_id = extract_episode_id(video_url)

    # 从meta获取duration，如果没有则从annotations的value中获取
    duration = video_meta.get('duration')
    if duration is None and video_data['annotations']:
        for result in video_data['annotations'][0]['result']:
            if 'duration' in result['value']:
                duration = result['value']['duration']
                break
    duration = int(duration) if duration else 0

    instruction = instructions.get(episode_id, "")

    if not instruction:
        errors.append(f"Episode {episode_id}: 未找到原始指令")

    frames = [{'frame': i, 'subtask': None, 'objects': []}
              for i in range(1, duration + 1)]

    subtask_ranges = []

    if video_data['annotations']:
        for result in video_data['annotations'][0]['result']:
            labels = result['value'].get('labels', [])
            sequence = result['value'].get('sequence', [])

            for label in labels:
                if label.startswith('subtask:'):
                    subtask_text = label[8:]

                    if instruction and subtask_text.lower() not in instruction.lower():
                        errors.append(f"Episode {episode_id}: subtask '{subtask_text}' 不在原始指令中")

                    enabled_frames = [s['frame'] for s in sequence if s.get('enabled', True)]
                    disabled_frames = [s['frame'] for s in sequence if not s.get('enabled', True)]

                    if enabled_frames:
                        start_frame = min(enabled_frames)
                        end_frame = max(disabled_frames) if disabled_frames else duration
                        subtask_ranges.append({'text': subtask_text, 'start': start_frame, 'end': end_frame})

                elif label.startswith('object:'):
                    object_name = label[7:]

                    if instruction and object_name.lower() not in instruction.lower():
                        errors.append(f"Episode {episode_id}: object '{object_name}' 不在原始指令中")

                    for seq in sequence:
                        if seq.get('enabled', True):
                            frame_idx = seq['frame'] - 1
                            if 0 <= frame_idx < duration:
                                frames[frame_idx]['objects'].append({
                                    'label': object_name,
                                    'bbox': {'x': seq['x'], 'y': seq['y'],
                                            'width': seq['width'], 'height': seq['height']}
                                })

    for subtask_range in subtask_ranges:
        for frame_num in range(subtask_range['start'], subtask_range['end'] + 1):
            if 1 <= frame_num <= duration:
                frames[frame_num - 1]['subtask'] = subtask_range['text']

    for frame in frames:
        if frame['subtask'] is None:
            errors.append(f"Episode {episode_id}, Frame {frame['frame']}: 缺少subtask标注")

    return {'episode_id': episode_id, 'video_url': video_url,
            'instruction': instruction, 'duration': duration, 'frames': frames}

def main():
    train_file = '/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json'

    instructions = load_instructions(train_file)

    all_results = []
    all_errors = []

    # 处理gengshuang_3和gengshuang_5
    for input_file in ['gengshuang_3_H.264.json', 'gengshuang_5_H.264.json']:
        with open(input_file, 'r') as f:
            data = json.load(f)

        for video_data in data:
            result = process_video(video_data, instructions, all_errors)
            all_results.append(result)

    # 加载已处理的gengshuang_1
    with open('gengshuang_1_H.264_0324_processed.json', 'r') as f:
        processed_data = json.load(f)

    all_results.extend(processed_data)

    # 按episode_id从大到小排序
    all_results.sort(key=lambda x: x['episode_id'], reverse=True)

    # 保存结果
    with open('merged_annotations.json', 'w') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    with open('merged_annotations_errors.txt', 'w') as f:
        for error in all_errors:
            f.write(error + '\n')

    print(f"处理完成！")
    print(f"- 总共处理了 {len(all_results)} 个视频")
    print(f"- 结果保存到: merged_annotations.json")
    print(f"- 错误日志: merged_annotations_errors.txt ({len(all_errors)} 条)")

if __name__ == '__main__':
    main()
