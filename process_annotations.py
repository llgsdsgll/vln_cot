#!/usr/bin/env python3
"""
视频标注数据处理脚本

用途：
  处理视频标注JSON文件，将标注数据按帧整理，验证subtask和object是否属于原始指令。

使用方法：
  python3 process_annotations.py

输入文件：
  - gengshuang_1_H.264_03182.json: 包含视频标注数据
  - /mnt/data-cpfs/gengshuang/vln_data/vln_ce/raw_data/r2r/train/train.json: 原始指令数据

输出文件：
  - gengshuang_1_H.264_03182_processed.json: 处理后的帧级标注数据
    格式: [
      {
        "episode_id": 1,
        "video_url": "https://...",
        "instruction": "原始指令文本",
        "duration": 39,
        "frames": [
          {
            "frame": 1,
            "subtask": "subtask文本",
            "objects": [
              {"label": "object名称", "bbox": {"x": 0, "y": 0, "width": 0, "height": 0}}
            ]
          }
        ]
      }
    ]

  - gengshuang_1_H.264_03182_errors.txt: 错误和警告日志
    包含：缺少subtask的帧、subtask/object不匹配原指令的警告
"""
import json
import re
from collections import defaultdict

def extract_episode_id(url):
    """从URL中提取episode_id，如 ep00001 -> 1"""
    match = re.search(r'ep(\d+)', url)
    return int(match.group(1)) if match else None

def load_instructions(train_json_path):
    """加载原始指令，返回 {episode_id: instruction_text} 字典"""
    with open(train_json_path, 'r') as f:
        data = json.load(f)
    return {ep['episode_id']: ep['instruction']['instruction_text']
            for ep in data['episodes']}

def process_video(video_data, instructions, errors):
    """处理单个视频的标注数据"""
    # 提取视频信息
    video_key = list(video_data['data']['meta'].keys())[0]
    video_url = video_data['data'][video_key]
    video_meta = video_data['data']['meta'][video_key]

    episode_id = extract_episode_id(video_url)
    duration = int(video_meta['duration'])

    # 获取原始指令
    instruction = instructions.get(episode_id, "")
    if not instruction:
        errors.append(f"Episode {episode_id}: 未找到原始指令")

    # 初始化每帧数据
    frames = [{
        'frame': i,
        'subtask': None,
        'objects': []
    } for i in range(1, duration + 1)]

    # 收集所有subtask的帧范围
    subtask_ranges = []

    # 处理标注
    if video_data['annotations']:
        for result in video_data['annotations'][0]['result']:
            labels = result['value'].get('labels', [])
            sequence = result['value'].get('sequence', [])

            for label in labels:
                if label.startswith('subtask:'):
                    subtask_text = label[8:]  # 去掉 "subtask:" 前缀

                    # 验证subtask是否在原始指令中
                    if instruction and subtask_text.lower() not in instruction.lower():
                        errors.append(f"Episode {episode_id}: subtask '{subtask_text}' 不在原始指令中\n  原指令: {instruction}")

                    # 确定subtask的帧范围
                    enabled_frames = [s['frame'] for s in sequence if s.get('enabled', True)]
                    disabled_frames = [s['frame'] for s in sequence if not s.get('enabled', True)]

                    if enabled_frames:
                        start_frame = min(enabled_frames)
                        # 如果有disabled帧，使用最后一个disabled帧作为结束
                        # 否则延续到视频最后一帧
                        end_frame = max(disabled_frames) if disabled_frames else duration

                        subtask_ranges.append({
                            'text': subtask_text,
                            'start': start_frame,
                            'end': end_frame
                        })

                elif label.startswith('object:'):
                    object_name = label[7:]  # 去掉 "object:" 前缀

                    # 验证object是否在原始指令中
                    if instruction and object_name.lower() not in instruction.lower():
                        errors.append(f"Episode {episode_id}: object '{object_name}' 不在原始指令中\n  原指令: {instruction}")

                    # 将object分配到对应的帧
                    for seq in sequence:
                        if seq.get('enabled', True):
                            frame_idx = seq['frame'] - 1
                            if 0 <= frame_idx < duration:
                                frames[frame_idx]['objects'].append({
                                    'label': object_name,
                                    'bbox': {
                                        'x': seq['x'],
                                        'y': seq['y'],
                                        'width': seq['width'],
                                        'height': seq['height']
                                    }
                                })

    # 将subtask填充到对应的帧范围
    for subtask_range in subtask_ranges:
        for frame_num in range(subtask_range['start'], subtask_range['end'] + 1):
            if 1 <= frame_num <= duration:
                frames[frame_num - 1]['subtask'] = subtask_range['text']

    # 检查每帧是否有subtask
    for frame in frames:
        if frame['subtask'] is None:
            errors.append(f"Episode {episode_id}, Frame {frame['frame']}: 缺少subtask标注")

    return {
        'episode_id': episode_id,
        'video_url': video_url,
        'instruction': instruction,
        'duration': duration,
        'frames': frames
    }

def main():
    input_file = 'gengshuang_1_H.264_0324.json'
    train_file = '/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train.json'
    output_file = 'gengshuang_1_H.264_0324_processed.json'
    error_file = 'gengshuang_1_H.264_0324_errors.txt'

    # 加载数据
    with open(input_file, 'r') as f:
        data = json.load(f)

    instructions = load_instructions(train_file)

    # 处理所有视频
    results = []
    errors = []

    for video_data in data:
        result = process_video(video_data, instructions, errors)
        results.append(result)

    # 保存结果
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # 保存错误日志
    with open(error_file, 'w') as f:
        for error in errors:
            f.write(error + '\n')

    print(f"处理完成！")
    print(f"- 处理了 {len(results)} 个视频")
    print(f"- 结果保存到: {output_file}")
    print(f"- 错误日志: {error_file} ({len(errors)} 条)")

if __name__ == '__main__':
    main()
