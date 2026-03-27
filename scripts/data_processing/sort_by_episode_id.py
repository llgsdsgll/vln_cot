'''
将合并后的annotations按episode_id排序
使用方法：
  python3 scripts/data_processing/sort_by_episode_id.py

默认输入：
  - data/processed/merged_annotations_0327.json

默认输出：
  - data/processed/merged_annotations_0327_sorted.json
'''

import json

input_file = '/home/gs/my_project/human_ann_cot/data/processed/merged_annotations_0327.json'
output_file = '/home/gs/my_project/human_ann_cot/data/processed/merged_annotations_0327_sorted.json'

with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

data.sort(key=lambda x: x['episode_id'])

with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"已排序完成，共 {len(data)} 条记录")
print(f"episode_id 范围: {data[0]['episode_id']} - {data[-1]['episode_id']}")
