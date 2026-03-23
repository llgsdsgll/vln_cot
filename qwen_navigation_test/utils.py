"""工具函数"""

import json
import os


def save_results(results, output_path):
    """保存测试结果"""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)


def load_episodes(split_file):
    """加载测试episodes"""
    # TODO: 实现episode加载逻辑
    pass
