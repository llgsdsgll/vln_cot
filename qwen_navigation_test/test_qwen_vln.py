#!/usr/bin/env python3
"""
Qwen3.5 VLN-CE Navigation Test Script
测试 Qwen3.5 模型在 VLN-CE 任务上的导航能力
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch
from habitat import Env, logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from tqdm import tqdm

from vlnce_baselines.config.default import get_config


class QwenNavigationAgent:
    """Qwen3.5 导航智能体"""

    def __init__(self, model_path=None):
        """
        初始化 Qwen3.5 模型

        Args:
            model_path: Qwen3.5 模型路径
        """
        self.model_path = model_path
        self.actions = [
            HabitatSimActions.STOP,
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]

        # TODO: 加载 Qwen3.5 模型
        # from transformers import AutoModelForCausalLM, AutoTokenizer
        # self.model = AutoModelForCausalLM.from_pretrained(model_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        logger.info(f"Initializing Qwen3.5 agent with model: {model_path}")
        self.step_count = 0

    def act(self, observations, instruction):
        """
        根据观察和指令生成动作

        Args:
            observations: 环境观察 (RGB, depth, etc.)
            instruction: 导航指令文本

        Returns:
            action: 动作字典 {"action": action_id}
        """
        # TODO: 实现 Qwen3.5 推理逻辑
        # 示例实现：随机选择动作（需要替换为实际的 Qwen3.5 推理）

        # 1. 处理视觉观察
        # rgb = observations.get('rgb', None)
        # depth = observations.get('depth', None)

        # 2. 构建提示词
        # prompt = f"Instruction: {instruction}\nCurrent step: {self.step_count}\nWhat action should I take?"

        # 3. 使用 Qwen3.5 生成动作
        # inputs = self.tokenizer(prompt, return_tensors="pt")
        # outputs = self.model.generate(**inputs)
        # action_text = self.tokenizer.decode(outputs[0])
        # action = self._parse_action(action_text)

        # 临时实现：简单的启发式策略
        if self.step_count < 30:
            action = HabitatSimActions.MOVE_FORWARD
        else:
            action = HabitatSimActions.STOP

        self.step_count += 1
        return {"action": action}

    def reset(self):
        """重置智能体状态"""
        self.step_count = 0


def setup_config(config, split):
    """参考 nonlearning_agents.py 的配置方式，设置评估所需的环境配置"""
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.TASK_CONFIG.TASK.NDTW.SPLIT = split
    config.TASK_CONFIG.TASK.SDTW.SPLIT = split
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.freeze()
    return config


def get_instruction(episode):
    """从 episode 中提取导航指令文本"""
    # habitat 中指令存储在 episode.instruction.instruction_text
    return episode.instruction.instruction_text


def print_metrics(stats, split):
    """打印评估指标"""
    logger.info(f"\n{'='*50}")
    logger.info(f"Evaluation Results on split: {split}")
    logger.info(f"{'='*50}")
    for key, val in stats.items():
        logger.info(f"  {key:30s}: {val:.4f}")
    logger.info(f"{'='*50}\n")


def test_navigation(config, agent, split, num_episodes, output_dir):
    """
    运行导航测试，参考 evaluate_agent() 的结构

    Args:
        config: 配置对象
        agent: Qwen 导航智能体
        split: 数据集划分 (val_seen / val_unseen)
        num_episodes: 测试的 episode 数量，-1 表示全部
        output_dir: 结果保存目录
    """
    # 配置环境（参考 nonlearning_agents.py）
    config = setup_config(config, split)

    # 初始化 Habitat 环境
    env = Env(config=config.TASK_CONFIG)

    # 确定测试数量
    total = len(env.episodes)
    num_episodes = total if num_episodes == -1 else min(num_episodes, total)
    logger.info(f"Testing on {num_episodes}/{total} episodes (split: {split})")

    # 累计评估指标（由 habitat 环境自动计算）
    stats = defaultdict(float)
    episode_results = []

    for _ in tqdm(range(num_episodes), desc=f"[eval:{split}]"):
        obs = env.reset()
        agent.reset()

        # 获取当前 episode 的导航指令
        instruction = get_instruction(env.current_episode)
        episode_id = env.current_episode.episode_id

        # 导航循环（参考 nonlearning_agents.py 的 while not env.episode_over）
        while not env.episode_over:
            action = agent.act(obs, instruction)
            obs = env.step(action)

        # 收集本 episode 的指标
        metrics = env.get_metrics()
        for m, v in metrics.items():
            stats[m] += v

        episode_results.append({
            "episode_id": episode_id,
            "instruction": instruction,
            "metrics": {k: float(v) for k, v in metrics.items()},
        })

    env.close()

    # 计算平均指标
    stats = {k: v / num_episodes for k, v in stats.items()}
    print_metrics(stats, split)

    # 保存结果
    results_path = os.path.join(output_dir, f"results_{split}.json")
    with open(results_path, "w") as f:
        json.dump({
            "split": split,
            "num_episodes": num_episodes,
            "averaged_metrics": stats,
            "episode_results": episode_results,
        }, f, indent=2)
    logger.info(f"Results saved to: {results_path}")

    return stats


def main():
    parser = argparse.ArgumentParser(description="Test Qwen3.5 on VLN-CE")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to VLN-CE config file"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to Qwen3.5 model"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val_unseen",
        choices=["train", "val_seen", "val_unseen"],
        help="Dataset split to evaluate on"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=-1,
        help="Number of episodes to test (-1 for all)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save results"
    )

    args = parser.parse_args()

    # 加载配置
    config = get_config(args.config)
    logger.info(f"Loaded config: {args.config}")

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)

    # 设置随机种子（参考 run.py）
    torch.manual_seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.backends.cudnn.deterministic = True

    # 初始化 Qwen 智能体
    agent = QwenNavigationAgent(model_path=args.model_path)

    # 运行测试
    test_navigation(
        config=config,
        agent=agent,
        split=args.split,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
