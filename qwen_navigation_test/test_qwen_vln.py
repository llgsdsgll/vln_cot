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
from openai import OpenAI
from tqdm import tqdm

from vlnce_baselines.config.default import get_config

# 动作名称到 HabitatSimActions 的映射，供模型输出解析使用
ACTION_MAP = {
    "stop":     HabitatSimActions.STOP,
    "forward":  HabitatSimActions.MOVE_FORWARD,
    "left":     HabitatSimActions.TURN_LEFT,
    "right":    HabitatSimActions.TURN_RIGHT,
}

SYSTEM_PROMPT = (
    "You are a navigation agent. Based on the instruction and current step, "
    "output exactly one action from: stop, forward, left, right. "
    "Reply with only the action word, nothing else."
)


class QwenNavigationAgent:
    """Qwen3.5 导航智能体，通过 vLLM HTTP 接口调用模型"""

    def __init__(self, server_url="http://0.0.0.0:8000", model_name=None):
        """
        初始化 vLLM 客户端

        Args:
            server_url: vLLM 服务地址，默认 http://0.0.0.0:8000
            model_name: 模型名称，若为 None 则自动从服务端查询
        """
        self.client = OpenAI(
            base_url=f"{server_url}/v1",
            api_key="token-abc123",  # vLLM 不校验 key，填任意非空字符串即可
        )

        # 自动获取模型名称（vLLM 部署时模型名即为路径或别名）
        if model_name is None:
            models = self.client.models.list()
            self.model_name = models.data[0].id
        else:
            self.model_name = model_name

        self.actions = list(ACTION_MAP.keys())
        self.step_count = 0
        logger.info(f"Connected to vLLM server at {server_url}, model: {self.model_name}")

    def _build_prompt(self, instruction):
        """构建发送给 Qwen3.5 的提示词"""
        return (
            f"Navigation instruction: {instruction}\n"
            f"Current step: {self.step_count}\n"
            f"What is your next action? Choose from: stop, forward, left, right."
        )

    def _parse_action(self, response_text):
        """将模型输出文本解析为 HabitatSimActions"""
        text = response_text.strip().lower()
        for key, action in ACTION_MAP.items():
            if key in text:
                return action
        # 无法解析时默认前进，避免提前停止
        logger.warning(f"Cannot parse action from: '{response_text}', defaulting to MOVE_FORWARD")
        return HabitatSimActions.MOVE_FORWARD

    def act(self, observations, instruction):
        """
        根据观察和指令调用 Qwen3.5 生成动作

        Args:
            observations: 环境观察 (RGB, depth 等，当前为纯文本模式暂不使用)
            instruction: 导航指令文本

        Returns:
            action: 动作字典 {"action": action_id}
        """
        prompt = self._build_prompt(instruction)

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=16,
            temperature=0.0,  # 贪心解码，保证可复现
        )

        response_text = response.choices[0].message.content
        action = self._parse_action(response_text)

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
        "--server-url",
        type=str,
        default="http://0.0.0.0:8000",
        help="vLLM server URL"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name on vLLM server (auto-detected if not specified)"
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

    # 初始化 Qwen 智能体（通过 vLLM HTTP 接口）
    agent = QwenNavigationAgent(
        server_url=args.server_url,
        model_name=args.model_name,
    )

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
