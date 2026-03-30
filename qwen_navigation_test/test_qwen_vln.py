#!/usr/bin/env python3
"""
Qwen3.5 VLN-CE Navigation Test Script
测试 Qwen3.5 模型在 VLN-CE 任务上的导航能力
"""

import argparse
import base64
import json
import os
import re
import time
from urllib import request
from collections import defaultdict

import cv2
import numpy as np
import requests
import torch
from habitat import Env, logger
from habitat.sims.habitat_simulator.actions import HabitatSimActions
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
    "You are an embodied navigation agent. You will be given a navigation instruction "
    "and a first-person RGB image of your current view from a simulator. "
    "Based on the image and the instruction, output exactly one action from: "
    "stop, forward, left, right. "
    "Reply with only the action word, nothing else."
)

LOCAL_API_BASE_URL = "http://127.0.0.1:8000/v1"
LOCAL_MODEL_DISCOVERY_URL = "http://localhost:8000/v1/models"
DASHSCOPE_API_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
DASHSCOPE_MODEL_NAME = "qwen3.5-plus"
API_PROVIDER_CHOICES = ("local", "dashscope")
THINKING_MODE_CHOICES = ("auto", "on", "off")


class QwenNavigationAgent:
    """Qwen 导航智能体，支持本地 Habitat + 远端 OpenAI 兼容多模态接口。"""

    def __init__(
        self,
        api_provider="local",
        api_base_url=None,
        model_name=None,
        api_key=None,
        api_key_env="DASHSCOPE_API_KEY",
        thinking_mode="off",
        max_retries=3,
    ):
        """
        初始化多模态 LLM 客户端。

        Args:
            api_provider: 接口提供方类型。local 表示自部署 OpenAI 兼容接口；
                dashscope 表示阿里云百炼兼容接口。
            api_base_url: OpenAI 兼容接口地址。既支持 http://host:port，
                也支持完整的 http://host:port/v1。
            model_name: 模型名称。local 默认自动探测，dashscope 默认 qwen3.5-plus。
            api_key: 显式传入 API Key。
            api_key_env: 当 provider 需要鉴权时，读取 API Key 的环境变量名。
            thinking_mode: thinking 模式开关，参考 vln_data_synthesizer.py。
            max_retries: 请求失败时的最大重试次数。
        """
        if api_provider not in API_PROVIDER_CHOICES:
            raise ValueError("unsupported api_provider: {}".format(api_provider))
        if thinking_mode not in THINKING_MODE_CHOICES:
            raise ValueError("unsupported thinking_mode: {}".format(thinking_mode))
        if max_retries < 1:
            raise ValueError("max_retries must be >= 1")

        self.api_provider = api_provider
        self.api_key_env = api_key_env
        self.thinking_mode = thinking_mode
        self.max_retries = max_retries
        self.base_url = self._resolve_api_base_url(api_provider, api_base_url)
        self.api_key = self._resolve_api_key(
            api_provider=api_provider,
            api_key=api_key,
            api_key_env=api_key_env,
        )
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer {}".format(self.api_key),
        }
        self.model_name = self._resolve_model_name(model_name)
        self.actions = list(ACTION_MAP.keys())
        self.step_count = 0
        logger.info(
            "Connected navigation agent | provider=%s | base_url=%s | model=%s | thinking_mode=%s",
            self.api_provider,
            self.base_url,
            self.model_name,
            self.thinking_mode,
        )

    @staticmethod
    def _resolve_api_base_url(api_provider, api_base_url):
        """根据 provider 解析并规范化 OpenAI 兼容接口地址。"""
        if api_base_url:
            base_url = api_base_url.rstrip("/")
        elif api_provider == "dashscope":
            base_url = DASHSCOPE_API_BASE_URL
        else:
            base_url = LOCAL_API_BASE_URL

        if base_url.endswith("/v1"):
            return base_url
        return base_url + "/v1"

    @staticmethod
    def _resolve_api_key(api_provider, api_key, api_key_env):
        """根据 provider 解析 API Key。"""
        if api_key:
            return api_key
        env_value = os.getenv(api_key_env)
        if env_value:
            return env_value
        if api_provider == "local":
            return "EMPTY"
        raise ValueError(
            "provider={} requires an API key. Set {} or pass --api-key.".format(
                api_provider,
                api_key_env,
            )
        )

    def _resolve_model_name(self, model_name):
        """解析模型名；local 未显式指定时固定从 localhost 模型列表探测。"""
        if model_name:
            return model_name
        if self.api_provider == "dashscope":
            return DASHSCOPE_MODEL_NAME

        try:
            with request.urlopen(LOCAL_MODEL_DISCOVERY_URL, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            models = payload.get("data") or []
            if models and models[0].get("id"):
                return models[0]["id"]
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to auto-detect model from %s: %s",
                LOCAL_MODEL_DISCOVERY_URL,
                exc,
            )

        raise ValueError(
            "Cannot auto-detect model from {}. Please pass --model explicitly.".format(
                LOCAL_MODEL_DISCOVERY_URL
            )
        )

    def _build_request_extra_body(self):
        """参考 vln_data_synthesizer.py 构造 provider 相关扩展参数。"""
        if self.api_provider == "dashscope":
            extra_body = {}
        else:
            extra_body = {"repetition_penalty": 1.01}

        if self.thinking_mode == "auto":
            return extra_body

        enable_thinking = self.thinking_mode == "on"
        if self.api_provider == "dashscope":
            extra_body["enable_thinking"] = enable_thinking
        else:
            extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
        return extra_body

    def _build_prompt(self, instruction):
        """构建发送给 Qwen3.5 的文本提示词"""
        return (
            "Navigation instruction: {}\n"
            "Current step: {}\n"
            "What is your next action? Choose from: stop, forward, left, right."
        ).format(instruction, self.step_count)

    @staticmethod
    def _encode_rgb(rgb_obs):
        """
        将 Habitat RGB 观察（numpy uint8 HxWx3，RGB 通道顺序）编码为 base64 JPEG 字符串。

        Args:
            rgb_obs: numpy array, shape (H, W, 3), dtype uint8, RGB 顺序

        Returns:
            data_uri: str, 形如 "data:image/jpeg;base64,<b64data>"
        """
        # Habitat 输出 RGB，cv2 编码需要 BGR
        bgr = cv2.cvtColor(rgb_obs.astype(np.uint8), cv2.COLOR_RGB2BGR)
        success, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            raise RuntimeError("cv2.imencode failed when encoding RGB observation")
        b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
        return "data:image/jpeg;base64," + b64

    @staticmethod
    def _extract_response_text(response):
        """兼容 OpenAI 兼容接口返回的字符串或分片结构内容。"""
        content = response["choices"][0]["message"]["content"]
        if content is None:
            return ""
        if isinstance(content, str):
            return content

        chunks = []
        for part in content:
            if isinstance(part, dict):
                if part.get("type") == "text" and part.get("text"):
                    chunks.append(part["text"])
            else:
                text = getattr(part, "text", None)
                if text:
                    chunks.append(text)
        return "\n".join(chunks).strip()

    def _parse_action(self, response_text):
        """将模型输出文本解析为 HabitatSimActions"""
        text = response_text.strip().lower()
        if text in ACTION_MAP:
            return ACTION_MAP[text]

        matches = re.findall(r"\b(stop|forward|left|right)\b", text)
        if matches:
            return ACTION_MAP[matches[-1]]

        # 无法解析时默认前进，避免提前停止
        logger.warning(f"Cannot parse action from: '{response_text}', defaulting to MOVE_FORWARD")
        return HabitatSimActions.MOVE_FORWARD

    def _request_action(self, prompt, image_data_uri):
        """调用远端多模态 LLM，返回原始文本。"""
        user_content = [
            {
                "type": "image_url",
                "image_url": {"url": image_data_uri},
            },
            {
                "type": "text",
                "text": prompt,
            },
        ]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ]

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            try:
                payload = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": 32,
                    "temperature": 0.0,
                    "top_p": 1.0,
                }
                payload.update(self._build_request_extra_body())

                response = requests.post(
                    self.base_url.rstrip("/") + "/chat/completions",
                    headers=self.headers,
                    json=payload,
                    timeout=60,
                )
                response.raise_for_status()
                response_json = response.json()
                return self._extract_response_text(response_json)

            except requests.exceptions.HTTPError as exc:
                last_error = exc
                logger.warning(
                    "VLM request returned HTTP error (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except requests.exceptions.ConnectionError as exc:
                last_error = exc
                logger.warning(
                    "VLM connection failed (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except requests.exceptions.Timeout as exc:
                last_error = exc
                logger.warning(
                    "VLM request timed out (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                logger.warning(
                    "VLM request failed unexpectedly (attempt %d/%d): %s",
                    attempt,
                    self.max_retries,
                    exc,
                )

            if attempt < self.max_retries:
                time.sleep(2 ** attempt)

        raise RuntimeError("LLM request failed after {} attempts: {}".format(self.max_retries, last_error))

    def act(self, observations, instruction):
        """
        根据第一人称 RGB 观察和导航指令调用 Qwen3.5 生成动作

        Args:
            observations: Habitat 环境观察字典，包含 "rgb" 键 (H x W x 3 uint8, RGB)
            instruction: 导航指令文本

        Returns:
            action: 动作字典 {"action": action_id}
        """
        prompt = self._build_prompt(instruction)

        # 提取并编码第一人称 RGB 图像
        rgb_obs = observations["rgb"]  # shape: (H, W, 3), dtype: uint8
        image_data_uri = self._encode_rgb(rgb_obs)

        response_text = self._request_action(prompt, image_data_uri)
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


def override_data_paths(config, dataset_root=None, scenes_dir=None):
    """按需覆盖数据集和场景路径。"""
    if dataset_root is None and scenes_dir is None:
        return config

    config.defrost()
    if dataset_root is not None:
        dataset_root = dataset_root.rstrip("/")
        config.TASK_CONFIG.DATASET.DATA_PATH = os.path.join(
            dataset_root,
            "{split}",
            "{split}.json.gz",
        )
        config.TASK_CONFIG.TASK.NDTW.GT_PATH = os.path.join(
            dataset_root,
            "{split}",
            "{split}_gt.json.gz",
        )
    if scenes_dir is not None:
        config.TASK_CONFIG.DATASET.SCENES_DIR = scenes_dir
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


def test_navigation(config, agent, split, num_episodes, output_dir, dataset_root=None, scenes_dir=None):
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
    config = override_data_paths(
        config,
        dataset_root=dataset_root,
        scenes_dir=scenes_dir,
    )

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
        "--api-provider",
        type=str,
        choices=API_PROVIDER_CHOICES,
        default="local",
        help="API provider: local=OpenAI-compatible server, dashscope=Alibaba DashScope"
    )
    parser.add_argument(
        "--api-base-url",
        "--server-url",
        dest="api_base_url",
        type=str,
        default=None,
        help="OpenAI-compatible API base URL; supports both http://host:port and http://host:port/v1"
    )
    parser.add_argument(
        "--model",
        "--model-name",
        dest="model_name",
        type=str,
        default=None,
        help="Model name; local defaults to auto-detection from http://localhost:8000/v1/models, dashscope defaults to qwen3.5-plus"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Explicit API key for authenticated providers"
    )
    parser.add_argument(
        "--api-key-env",
        type=str,
        default="DASHSCOPE_API_KEY",
        help="Environment variable name used to load API key when needed"
    )
    parser.add_argument(
        "--thinking-mode",
        type=str,
        choices=THINKING_MODE_CHOICES,
        default="off",
        help="Thinking mode: auto=backend default, on=enable, off=disable"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries for remote model requests"
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
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        help="Dataset root containing train/val_seen/val_unseen split folders"
    )
    parser.add_argument(
        "--scenes-dir",
        type=str,
        default=None,
        help="Scene dataset root directory used by Habitat"
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
        api_provider=args.api_provider,
        api_base_url=args.api_base_url,
        model_name=args.model_name,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        thinking_mode=args.thinking_mode,
        max_retries=args.max_retries,
    )

    # 运行测试
    test_navigation(
        config=config,
        agent=agent,
        split=args.split,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        dataset_root=args.dataset_root,
        scenes_dir=args.scenes_dir,
    )


if __name__ == "__main__":
    main()
