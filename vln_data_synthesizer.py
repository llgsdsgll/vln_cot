#!/usr/bin/env python3
"""
VLN CoT 数据合成脚本

功能：
  读取人工标注后的 processed.json（含 subtask/object/bbox）和 train_gt.json（含真值动作序列），
  结合视频帧，调用本地部署的 Qwen VLM（OpenAI 兼容接口）生成 Chain-of-Thought 推理文本，
  输出为 OpenAI/HuggingFace 标准的对话微调格式（JSONL）。

使用方法：
  python3 vln_data_synthesizer.py \\
      --input   gengshuang_1_H.264_0324_processed.json \\
      --gt      /home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json \\
      --video-dir /home/gs/my_test/vln_dataset/vln_ce_video \\
      --output  output_cot.jsonl

  可选参数：
      --api-base-url  本地 VLM 服务地址（默认 http://localhost:8000/v1）
      --model         模型名称，与服务端部署名一致（默认 /mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8）
      --max-retries   VLM API 最大重试次数（默认 3）
      --episode-ids   只处理指定 episode（如 --episode-ids 1 6 7）

本地服务映射命令（参考）：
  ssh -fN -L 8000:localhost:8000 root@139.196.171.150 -p 6222
"""

import argparse
import base64
import json
import logging
import os
import time
from typing import Any, Optional

import cv2
import openai

import re

# ---------------------------------------------------------------------------
# Prompt 模板 —— 抽离为全局变量，方便随时修改
# ---------------------------------------------------------------------------

# 有目标物体可见时使用
PROMPT_TEMPLATE = """\
You are an expert embodied AI data synthesis specialist. Your task is to generate a rigorous "Chain-of-Thought" (CoT) reasoning text for a Vision-Language Navigation (VLN) agent.

[Current Ground-Truth State]:
- Global Instruction: "{global_instruction}"
- Pre-defined Sub-tasks: {global_subtasks_list}
- Current Completed Sub-task Index: {completed_index}
- Historical Trajectory Memory: "{script_memory}"
- Target Landmark/Object in Current Frame: {target_object_name}
- Ground-Truth Target BBox (Normalized 0-1000): {bbox_norm}
- Ground-Truth Next Action: {next_action}

[Visual Input]:
<Image>

[Task Requirements]:
Based on the image and the ground-truth state, generate the reasoning text strictly following the Markdown format below. Do NOT output any other conversational text.

**Reasoning Process:**
Step 1: Task Progress. Briefly state what has been completed based on the memory and what the current sub-task is.
Step 2: Spatial Perception. Explicitly state that you see the '{target_object_name}' and it is located at the exact bounding box {bbox_norm}. Describe its relative screen position.
Step 3: Decision Logic. Explain exactly why the action '{next_action}' must be taken to approach or interact with the target.
Step 4: Memory Update. Provide a one-sentence summary of this step's observation and decision.

**Final Action:**
{next_action}\
"""

# 目标物体不可见时使用
PROMPT_TEMPLATE_NO_TARGET = """\
You are an expert embodied AI data synthesis specialist. Your task is to generate a rigorous "Chain-of-Thought" (CoT) reasoning text for a Vision-Language Navigation (VLN) agent.

[Current Ground-Truth State]:
- Global Instruction: "{global_instruction}"
- Pre-defined Sub-tasks: {global_subtasks_list}
- Current Completed Sub-task Index: {completed_index}
- Historical Trajectory Memory: "{script_memory}"
- Target Landmark/Object: (Not visible in current frame)
- Ground-Truth Next Action: {next_action}

[Visual Input]:
<Image>

[Task Requirements]:
Based on the image and the ground-truth state, generate the reasoning text strictly following the Markdown format below. Do NOT output any other conversational text.

**Reasoning Process:**
Step 1: Task Progress. Briefly state what has been completed based on the memory and what the current sub-task is.
Step 2: Spatial Perception. State that the target object is currently not visible in the field of view. Do NOT hallucinate any bounding boxes.
Step 3: Decision Logic. Explain why exploring via the action '{next_action}' is the most reasonable choice.
Step 4: Memory Update. Provide a one-sentence summary stating the target was unseen and the exploration decision made.

**Final Action:**
{next_action}\
"""

# 动作编码映射
ACTION_MAP: dict[int, str] = {
    0: "STOP",
    1: "MOVE_FORWARD",
    2: "TURN_LEFT",
    3: "TURN_RIGHT",
}

# ---------------------------------------------------------------------------
# 日志配置
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("VLNDataSynthesizer")


# ---------------------------------------------------------------------------
# 主类
# ---------------------------------------------------------------------------

class VLNDataSynthesizer:
    """
    VLN Chain-of-Thought 数据合成器。

    处理流程：
      1. 读取 processed.json（帧级标注）和 train_gt.json（真值动作）。
      2. 按 episode 遍历，对每一帧提取视频帧、构建 prompt、调用 VLM API。
      3. 将结果保存为 JSONL 格式，供后续微调使用。
    """

    def __init__(
        self,
        max_retries: int = 3,
        api_base_url: str = "http://localhost:8000/v1",
        model_name: str = "/mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8",
    ) -> None:
        """
        初始化合成器。

        Args:
            max_retries:   VLM API 调用失败时的最大重试次数。
            api_base_url:  本地 VLM 服务的 OpenAI 兼容接口地址。
                           默认指向 SSH 隧道映射的 8000 端口。
            model_name:    服务端部署的模型名称，需与 vLLM --served-model-name 一致。
                           可通过 GET /v1/models 确认实际名称。
        """
        self.max_retries = max_retries
        self._client = openai.OpenAI(
            api_key="EMPTY",        # vLLM 本地服务不校验 API key，填任意非空字符串
            base_url=api_base_url,
        )
        self._model_name = model_name
        logger.info(
            "VLM 客户端初始化完成 | base_url=%s | model=%s",
            api_base_url,
            model_name,
        )

    # ------------------------------------------------------------------
    # 静态/工具方法
    # ------------------------------------------------------------------

    @staticmethod
    def normalize_bbox(bbox: dict[str, float]) -> list[int]:
        """
        将百分比格式的 bbox 归一化到 [0, 1000] 整数区间。

        Args:
            bbox: 标注数据中的 bbox 字典，格式为
                  ``{"x": float, "y": float, "width": float, "height": float}``，
                  各字段均为百分比值（0-100）。

        Returns:
            ``[xmin, ymin, xmax, ymax]`` 归一化到 [0, 1000] 的整数列表。
        """
        xmin = int(round(bbox["x"] * 10))
        ymin = int(round(bbox["y"] * 10))
        xmax = int(round((bbox["x"] + bbox["width"]) * 10))
        ymax = int(round((bbox["y"] + bbox["height"]) * 10))
        # 限制在合法范围内
        xmin = max(0, min(1000, xmin))
        ymin = max(0, min(1000, ymin))
        xmax = max(0, min(1000, xmax))
        ymax = max(0, min(1000, ymax))
        return [xmin, ymin, xmax, ymax]

    @staticmethod
    def update_script_memory(
        script_memory: list[str],
        frame_idx: int,
        action_str: str,
    ) -> None:
        """
        将本步骤的真值动作追加到历史记忆列表（in-place 修改）。

        使用真值动作维护历史，防止模型幻觉在步骤间累积。

        Args:
            script_memory: 存储历史动作文本的列表（将被修改）。
            frame_idx:      当前帧序号（1-indexed）。
            action_str:     当前帧执行的动作字符串（如 ``"MOVE_FORWARD"``）。
        """
        script_memory.append(f"Step {frame_idx}: Executed {action_str}.")

    def build_prompt(
        self,
        frame_data: dict[str, Any],
        script_memory: list[str],
        global_instruction: str,
        global_subtasks: list[str],
        completed_index: int,
        next_action: str,
    ) -> str:
        """
        根据当前帧的真值数据构建填充后的 prompt 字符串。

        当帧中没有 objects 标注时，使用无目标物体版本的模板，
        跳过坐标输出并改为探索性描述。

        Args:
            frame_data:        processed.json 中单帧的字典数据。
            script_memory:     截至上一步的历史动作文本列表。
            global_instruction: 当前 episode 的全局导航指令。
            global_subtasks:   当前 episode 的子任务列表（有序、去重）。
            completed_index:   当前子任务在列表中的索引（之前的均已完成）。
            next_action:       当前帧的真值动作字符串。

        Returns:
            填充完毕的 prompt 字符串，可直接传给 VLM API。
        """
        memory_str = " ".join(script_memory) if script_memory else "None"
        subtasks_str = str(global_subtasks)

        objects = frame_data.get("objects", [])

        if objects:
            # 取第一个 object 作为当前帧的主要目标
            obj = objects[0]
            target_name = obj["label"]
            bbox_norm = self.normalize_bbox(obj["bbox"])
            bbox_str = str(bbox_norm)

            prompt = PROMPT_TEMPLATE.format(
                global_instruction=global_instruction,
                global_subtasks_list=subtasks_str,
                completed_index=completed_index,
                script_memory=memory_str,
                target_object_name=target_name,
                bbox_norm=bbox_str,
                next_action=next_action,
            )
        else:
            prompt = PROMPT_TEMPLATE_NO_TARGET.format(
                global_instruction=global_instruction,
                global_subtasks_list=subtasks_str,
                completed_index=completed_index,
                script_memory=memory_str,
                next_action=next_action,
            )

        return prompt

    @staticmethod
    def extract_frame(video_path: str, frame_number: int) -> Optional[str]:
        """
        从视频文件中提取指定帧，返回 base64 编码的 JPEG 字符串。

        Args:
            video_path:   视频文件的绝对路径。
            frame_number: 目标帧号（1-indexed）。

        Returns:
            base64 编码的 JPEG 图像字符串；提取失败时返回 ``None``。
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error("无法打开视频文件: %s", video_path)
            return None

        try:
            # OpenCV 帧索引从 0 开始
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
            ret, frame = cap.read()
            if not ret or frame is None:
                logger.warning(
                    "无法读取帧 %d，视频: %s", frame_number, video_path
                )
                return None

            success, buffer = cv2.imencode(".jpg", frame)
            if not success:
                logger.warning("JPEG 编码失败，帧 %d，视频: %s", frame_number, video_path)
                return None

            return base64.b64encode(buffer).decode("utf-8")
        finally:
            cap.release()

    def call_vlm_api(self, prompt: str, image_b64: Optional[str]) -> Optional[str]:
        # 构建 user message content：图像（若有）+ 文本
        user_content: list[dict[str, Any]]
        if image_b64 is not None:
            user_content = [
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
                {"type": "text", "text": prompt},
            ]
        else:
            user_content = [{"type": "text", "text": prompt}]

        messages = [{"role": "user", "content": user_content}]

        for attempt in range(1, self.max_retries + 1):
            try:
                logger.debug(
                    "调用 VLM API (attempt %d/%d) model=%s image=%s",
                    attempt,
                    self.max_retries,
                    self._model_name,
                    "有" if image_b64 else "无",
                )
                
                # 首先尝试：开启思考模式 (Teacher Mode)
                response = self._client.chat.completions.create(
                    model=self._model_name,
                    messages=messages,
                    temperature=0.3,
                    top_p=0.85,
                    frequency_penalty=0.3,
                    presence_penalty=0.0,
                    max_tokens=2048,
                    # 输出到结束标记立即停止，防止重复退化
                    stop=["[ACTION_DECISION_END]"],
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                        # vLLM 原生重复惩罚（比 frequency_penalty 更有效）
                        "repetition_penalty": 1.05,
                    },
                )
                
                msg = response.choices[0].message
                # content = 模型最终答案（</think> 之后），存入训练数据
                # reasoning_content = Qwen3 内部思考过程，仅用于空值兜底判断，不写入 JSONL
                answer = msg.content or ""
                reasoning = getattr(msg, "reasoning_content", None) or ""
                raw_text = answer if answer.strip() else reasoning

                # 如果思考模式返回为空，触发柔性降级 (Fallback)
                if not raw_text.strip():
                    logger.warning("思考模式返回为空！打印原始响应以便排查: %s", response.model_dump_json())
                    logger.warning("尝试降级为【非思考模式】重试本帧...")
                    
                    fallback_response = self._client.chat.completions.create(
                        model=self._model_name,
                        messages=messages,
                        temperature=0.5,
                        top_p=0.85,
                        frequency_penalty=0.5,
                        presence_penalty=0.1,
                        max_tokens=1024,
                        stop=["[ACTION_DECISION_END]"],
                        extra_body={"repetition_penalty": 1.1},
                    )
                    
                    msg_fallback = fallback_response.choices[0].message
                    raw_text = msg_fallback.content or ""
                    
                    if not raw_text.strip():
                        raise ValueError("降级模式依然返回空内容，可能是输入图像/文本触发了拒答。")
                
                return raw_text

            except openai.APIConnectionError as exc:
                wait = 2 ** attempt
                logger.warning("VLM 连接失败 (attempt %d/%d): %s — %ds 后重试", attempt, self.max_retries, exc, wait)
                if attempt < self.max_retries: time.sleep(wait)
                else: return None

            except Exception as exc:  # noqa: BLE001
                wait = 2 ** attempt
                logger.warning("VLM 未知异常 (attempt %d/%d): %s — %ds 后重试", attempt, self.max_retries, exc, wait)
                if attempt < self.max_retries: time.sleep(wait)
                else: return None

        return None

    def process_trajectory(
        self,
        episode_data: dict[str, Any],
        gt_actions: list[int],
        video_dir: str,
        out_f,
    ) -> int:
        episode_id: int = episode_data["episode_id"]
        instruction: str = episode_data["instruction"]
        frames: list[dict[str, Any]] = episode_data["frames"]

        global_subtasks: list[str] = list(
            dict.fromkeys(f["subtask"] for f in frames if f.get("subtask") is not None)
        )

        video_path = os.path.join(video_dir, f"ep{episode_id:05d}.mp4")
        # 帧图像保存目录：与输出 JSONL 同级的 frames/ 子目录
        frames_dir = os.path.join(
            os.path.dirname(os.path.abspath(out_f.name)), "frames"
        )
        os.makedirs(frames_dir, exist_ok=True)

        script_memory: list[str] = []
        success_count = 0

        for frame_data in frames:
            frame_number: int = frame_data["frame"]
            subtask: Optional[str] = frame_data.get("subtask")

            if subtask is None:
                continue
            action_idx = frame_number - 1
            if action_idx >= len(gt_actions):
                continue

            action_code: int = gt_actions[action_idx]
            next_action: str = ACTION_MAP.get(action_code, f"UNKNOWN_{action_code}")

            # 帧级兜底：任何未预期异常只跳过本帧，不崩溃整个进程
            try:
                try:
                    completed_index = global_subtasks.index(subtask)
                except ValueError:
                    completed_index = 0

                # 提取帧：base64 仅用于 API 调用，同时保存 JPEG 文件供 JSONL 引用
                image_b64: Optional[str] = None
                image_path: Optional[str] = None
                if os.path.exists(video_path):
                    image_b64 = self.extract_frame(video_path, frame_number)
                    if image_b64 is not None:
                        fname = f"ep{episode_id:05d}_frame{frame_number:04d}.jpg"
                        image_path = os.path.join(frames_dir, fname)
                        with open(image_path, "wb") as img_f:
                            img_f.write(base64.b64decode(image_b64))

                prompt = self.build_prompt(
                    frame_data=frame_data,
                    script_memory=script_memory,
                    global_instruction=instruction,
                    global_subtasks=global_subtasks,
                    completed_index=completed_index,
                    next_action=next_action,
                )

                vlm_response = self.call_vlm_api(prompt, image_b64)

                if vlm_response is None:
                    logger.error("episode %d frame %d VLM 调用失败，跳过", episode_id, frame_number)
                    self.update_script_memory(script_memory, frame_number, next_action)
                    continue

                # ------------------------------------------------------------------
                # 提取自定义标记块
                # stop=["[ACTION_DECISION_END]"] 会截断输出，需在此补回结束标记
                # ------------------------------------------------------------------
                raw = vlm_response
                if "[ACTION_DECISION_START]" in raw and "[ACTION_DECISION_END]" not in raw:
                    raw = raw + "\n[ACTION_DECISION_END]"

                think_match = re.search(
                    r'\[ANNOTATION_THINK_START\](.*?)\[ANNOTATION_THINK_END\]',
                    raw, re.DOTALL,
                )
                action_match = re.search(
                    r'\[ACTION_DECISION_START\](.*?)\[ACTION_DECISION_END\]',
                    raw, re.DOTALL,
                )

                if think_match and action_match:
                    think_part  = think_match.group(1).strip()
                    action_part = action_match.group(1).strip()
                    formatted_response = (
                        f"[ANNOTATION_THINK_START]\n{think_part}\n[ANNOTATION_THINK_END]\n"
                        f"[ACTION_DECISION_START]\n{action_part}\n[ACTION_DECISION_END]"
                    )
                else:
                    logger.warning(
                        "episode %d frame %d 自定义标记提取失败，保存原始输出。raw前200字符: %s",
                        episode_id, frame_number, raw[:200],
                    )
                    formatted_response = raw

                # JSONL 中 user_content 只存路径引用，不内嵌 base64
                user_content: list[dict[str, Any]]
                if image_path is not None:
                    user_content = [
                        {"type": "image_path", "image_path": image_path},
                        {"type": "text", "text": prompt},
                    ]
                else:
                    user_content = [{"type": "text", "text": prompt}]

                record: dict[str, Any] = {
                    "episode_id": episode_id,
                    "frame": frame_number,
                    "messages": [
                        {"role": "user", "content": user_content},
                        {"role": "assistant", "content": formatted_response},
                    ],
                }

                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                success_count += 1
                logger.info("  => 成功写入 Episode %d Frame %d", episode_id, frame_number)

            except Exception as exc:  # noqa: BLE001
                # 单帧任何异常均只跳过本帧，不中断整个 episode
                logger.error(
                    "episode %d frame %d 处理异常，跳过本帧: %s: %s",
                    episode_id, frame_number, type(exc).__name__, exc,
                )
            finally:
                # 无论成功或失败，始终用真值更新记忆，保证后续帧历史不断链
                self.update_script_memory(script_memory, frame_number, next_action)

        return success_count

    def run(
        self,
        input_json: str,
        gt_json: str,
        video_dir: str,
        output_jsonl: str,
        episode_ids: Optional[list[int]] = None,
    ) -> None:
        """
        主入口：遍历所有（或指定）episodes，合成 CoT 数据并写入 JSONL 文件。

        Args:
            input_json:   processed.json 文件路径。
            gt_json:      train_gt.json 文件路径。
            video_dir:    视频文件所在目录。
            output_jsonl: 输出 JSONL 文件路径。
            episode_ids:  若指定，则只处理列表中的 episode_id；为 ``None`` 时处理全部。
        """
        logger.info("加载标注数据: %s", input_json)
        with open(input_json, encoding="utf-8") as f:
            episodes: list[dict[str, Any]] = json.load(f)

        logger.info("加载真值动作数据: %s", gt_json)
        with open(gt_json, encoding="utf-8") as f:
            gt_data: dict[str, Any] = json.load(f)

        if episode_ids is not None:
            episodes = [ep for ep in episodes if ep["episode_id"] in episode_ids]
            logger.info("过滤后处理 %d 个 episode: %s", len(episodes), episode_ids)
        else:
            logger.info("共处理 %d 个 episode", len(episodes))

        total_records = 0
        with open(output_jsonl, "w", encoding="utf-8") as out_f:
            for ep_idx, episode in enumerate(episodes, start=1):
                episode_id = episode["episode_id"]
                ep_key = str(episode_id)

                if ep_key not in gt_data:
                    logger.warning(
                        "episode %d 在 train_gt.json 中未找到，跳过", episode_id
                    )
                    continue

                gt_actions: list[int] = gt_data[ep_key]["actions"]

                logger.info(
                    "[%d/%d] 处理 episode %d ...", ep_idx, len(episodes), episode_id
                )

                # 将 out_f 传给 process_trajectory，并获取成功条数
                records_count = self.process_trajectory(episode, gt_actions, video_dir, out_f)
                total_records += records_count

        logger.info(
            "全部处理完成！共生成 %d 条 CoT 记录，输出文件: %s",
            total_records,
            output_jsonl,
        )


# ---------------------------------------------------------------------------
# 命令行入口
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(
        description="VLN CoT 数据合成脚本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default="gengshuang_1_H.264_0324_processed.json",
        help="processed.json 文件路径",
    )
    parser.add_argument(
        "--gt",
        default="/home/gs/my_test/vln_dataset/data/datasets/r2r/train/train_gt.json",
        help="train_gt.json 文件路径",
    )
    parser.add_argument(
        "--video-dir",
        default="/home/gs/my_test/vln_dataset/vln_ce_video",
        help="视频文件目录（包含 ep00001.mp4 等文件）",
    )
    parser.add_argument(
        "--output",
        default="output_cot.jsonl",
        help="输出 JSONL 文件路径",
    )
    parser.add_argument(
        "--api-base-url",
        default="http://localhost:8000/v1",
        help="本地 VLM 服务的 OpenAI 兼容接口地址（SSH 隧道映射后的地址）",
    )
    parser.add_argument(
        "--model",
        default="/mnt/data-cpfs/gengshuang/models/Qwen3.5-397B-A17B-FP8",
        help="服务端部署的模型名称，需与 vLLM --served-model-name 一致；"
             "可通过 curl http://localhost:8000/v1/models 确认",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="VLM API 最大重试次数",
    )
    parser.add_argument(
        "--episode-ids",
        type=int,
        nargs="+",
        default=None,
        help="只处理指定 episode_id（空格分隔），不指定则处理全部",
    )
    return parser.parse_args()


def main() -> None:
    """脚本主函数。"""
    args = parse_args()
    synthesizer = VLNDataSynthesizer(
        max_retries=args.max_retries,
        api_base_url=args.api_base_url,
        model_name=args.model,
    )
    synthesizer.run(
        input_json=args.input,
        gt_json=args.gt,
        video_dir=args.video_dir,
        output_jsonl=args.output,
        episode_ids=args.episode_ids,
    )


if __name__ == "__main__":
    main()
