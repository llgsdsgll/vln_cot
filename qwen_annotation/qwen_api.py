#!/usr/bin/env python3

import argparse
import base64
import json
import mimetypes
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from openai import OpenAI


DEFAULT_BASE_URL = os.getenv(
    "DASHSCOPE_BASE_URL",
    "https://dashscope.aliyuncs.com/compatible-mode/v1",
)
DEFAULT_VL_MODEL = os.getenv("QWEN_VL_MODEL", "qwen3.6-plus")


def require_api_key(api_key: Optional[str] = None) -> str:
    resolved_api_key = api_key or os.getenv("DASHSCOPE_API_KEY")
    if not resolved_api_key:
        raise RuntimeError(
            "Missing DASHSCOPE_API_KEY. Set the environment variable before "
            "calling the Qwen API."
        )
    return resolved_api_key


def build_client(
    api_key: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
) -> OpenAI:
    return OpenAI(api_key=require_api_key(api_key), base_url=base_url)


def guess_mime_type(path: Path) -> str:
    mime_type, _ = mimetypes.guess_type(str(path))
    return mime_type or "application/octet-stream"


def encode_file_to_data_url(path: Path) -> str:
    mime_type = guess_mime_type(path)
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def build_image_content_item(
    image_path: Path,
    detail: Optional[str] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    image_url: Dict[str, Any] = {"url": encode_file_to_data_url(image_path)}
    if detail is not None:
        image_url["detail"] = detail
    if min_pixels is not None:
        image_url["min_pixels"] = min_pixels
    if max_pixels is not None:
        image_url["max_pixels"] = max_pixels
    return {"type": "image_url", "image_url": image_url}


def build_video_url_content_item(
    video_path: Path,
    fps: Optional[float] = None,
    min_pixels: Optional[int] = None,
    max_pixels: Optional[int] = None,
) -> Dict[str, Any]:
    video_url: Dict[str, Any] = {"url": encode_file_to_data_url(video_path)}
    if fps is not None:
        video_url["fps"] = fps
    if min_pixels is not None:
        video_url["min_pixels"] = min_pixels
    if max_pixels is not None:
        video_url["max_pixels"] = max_pixels
    return {"type": "video_url", "video_url": video_url}


def build_video_frames_content_item(
    frame_paths: Sequence[Path],
    fps: Optional[float] = None,
) -> Dict[str, Any]:
    content: Dict[str, Any] = {
        "type": "video",
        "video": [encode_file_to_data_url(path) for path in frame_paths],
    }
    if fps is not None:
        content["fps"] = fps
    return content


class QwenClient:
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        default_model: str = DEFAULT_VL_MODEL,
        default_temperature: Optional[float] = None,
        default_top_p: Optional[float] = None,
        default_result_format: Optional[str] = None,
        default_enable_thinking: bool = False,
        default_thinking_budget: Optional[int] = None,
        timeout_seconds: int = 300,
        max_retries: int = 3,
        retry_sleep_seconds: float = 2.0,
    ) -> None:
        self.client = build_client(api_key=api_key, base_url=base_url)
        self.default_model = default_model
        self.default_temperature = default_temperature
        self.default_top_p = default_top_p
        self.default_result_format = default_result_format
        self.default_enable_thinking = default_enable_thinking
        self.default_thinking_budget = default_thinking_budget
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_sleep_seconds = retry_sleep_seconds

    def _build_extra_body(
        self,
        extra_body: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        merged: Dict[str, Any] = {
            "enable_thinking": self.default_enable_thinking,
        }
        if self.default_enable_thinking and self.default_thinking_budget is not None:
            merged["thinking_budget"] = self.default_thinking_budget
        if self.default_result_format is not None:
            merged["result_format"] = self.default_result_format
        if extra_body:
            merged.update(extra_body)
        if not merged.get("enable_thinking", False):
            merged.pop("thinking_budget", None)
        return merged

    def chat_completion(
        self,
        messages: Sequence[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Any:
        request: Dict[str, Any] = {
            "model": model or self.default_model,
            "messages": list(messages),
            "timeout": self.timeout_seconds,
        }
        resolved_temperature = (
            self.default_temperature if temperature is None else temperature
        )
        if resolved_temperature is not None:
            request["temperature"] = resolved_temperature
        resolved_top_p = self.default_top_p if top_p is None else top_p
        if resolved_top_p is not None:
            request["top_p"] = resolved_top_p
        if max_tokens is not None:
            request["max_tokens"] = max_tokens
        if response_format is not None:
            request["response_format"] = response_format
        request["extra_body"] = self._build_extra_body(extra_body)

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return self.client.chat.completions.create(**request)
            except Exception as exc:  # pragma: no cover - network/API dependent
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(self.retry_sleep_seconds * attempt)

        raise RuntimeError(
            f"Qwen API request failed after {self.max_retries} attempts."
        ) from last_error

    def chat_text(
        self,
        messages: Sequence[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> str:
        completion = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            extra_body=extra_body,
        )
        message = completion.choices[0].message
        content = getattr(message, "content", "")
        return self._content_to_text(content)

    def _content_to_text(self, content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts: List[str] = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        text_parts.append(str(item.get("text", "")))
                    elif "text" in item:
                        text_parts.append(str(item.get("text", "")))
                    else:
                        text_parts.append(json.dumps(item, ensure_ascii=False))
                else:
                    text_parts.append(str(item))
            return "".join(text_parts)
        if isinstance(content, dict):
            if "text" in content:
                return str(content.get("text", ""))
            return json.dumps(content, ensure_ascii=False)
        if content is None:
            return ""
        return str(content)

    def _extract_json_text(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            return stripped

        start = stripped.find("{")
        if start < 0:
            raise json.JSONDecodeError("No JSON object found", stripped, 0)

        in_string = False
        escape = False
        depth = 0
        end = -1
        for index in range(start, len(stripped)):
            char = stripped[index]
            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    end = index + 1
                    break

        if end < 0:
            raise json.JSONDecodeError("Unterminated JSON object", stripped, start)
        return stripped[start:end]

    def _serialize_debug_value(self, value: Any) -> Any:
        try:
            if hasattr(value, "model_dump"):
                return value.model_dump()
        except Exception:
            pass
        try:
            if hasattr(value, "to_dict"):
                return value.to_dict()
        except Exception:
            pass
        return repr(value)

    def chat_json(
        self,
        messages: Sequence[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        max_tokens: Optional[int] = None,
        extra_body: Optional[Dict[str, Any]] = None,
        raw_text_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        completion = self.chat_completion(
            messages=messages,
            model=model,
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
            extra_body=extra_body,
        )
        choice = completion.choices[0]
        message = choice.message
        content = message.content
        text = self._content_to_text(content)
        if raw_text_path is not None:
            raw_text_path.parent.mkdir(parents=True, exist_ok=True)
            raw_text_path.write_text(text, encoding="utf-8")
            debug_payload = {
                "message": self._serialize_debug_value(message),
                "choice": self._serialize_debug_value(choice),
            }
            debug_path = raw_text_path.parent / f"{raw_text_path.stem}_response.json"
            debug_path.write_text(
                json.dumps(debug_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return json.loads(self._extract_json_text(text))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal CLI for testing Qwen compatible-mode multimodal calls."
    )
    parser.add_argument("prompt", help="User prompt to send to the model.")
    parser.add_argument("--image", type=Path, action="append", default=[])
    parser.add_argument("--video", type=Path)
    parser.add_argument("--video-fps", type=float, default=None)
    parser.add_argument("--model", default=DEFAULT_VL_MODEL)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--result-format", default=None)
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable or disable Qwen thinking mode.",
    )
    parser.add_argument("--thinking-budget", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Force JSON output.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = QwenClient(
        default_model=args.model,
        default_temperature=args.temperature,
        default_top_p=args.top_p,
        default_result_format=args.result_format,
        default_enable_thinking=args.thinking,
        default_thinking_budget=args.thinking_budget,
    )

    user_content: List[Dict[str, Any]] = []
    for image_path in args.image:
        user_content.append(build_image_content_item(image_path))
    if args.video is not None:
        user_content.append(
            build_video_url_content_item(args.video, fps=args.video_fps)
        )
    user_content.append({"type": "text", "text": args.prompt})

    messages = [{"role": "user", "content": user_content}]
    if args.json:
        payload = client.chat_json(messages=messages, model=args.model)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
    else:
        print(client.chat_text(messages=messages, model=args.model))


if __name__ == "__main__":
    main()
