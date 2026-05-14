#!/usr/bin/env python3
"""
将 CoT JSONL 导出为按帧拆分的 Markdown 文件。

示例:
    python3 scripts/cot_annotation/jsonl_to_frame_markdown.py \
        --input data/processed/output_cot.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export each JSONL record into an individual markdown file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Input JSONL path.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for markdown files. Defaults to <input_stem>_md beside the input file.",
    )
    parser.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include the user text prompt in each markdown file.",
    )
    return parser.parse_args()


def format_user_prompt(messages: list[dict[str, Any]]) -> tuple[str | None, str | None]:
    image_path: str | None = None
    prompt_text: str | None = None

    for message in messages:
        if message.get("role") != "user":
            continue
        for item in message.get("content", []):
            item_type = item.get("type")
            if item_type == "image_path":
                image_path = item.get("image_path")
            elif item_type == "text":
                prompt_text = item.get("text")
        break

    return image_path, prompt_text


def get_assistant_content(messages: list[dict[str, Any]]) -> str:
    for message in messages:
        if message.get("role") == "assistant":
            return message.get("content", "")
    return ""


def build_markdown(
    episode_id: int,
    frame_number: int,
    image_path: str | None,
    history_memory_input: str | None,
    history_source_frame: int | None,
    assistant_content: str,
    prompt_text: str | None,
    include_prompt: bool,
) -> str:
    lines: list[str] = [
        f"# Episode {episode_id} Frame {frame_number}",
        "",
        f"- Episode ID: {episode_id}",
        f"- Frame: {frame_number}",
    ]

    if image_path:
        lines.append(f"- Image Path: `{image_path}`")

    if history_memory_input is not None:
        lines.append(f'- Historical Memory Input: "{history_memory_input}"')
    if history_source_frame is not None:
        lines.append(f"- History Source Frame: {history_source_frame}")

    lines.extend(
        [
            "",
            "## Assistant Output",
            "",
            assistant_content.rstrip(),
            "",
        ]
    )

    if include_prompt and prompt_text:
        lines.extend(
            [
                "## User Prompt",
                "",
                "```text",
                prompt_text.rstrip(),
                "```",
                "",
            ]
        )

    return "\n".join(lines).rstrip() + "\n"


def write_index(records: list[tuple[int, int, Path]], output_dir: Path) -> None:
    lines = [
        "# Frame Index",
        "",
        f"- Total Frames: {len(records)}",
        "",
    ]
    for episode_id, frame_number, md_path in records:
        lines.append(
            f"- [Episode {episode_id} Frame {frame_number}]({md_path.name})"
        )
    (output_dir / "index.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Input JSONL not found: {input_path}")

    if args.output_dir:
        output_dir = Path(args.output_dir).expanduser().resolve()
    else:
        output_dir = input_path.with_name(f"{input_path.stem}_md")
    output_dir.mkdir(parents=True, exist_ok=True)

    written_records: list[tuple[int, int, Path]] = []

    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue

            record = json.loads(stripped)
            episode_id = int(record["episode_id"])
            frame_number = int(record["frame"])
            messages = record.get("messages", [])
            input_state = record.get("input_state", {})
            image_path, prompt_text = format_user_prompt(messages)
            assistant_content = get_assistant_content(messages)
            history_memory_input = input_state.get("historical_trajectory_memory")
            history_source_frame = input_state.get("history_source_frame")

            filename = f"ep{episode_id:05d}_frame{frame_number:04d}.md"
            md_path = output_dir / filename
            md_path.write_text(
                build_markdown(
                    episode_id=episode_id,
                    frame_number=frame_number,
                    image_path=image_path,
                    history_memory_input=history_memory_input,
                    history_source_frame=history_source_frame,
                    assistant_content=assistant_content,
                    prompt_text=prompt_text,
                    include_prompt=args.include_prompt,
                ),
                encoding="utf-8",
            )
            written_records.append((episode_id, frame_number, md_path))

    write_index(written_records, output_dir)
    print(f"Exported {len(written_records)} frame markdown files to: {output_dir}")
    print(f"Index file: {output_dir / 'index.md'}")


if __name__ == "__main__":
    main()
