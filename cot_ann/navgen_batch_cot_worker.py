#!/usr/bin/env python3
"""
处理单个远程 NavGen success item 的 CoT 批量生成。

流程：
1. 远程检查 item/step_task 是否就绪。
2. rsync step_task 到本地临时目录。
3. 对 item 下每个 step_task 的 *.frame_info.json 生成对应 *.cot.jsonl。
4. 将 *.cot.jsonl 回传到远程 step_task 目录。
5. 成功后删除本地临时目录（可配置关闭）。
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from navgen_cot_synthesizer import NavGenFrameInfoSynthesizer
from vln_data_synthesizer import (
    API_PROVIDER_CHOICES,
    THINKING_MODE_CHOICES,
)


LOGGER = logging.getLogger("navgen_batch_cot_worker")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REMOTE_HOST = "139.196.171.150"
DEFAULT_REMOTE_PORT = 6222
DEFAULT_REMOTE_USER = "root"
DEFAULT_REMOTE_ROOT = (
    "/mnt/data-cpfs/gengshuang/lhvln_dataset/"
    "navgen_batch_20260514_train1000_step_assets_novideo_v2"
)
DEFAULT_LOCAL_STAGE_ROOT = PROJECT_ROOT / "debug" / "navgen_batch_live_stage"
DEFAULT_OUTPUT_SUFFIX = ".cot.jsonl"
DEFAULT_STABLE_SECONDS = 90.0
DEFAULT_NETWORK_COMMAND_RETRIES = 6
DEFAULT_NETWORK_RETRY_DELAY_SECONDS = 5.0
DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS = 20
DEFAULT_SSH_SERVER_ALIVE_INTERVAL_SECONDS = 30
DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX = 6

NETWORK_ERROR_MARKERS = (
    "connection closed by",
    "connection reset by peer",
    "broken pipe",
    "connection timed out",
    "operation timed out",
    "timed out",
    "network is unreachable",
    "connection refused",
    "kex_exchange_identification",
    "unexpected eof",
)

REMOTE_INSPECT_SCRIPT = r"""
import json
import os
import sys
import time

item_dir = sys.argv[1]
stable_seconds = float(sys.argv[2])
output_suffix = sys.argv[3]

now = time.time()
step_task_dir = os.path.join(item_dir, "step_task")
result = {
    "item_id": os.path.basename(item_dir.rstrip("/")),
    "remote_item_dir": item_dir,
    "step_task_dir": step_task_dir,
    "ready": False,
    "stable_seconds": stable_seconds,
    "elapsed_since_update": None,
    "latest_mtime": None,
    "tasks": [],
    "problems": [],
}

if not os.path.isdir(step_task_dir):
    result["problems"].append("missing_step_task_dir")
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)

frame_info_names = sorted(
    name for name in os.listdir(step_task_dir)
    if name.endswith(".frame_info.json")
)
if not frame_info_names:
    result["problems"].append("no_frame_info_json")
    print(json.dumps(result, ensure_ascii=False))
    sys.exit(0)

latest_mtime = 0.0
all_ready = True

for frame_info_name in frame_info_names:
    stem = frame_info_name[:-len(".frame_info.json")]
    frame_info_path = os.path.join(step_task_dir, frame_info_name)
    step_json_path = os.path.join(step_task_dir, stem + ".json")
    subtasks_json_path = os.path.join(step_task_dir, stem + ".subtasks.json")
    assets_dir = os.path.join(step_task_dir, stem + "_frame_assets")
    frames_rgb_dir = os.path.join(assets_dir, "frames_rgb")
    output_jsonl_path = os.path.join(step_task_dir, stem + output_suffix)

    task_problems = []
    if not os.path.isfile(frame_info_path):
        task_problems.append("missing_frame_info")
    if not os.path.isfile(step_json_path):
        task_problems.append("missing_step_json")
    if not os.path.isdir(assets_dir):
        task_problems.append("missing_assets_dir")
    if not os.path.isdir(frames_rgb_dir):
        task_problems.append("missing_frames_rgb_dir")

    rgb_frame_count = 0
    if os.path.isdir(frames_rgb_dir):
        try:
            rgb_frame_count = sum(
                1 for name in os.listdir(frames_rgb_dir)
                if name.lower().endswith(".png")
            )
        except OSError:
            task_problems.append("unreadable_frames_rgb_dir")
        if rgb_frame_count == 0:
            task_problems.append("empty_frames_rgb_dir")

    task_mtimes = []
    for path in (frame_info_path, step_json_path, subtasks_json_path, frames_rgb_dir):
        if os.path.exists(path):
            try:
                task_mtimes.append(os.path.getmtime(path))
            except OSError:
                task_problems.append(f"stat_failed:{os.path.basename(path)}")

    task_latest_mtime = max(task_mtimes) if task_mtimes else None
    if task_latest_mtime is not None:
        latest_mtime = max(latest_mtime, task_latest_mtime)

    output_exists = os.path.isfile(output_jsonl_path) and os.path.getsize(output_jsonl_path) > 0
    task_ready = len(task_problems) == 0
    all_ready = all_ready and task_ready

    result["tasks"].append(
        {
            "stem": stem,
            "frame_info_name": frame_info_name,
            "frame_info_path": frame_info_path,
            "step_json_path": step_json_path,
            "subtasks_json_path": subtasks_json_path,
            "assets_dir": assets_dir,
            "frames_rgb_dir": frames_rgb_dir,
            "rgb_frame_count": rgb_frame_count,
            "output_jsonl_path": output_jsonl_path,
            "output_exists": output_exists,
            "ready": task_ready,
            "latest_mtime": task_latest_mtime,
            "problems": task_problems,
        }
    )

if latest_mtime > 0:
    result["latest_mtime"] = latest_mtime
    result["elapsed_since_update"] = now - latest_mtime
else:
    result["problems"].append("no_valid_mtime")

if not all_ready:
    result["problems"].append("task_assets_incomplete")

if result["elapsed_since_update"] is None or result["elapsed_since_update"] < stable_seconds:
    result["problems"].append("item_not_stable_yet")

result["ready"] = (
    all_ready
    and result["elapsed_since_update"] is not None
    and result["elapsed_since_update"] >= stable_seconds
)

print(json.dumps(result, ensure_ascii=False))
"""


@dataclass(frozen=True)
class SSHConfig:
    host: str
    port: int
    user: str
    network_command_retries: int = DEFAULT_NETWORK_COMMAND_RETRIES
    network_retry_delay_seconds: float = DEFAULT_NETWORK_RETRY_DELAY_SECONDS
    connect_timeout_seconds: int = DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS
    server_alive_interval_seconds: int = DEFAULT_SSH_SERVER_ALIVE_INTERVAL_SECONDS
    server_alive_count_max: int = DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX


@dataclass(frozen=True)
class WorkerConfig:
    ssh: SSHConfig
    local_stage_root: Path
    output_suffix: str
    stable_seconds: float
    delete_local_on_success: bool
    skip_existing_remote_output: bool
    export_markdown: bool
    include_prompt_in_markdown: bool
    api_provider: str
    api_base_url: Optional[str]
    model_name: Optional[str]
    api_key: Optional[str]
    api_key_env: str
    thinking_mode: str
    max_retries: int
    invalid_frame_policy: str
    max_frames: Optional[int]
    max_tasks: Optional[int]


@dataclass
class ProcessResult:
    item_id: str
    remote_item_dir: str
    total_tasks: int
    processed_tasks: int
    uploaded_tasks: int
    skipped_existing_tasks: int
    local_stage_dir: str


def configure_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        force=True,
    )
    # Ensure stderr handler flushes after every record (critical for nohup/redirected output)
    for handler in logging.getLogger().handlers:
        handler.flush = lambda: handler.stream.flush()


def _ssh_host(config: SSHConfig) -> str:
    return f"{config.user}@{config.host}"


def _ssh_options(config: SSHConfig) -> list[str]:
    return [
        "-o",
        "BatchMode=yes",
        "-o",
        f"ConnectTimeout={config.connect_timeout_seconds}",
        "-o",
        f"ServerAliveInterval={config.server_alive_interval_seconds}",
        "-o",
        f"ServerAliveCountMax={config.server_alive_count_max}",
        "-o",
        "TCPKeepAlive=yes",
    ]


def _ssh_command(config: SSHConfig, *remote_args: str) -> list[str]:
    return [
        "ssh",
        "-p",
        str(config.port),
        *_ssh_options(config),
        _ssh_host(config),
        *remote_args,
    ]


def _ssh_rsync_shell(config: SSHConfig) -> str:
    options = " ".join(_ssh_options(config))
    return f"ssh -p {config.port} {options}"


def _is_retryable_network_failure(returncode: int, stdout: str, stderr: str) -> bool:
    combined = f"{stdout}\n{stderr}".lower()
    if returncode == 255:
        return True
    return any(marker in combined for marker in NETWORK_ERROR_MARKERS)


def run_command(
    cmd: list[str],
    *,
    input_text: Optional[str] = None,
    check: bool = True,
    retries: int = 0,
    retry_delay_seconds: float = DEFAULT_NETWORK_RETRY_DELAY_SECONDS,
    retry_on_network_failure: bool = False,
) -> subprocess.CompletedProcess[str]:
    attempt = 0
    while True:
        attempt += 1
        LOGGER.debug("Running command (attempt %d): %s", attempt, " ".join(cmd))
        result = subprocess.run(
            cmd,
            input=input_text,
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode == 0 or not check:
            if result.returncode == 0:
                return result
            if not retry_on_network_failure:
                return result
            if attempt > retries or not _is_retryable_network_failure(
                result.returncode,
                result.stdout,
                result.stderr,
            ):
                return result
        elif result.returncode == 0:
            return result

        retryable = retry_on_network_failure and _is_retryable_network_failure(
            result.returncode,
            result.stdout,
            result.stderr,
        )
        if retryable and attempt <= retries:
            wait_seconds = retry_delay_seconds * (2 ** (attempt - 1))
            LOGGER.warning(
                "network command failed (attempt %d/%d), retry after %.1fs: %s | stderr=%s",
                attempt,
                retries + 1,
                wait_seconds,
                " ".join(cmd),
                result.stderr.strip(),
            )
            time.sleep(wait_seconds)
            continue

        if check:
            raise RuntimeError(
                f"command failed ({result.returncode}): {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )
        return result


def run_remote_python(config: SSHConfig, script: str, args: list[str]) -> str:
    cmd = _ssh_command(config, "python3", "-", *args)
    result = run_command(
        cmd,
        input_text=script,
        retries=config.network_command_retries,
        retry_delay_seconds=config.network_retry_delay_seconds,
        retry_on_network_failure=True,
    )
    return result.stdout


def inspect_remote_item(
    ssh_config: SSHConfig,
    remote_item_dir: str,
    stable_seconds: float,
    output_suffix: str,
) -> dict[str, Any]:
    raw_text = run_remote_python(
        ssh_config,
        REMOTE_INSPECT_SCRIPT,
        [remote_item_dir, str(stable_seconds), output_suffix],
    )
    try:
        return json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"failed to parse remote inspect result for {remote_item_dir}: {exc}\n"
            f"raw output:\n{raw_text}"
        ) from exc


def rsync_step_task_dir(
    ssh_config: SSHConfig,
    remote_item_dir: str,
    local_item_dir: Path,
) -> Path:
    local_step_task_dir = local_item_dir / "step_task"
    local_step_task_dir.parent.mkdir(parents=True, exist_ok=True)
    remote_step_task_dir = remote_item_dir.rstrip("/") + "/step_task/"
    remote_source = f"{_ssh_host(ssh_config)}:{remote_step_task_dir}"
    run_command(
        [
            "rsync",
            "-a",
            "-s",
            "--delete",
            "-e",
            _ssh_rsync_shell(ssh_config),
            remote_source,
            str(local_step_task_dir),
        ],
        retries=ssh_config.network_command_retries,
        retry_delay_seconds=ssh_config.network_retry_delay_seconds,
        retry_on_network_failure=True,
    )
    return local_step_task_dir


def upload_file_to_remote(
    ssh_config: SSHConfig,
    local_path: Path,
    remote_dir: str,
) -> None:
    remote_target = f"{_ssh_host(ssh_config)}:{remote_dir.rstrip('/')}/"
    run_command(
        [
            "rsync",
            "-a",
            "-s",
            "-e",
            _ssh_rsync_shell(ssh_config),
            str(local_path),
            remote_target,
        ],
        retries=ssh_config.network_command_retries,
        retry_delay_seconds=ssh_config.network_retry_delay_seconds,
        retry_on_network_failure=True,
    )


def verify_remote_file_exists(
    ssh_config: SSHConfig,
    remote_file_path: str,
) -> bool:
    script = f"""
import os

path = {json.dumps(remote_file_path)}
ok = os.path.isfile(path) and os.path.getsize(path) > 0
print("1" if ok else "0")
"""
    result = run_command(
        _ssh_command(ssh_config, "python3", "-"),
        input_text=script,
        retries=ssh_config.network_command_retries,
        retry_delay_seconds=ssh_config.network_retry_delay_seconds,
        retry_on_network_failure=True,
    )
    output = result.stdout.strip()
    return output == "1"


def export_markdown(
    output_jsonl_path: Path,
    include_prompt: bool,
) -> Path:
    output_dir = output_jsonl_path.with_name(f"{output_jsonl_path.stem}_md")
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "cot_ann" / "jsonl_to_frame_markdown.py"),
        "--input",
        str(output_jsonl_path),
        "--output-dir",
        str(output_dir),
    ]
    if include_prompt:
        cmd.append("--include-prompt")
    run_command(cmd)
    return output_dir


def create_synthesizer(config: WorkerConfig) -> NavGenFrameInfoSynthesizer:
    return NavGenFrameInfoSynthesizer(
        max_retries=config.max_retries,
        api_provider=config.api_provider,
        api_base_url=config.api_base_url,
        model_name=config.model_name,
        api_key=config.api_key,
        api_key_env=config.api_key_env,
        thinking_mode=config.thinking_mode,
        invalid_frame_policy=config.invalid_frame_policy,
    )


def process_remote_item(
    config: WorkerConfig,
    remote_item_dir: str,
) -> ProcessResult:
    inspect_result = inspect_remote_item(
        ssh_config=config.ssh,
        remote_item_dir=remote_item_dir,
        stable_seconds=config.stable_seconds,
        output_suffix=config.output_suffix,
    )
    item_id = str(inspect_result["item_id"])
    tasks: list[dict[str, Any]] = inspect_result.get("tasks", [])
    if not tasks:
        raise RuntimeError(f"{item_id} has no step_task frame_info files")
    if not inspect_result.get("ready"):
        raise RuntimeError(
            f"{item_id} is not ready yet: {inspect_result.get('problems')}"
        )

    if config.max_tasks is not None:
        tasks = tasks[: config.max_tasks]

    if config.skip_existing_remote_output and all(
        task.get("output_exists") for task in tasks
    ):
        LOGGER.info(
            "%s already has %d/%d remote CoT outputs, skipping generation.",
            item_id,
            len(tasks),
            len(tasks),
        )
        return ProcessResult(
            item_id=item_id,
            remote_item_dir=remote_item_dir,
            total_tasks=len(tasks),
            processed_tasks=0,
            uploaded_tasks=0,
            skipped_existing_tasks=len(tasks),
            local_stage_dir=str(config.local_stage_root / item_id),
        )

    local_item_dir = config.local_stage_root / item_id
    local_step_task_dir = rsync_step_task_dir(
        ssh_config=config.ssh,
        remote_item_dir=remote_item_dir,
        local_item_dir=local_item_dir,
    )
    synthesizer = create_synthesizer(config)

    processed_tasks = 0
    uploaded_tasks = 0
    skipped_existing_tasks = 0
    remote_step_task_dir = inspect_result["step_task_dir"]

    try:
        for task in tasks:
            stem = str(task["stem"])
            local_frame_info = local_step_task_dir / f"{stem}.frame_info.json"
            local_step_json = local_step_task_dir / f"{stem}.json"
            local_output_jsonl = local_step_task_dir / f"{stem}{config.output_suffix}"
            remote_output_jsonl = f"{remote_step_task_dir}/{stem}{config.output_suffix}"

            if config.skip_existing_remote_output and task.get("output_exists"):
                LOGGER.info(
                    "%s | task '%s' already has remote output, skip.",
                    item_id,
                    stem,
                )
                skipped_existing_tasks += 1
                continue

            if not local_frame_info.exists():
                raise FileNotFoundError(f"missing local frame_info: {local_frame_info}")
            if not local_step_json.exists():
                raise FileNotFoundError(f"missing local step json: {local_step_json}")

            if local_output_jsonl.exists():
                local_output_jsonl.unlink()
            LOGGER.info(
                "%s | generating CoT for task '%s'.",
                item_id,
                stem,
            )
            synthesizer.run_frame_info(
                frame_info=str(local_frame_info),
                step_task_json=str(local_step_json),
                output_jsonl=str(local_output_jsonl),
                max_frames=config.max_frames,
            )
            if not local_output_jsonl.exists() or local_output_jsonl.stat().st_size == 0:
                raise RuntimeError(
                    f"generated output missing or empty: {local_output_jsonl}"
                )

            if config.export_markdown:
                export_markdown(
                    output_jsonl_path=local_output_jsonl,
                    include_prompt=config.include_prompt_in_markdown,
                )

            upload_file_to_remote(
                ssh_config=config.ssh,
                local_path=local_output_jsonl,
                remote_dir=remote_step_task_dir,
            )
            if not verify_remote_file_exists(config.ssh, remote_output_jsonl):
                raise RuntimeError(
                    f"remote upload verification failed: {remote_output_jsonl}"
                )
            processed_tasks += 1
            uploaded_tasks += 1
            LOGGER.info(
                "%s | uploaded '%s' to remote step_task dir.",
                item_id,
                local_output_jsonl.name,
            )

        return ProcessResult(
            item_id=item_id,
            remote_item_dir=remote_item_dir,
            total_tasks=len(tasks),
            processed_tasks=processed_tasks,
            uploaded_tasks=uploaded_tasks,
            skipped_existing_tasks=skipped_existing_tasks,
            local_stage_dir=str(local_item_dir),
        )
    finally:
        if config.delete_local_on_success and local_item_dir.exists():
            if processed_tasks + skipped_existing_tasks == len(tasks):
                shutil.rmtree(local_item_dir)
                LOGGER.info("%s | removed local stage dir %s", item_id, local_item_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate CoT for every step_task under one remote success item.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--remote-item-dir",
        required=True,
        help="Remote item directory, for example .../success/item_0008",
    )
    parser.add_argument(
        "--remote-host",
        default=DEFAULT_REMOTE_HOST,
        help="Remote SSH host.",
    )
    parser.add_argument(
        "--remote-port",
        type=int,
        default=DEFAULT_REMOTE_PORT,
        help="Remote SSH port.",
    )
    parser.add_argument(
        "--remote-user",
        default=DEFAULT_REMOTE_USER,
        help="Remote SSH user.",
    )
    parser.add_argument(
        "--local-stage-root",
        default=str(DEFAULT_LOCAL_STAGE_ROOT),
        help="Local staging root. Item data is synced here temporarily.",
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help="Remote/local output suffix appended to each step_task stem.",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=DEFAULT_STABLE_SECONDS,
        help="Require remote item to stay unchanged for this many seconds before processing.",
    )
    parser.add_argument(
        "--delete-local-on-success",
        dest="delete_local_on_success",
        action="store_true",
        help="Delete the local staged item directory after every task succeeds.",
    )
    parser.add_argument(
        "--keep-local-on-success",
        dest="delete_local_on_success",
        action="store_false",
        help="Keep local staged files after success for debugging.",
    )
    parser.set_defaults(delete_local_on_success=True)
    parser.add_argument(
        "--skip-existing-remote-output",
        dest="skip_existing_remote_output",
        action="store_true",
        help="Skip a task if remote <stem><output_suffix> already exists and is non-empty.",
    )
    parser.add_argument(
        "--force-regenerate-remote-output",
        dest="skip_existing_remote_output",
        action="store_false",
        help="Always regenerate and re-upload even if remote output already exists.",
    )
    parser.set_defaults(skip_existing_remote_output=True)
    parser.add_argument(
        "--export-markdown",
        action="store_true",
        help="Export local per-frame markdown beside each generated JSONL before cleanup.",
    )
    parser.add_argument(
        "--include-prompt-in-markdown",
        action="store_true",
        help="When exporting markdown, include the prompt text.",
    )
    parser.add_argument(
        "--api-provider",
        choices=API_PROVIDER_CHOICES,
        default="dashscope",
        help="API provider type.",
    )
    parser.add_argument(
        "--api-base-url",
        default=None,
        help="OpenAI compatible API base url.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Explicit API key.",
    )
    parser.add_argument(
        "--api-key-env",
        default="DASHSCOPE_API_KEY",
        help="Environment variable name for API key.",
    )
    parser.add_argument(
        "--thinking-mode",
        choices=THINKING_MODE_CHOICES,
        default="off",
        help="Thinking mode.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum API retry count.",
    )
    parser.add_argument(
        "--invalid-frame-policy",
        choices=("skip", "template"),
        default="template",
        help="Policy when the model repeatedly returns invalid output.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional debug limit on the number of frames processed per step_task.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional debug limit on the number of step_task files processed in the item.",
    )
    parser.add_argument(
        "--network-command-retries",
        type=int,
        default=DEFAULT_NETWORK_COMMAND_RETRIES,
        help="Retries for ssh/rsync/network shell commands.",
    )
    parser.add_argument(
        "--network-retry-delay-seconds",
        type=float,
        default=DEFAULT_NETWORK_RETRY_DELAY_SECONDS,
        help="Base retry delay for ssh/rsync/network shell commands.",
    )
    parser.add_argument(
        "--ssh-connect-timeout-seconds",
        type=int,
        default=DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS,
        help="SSH connect timeout in seconds.",
    )
    parser.add_argument(
        "--ssh-server-alive-interval-seconds",
        type=int,
        default=DEFAULT_SSH_SERVER_ALIVE_INTERVAL_SECONDS,
        help="SSH ServerAliveInterval in seconds.",
    )
    parser.add_argument(
        "--ssh-server-alive-count-max",
        type=int,
        default=DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX,
        help="SSH ServerAliveCountMax.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_logging(verbose=args.verbose)

    worker_config = WorkerConfig(
        ssh=SSHConfig(
            host=args.remote_host,
            port=args.remote_port,
            user=args.remote_user,
            network_command_retries=args.network_command_retries,
            network_retry_delay_seconds=args.network_retry_delay_seconds,
            connect_timeout_seconds=args.ssh_connect_timeout_seconds,
            server_alive_interval_seconds=args.ssh_server_alive_interval_seconds,
            server_alive_count_max=args.ssh_server_alive_count_max,
        ),
        local_stage_root=Path(args.local_stage_root).expanduser().resolve(),
        output_suffix=args.output_suffix,
        stable_seconds=args.stable_seconds,
        delete_local_on_success=args.delete_local_on_success,
        skip_existing_remote_output=args.skip_existing_remote_output,
        export_markdown=args.export_markdown,
        include_prompt_in_markdown=args.include_prompt_in_markdown,
        api_provider=args.api_provider,
        api_base_url=args.api_base_url,
        model_name=args.model,
        api_key=args.api_key,
        api_key_env=args.api_key_env,
        thinking_mode=args.thinking_mode,
        max_retries=args.max_retries,
        invalid_frame_policy=args.invalid_frame_policy,
        max_frames=args.max_frames,
        max_tasks=args.max_tasks,
    )

    result = process_remote_item(
        config=worker_config,
        remote_item_dir=args.remote_item_dir,
    )
    print(json.dumps(result.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
