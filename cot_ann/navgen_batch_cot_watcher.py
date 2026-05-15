#!/usr/bin/env python3
"""
监听远程 batch_summary.tsv，并对新成功的 success/item_xxxx 逐个生成 CoT。

特性：
1. 轮询远程 batch_summary.tsv，发现 status=ok 的新 item。
2. 先检查 item/step_task 是否完整且稳定，再开始处理。
3. 调用 navgen_batch_cot_worker.py 中的逻辑处理单个 item。
4. 将状态写入本地 sqlite，支持断点续跑与失败重试。
"""

from __future__ import annotations

import argparse
import csv
from concurrent.futures import Future, ProcessPoolExecutor
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from navgen_batch_cot_worker import (
    DEFAULT_LOCAL_STAGE_ROOT,
    DEFAULT_NETWORK_COMMAND_RETRIES,
    DEFAULT_NETWORK_RETRY_DELAY_SECONDS,
    DEFAULT_OUTPUT_SUFFIX,
    DEFAULT_REMOTE_HOST,
    DEFAULT_REMOTE_PORT,
    DEFAULT_REMOTE_ROOT,
    DEFAULT_REMOTE_USER,
    DEFAULT_SSH_CONNECT_TIMEOUT_SECONDS,
    DEFAULT_SSH_SERVER_ALIVE_COUNT_MAX,
    DEFAULT_SSH_SERVER_ALIVE_INTERVAL_SECONDS,
    DEFAULT_STABLE_SECONDS,
    ProcessResult,
    SSHConfig,
    WorkerConfig,
    configure_logging,
    inspect_remote_item,
    process_remote_item,
    run_command,
)
from vln_data_synthesizer import API_PROVIDER_CHOICES, THINKING_MODE_CHOICES


LOGGER = logging.getLogger("navgen_batch_cot_watcher")
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_STATE_DB = PROJECT_ROOT / "debug" / "navgen_batch_cot_state.sqlite3"
DEFAULT_STALE_PROCESSING_SECONDS = 1800.0


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def init_db(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS items (
                item_id TEXT PRIMARY KEY,
                summary_status TEXT NOT NULL,
                summary_line INTEGER NOT NULL,
                remote_item_dir TEXT NOT NULL,
                state TEXT NOT NULL,
                attempts INTEGER NOT NULL DEFAULT 0,
                last_error TEXT,
                last_seen_at TEXT,
                ready_checked_at TEXT,
                processing_started_at TEXT,
                finished_at TEXT,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()


def open_db(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def fetch_remote_summary_text(ssh_config: SSHConfig, summary_path: str) -> str:
    result = run_command(
        [
            "ssh",
            "-p",
            str(ssh_config.port),
            *[
                "-o",
                "BatchMode=yes",
                "-o",
                f"ConnectTimeout={ssh_config.connect_timeout_seconds}",
                "-o",
                f"ServerAliveInterval={ssh_config.server_alive_interval_seconds}",
                "-o",
                f"ServerAliveCountMax={ssh_config.server_alive_count_max}",
                "-o",
                "TCPKeepAlive=yes",
            ],
            f"{ssh_config.user}@{ssh_config.host}",
            "cat",
            summary_path,
        ],
        retries=ssh_config.network_command_retries,
        retry_delay_seconds=ssh_config.network_retry_delay_seconds,
        retry_on_network_failure=True,
    )
    return result.stdout


def parse_summary_rows(summary_text: str) -> list[list[str]]:
    rows: list[list[str]] = []
    reader = csv.reader(summary_text.splitlines(), delimiter="\t")
    for row in reader:
        if not row:
            continue
        rows.append(row)
    return rows


def derive_remote_item_dir(remote_root: str, item_id: str) -> str:
    return f"{remote_root.rstrip('/')}/success/{item_id}"


def upsert_items_from_summary(
    conn: sqlite3.Connection,
    rows: list[list[str]],
    remote_root: str,
) -> int:
    now = utc_now_iso()
    inserted_or_updated = 0
    for line_no, row in enumerate(rows, start=1):
        if len(row) < 2:
            continue
        item_id = row[0].strip()
        status = row[1].strip()
        if not item_id.startswith("item_"):
            continue
        if status != "ok":
            continue

        remote_item_dir = derive_remote_item_dir(remote_root, item_id)
        current = conn.execute(
            "SELECT item_id, summary_line, state FROM items WHERE item_id = ?",
            (item_id,),
        ).fetchone()

        if current is None:
            conn.execute(
                """
                INSERT INTO items (
                    item_id, summary_status, summary_line, remote_item_dir,
                    state, attempts, last_error, last_seen_at,
                    ready_checked_at, processing_started_at, finished_at, updated_at
                ) VALUES (?, ?, ?, ?, 'discovered', 0, NULL, ?, NULL, NULL, NULL, ?)
                """,
                (
                    item_id,
                    status,
                    line_no,
                    remote_item_dir,
                    now,
                    now,
                ),
            )
            inserted_or_updated += 1
            continue

        conn.execute(
            """
            UPDATE items
            SET summary_status = ?, summary_line = ?, remote_item_dir = ?,
                last_seen_at = ?, updated_at = ?
            WHERE item_id = ?
            """,
            (
                status,
                line_no,
                remote_item_dir,
                now,
                now,
                item_id,
            ),
        )
        inserted_or_updated += 1
    conn.commit()
    return inserted_or_updated


def select_candidate_items(
    conn: sqlite3.Connection,
    max_item_attempts: int,
    max_items_per_cycle: int,
) -> list[sqlite3.Row]:
    query = """
        SELECT *
        FROM items
        WHERE summary_status = 'ok'
          AND state IN ('discovered', 'waiting_remote', 'failed')
          AND attempts < ?
        ORDER BY summary_line ASC, item_id ASC
        LIMIT ?
    """
    return list(
        conn.execute(query, (max_item_attempts, max_items_per_cycle)).fetchall()
    )


def reset_processing_items(
    conn: sqlite3.Connection,
    stale_processing_seconds: float,
) -> int:
    now = utc_now_iso()
    cutoff_expr = f"-{int(max(1, stale_processing_seconds))} seconds"
    cursor = conn.execute(
        """
        UPDATE items
        SET state = 'discovered',
            last_error = COALESCE(
                last_error,
                'watcher restarted and stale processing item was re-queued'
            ),
            updated_at = ?
        WHERE state = 'processing'
          AND (
            processing_started_at IS NULL
            OR datetime(processing_started_at) <= datetime('now', ?)
          )
        """,
        (now, cutoff_expr),
    )
    conn.commit()
    return cursor.rowcount


def mark_item_waiting(
    conn: sqlite3.Connection,
    item_id: str,
    error_text: str,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE items
        SET state = 'waiting_remote',
            last_error = ?,
            ready_checked_at = ?,
            updated_at = ?
        WHERE item_id = ?
        """,
        (error_text, now, now, item_id),
    )
    conn.commit()


def mark_item_processing(
    conn: sqlite3.Connection,
    item_id: str,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE items
        SET state = 'processing',
            attempts = attempts + 1,
            last_error = NULL,
            processing_started_at = ?,
            updated_at = ?
        WHERE item_id = ?
        """,
        (now, now, item_id),
    )
    conn.commit()


def mark_item_done(
    conn: sqlite3.Connection,
    item_id: str,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE items
        SET state = 'done',
            last_error = NULL,
            finished_at = ?,
            updated_at = ?
        WHERE item_id = ?
        """,
        (now, now, item_id),
    )
    conn.commit()


def mark_item_failed(
    conn: sqlite3.Connection,
    item_id: str,
    error_text: str,
) -> None:
    now = utc_now_iso()
    conn.execute(
        """
        UPDATE items
        SET state = 'failed',
            last_error = ?,
            updated_at = ?
        WHERE item_id = ?
        """,
        (error_text, now, item_id),
    )
    conn.commit()


def summarize_readiness(readiness: dict[str, Any]) -> str:
    problems = readiness.get("problems") or []
    elapsed = readiness.get("elapsed_since_update")
    if elapsed is not None:
        return (
            f"ready={readiness.get('ready')} problems={problems} "
            f"elapsed_since_update={elapsed:.1f}s"
        )
    return f"ready={readiness.get('ready')} problems={problems}"


def inspect_candidate_item(
    worker_config: WorkerConfig,
    item_row: sqlite3.Row,
) -> tuple[bool, str]:
    item_id = str(item_row["item_id"])
    remote_item_dir = str(item_row["remote_item_dir"])
    readiness = inspect_remote_item(
        ssh_config=worker_config.ssh,
        remote_item_dir=remote_item_dir,
        stable_seconds=worker_config.stable_seconds,
        output_suffix=worker_config.output_suffix,
    )
    if not readiness.get("ready"):
        message = summarize_readiness(readiness)
        return False, message
    return True, summarize_readiness(readiness)


def poll_running_jobs(
    conn: sqlite3.Connection,
    running_jobs: dict[str, dict[str, Any]],
) -> int:
    completed = 0
    for item_id, job in list(running_jobs.items()):
        future: Future = job["future"]
        if not future.done():
            continue
        completed += 1
        try:
            result: ProcessResult = future.result()
        except Exception as exc:  # noqa: BLE001
            error_text = f"{type(exc).__name__}: {exc}"
            mark_item_failed(conn, item_id, error_text)
            LOGGER.exception("%s processing failed", item_id)
        else:
            mark_item_done(conn, item_id)
            LOGGER.info(
                "%s done | total_tasks=%d processed=%d skipped_existing=%d uploaded=%d",
                item_id,
                result.total_tasks,
                result.processed_tasks,
                result.skipped_existing_tasks,
                result.uploaded_tasks,
            )
        finally:
            running_jobs.pop(item_id, None)
    return completed


def run_watcher(args: argparse.Namespace) -> None:
    db_path = Path(args.state_db).expanduser().resolve()
    init_db(db_path)

    ssh_config = SSHConfig(
        host=args.remote_host,
        port=args.remote_port,
        user=args.remote_user,
        network_command_retries=args.network_command_retries,
        network_retry_delay_seconds=args.network_retry_delay_seconds,
        connect_timeout_seconds=args.ssh_connect_timeout_seconds,
        server_alive_interval_seconds=args.ssh_server_alive_interval_seconds,
        server_alive_count_max=args.ssh_server_alive_count_max,
    )
    worker_config = WorkerConfig(
        ssh=ssh_config,
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

    with open_db(db_path) as conn:
        reset_count = reset_processing_items(
            conn,
            stale_processing_seconds=args.stale_processing_seconds,
        )
        if reset_count:
            LOGGER.warning(
                "reset %d processing item(s) back to discovered after watcher restart.",
                reset_count,
            )

        running_jobs: dict[str, dict[str, Any]] = {}
        total_started = 0

        with ProcessPoolExecutor(max_workers=args.max_concurrent_workers) as executor:
            while True:
                poll_running_jobs(conn, running_jobs)

                try:
                    summary_text = fetch_remote_summary_text(ssh_config, args.summary_path)
                    rows = parse_summary_rows(summary_text)
                    upsert_count = upsert_items_from_summary(
                        conn=conn,
                        rows=rows,
                        remote_root=args.remote_root,
                    )
                    LOGGER.info(
                        "summary scan complete: %d rows, %d ok items upserted.",
                        len(rows),
                        upsert_count,
                    )

                    available_slots = max(0, args.max_concurrent_workers - len(running_jobs))
                    if available_slots == 0:
                        LOGGER.info(
                            "all %d worker slot(s) are busy.",
                            args.max_concurrent_workers,
                        )
                    else:
                        candidates = select_candidate_items(
                            conn=conn,
                            max_item_attempts=args.max_item_attempts,
                            max_items_per_cycle=min(args.max_items_per_cycle, available_slots),
                        )
                        if not candidates:
                            LOGGER.info("no pending ok items found.")
                        else:
                            LOGGER.info(
                                "found %d candidate item(s) this cycle, %d slot(s) available.",
                                len(candidates),
                                available_slots,
                            )

                        for item_row in candidates:
                            item_id = str(item_row["item_id"])
                            remote_item_dir = str(item_row["remote_item_dir"])
                            try:
                                ready, message = inspect_candidate_item(worker_config, item_row)
                            except Exception as exc:  # noqa: BLE001
                                error_text = f"readiness check failed: {type(exc).__name__}: {exc}"
                                LOGGER.warning("%s readiness check failed: %s", item_id, error_text)
                                mark_item_waiting(conn, item_id, error_text)
                                continue

                            if not ready:
                                LOGGER.info("%s not ready yet: %s", item_id, message)
                                mark_item_waiting(conn, item_id, message)
                                continue

                            mark_item_processing(conn, item_id)
                            future = executor.submit(
                                process_remote_item,
                                worker_config,
                                remote_item_dir,
                            )
                            running_jobs[item_id] = {
                                "future": future,
                                "remote_item_dir": remote_item_dir,
                            }
                            total_started += 1
                            LOGGER.info(
                                "%s is ready. Submitted to worker pool (%d/%d running).",
                                item_id,
                                len(running_jobs),
                                args.max_concurrent_workers,
                            )
                            if args.once and args.max_total_items is not None:
                                if total_started >= args.max_total_items:
                                    break
                except Exception as exc:  # noqa: BLE001
                    LOGGER.exception(
                        "summary scan failed but watcher will keep running: %s",
                        exc,
                    )

                if args.once:
                    while running_jobs:
                        poll_running_jobs(conn, running_jobs)
                        if running_jobs:
                            time.sleep(1.0)
                    LOGGER.info("--once mode finished one summary scan.")
                    return

                LOGGER.info(
                    "sleeping %.1f seconds before next poll. running_jobs=%d",
                    args.poll_interval,
                    len(running_jobs),
                )
                time.sleep(args.poll_interval)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Watch remote batch_summary.tsv and continuously annotate new success items.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--remote-root",
        default=DEFAULT_REMOTE_ROOT,
        help="Remote batch root directory.",
    )
    parser.add_argument(
        "--summary-path",
        default=None,
        help="Remote batch_summary.tsv path. Defaults to <remote-root>/batch_summary.tsv.",
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
        "--state-db",
        default=str(DEFAULT_STATE_DB),
        help="Local sqlite state database path.",
    )
    parser.add_argument(
        "--poll-interval",
        type=float,
        default=120.0,
        help="Polling interval in seconds.",
    )
    parser.add_argument(
        "--stable-seconds",
        type=float,
        default=DEFAULT_STABLE_SECONDS,
        help="Require remote item to stay unchanged for this many seconds.",
    )
    parser.add_argument(
        "--max-item-attempts",
        type=int,
        default=20,
        help="Maximum processing attempts per item.",
    )
    parser.add_argument(
        "--max-items-per-cycle",
        type=int,
        default=5,
        help="Maximum candidate items processed in one polling cycle.",
    )
    parser.add_argument(
        "--max-concurrent-workers",
        type=int,
        default=5,
        help="Maximum number of item workers running in parallel.",
    )
    parser.add_argument(
        "--stale-processing-seconds",
        type=float,
        default=DEFAULT_STALE_PROCESSING_SECONDS,
        help="Only reset processing items older than this threshold on watcher startup.",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one scan/process cycle and exit.",
    )
    parser.add_argument(
        "--max-total-items",
        type=int,
        default=None,
        help="When used with --once, stop after successfully processing this many items.",
    )
    parser.add_argument(
        "--local-stage-root",
        default=str(DEFAULT_LOCAL_STAGE_ROOT),
        help="Local staging root used by the item worker.",
    )
    parser.add_argument(
        "--output-suffix",
        default=DEFAULT_OUTPUT_SUFFIX,
        help="Remote/local output suffix appended to each step_task stem.",
    )
    parser.add_argument(
        "--delete-local-on-success",
        dest="delete_local_on_success",
        action="store_true",
        help="Delete local staged files after a full item succeeds.",
    )
    parser.add_argument(
        "--keep-local-on-success",
        dest="delete_local_on_success",
        action="store_false",
        help="Keep local staged files after success.",
    )
    parser.set_defaults(delete_local_on_success=True)
    parser.add_argument(
        "--skip-existing-remote-output",
        dest="skip_existing_remote_output",
        action="store_true",
        help="Skip a step_task when remote <stem><output_suffix> already exists.",
    )
    parser.add_argument(
        "--force-regenerate-remote-output",
        dest="skip_existing_remote_output",
        action="store_false",
        help="Always regenerate and re-upload remote outputs.",
    )
    parser.set_defaults(skip_existing_remote_output=True)
    parser.add_argument(
        "--export-markdown",
        action="store_true",
        help="Export local markdown beside generated JSONL before optional cleanup.",
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
        help="Optional debug limit on the number of frames per step_task.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=None,
        help="Optional debug limit on the number of step_task files per item.",
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
    if args.summary_path is None:
        args.summary_path = f"{args.remote_root.rstrip('/')}/batch_summary.tsv"
    configure_logging(verbose=args.verbose)
    LOGGER.info(
        "watching %s via %s@%s:%s",
        args.summary_path,
        args.remote_user,
        args.remote_host,
        args.remote_port,
    )
    run_watcher(args)


if __name__ == "__main__":
    main()
