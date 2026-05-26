#!/usr/bin/env python3
"""Submit a shell command to a shared-directory job queue."""

from __future__ import annotations

import argparse
import getpass
import json
import os
import socket
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_QUEUE_ROOT = "/inspire/hdd/global_user/wanrui-p-wanrui/shared_job_queue"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def compact_timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_queue(root: Path) -> dict[str, Path]:
    dirs = {
        "pending": root / "pending",
        "running": root / "running",
        "done": root / "done",
        "failed": root / "failed",
        "logs": root / "logs",
        "workers": root / "workers",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def parse_env_pairs(pairs: list[str]) -> dict[str, str]:
    env: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--env expects KEY=VALUE, got: {pair}")
        key, value = pair.split("=", 1)
        if not key:
            raise ValueError(f"--env has empty key: {pair}")
        env[key] = value
    return env


def load_command(args: argparse.Namespace) -> str:
    if args.cmd and args.cmd_file:
        raise ValueError("use either --cmd or --cmd-file, not both")
    if args.cmd_file:
        return Path(args.cmd_file).expanduser().read_text(encoding="utf-8").strip()
    if args.cmd:
        return args.cmd.strip()
    raise ValueError("one of --cmd or --cmd-file is required")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-root",
        default=os.environ.get("SHARED_JOB_QUEUE", DEFAULT_QUEUE_ROOT),
        help=f"shared queue directory (default: {DEFAULT_QUEUE_ROOT})",
    )
    parser.add_argument("--name", default="job", help="human-readable job name")
    parser.add_argument("--workdir", default=os.getcwd(), help="directory where the command runs")
    parser.add_argument("--cmd", help="shell command executed by the worker with /bin/bash -lc")
    parser.add_argument("--cmd-file", help="read shell command from this UTF-8 text file")
    parser.add_argument("--env", action="append", default=[], help="extra environment KEY=VALUE")
    parser.add_argument("--timeout-sec", type=int, default=0, help="0 means no timeout")
    parser.add_argument(
        "--token",
        default=os.environ.get("SHARED_JOB_QUEUE_TOKEN"),
        help="optional shared token; only needed if worker was started with --token",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.timeout_sec < 0:
        raise ValueError("--timeout-sec must be >= 0")

    queue_root = Path(args.queue_root).expanduser().resolve()
    dirs = ensure_queue(queue_root)
    command = load_command(args)
    env = parse_env_pairs(args.env)

    job_id = f"{compact_timestamp()}-{args.name}-{uuid.uuid4().hex[:8]}"
    safe_job_id = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in job_id)
    job: dict[str, Any] = {
        "id": safe_job_id,
        "name": args.name,
        "cmd": command,
        "workdir": str(Path(args.workdir).expanduser().resolve()),
        "env": env,
        "timeout_sec": args.timeout_sec,
        "status": "pending",
        "created_at": utc_now(),
        "created_by": getpass.getuser(),
        "created_on": socket.gethostname(),
    }
    if args.token is not None:
        job["token"] = args.token

    job_path = dirs["pending"] / f"{safe_job_id}.json"
    atomic_write_json(job_path, job)

    print(f"submitted: {safe_job_id}")
    print(f"job_file: {job_path}")
    print(f"log_file: {dirs['logs'] / (safe_job_id + '.log')}")
    print(f"queue_root: {queue_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
