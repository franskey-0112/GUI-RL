#!/usr/bin/env python3
"""Poll a shared-directory job queue and execute submitted shell commands."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_QUEUE_ROOT = "/inspire/hdd/global_user/wanrui-p-wanrui/shared_job_queue"
DEFAULT_ALLOWED_ROOT = "/inspire/hdd/global_user/wanrui-p-wanrui"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def queue_dirs(root: Path) -> dict[str, Path]:
    return {
        "pending": root / "pending",
        "running": root / "running",
        "done": root / "done",
        "failed": root / "failed",
        "logs": root / "logs",
        "workers": root / "workers",
    }


def ensure_queue(root: Path) -> dict[str, Path]:
    dirs = queue_dirs(root)
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def resolve_under(path: Path, allowed_roots: list[Path]) -> bool:
    resolved = path.resolve()
    for root in allowed_roots:
        try:
            resolved.relative_to(root.resolve())
            return True
        except ValueError:
            continue
    return False


def claim_one_job(dirs: dict[str, Path]) -> Path | None:
    for pending_path in sorted(dirs["pending"].glob("*.json")):
        running_path = dirs["running"] / pending_path.name
        try:
            os.replace(pending_path, running_path)
            return running_path
        except FileNotFoundError:
            continue
        except OSError as exc:
            print(f"[{utc_now()}] claim failed for {pending_path}: {exc}", flush=True)
            continue
    return None


def write_heartbeat(dirs: dict[str, Path], worker_id: str, processed_jobs: int) -> None:
    heartbeat = {
        "worker_id": worker_id,
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "updated_at": utc_now(),
        "processed_jobs": processed_jobs,
    }
    atomic_write_json(dirs["workers"] / f"{worker_id}.json", heartbeat)


def append_job_event(log_path: Path, message: str) -> None:
    with log_path.open("a", encoding="utf-8") as f:
        f.write(f"[{utc_now()}] {message}\n")


def fail_without_running(
    job_path: Path,
    dirs: dict[str, Path],
    job: dict[str, Any],
    reason: str,
    log_path: Path,
) -> None:
    job.update(
        {
            "status": "failed",
            "exit_code": None,
            "failure_reason": reason,
            "finished_at": utc_now(),
            "log_path": str(log_path),
        }
    )
    append_job_event(log_path, f"FAILED BEFORE START: {reason}")
    atomic_write_json(job_path, job)
    os.replace(job_path, dirs["failed"] / job_path.name)


def validate_job(
    job: dict[str, Any],
    allowed_roots: list[Path],
    expected_token: str | None,
) -> tuple[str, Path, dict[str, str], int]:
    if expected_token is not None and job.get("token") != expected_token:
        raise ValueError("job token does not match worker token")

    cmd = job.get("cmd")
    if not isinstance(cmd, str) or not cmd.strip():
        raise ValueError("job field 'cmd' must be a non-empty string")

    workdir_value = job.get("workdir")
    if not isinstance(workdir_value, str) or not workdir_value.strip():
        raise ValueError("job field 'workdir' must be a non-empty string")
    workdir = Path(workdir_value).expanduser().resolve()
    if not workdir.is_dir():
        raise ValueError(f"workdir does not exist: {workdir}")
    if allowed_roots and not resolve_under(workdir, allowed_roots):
        roots = ", ".join(str(root.resolve()) for root in allowed_roots)
        raise ValueError(f"workdir {workdir} is outside allowed roots: {roots}")

    env = job.get("env", {})
    if env is None:
        env = {}
    if not isinstance(env, dict):
        raise ValueError("job field 'env' must be an object")
    str_env = {str(k): str(v) for k, v in env.items()}

    timeout_sec = int(job.get("timeout_sec") or 0)
    if timeout_sec < 0:
        raise ValueError("timeout_sec must be >= 0")

    return cmd, workdir, str_env, timeout_sec


def terminate_process_group(proc: subprocess.Popen[Any], grace_sec: int = 30) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + grace_sec
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            return
        time.sleep(1)
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return


def run_job(
    job_path: Path,
    dirs: dict[str, Path],
    worker_id: str,
    allowed_roots: list[Path],
    expected_token: str | None,
) -> int | None:
    job = read_json(job_path)
    job_id = str(job.get("id") or job_path.stem)
    log_path = dirs["logs"] / f"{job_id}.log"

    try:
        cmd, workdir, job_env, timeout_sec = validate_job(job, allowed_roots, expected_token)
    except Exception as exc:  # noqa: BLE001 - validation errors should be reflected in job state.
        fail_without_running(job_path, dirs, job, str(exc), log_path)
        return None

    env = os.environ.copy()
    env.update(job_env)
    env["PYTHONUNBUFFERED"] = "1"

    job.update(
        {
            "status": "running",
            "worker_id": worker_id,
            "hostname": socket.gethostname(),
            "pid": os.getpid(),
            "started_at": utc_now(),
            "log_path": str(log_path),
        }
    )
    atomic_write_json(job_path, job)

    append_job_event(log_path, f"START job={job_id} worker={worker_id} cwd={workdir}")
    append_job_event(log_path, f"CMD {cmd}")

    start_monotonic = time.monotonic()
    timed_out = False
    with log_path.open("a", encoding="utf-8", buffering=1) as log_file:
        proc = subprocess.Popen(
            ["/bin/bash", "-lc", cmd],
            cwd=str(workdir),
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        while True:
            exit_code = proc.poll()
            if exit_code is not None:
                break
            if timeout_sec and time.monotonic() - start_monotonic > timeout_sec:
                timed_out = True
                log_file.write(f"[{utc_now()}] TIMEOUT after {timeout_sec}s, terminating\n")
                log_file.flush()
                terminate_process_group(proc)
                exit_code = proc.poll()
                if exit_code is None:
                    exit_code = -signal.SIGKILL
                break
            time.sleep(1)

    finished_at = utc_now()
    status = "failed" if timed_out or exit_code != 0 else "done"
    job.update(
        {
            "status": status,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "finished_at": finished_at,
            "duration_sec": round(time.monotonic() - start_monotonic, 3),
        }
    )
    if timed_out:
        job["failure_reason"] = f"timeout after {timeout_sec}s"

    append_job_event(log_path, f"FINISH job={job_id} status={status} exit_code={exit_code}")
    atomic_write_json(job_path, job)
    os.replace(job_path, dirs[status] / job_path.name)
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-root",
        default=os.environ.get("SHARED_JOB_QUEUE", DEFAULT_QUEUE_ROOT),
        help=f"shared queue directory (default: {DEFAULT_QUEUE_ROOT})",
    )
    parser.add_argument(
        "--worker-id",
        default=f"{socket.gethostname()}-{os.getpid()}",
        help="stable worker id written to queue_root/workers",
    )
    parser.add_argument("--poll-interval", type=float, default=5.0, help="seconds between polls")
    parser.add_argument("--once", action="store_true", help="exit after one claimed job")
    parser.add_argument("--max-jobs", type=int, default=0, help="exit after N jobs; 0 means forever")
    parser.add_argument(
        "--allowed-root",
        action="append",
        default=[DEFAULT_ALLOWED_ROOT],
        help="only run jobs whose workdir is under this root; repeatable",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("SHARED_JOB_QUEUE_TOKEN"),
        help="optional shared token; when set, jobs must include the same token",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    queue_root = Path(args.queue_root).expanduser().resolve()
    dirs = ensure_queue(queue_root)
    allowed_roots = [Path(root).expanduser().resolve() for root in args.allowed_root]

    processed_jobs = 0
    print(
        f"[{utc_now()}] worker={args.worker_id} queue={queue_root} "
        f"allowed_roots={[str(root) for root in allowed_roots]}",
        flush=True,
    )
    while True:
        write_heartbeat(dirs, args.worker_id, processed_jobs)
        job_path = claim_one_job(dirs)
        if job_path is None:
            time.sleep(args.poll_interval)
            continue

        print(f"[{utc_now()}] claimed {job_path.name}", flush=True)
        run_job(job_path, dirs, args.worker_id, allowed_roots, args.token)
        processed_jobs += 1

        if args.once or (args.max_jobs and processed_jobs >= args.max_jobs):
            write_heartbeat(dirs, args.worker_id, processed_jobs)
            return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print(f"[{utc_now()}] interrupted", file=sys.stderr, flush=True)
        raise
