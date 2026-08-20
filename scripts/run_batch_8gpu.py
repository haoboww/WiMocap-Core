#!/usr/bin/env python3
"""Run resumable WiMocap session processing on a pool of GPUs.

Each GPU owns at most one full session/radar task at a time.  Individual task
failures are recorded and isolated so that the remaining queue can continue.
The underlying process_radar_session.py remains responsible for step-level
resume and output validation.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import queue
import subprocess
import sys
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE = PROJECT_ROOT / "data_pipeline" / "process_radar_session.py"
RADAR_DIRS = {
    "bgt60": "BGT60_4T4R",
    "ti": "TI_IWR6843",
    "calterah": "Calterah",
}

# Wall-clock estimates derived from completed 600 s BGT60/TI jobs on this host,
# plus the Calterah smoke throughput and its larger aligned frame count.
MINUTES_PER_CAPTURE_SECOND = {
    "bgt60": 0.10,
    "ti": 0.125,
    "calterah": 0.17,
}


@dataclass(frozen=True)
class Task:
    radar: str
    session: str
    group: str
    duration_s: float
    estimated_minutes: float
    output_root: str

    @property
    def key(self) -> str:
        return f"{self.radar}/{self.group}/{Path(self.session).name}"


def now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def capture_duration(session: Path) -> float:
    meta_path = session / "cam0" / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        timestamps = meta.get("timestamps", [])
        if len(timestamps) >= 2:
            return float(timestamps[-1]) - float(timestamps[0])
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        pass
    return 0.0


def radar_data_issue(session: Path, radar: str):
    """Return a reason when metadata proves that a radar has no usable frames."""
    meta_path = session / RADAR_DIRS[radar] / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
        return f"unreadable radar metadata: {exc}"
    if radar == "bgt60":
        count = int(meta.get("pointcloud", {}).get("frame_count") or 0)
        if count <= 0 or not (session / RADAR_DIRS[radar] / "pointcloud" / "index.csv").is_file():
            return "no BGT60 pointcloud frames/index"
    elif radar == "calterah":
        count = int(meta.get("full_frame_count") or 0)
        if count <= 0 or not meta.get("timestamps"):
            return "no Calterah frames/timestamps"
    elif radar == "ti":
        count = int(meta.get("frame_count") or 0)
        if count <= 0 or not meta.get("hw_timestamps_ms") or not meta.get("data_files"):
            return "no TI raw frames/hardware timestamps"
    return None


def discover_tasks(args: argparse.Namespace) -> tuple[list[Task], list[dict]]:
    excluded_names = set(args.exclude)
    tasks: list[Task] = []
    skipped: list[dict] = []
    sessions: list[Path] = []
    for raw_root in args.data_roots:
        root = Path(raw_root).expanduser().resolve()
        sessions.extend(sorted(path for path in root.glob("capture_*") if path.is_dir()))

    for session in sessions:
        if session.name in excluded_names or str(session) in excluded_names:
            skipped.append({"session": str(session), "reason": "explicitly excluded"})
            continue
        duration_s = capture_duration(session)
        missing_cameras = [
            cam for cam in ("cam0", "cam2", "cam4", "cam6")
            if not (session / cam / "meta.json").is_file()
        ]
        if missing_cameras:
            skipped.append({
                "session": str(session),
                "reason": f"missing camera metadata: {','.join(missing_cameras)}",
            })
            continue
        for radar in args.radars:
            radar_meta = session / RADAR_DIRS[radar] / "meta.json"
            if not radar_meta.is_file():
                skipped.append({
                    "session": str(session),
                    "radar": radar,
                    "reason": f"missing {radar_meta}",
                })
                continue
            issue = radar_data_issue(session, radar)
            if issue:
                skipped.append({
                    "session": str(session),
                    "radar": radar,
                    "reason": issue,
                })
                continue
            tasks.append(Task(
                radar=radar,
                session=str(session),
                group=session.parent.name,
                duration_s=round(duration_s, 3),
                estimated_minutes=round(
                    duration_s * MINUTES_PER_CAPTURE_SECOND[radar], 1
                ),
                output_root=str(Path(args.output_base).expanduser().resolve() / radar),
            ))

    # Longest-processing-time first keeps the final straggler small.
    tasks.sort(key=lambda task: (-task.estimated_minutes, task.key))
    return tasks, skipped


class State:
    def __init__(self, tasks: list[Task], run_dir: Path, skipped: list[dict]) -> None:
        self.tasks = tasks
        self.run_dir = run_dir
        self.skipped = skipped
        self.lock = threading.Lock()
        self.running: dict[str, dict] = {}
        self.succeeded: list[dict] = []
        self.failed: list[dict] = []
        self.started_at = now()
        self.events = (run_dir / "events.jsonl").open("a", encoding="utf-8", buffering=1)

    def event(self, event: str, **fields: object) -> None:
        record = {"time": now(), "event": event, **fields}
        with self.lock:
            self.events.write(json.dumps(record, ensure_ascii=False) + "\n")
            print(json.dumps(record, ensure_ascii=False), flush=True)
            self._write_summary_locked()

    def _write_summary_locked(self) -> None:
        completed = len(self.succeeded) + len(self.failed)
        summary = {
            "started_at": self.started_at,
            "updated_at": now(),
            "total_tasks": len(self.tasks),
            "completed_tasks": completed,
            "pending_tasks": len(self.tasks) - completed - len(self.running),
            "running": self.running,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "skipped_sessions": self.skipped,
        }
        temp = self.run_dir / "summary.json.tmp"
        temp.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        os.replace(temp, self.run_dir / "summary.json")

    def start(self, task: Task, gpu: int) -> None:
        with self.lock:
            self.running[task.key] = {"gpu": gpu, "started_at": now()}
        self.event("task_started", task=task.key, gpu=gpu)

    def finish(self, task: Task, gpu: int, returncode: int, elapsed_s: float) -> None:
        result = {
            "task": task.key,
            "gpu": gpu,
            "returncode": returncode,
            "elapsed_s": round(elapsed_s, 1),
            "finished_at": now(),
        }
        with self.lock:
            self.running.pop(task.key, None)
            (self.succeeded if returncode == 0 else self.failed).append(result)
        self.event(
            "task_succeeded" if returncode == 0 else "task_failed",
            **result,
        )


def run_worker(
    gpu: int,
    task_queue: "queue.Queue[Task]",
    args: argparse.Namespace,
    state: State,
) -> None:
    while True:
        try:
            task = task_queue.get_nowait()
        except queue.Empty:
            return

        state.start(task, gpu)
        task_output = Path(task.output_root) / task.group / Path(task.session).name
        task_log = task_output / "logs" / "batch_task.log"
        task_log.parent.mkdir(parents=True, exist_ok=True)
        command = [
            args.python,
            str(PIPELINE),
            "--data-root", task.session,
            "--radar", task.radar,
            "--output-root", task.output_root,
            "--threads", str(args.threads),
        ]
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
        env["PYTHONUNBUFFERED"] = "1"
        for name in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[name] = str(args.threads)

        started = time.monotonic()
        with task_log.open("a", encoding="utf-8", buffering=1) as log:
            log.write(f"\n[{now()}] GPU={gpu} command={command!r}\n")
            try:
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
                returncode = result.returncode
            except Exception as exc:  # Keep the worker alive for later tasks.
                log.write(f"scheduler exception: {exc!r}\n")
                returncode = 125
        state.finish(task, gpu, returncode, time.monotonic() - started)
        task_queue.task_done()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-roots", nargs="+", required=True)
    parser.add_argument("--output-base", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--gpus", nargs="+", type=int, default=list(range(8)))
    parser.add_argument(
        "--radars", nargs="+", choices=sorted(RADAR_DIRS),
        default=["bgt60", "ti", "calterah"],
    )
    parser.add_argument("--exclude", nargs="*", default=[])
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--wait-for-lock",
        action="store_true",
        help="wait for an earlier scheduler using the same output base",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
    )
    args = parser.parse_args()

    if len(set(args.gpus)) != len(args.gpus):
        parser.error("--gpus must not contain duplicates")
    if args.threads <= 0:
        parser.error("--threads must be positive")

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    output_base = Path(args.output_base).expanduser().resolve()
    output_base.mkdir(parents=True, exist_ok=True)

    lock_path = output_base / ".batch_8gpu.lock"
    lock_file = lock_path.open("a+", encoding="utf-8")
    lock_flags = fcntl.LOCK_EX
    if not args.wait_for_lock:
        lock_flags |= fcntl.LOCK_NB
    elif lock_path.stat().st_size:
        print(f"Waiting for earlier batch scheduler to release {lock_path}", flush=True)
    try:
        fcntl.flock(lock_file, lock_flags)
    except BlockingIOError:
        print(f"Another batch scheduler holds {lock_path}", file=sys.stderr)
        return 2
    lock_file.seek(0)
    lock_file.truncate()
    lock_file.write(str(os.getpid()) + "\n")
    lock_file.flush()

    tasks, skipped = discover_tasks(args)
    manifest = {
        "created_at": now(),
        "gpus": args.gpus,
        "threads_per_task": args.threads,
        "tasks": [asdict(task) for task in tasks],
        "skipped": skipped,
        "estimated_gpu_hours": round(
            sum(task.estimated_minutes for task in tasks) / 60.0, 2
        ),
    }
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    state = State(tasks, run_dir, skipped)
    state.event(
        "batch_started",
        pid=os.getpid(),
        tasks=len(tasks),
        gpus=args.gpus,
        estimated_gpu_hours=manifest["estimated_gpu_hours"],
    )

    task_queue: "queue.Queue[Task]" = queue.Queue()
    for task in tasks:
        task_queue.put(task)

    workers = [
        threading.Thread(
            target=run_worker,
            name=f"gpu-{gpu}",
            args=(gpu, task_queue, args, state),
        )
        for gpu in args.gpus
    ]
    for worker in workers:
        worker.start()
    for worker in workers:
        worker.join()

    state.event(
        "batch_finished",
        succeeded=len(state.succeeded),
        failed=len(state.failed),
    )
    return 1 if state.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
