#!/usr/bin/env python3
"""Portable Calterah/TI/BGT60 post-processing for the current capture layout."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_DIR = PROJECT_ROOT / "data_pipeline"
VALID_STEPS = ("pointcloud", "match", "easymocap")
RADAR_DIRS = {
    "bgt60": "BGT60_4T4R",
    "calterah": "Calterah",
    "ti": "TI_IWR6843",
}


def split_steps(value: str) -> list[str]:
    steps = [item.strip() for item in value.split(",") if item.strip()]
    invalid = [item for item in steps if item not in VALID_STEPS]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"invalid steps {invalid}; choose from {', '.join(VALID_STEPS)}"
        )
    return steps


def is_session(path: Path, radar: str) -> bool:
    return (path / RADAR_DIRS[radar] / "meta.json").is_file()


def find_sessions(data_root: Path, radar: str) -> list[Path]:
    root = data_root.expanduser().resolve()
    if is_session(root, radar):
        return [root]
    if not root.is_dir():
        return []
    return sorted(
        path for path in root.rglob("capture_*")
        if path.is_dir() and is_session(path, radar)
    )


def default_output_root(radar: str) -> Path:
    return PROJECT_ROOT / f"processed_{radar}"


def output_dir_for_session(session: Path, args: argparse.Namespace) -> Path:
    root = (
        Path(args.output_root).expanduser().resolve()
        if args.output_root
        else default_output_root(args.radar)
    )
    return root / session.parent.name / session.name


def full_pointcloud_path(output_dir: Path, radar: str) -> Path:
    if radar == "calterah":
        return output_dir / "Calterah" / "pointcloud.npz"
    if radar == "ti":
        return output_dir / "TI_IWR6843" / "pointclouds" / "ti_iwr6843_pointcloud.npz"
    return output_dir / "BGT60_4T4R" / "pointcloud.npz"


def step_is_complete(output_dir: Path, step: str, radar: str) -> bool:
    if step == "pointcloud":
        return full_pointcloud_path(output_dir, radar).is_file()
    if step == "match":
        return (
            (output_dir / "alignment.json").is_file()
            and (output_dir / "pointcloud.npz").is_file()
        )
    if step == "easymocap":
        smpl_dir = output_dir / "output" / "smpl"
        if not smpl_dir.is_dir():
            return False
        try:
            alignment = json.loads((output_dir / "alignment.json").read_text())
            expected = int(alignment["aligned_frames"])
        except (FileNotFoundError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False
        return len(list(smpl_dir.glob("*.json"))) >= expected
    return False


def build_command(
    session: Path,
    output_dir: Path,
    step: str,
    args: argparse.Namespace,
) -> list[str]:
    python = args.python
    full_pc = full_pointcloud_path(output_dir, args.radar)

    if step == "pointcloud" and args.radar == "calterah":
        command = [
            python,
            str(PIPELINE_DIR / "calterah" / "generate_pointcloud.py"),
            "--capture-dir", str(session),
            "--npz-output", str(full_pc),
            "--output", str(output_dir / "Calterah" / "pointcloud_artifacts"),
            "--log-every", str(args.log_every),
            "--max-range", str(args.max_range),
            "--max-detections", str(args.max_detections),
            "--cfar-mul", str(args.cfar_mul),
            "--dpk-threshold", str(args.dpk_threshold),
        ]
        command.append("--local-max" if args.local_max else "--no-local-max")
        command.append("--no-elevation" if args.no_elevation else "--enable-elevation")
        return command

    if step == "pointcloud" and args.radar == "ti":
        command = [
            python,
            str(PIPELINE_DIR / "ti" / "generate_pointcloud.py"),
            "--input-dir", str(session / "TI_IWR6843"),
            "--output", str(output_dir / "TI_IWR6843"),
            "--max-range", str(args.max_range),
            "--max-detections", str(args.max_detections),
            "--cfar-mul", str(args.cfar_mul),
            "--dpk-threshold", str(args.dpk_threshold),
        ]
        if not args.local_max:
            command.append("--no-local-max")
        if args.no_elevation:
            command.append("--no-elevation")
        if not args.save_rdmap:
            command.append("--no-rdmap")
        return command

    if step == "pointcloud" and args.radar == "bgt60":
        return [
            python,
            str(PIPELINE_DIR / "bgt60" / "convert_pointcloud.py"),
            "--input-dir", str(session / "BGT60_4T4R"),
            "--output", str(full_pc),
            "--log-every", str(args.log_every),
        ]

    if step == "match":
        matcher = {
            "bgt60": "match_bgt60_cameras.py",
            "calterah": "match_cal_cameras.py",
            "ti": "match_ti6843_cameras.py",
        }[args.radar]
        command = [
            python,
            str(PIPELINE_DIR / matcher),
            "--capture-dir", str(session),
            "--output-dir", str(output_dir),
            "--max-delta", str(args.max_delta),
            "--subs", *args.subs,
            "--link-mode", "symlink",
            "--write-pointcloud",
            "--pointcloud", str(full_pc),
        ]
        if args.radar == "calterah" and args.drop_duplicate_camera_combos:
            command.append("--drop-duplicate-camera-combos")
        return command

    if step == "easymocap":
        command = [
            python,
            str(PIPELINE_DIR / "run_easymocap.py"),
            "--input", str(output_dir),
            "--subs", *args.subs,
            "--skip-vis",
        ]
        if not args.force:
            command.append("--skip-existing")
        return command

    raise ValueError(f"unsupported step: {step}")


def run_command(
    command: list[str],
    log_path: Path,
    args: argparse.Namespace,
) -> bool:
    print(f"Command: {shlex.join(command)}")
    print(f"Log: {log_path}")
    if args.dry_run:
        return True

    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[name] = str(args.threads)

    with log_path.open("w", encoding="utf-8") as log_file:
        result = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )
    if result.returncode != 0:
        print(f"Step failed with exit code {result.returncode}; see {log_path}")
    return result.returncode == 0


def output_counts(output_dir: Path, radar: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    full_pc = full_pointcloud_path(output_dir, radar)
    aligned_pc = output_dir / "pointcloud.npz"
    for label, path in (("full_pointcloud_frames", full_pc), ("aligned_pointcloud_frames", aligned_pc)):
        if not path.is_file():
            continue
        try:
            with np.load(path, allow_pickle=False) as data:
                counts[label] = int(len(data["frames"]))
                counts[label.replace("frames", "points")] = int(len(data["points"]))
        except (OSError, KeyError, ValueError):
            pass

    for label, path, pattern in (
        ("keypoints3d_files", output_dir / "output" / "keypoints3d", "*.json"),
        ("smpl_files", output_dir / "output" / "smpl", "*.json"),
    ):
        if path.is_dir():
            counts[label] = len(list(path.glob(pattern)))
    return counts


def write_status(output_dir: Path, status: dict) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "processing_status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def process_session(session: Path, args: argparse.Namespace) -> bool:
    output_dir = output_dir_for_session(session, args)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "radar": args.radar,
        "session": str(session),
        "output_dir": str(output_dir),
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "steps": {},
    }
    print(f"\nSession: {session}")
    print(f"Output:  {output_dir}")

    for step in args.steps:
        if not args.force and step_is_complete(output_dir, step, args.radar):
            print(f"[{step}] skip existing")
            status["steps"][step] = "skipped_existing"
            continue

        print(f"[{step}] start")
        command = build_command(session, output_dir, step, args)
        ok = run_command(command, output_dir / "logs" / f"{step}.log", args)
        status["steps"][step] = "ok" if ok else "failed"
        if not args.dry_run:
            status["counts"] = output_counts(output_dir, args.radar)
            write_status(output_dir, status)
        if not ok:
            status["success"] = False
            status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
            if not args.dry_run:
                write_status(output_dir, status)
            return False
        print(f"[{step}] ok")

    status["counts"] = output_counts(output_dir, args.radar)
    status["success"] = True
    status["finished_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
    if not args.dry_run:
        write_status(output_dir, status)
    print(f"Completed: {output_dir}")
    print(f"Counts: {status['counts']}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Process Calterah, TI, or BGT60 sessions without writing to the raw-data disk",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", required=True, help="capture_* session or a parent directory")
    parser.add_argument("--radar", required=True, choices=sorted(RADAR_DIRS))
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--steps", type=split_steps, default=list(VALID_STEPS))
    parser.add_argument("--subs", nargs="+", default=["cam0", "cam2", "cam4", "cam6"])
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-error", action="store_true")

    parser.add_argument("--max-range", type=float, default=6.0)
    parser.add_argument("--max-detections", type=int, default=400)
    parser.add_argument("--cfar-mul", type=float, default=2.0)
    parser.add_argument("--dpk-threshold", type=float, default=3.0)
    parser.add_argument("--local-max", action="store_true")
    parser.add_argument("--no-elevation", action="store_true")
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-rdmap", action="store_true", help="TI only; disabled by default")
    parser.add_argument("--drop-duplicate-camera-combos", action="store_true", help="Calterah only")
    args = parser.parse_args()

    if args.threads <= 0:
        parser.error("--threads must be positive")

    sessions = find_sessions(Path(args.data_root), args.radar)
    if not sessions:
        print(f"No {args.radar} sessions found under {args.data_root}", file=sys.stderr)
        return 1

    print(f"Found {len(sessions)} {args.radar} session(s)")
    failed = 0
    for session in sessions:
        if not process_session(session, args):
            failed += 1
            if args.stop_on_error:
                break
    print(f"Summary: success={len(sessions) - failed}, failed={failed}, total={len(sessions)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
