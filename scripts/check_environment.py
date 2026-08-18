#!/usr/bin/env python3
"""Check the portable post-processing runtime and optional capture layout."""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EASYMOCAP_ROOT = PROJECT_ROOT / "third_party" / "EasyMocapFork"

REQUIRED_FILES = (
    "configs/camera_intrinsics/cam0.json",
    "configs/camera_intrinsics/cam2.json",
    "configs/camera_intrinsics/cam4.json",
    "configs/camera_intrinsics/cam6.json",
    "configs/camera_extrinsics/unified.json",
    "configs/easymocap/detect_triangulate_fitSMPL.yml",
    "third_party/EasyMocapFork/data/models/pose_hrnet_w48_384x288.pth",
    "third_party/EasyMocapFork/models/yolov5m.pt",
    "third_party/EasyMocapFork/models/pare/data/body_models/smpl/SMPL_NEUTRAL.pkl",
    "third_party/EasyMocapFork/models/J_regressor_body25.npy",
    "third_party/EasyMocapFork/vendor/yolov5/hubconf.py",
)

MODULES = (
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("OpenCV", "cv2"),
    ("PyYAML", "yaml"),
    ("PyTorch", "torch"),
    ("torchvision", "torchvision"),
    ("pandas", "pandas"),
    ("tqdm", "tqdm"),
    ("tabulate", "tabulate"),
    ("termcolor", "termcolor"),
    ("ultralytics", "ultralytics"),
)


def check_session(session: Path) -> list[str]:
    errors = []
    for camera in ("cam0", "cam2", "cam4", "cam6"):
        camera_dir = session / camera
        meta_path = camera_dir / "meta.json"
        if not camera_dir.is_dir() or not meta_path.is_file():
            errors.append(f"missing {camera}/meta.json")

    for radar in ("Calterah", "TI_IWR6843", "BGT60_4T4R"):
        radar_dir = session / radar
        if radar_dir.exists() and not (radar_dir / "meta.json").is_file():
            errors.append(f"missing {radar}/meta.json")

    print(f"Session: {session}")
    for radar in ("Calterah", "TI_IWR6843"):
        meta_path = session / radar / "meta.json"
        if not meta_path.is_file():
            print(f"  {radar}: absent")
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid {radar}/meta.json: {exc}")
            continue
        timestamps = meta.get("timestamps", meta.get("hw_timestamps_ms", []))
        frame_count = meta.get("full_frame_count", meta.get("frame_count", "unknown"))
        print(f"  {radar}: frame_count={frame_count}, timestamps={len(timestamps)}")

    bgt_dir = session / "BGT60_4T4R"
    if bgt_dir.is_dir():
        index_path = bgt_dir / "pointcloud" / "index.csv"
        bin_files = sorted((bgt_dir / "pointcloud").glob("pcd_*.bin"))
        if not index_path.is_file():
            errors.append("missing BGT60_4T4R/pointcloud/index.csv")
        if not bin_files:
            errors.append("missing BGT60_4T4R/pointcloud/pcd_*.bin")
        index_frames = 0
        if index_path.is_file():
            try:
                with index_path.open(newline="", encoding="utf-8") as handle:
                    index_frames = sum(1 for _ in csv.DictReader(handle))
            except (OSError, csv.Error) as exc:
                errors.append(f"invalid BGT60 pointcloud index: {exc}")
        try:
            meta = json.loads((bgt_dir / "meta.json").read_text())
            meta_frames = meta.get("pointcloud", {}).get("frame_count", "unknown")
        except (OSError, json.JSONDecodeError) as exc:
            errors.append(f"invalid BGT60_4T4R/meta.json: {exc}")
            meta_frames = "unknown"
        print(
            f"  BGT60_4T4R: pointcloud_meta={meta_frames}, "
            f"pointcloud_index={index_frames}, bin_parts={len(bin_files)}"
        )
        if isinstance(meta_frames, int) and meta_frames != index_frames:
            errors.append(
                "BGT60 pointcloud frame count mismatch: "
                f"meta={meta_frames}, index={index_frames}"
            )
    else:
        print("  BGT60_4T4R: absent")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session", type=Path, default=None)
    args = parser.parse_args()
    errors: list[str] = []

    print(f"Project: {PROJECT_ROOT}")
    print(f"Python:  {sys.executable} ({sys.version.split()[0]})")

    for name, module_name in MODULES:
        try:
            module = importlib.import_module(module_name)
            version = getattr(module, "__version__", "ok")
            print(f"[ok] {name}: {version}")
        except Exception as exc:
            errors.append(f"cannot import {name}: {exc}")

    if str(EASYMOCAP_ROOT) not in sys.path:
        sys.path.insert(0, str(EASYMOCAP_ROOT))
    try:
        importlib.import_module("apps.mocap.run")
        importlib.import_module("myeasymocap.operations.triangulate")
        print("[ok] bundled EasyMocap imports")
    except Exception as exc:
        errors.append(f"cannot import bundled EasyMocap: {exc}")

    try:
        import torch

        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA runtime:   {torch.version.cuda}")
            print(f"GPU:            {torch.cuda.get_device_name(0)}")
        else:
            errors.append("CUDA is unavailable; EasyMocap will be impractically slow")
    except Exception:
        pass

    for relative in REQUIRED_FILES:
        path = PROJECT_ROOT / relative
        if path.is_file():
            print(f"[ok] {relative} ({path.stat().st_size / 1048576:.1f} MiB)")
        else:
            errors.append(f"missing required file: {relative}")

    if args.session:
        errors.extend(check_session(args.session.expanduser().resolve()))

    if errors:
        print("\nFAILED")
        for error in errors:
            print(f"- {error}")
        return 1
    print("\nEnvironment check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
