#!/usr/bin/env python3
"""Match cameras to the capture-time BGT60 point-cloud timeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from match_ti6843_cameras import CAM_IDS, run


def load_pointcloud_timestamps(capture_dir: Path) -> np.ndarray:
    index_path = capture_dir / "BGT60_4T4R" / "pointcloud" / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"未找到 {index_path}")

    frame_ids: list[int] = []
    timestamps: list[float] = []
    with index_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            frame_ids.append(int(row["frame_index"]))
            timestamps.append(int(row["timestamp_ns"]) / 1_000_000_000.0)

    expected = list(range(len(frame_ids)))
    if frame_ids != expected:
        raise ValueError("BGT60 pointcloud/index.csv 的 frame_index 必须从 0 连续递增")
    if not timestamps:
        raise ValueError("BGT60 pointcloud/index.csv 没有帧")
    return np.asarray(timestamps, dtype=np.float64)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="以 BGT60 采集点云时间戳为参考匹配相机图像",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("out"))
    parser.add_argument("--max-delta", type=float, default=0.2)
    parser.add_argument("--subs", nargs="+", default=list(CAM_IDS))
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="copy",
    )
    parser.add_argument("--pointcloud", type=Path, default=None)
    parser.add_argument("--write-pointcloud", action="store_true")
    args = parser.parse_args()

    capture_dir = args.capture_dir.expanduser().resolve()
    run(
        capture_dir=capture_dir,
        output_dir=args.output_dir,
        max_delta=args.max_delta,
        cam_ids=tuple(args.subs),
        link_mode=args.link_mode,
        pointcloud_path=args.pointcloud,
        write_pointcloud=args.write_pointcloud,
        radar_timestamps=load_pointcloud_timestamps(capture_dir),
        radar_name="BGT60_4T4R",
    )


if __name__ == "__main__":
    main()
