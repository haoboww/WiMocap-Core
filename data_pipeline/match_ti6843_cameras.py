#!/usr/bin/env python3
"""
以 TI_IWR6843 雷达时间戳为参考，匹配相机图像并链接或复制到输出目录。

数据格式要求：
- TI_IWR6843/meta.json: hw_timestamps_ms 或 timestamps
- cam0/, cam2/, cam4/, cam6/meta.json: timestamps (秒)
- 各相机目录下 000000.jpg, 000001.jpg, ...

用法:
  python match_ti6843_cameras.py \
      --capture-dir /media/wais/data2t/first_cap/qian_test/capture_20260310_23392222 \
      --output-dir out \
      --max-delta 0.2 \
      --link-mode symlink
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
from pathlib import Path

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

CAM_IDS = ("cam0", "cam2", "cam4", "cam6")


def clear_old_images(directory: Path) -> None:
    for path in directory.glob("*.jpg"):
        path.unlink()


def link_or_copy(src: Path, dst: Path, mode: str) -> str:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "copy":
        shutil.copy2(src, dst)
        return "copy"
    if mode == "symlink":
        dst.symlink_to(src.resolve())
        return "symlink"
    if mode == "hardlink":
        os.link(src, dst)
        return "hardlink"

    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        try:
            dst.symlink_to(src.resolve())
            return "symlink"
        except OSError:
            shutil.copy2(src, dst)
            return "copy"


def write_matches_csv(output_dir: Path, details: list[dict], cam_ids: tuple[str, ...]) -> None:
    fields = ["out_frame_id", "radar_frame_id", "radar_ts"]
    fields.extend(f"{cam_id}_frame_id" for cam_id in cam_ids)
    fields.extend(f"{cam_id}_delta_s" for cam_id in cam_ids)
    with (output_dir / "matches.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in details:
            row = {
                "out_frame_id": item["out_frame_id"],
                "radar_frame_id": item["radar_frame_id"],
                "radar_ts": item["radar_ts"],
            }
            for cam_id in cam_ids:
                row[f"{cam_id}_frame_id"] = item["video"][cam_id]
                row[f"{cam_id}_delta_s"] = item["delta_s"][cam_id]
            writer.writerow(row)


def write_aligned_pointcloud(
    pointcloud_path: Path,
    output_dir: Path,
    details: list[dict],
) -> Path:
    data = np.load(pointcloud_path, allow_pickle=False)
    frames = data["frames"]
    points = data["points"]
    frame_to_row = {int(frame["frame_id"]): i for i, frame in enumerate(frames)}

    out_frames = np.zeros(len(details), dtype=frames.dtype)
    point_parts: list[np.ndarray] = []
    offset = 0
    missing = 0
    for out_i, item in enumerate(details):
        radar_frame_id = int(item["radar_frame_id"])
        row_idx = frame_to_row.get(radar_frame_id)
        if row_idx is None:
            frame_points = np.zeros(0, dtype=points.dtype)
            out_frame = np.zeros((), dtype=frames.dtype)
            missing += 1
        else:
            source_frame = frames[row_idx]
            start = int(source_frame["offset"])
            end = start + int(source_frame["num_points"])
            frame_points = points[start:end]
            out_frame = source_frame.copy()

        out_frame["frame_id"] = out_i
        out_frame["timestamp"] = float(item["radar_ts"])
        out_frame["num_points"] = len(frame_points)
        out_frame["offset"] = offset
        out_frames[out_i] = out_frame
        point_parts.append(frame_points)
        offset += len(frame_points)

    out_points = (
        np.concatenate(point_parts)
        if point_parts and offset > 0
        else np.zeros(0, dtype=points.dtype)
    )
    payload = {"frames": out_frames, "points": out_points}
    if "meta" in data.files:
        payload["meta"] = data["meta"]
    out_path = output_dir / "pointcloud.npz"
    np.savez_compressed(out_path, **payload)
    if missing:
        logger.warning("点云缺少 %d 个匹配帧，已写空点云帧", missing)
    logger.info(
        "已写对齐点云: %s (%d 帧, %d 点)",
        out_path,
        len(out_frames),
        len(out_points),
    )
    return out_path


def load_radar_timestamps(capture_dir: Path) -> tuple[np.ndarray, int]:
    """从 TI_IWR6843/meta.json 加载雷达时间戳（秒）"""
    meta_path = capture_dir / "TI_IWR6843" / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"未找到 {meta_path}")

    with open(meta_path) as f:
        meta = json.load(f)

    if "hw_timestamps_ms" in meta:
        ts_ms = np.array(meta["hw_timestamps_ms"], dtype=np.float64)
        ts = ts_ms / 1000.0
    elif "timestamps" in meta:
        ts = np.array(meta["timestamps"], dtype=np.float64)
    else:
        raise ValueError("meta.json 中缺少 hw_timestamps_ms 或 timestamps")

    return ts, len(ts)


def load_camera_timestamps(capture_dir: Path, cam_id: str) -> tuple[np.ndarray, list[Path]] | None:
    """从 camX/meta.json 加载相机时间戳和图像路径"""
    meta_path = capture_dir / cam_id / "meta.json"
    cam_dir = capture_dir / cam_id
    if not meta_path.exists() or not cam_dir.is_dir():
        return None

    with open(meta_path) as f:
        meta = json.load(f)

    ts = np.array(meta["timestamps"], dtype=np.float64)
    n = len(ts)
    file_paths = [cam_dir / f"{i:06d}.jpg" for i in range(n)]
    return ts, file_paths


def nearest_neighbor_match(
    query_ts: np.ndarray,
    target_ts: np.ndarray,
    max_delta: float = float("inf"),
) -> tuple[np.ndarray, np.ndarray]:
    """对每个 query 时间戳，在 target 中找最邻近的索引"""
    if len(target_ts) == 0:
        return np.full(len(query_ts), -1, dtype=np.int64), np.full(len(query_ts), np.nan)

    idx_r = np.searchsorted(target_ts, query_ts, side="left")
    idx_r = np.clip(idx_r, 0, len(target_ts) - 1)
    idx_l = np.maximum(idx_r - 1, 0)

    delta_r = target_ts[idx_r] - query_ts
    delta_l = target_ts[idx_l] - query_ts

    use_right = np.abs(delta_r) <= np.abs(delta_l)
    matched_idx = np.where(use_right, idx_r, idx_l)
    deltas = np.where(use_right, delta_r, delta_l)

    invalid = np.abs(deltas) > max_delta
    matched_idx = np.where(invalid, -1, matched_idx)
    deltas = np.where(invalid, np.nan, deltas)

    return matched_idx, deltas


def run(
    capture_dir: Path,
    output_dir: Path,
    max_delta: float = 0.2,
    cam_ids: tuple[str, ...] = CAM_IDS,
    link_mode: str = "copy",
    pointcloud_path: Path | None = None,
    write_pointcloud: bool = False,
    radar_timestamps: np.ndarray | None = None,
    radar_name: str = "TI_IWR6843",
) -> None:
    capture_dir = Path(capture_dir)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = capture_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载雷达时间戳
    if radar_timestamps is None:
        radar_ts, n_radar = load_radar_timestamps(capture_dir)
    else:
        radar_ts = np.asarray(radar_timestamps, dtype=np.float64)
        n_radar = len(radar_ts)
    logger.info("[%s] 雷达帧数: %d", radar_name, n_radar)

    # 加载各相机并匹配
    cam_data: dict[str, tuple[np.ndarray, list[Path]]] = {}
    matches: dict[str, np.ndarray] = {}
    match_deltas: dict[str, np.ndarray] = {}

    for cam_id in cam_ids:
        data = load_camera_timestamps(capture_dir, cam_id)
        if data is None:
            logger.warning(f"[{cam_id}] 未找到数据，跳过")
            continue

        cam_ts, cam_paths = data
        matched_idx, deltas = nearest_neighbor_match(radar_ts, cam_ts, max_delta)

        valid = matched_idx >= 0
        n_valid = int(np.sum(valid))
        mean_delta = float(np.nanmean(deltas[valid])) * 1000 if n_valid > 0 else 0
        max_abs_delta = float(np.nanmax(np.abs(deltas[valid]))) * 1000 if n_valid > 0 else 0

        logger.info(
            f"[{cam_id}] 匹配 {n_valid}/{n_radar} 帧, "
            f"平均偏差={mean_delta:.2f}ms, 最大|Δ|={max_abs_delta:.2f}ms"
        )

        cam_data[cam_id] = (cam_ts, cam_paths)
        matches[cam_id] = matched_idx
        match_deltas[cam_id] = deltas

    if not cam_data:
        raise RuntimeError("未找到任何相机数据")

    # 仅输出所有相机都有有效匹配的雷达帧
    all_valid = np.ones(n_radar, dtype=bool)
    for cam_id, midx in matches.items():
        all_valid &= midx >= 0

    n_out = int(np.sum(all_valid))
    out_indices = np.where(all_valid)[0]
    logger.info(f"共 {n_out}/{n_radar} 帧所有相机均有效匹配，将输出到 {output_dir}")

    details = []
    for radar_i in out_indices:
        details.append({
            "out_frame_id": len(details),
            "radar_frame_id": int(radar_i),
            "radar_ts": float(radar_ts[radar_i]),
            "video": {
                cam_id: int(matches[cam_id][radar_i])
                for cam_id in cam_data
            },
            "delta_s": {
                cam_id: float(match_deltas[cam_id][radar_i])
                for cam_id in cam_data
            },
        })

    # 创建 out/cam0, out/cam2, out/cam4, out/cam6
    for cam_id in cam_data:
        out_cam = output_dir / cam_id
        out_cam.mkdir(parents=True, exist_ok=True)
        clear_old_images(out_cam)

    mode_counts: dict[str, int] = {}
    for item in details:
        out_i = int(item["out_frame_id"])
        for cam_id in cam_data:
            src = cam_data[cam_id][1][int(item["video"][cam_id])]
            dst = output_dir / cam_id / f"{out_i:06d}.jpg"
            mode = link_or_copy(src, dst, link_mode)
            mode_counts[mode] = mode_counts.get(mode, 0) + 1

    alignment = {
        "capture_dir": str(capture_dir.resolve()),
        "output_dir": str(output_dir.resolve()),
        "radar": radar_name,
        "max_delta_s": max_delta,
        "cameras": list(cam_data.keys()),
        "radar_frames_total": n_radar,
        "aligned_frames": len(details),
        "link_mode": link_mode,
        "link_counts": mode_counts,
        "details": details,
    }
    with (output_dir / "alignment.json").open("w", encoding="utf-8") as f:
        json.dump(alignment, f, indent=2, ensure_ascii=False)
    write_matches_csv(output_dir, details, tuple(cam_data.keys()))

    if write_pointcloud:
        if pointcloud_path is None:
            raise ValueError("--write-pointcloud requires --pointcloud")
        pc_out = write_aligned_pointcloud(Path(pointcloud_path), output_dir, details)
        alignment["pointcloud"] = str(pc_out.resolve())
        with (output_dir / "alignment.json").open("w", encoding="utf-8") as f:
            json.dump(alignment, f, indent=2, ensure_ascii=False)

    mode_summary = ", ".join(f"{v} {k}" for k, v in sorted(mode_counts.items()))
    logger.info(
        "已输出 %d 帧 × %d 路相机 → %s (%s)",
        len(details),
        len(cam_data),
        output_dir,
        mode_summary,
    )


def main():
    parser = argparse.ArgumentParser(
        description="以 TI_IWR6843 雷达时间戳为参考，最邻近匹配相机图像",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--capture-dir",
        type=str,
        required=True,
        help="采集数据目录路径",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="out",
        help="输出目录（相对或绝对路径，默认 capture_dir/out）",
    )
    parser.add_argument(
        "--max-delta",
        type=float,
        default=0.2,
        help="最大允许时间差 (秒)，超过视为无匹配",
    )
    parser.add_argument("--subs", nargs="+", default=list(CAM_IDS))
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="copy",
    )
    parser.add_argument("--pointcloud", type=str, default=None)
    parser.add_argument("--write-pointcloud", action="store_true")
    args = parser.parse_args()

    run(
        capture_dir=args.capture_dir,
        output_dir=args.output_dir,
        max_delta=args.max_delta,
        cam_ids=tuple(args.subs),
        link_mode=args.link_mode,
        pointcloud_path=Path(args.pointcloud) if args.pointcloud else None,
        write_pointcloud=args.write_pointcloud,
    )


if __name__ == "__main__":
    main()
