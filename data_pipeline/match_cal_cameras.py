#!/usr/bin/env python3
"""
以 Calterah 雷达时间戳为参考，匹配相机图像并链接到 out 文件夹。

数据格式要求：
- Calterah/meta.json: timestamps
- cam0/, cam2/, cam4/, cam6/meta.json: timestamps (秒)
- 各相机目录下 000000.jpg, 000001.jpg, ...

用法:
  python match_cal_cameras.py \
      --capture-dir /media/wais/data2t/first_cap/qian_test/capture_20260310_23392222 \
      --output-dir out \
      --max-delta 0.2
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


def link_or_copy(src: Path, dst: Path, mode: str = "auto") -> str:
    """Create dst without duplicating image bytes when possible."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    src_resolved = src.resolve()
    if mode in ("auto", "hardlink"):
        try:
            os.link(src_resolved, dst)
            return "hardlink"
        except OSError:
            if mode == "hardlink":
                raise

    if mode in ("auto", "symlink"):
        try:
            dst.symlink_to(src_resolved)
            return "symlink"
        except OSError:
            if mode == "symlink":
                raise

    shutil.copy2(src, dst)
    return "copy"


def clear_old_images(cam_dir: Path) -> None:
    if not cam_dir.exists():
        return
    for path in cam_dir.glob("*.jpg"):
        path.unlink()


def write_matches_csv(output_dir: Path, details: list[dict], cam_ids: tuple[str, ...]) -> None:
    csv_path = output_dir / "matches.csv"
    fields = ["out_frame_id", "radar_frame_id", "radar_ts"]
    for cam_id in cam_ids:
        fields.extend([f"{cam_id}_frame_id", f"{cam_id}_delta_s"])

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for item in details:
            row = {
                "out_frame_id": item["out_frame_id"],
                "radar_frame_id": item["radar_frame_id"],
                "radar_ts": item["radar_ts"],
            }
            for cam_id in cam_ids:
                row[f"{cam_id}_frame_id"] = item["video"].get(cam_id, "")
                row[f"{cam_id}_delta_s"] = item["delta_s"].get(cam_id, "")
            writer.writerow(row)


def write_aligned_pointcloud(pointcloud_path: Path, output_dir: Path, details: list[dict]) -> Path:
    """Write output_dir/pointcloud.npz with frame_id matching EasyMocap frame ids."""
    pointcloud_path = Path(pointcloud_path)
    if not pointcloud_path.exists():
        raise FileNotFoundError(f"未找到点云文件: {pointcloud_path}")

    npz = np.load(pointcloud_path, allow_pickle=True)
    frames = npz["frames"]
    points = npz["points"]
    frame_to_row = {int(f["frame_id"]): i for i, f in enumerate(frames)}

    out_frames = np.zeros(len(details), dtype=frames.dtype)
    point_chunks = []
    offset = 0
    missing = 0

    for item in details:
        out_fid = int(item["out_frame_id"])
        radar_fid = int(item["radar_frame_id"])
        row = frame_to_row.get(radar_fid)
        if row is None:
            pts = np.zeros(0, dtype=points.dtype)
            missing += 1
        else:
            f = frames[row]
            start = int(f["offset"])
            end = start + int(f["num_points"])
            pts = points[start:end]

        out_frames[out_fid] = (
            out_fid,
            float(item["radar_ts"]),
            len(pts),
            offset,
        )
        point_chunks.append(pts)
        offset += len(pts)

    out_points = (
        np.concatenate(point_chunks)
        if point_chunks
        else np.zeros(0, dtype=points.dtype)
    )
    out_path = output_dir / "pointcloud.npz"
    np.savez_compressed(out_path, frames=out_frames, points=out_points)
    if missing:
        logger.warning("点云缺少 %d 个匹配帧，已写空点云帧", missing)
    logger.info("已写对齐点云: %s (%d 帧, %d 点)", out_path, len(out_frames), len(out_points))
    return out_path


def load_radar_timestamps(capture_dir: Path) -> tuple[np.ndarray, int]:
    """从 Calterah/meta.json 加载雷达时间戳（秒）"""
    meta_path = capture_dir / "Calterah" / "meta.json"
    # meta_path = capture_dir / "meta.json"
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
    link_mode: str = "auto",
    drop_duplicate_camera_combos: bool = False,
    pointcloud_path: Path | None = None,
    write_pointcloud: bool = False,
    start_frame: int = 0,
    end_frame: int | None = None,
) -> None:
    capture_dir = Path(capture_dir)
    output_dir = Path(output_dir)
    if not output_dir.is_absolute():
        output_dir = capture_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载雷达时间戳
    radar_ts_all, n_radar_total = load_radar_timestamps(capture_dir)
    start_frame = max(0, int(start_frame))
    end_frame = n_radar_total if end_frame is None else min(n_radar_total, int(end_frame))
    if start_frame >= end_frame:
        raise ValueError(
            f"无效雷达帧范围 [{start_frame}, {end_frame}); 总帧数 {n_radar_total}"
        )

    radar_frame_ids = np.arange(n_radar_total, dtype=np.int64)[start_frame:end_frame]
    radar_ts = radar_ts_all[start_frame:end_frame]
    n_radar = len(radar_ts)
    logger.info(
        f"[Calterah] 雷达帧数: {n_radar_total}, "
        f"本次匹配范围 [{start_frame}, {end_frame}) → {n_radar} 帧"
    )

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
    logger.info(f"共 {n_out}/{n_radar} 帧所有相机均有效匹配，将复制到 {output_dir}")

    details = []
    seen_combos = set()
    duplicate_count = 0
    for radar_i in out_indices:
        radar_frame_id = int(radar_frame_ids[radar_i])
        video = {cam_id: int(matches[cam_id][radar_i]) for cam_id in cam_data}
        combo = tuple(sorted(video.items()))
        if drop_duplicate_camera_combos and combo in seen_combos:
            duplicate_count += 1
            continue
        seen_combos.add(combo)

        out_i = len(details)
        details.append({
            "out_frame_id": out_i,
            "radar_frame_id": radar_frame_id,
            "radar_ts": float(radar_ts[radar_i]),
            "video": video,
            "delta_s": {
                cam_id: float(match_deltas[cam_id][radar_i])
                for cam_id in cam_data
            },
        })

    if duplicate_count:
        logger.info("去除了 %d 个重复相机组合", duplicate_count)

    # 创建 out/cam0, out/cam2, out/cam4, out/cam6
    for cam_id in cam_data:
        out_cam = output_dir / cam_id
        out_cam.mkdir(parents=True, exist_ok=True)
        clear_old_images(out_cam)

    # 按 000000.jpg, 000001.jpg, ... 顺序链接/复制
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
        "radar": "Calterah",
        "max_delta_s": max_delta,
        "cameras": list(cam_data.keys()),
        "radar_frames_total": n_radar_total,
        "radar_frame_start": start_frame,
        "radar_frame_end": end_frame,
        "radar_frames": n_radar,
        "matched_frames_before_duplicate_filter": n_out,
        "aligned_frames": len(details),
        "drop_duplicate_camera_combos": drop_duplicate_camera_combos,
        "duplicate_camera_combos_dropped": duplicate_count,
        "link_mode": link_mode,
        "link_counts": mode_counts,
        "details": details,
    }
    with (output_dir / "alignment.json").open("w", encoding="utf-8") as f:
        json.dump(alignment, f, indent=2, ensure_ascii=False)
    write_matches_csv(output_dir, details, tuple(cam_data.keys()))

    if write_pointcloud:
        if pointcloud_path is None:
            pointcloud_path = capture_dir / "Calterah" / "pointcloud.npz"
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
        description="以 Calterah 雷达时间戳为参考，最邻近匹配相机图像并链接到 out 文件夹",
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
    parser.add_argument(
        "--subs",
        nargs="+",
        default=list(CAM_IDS),
        help="需要匹配的相机目录名",
    )
    parser.add_argument(
        "--link-mode",
        choices=["auto", "hardlink", "symlink", "copy"],
        default="auto",
        help="输出图像方式；auto 优先 hardlink，再 symlink，最后 copy",
    )
    parser.add_argument(
        "--drop-duplicate-camera-combos",
        action="store_true",
        help="丢弃映射到同一组相机帧的重复雷达帧",
    )
    parser.add_argument(
        "--pointcloud",
        default=None,
        help="Calterah 全量点云 NPZ，默认 capture_dir/Calterah/pointcloud.npz",
    )
    parser.add_argument(
        "--write-pointcloud",
        action="store_true",
        help="按 alignment 生成 output_dir/pointcloud.npz，帧号与 EasyMocap 输出一致",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="雷达起始帧，闭区间左端",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="雷达结束帧，开区间右端；默认到最后一帧",
    )
    args = parser.parse_args()

    run(
        capture_dir=args.capture_dir,
        output_dir=args.output_dir,
        max_delta=args.max_delta,
        cam_ids=tuple(args.subs),
        link_mode=args.link_mode,
        drop_duplicate_camera_combos=args.drop_duplicate_camera_combos,
        pointcloud_path=Path(args.pointcloud) if args.pointcloud else None,
        write_pointcloud=args.write_pointcloud,
        start_frame=args.start_frame,
        end_frame=args.end_frame,
    )


if __name__ == "__main__":
    main()
