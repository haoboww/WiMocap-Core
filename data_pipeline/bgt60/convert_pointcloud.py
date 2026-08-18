#!/usr/bin/env python3
"""Convert captured SKY32B750/BGT60 point-cloud packets to training NPZ."""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np


logger = logging.getLogger(__name__)

FRAME_HEADER = b"\x55\xAA"
RANGE_RESOLUTION_M = 0.02478966346153846
VELOCITY_RESOLUTION_MPS = 0.1252003205128205
ANGLE_BINS = 128
ANGLE_DIVISOR = ANGLE_BINS // 2
VELOCITY_OFFSET = 32

RAW_POINT_DTYPE = np.dtype([
    ("range_idx", "<u2"),
    ("doppler_idx", "u1"),
    ("azimuth_idx", "u1"),
    ("elevation_idx", "u1"),
    ("power_packed", "u1"),
])
POINT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("range", "<f4"), ("velocity", "<f4"),
    ("azimuth", "<f4"), ("elevation", "<f4"),
    ("power", "<f4"),
])
FRAME_DTYPE = np.dtype([
    ("frame_id", "<i4"), ("timestamp", "<f8"),
    ("num_points", "<i4"), ("offset", "<i4"),
])


def index_to_angle_rad(indices: np.ndarray) -> np.ndarray:
    values = indices.astype(np.float64)
    sine = np.where(values < ANGLE_DIVISOR, values, values - ANGLE_BINS)
    return np.arcsin(np.clip(sine / ANGLE_DIVISOR, -1.0, 1.0))


def parse_protocol_frame(frame: bytes) -> tuple[np.ndarray, dict[int, int], int]:
    """Parse all dynamic and static TLVs from one captured serial frame."""
    if len(frame) < 12 or frame[:2] != FRAME_HEADER:
        raise ValueError("invalid 55 AA point-cloud frame header")

    payload_size = int.from_bytes(frame[2:6], "little", signed=False)
    if len(frame) != payload_size + 6:
        raise ValueError(
            f"point-cloud frame length mismatch: {len(frame)} != {payload_size + 6}"
        )

    payload = memoryview(frame)[6:]
    if len(payload) < 6:
        raise ValueError("point-cloud payload is too short")

    frame_time_ms = int.from_bytes(payload[0:2], "little", signed=False)
    offset = 3  # The firmware's numTLVs byte is fixed to 1; parse to payload end.
    chunks: list[np.ndarray] = []
    tlv_counts: dict[int, int] = {}

    while offset < len(payload):
        if offset + 3 > len(payload):
            raise ValueError("truncated point-cloud TLV header")
        tlv_type = int(payload[offset])
        target_count = int.from_bytes(payload[offset + 1:offset + 3], "little")
        offset += 3
        point_bytes = target_count * RAW_POINT_DTYPE.itemsize
        if offset + point_bytes > len(payload):
            raise ValueError(
                f"truncated TLV {tlv_type}: {target_count} points exceed payload"
            )
        if not 1 <= tlv_type <= 7:
            raise ValueError(f"unexpected point-cloud TLV type {tlv_type}")

        if target_count:
            chunk = np.frombuffer(
                payload[offset:offset + point_bytes],
                dtype=RAW_POINT_DTYPE,
                count=target_count,
            ).copy()
            if np.any(chunk["azimuth_idx"] >= ANGLE_BINS):
                raise ValueError(f"TLV {tlv_type} contains an invalid azimuth index")
            if np.any(chunk["elevation_idx"] >= ANGLE_BINS):
                raise ValueError(f"TLV {tlv_type} contains an invalid elevation index")
            chunks.append(chunk)
        tlv_counts[tlv_type] = target_count
        offset += point_bytes

    raw = (
        np.concatenate(chunks)
        if chunks
        else np.zeros(0, dtype=RAW_POINT_DTYPE)
    )
    return raw, tlv_counts, frame_time_ms


def decode_points(raw: np.ndarray) -> np.ndarray:
    points = np.zeros(len(raw), dtype=POINT_DTYPE)
    if not len(raw):
        return points

    range_m = raw["range_idx"].astype(np.float64) * RANGE_RESOLUTION_M
    velocity = (
        VELOCITY_OFFSET - raw["doppler_idx"].astype(np.float64)
    ) * VELOCITY_RESOLUTION_MPS
    azimuth = index_to_angle_rad(raw["azimuth_idx"])
    elevation = index_to_angle_rad(raw["elevation_idx"])
    horizontal_range = range_m * np.cos(elevation)

    # Match the Calterah/TI training convention: x lateral, y forward, z up.
    points["x"] = horizontal_range * np.sin(azimuth)
    points["y"] = horizontal_range * np.cos(azimuth)
    points["z"] = range_m * np.sin(elevation)
    points["range"] = range_m
    points["velocity"] = velocity
    points["azimuth"] = np.degrees(azimuth)
    points["elevation"] = np.degrees(elevation)
    points["power"] = raw["power_packed"].astype(np.float32)
    return points


def load_index(index_path: Path) -> list[dict[str, str]]:
    required = {
        "frame_index", "timestamp_ns", "file", "file_offset", "frame_bytes",
    }
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            missing = sorted(required.difference(reader.fieldnames or []))
            raise ValueError(f"point-cloud index is missing columns: {missing}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"point-cloud index is empty: {index_path}")
    return rows


def convert(
    input_dir: Path,
    output_path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    log_every: int = 100,
    strict: bool = False,
) -> tuple[int, int]:
    pointcloud_dir = input_dir / "pointcloud"
    index_path = pointcloud_dir / "index.csv"
    if not index_path.is_file():
        raise FileNotFoundError(f"missing BGT60 point-cloud index: {index_path}")

    rows = load_index(index_path)
    selected = [
        row for row in rows
        if int(row["frame_index"]) >= start_frame
        and (end_frame is None or int(row["frame_index"]) < end_frame)
    ]
    if not selected:
        raise ValueError("the requested BGT60 frame range is empty")

    frame_records = np.zeros(len(selected), dtype=FRAME_DTYPE)
    point_parts: list[np.ndarray] = []
    tlv_totals: dict[int, int] = {}
    bad_frames: list[dict[str, object]] = []
    open_files: dict[Path, object] = {}
    point_offset = 0

    try:
        for out_index, row in enumerate(selected):
            source_frame_id = int(row["frame_index"])
            source_path = pointcloud_dir / row["file"]
            if source_path not in open_files:
                open_files[source_path] = source_path.open("rb")
            handle = open_files[source_path]
            handle.seek(int(row["file_offset"]))
            frame = handle.read(int(row["frame_bytes"]))
            if len(frame) != int(row["frame_bytes"]):
                raise ValueError(f"short read for BGT60 frame {source_frame_id}")

            try:
                raw, tlv_counts, _ = parse_protocol_frame(frame)
            except ValueError as exc:
                if strict:
                    raise ValueError(
                        f"BGT60 frame {source_frame_id}: {exc}"
                    ) from exc
                logger.warning(
                    "BGT60 frame %d is malformed; preserving it as an empty frame: %s",
                    source_frame_id,
                    exc,
                )
                bad_frames.append({
                    "frame_id": source_frame_id,
                    "reason": str(exc),
                })
                raw = np.zeros(0, dtype=RAW_POINT_DTYPE)
                tlv_counts = {}
            points = decode_points(raw)
            timestamp = int(row["timestamp_ns"]) / 1_000_000_000.0
            frame_records[out_index] = (
                source_frame_id, timestamp, len(points), point_offset,
            )
            point_parts.append(points)
            point_offset += len(points)
            for tlv_type, count in tlv_counts.items():
                tlv_totals[tlv_type] = tlv_totals.get(tlv_type, 0) + count

            if (out_index + 1) % log_every == 0 or out_index + 1 == len(selected):
                logger.info(
                    "decoded %d/%d BGT60 frames (%d points)",
                    out_index + 1,
                    len(selected),
                    point_offset,
                )
    finally:
        for handle in open_files.values():
            handle.close()

    all_points = (
        np.concatenate(point_parts)
        if point_offset
        else np.zeros(0, dtype=POINT_DTYPE)
    )
    metadata = {
        "source": str(input_dir.resolve()),
        "index": str(index_path.resolve()),
        "source_frames": len(rows),
        "output_frames": len(frame_records),
        "output_points": len(all_points),
        "protocol": "55 AA + uint32_le(payload_bytes) + multi-TLV 6-byte points",
        "range_resolution_m": RANGE_RESOLUTION_M,
        "velocity_resolution_mps": VELOCITY_RESOLUTION_MPS,
        "coordinate_system": "x_lateral_y_forward_z_up",
        "power": "firmware packed uint8: (pow3dAbs >> 12) & 0xff",
        "tlv_point_totals": {str(key): value for key, value in sorted(tlv_totals.items())},
        "malformed_frames": bad_frames,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp.npz")
    np.savez_compressed(
        temporary,
        frames=frame_records,
        points=all_points,
        meta=np.asarray([json.dumps(metadata, ensure_ascii=False)]),
    )
    temporary.replace(output_path)
    logger.info(
        "wrote %s (%d frames, %d points)",
        output_path,
        len(frame_records),
        len(all_points),
    )
    return len(frame_records), len(all_points)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert captured BGT60/SKY32B750 point-cloud packets to NPZ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input-dir", type=Path, required=True,
        help="BGT60_4T4R directory containing pointcloud/index.csv",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument(
        "--strict", action="store_true",
        help="abort on a malformed captured point-cloud frame instead of writing an empty frame",
    )
    args = parser.parse_args()

    if args.start_frame < 0:
        parser.error("--start-frame must be non-negative")
    if args.end_frame is not None and args.end_frame <= args.start_frame:
        parser.error("--end-frame must be greater than --start-frame")
    if args.log_every <= 0:
        parser.error("--log-every must be positive")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    convert(
        args.input_dir.expanduser().resolve(),
        args.output.expanduser().resolve(),
        args.start_frame,
        args.end_frame,
        args.log_every,
        args.strict,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
