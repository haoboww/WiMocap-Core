from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "data_pipeline" / "bgt60" / "convert_pointcloud.py"
)
SPEC = importlib.util.spec_from_file_location("bgt60_convert", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def make_point(range_idx: int, doppler: int, azimuth: int, elevation: int, power: int) -> bytes:
    return (
        range_idx.to_bytes(2, "little")
        + bytes((doppler, azimuth, elevation, power))
    )


class Bgt60PointcloudTest(unittest.TestCase):
    def test_multi_tlv_frame(self) -> None:
        payload = bytearray((0, 0, 1))
        payload.extend((1, 1, 0))
        payload.extend(make_point(100, 31, 0, 0, 42))
        payload.extend((2, 1, 0))
        payload.extend(make_point(80, 32, 64, 64, 7))
        frame = b"\x55\xAA" + len(payload).to_bytes(4, "little") + payload

        raw, tlv_counts, frame_time = MODULE.parse_protocol_frame(frame)
        points = MODULE.decode_points(raw)

        self.assertEqual(frame_time, 0)
        self.assertEqual(tlv_counts, {1: 1, 2: 1})
        self.assertEqual(len(points), 2)
        self.assertAlmostEqual(float(points[0]["range"]), 100 * MODULE.RANGE_RESOLUTION_M, places=5)
        self.assertAlmostEqual(float(points[0]["velocity"]), MODULE.VELOCITY_RESOLUTION_MPS, places=5)
        self.assertAlmostEqual(float(points[0]["x"]), 0.0, places=5)
        self.assertAlmostEqual(float(points[0]["y"]), float(points[0]["range"]), places=5)
        self.assertAlmostEqual(float(points[1]["velocity"]), 0.0, places=5)
        self.assertAlmostEqual(float(points[1]["z"]), -float(points[1]["range"]), places=5)
        np.testing.assert_array_equal(points["power"], np.asarray([42.0, 7.0]))

    def test_rejects_bad_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "length mismatch"):
            MODULE.parse_protocol_frame(b"\x55\xAA\x07\x00\x00\x00" + b"\x00" * 6)


if __name__ == "__main__":
    unittest.main()
