#!/usr/bin/env python3
"""
End-to-end TI IWR6843 raw ADC to point cloud.

This script is intentionally self-contained.  It does not import mmwave_dsp.
The DSP chain mirrors D:/radar_data/generate_pointcloud.py where practical:

  1. Parse radar.cfg and load DCA1000 adc_data_Raw_*.bin
  2. Convert Q-first IQ int16 stream to complex ADC frames
  3. Range FFT with windowing
  4. TDM-MIMO de-interleave to a virtual antenna cube
  5. Previous-frame mean subtraction + Doppler FFT
  6. ABS SUM detection map
  7. 1D SO-CFAR along Doppler, then 2D local peak pruning
  8. TI 3TX/4RX virtual-array Doppler compensation and DPK azimuth
  9. Optional elevation from the elevated TX row
 10. Save NPZ, RDMap images, point-cloud preview images, and statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)

C_LIGHT = 299_792_458.0

PT_DTYPE = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
    ("range", "<f4"), ("velocity", "<f4"),
    ("azimuth", "<f4"), ("elevation", "<f4"),
    ("power", "<f4"),
])

FR_DTYPE = np.dtype([
    ("frame_id", "<i4"), ("timestamp", "<f8"),
    ("num_points", "<i4"), ("offset", "<i4"),
])

MOUNT_FLIP_BORESIGHT = np.array([
    [-1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=np.float64)

MOUNT_SWAP_XY_FLIP_Z = np.array([
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
], dtype=np.float64)

MOUNT_PRESETS = {
    "none": None,
    "flip_boresight": MOUNT_FLIP_BORESIGHT,
    "swap_xy_flip_z": MOUNT_SWAP_XY_FLIP_Z,
}


def make_window(kind: str, n: int, params: tuple[float, ...] = (80.0,)) -> np.ndarray:
    kind = (kind or "none").lower()
    if kind in {"cheb", "chebwin", "chebyshev"}:
        try:
            from scipy.signal.windows import chebwin

            return chebwin(n, at=float(params[0]), sym=False).astype(np.float64)
        except Exception:
            return np.hanning(n).astype(np.float64)
    if kind in {"hann", "hanning"}:
        return np.hanning(n).astype(np.float64)
    if kind == "hamming":
        return np.hamming(n).astype(np.float64)
    if kind == "blackman":
        return np.blackman(n).astype(np.float64)
    return np.ones(n, dtype=np.float64)


def make_angle_grid(left_deg: float, right_deg: float, n: int) -> np.ndarray:
    if n <= 1:
        return np.array([(left_deg + right_deg) * 0.5], dtype=np.float64)
    s0 = np.sin(np.radians(left_deg))
    s1 = np.sin(np.radians(right_deg))
    sin_grid = np.linspace(s0, s1, n)
    return np.degrees(np.arcsin(np.clip(sin_grid, -1.0, 1.0)))


def shifted_doppler_indices(nfft: int) -> tuple[np.ndarray, np.ndarray]:
    half = nfft // 2
    idx = np.concatenate([np.arange(nfft - half, nfft), np.arange(0, half)])
    signed = np.concatenate([np.arange(-half, 0), np.arange(0, half)])
    return idx.astype(int), signed.astype(int)


def bit_indices(mask: int, max_bits: int = 8) -> list[int]:
    return [i for i in range(max_bits) if int(mask) & (1 << i)]


def _parse_number(value: str) -> Any:
    try:
        return float(value) if "." in value or "e" in value.lower() else int(value)
    except ValueError:
        return value


def parse_cfg_file(path: str | os.PathLike[str]) -> dict[str, list[list[Any]]]:
    config: dict[str, list[list[Any]]] = defaultdict(list)
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("%") or line.startswith("#"):
                continue
            parts = line.split()
            if not parts:
                continue
            config[parts[0]].append([_parse_number(v) for v in parts[1:]])
    return config


def build_ti_virtual_ant_pos(tx_order: tuple[int, ...], num_rx: int) -> np.ndarray:
    """Return TI virtual antenna positions in wavelength units.

    IWR6843 3TX/4RX layout used here:
      TX0 + RX0..3: x = 0..3 half-lambda, z = 0
      TX1 + RX0..3: x = 2..5 half-lambda, z = 1 half-lambda
      TX2 + RX0..3: x = 4..7 half-lambda, z = 0
    """

    def tx_base(tx_id: int) -> tuple[float, float]:
        if tx_id == 0:
            return 0.0, 0.0
        if tx_id == 1:
            return 1.0, 0.5
        if tx_id == 2:
            return 2.0, 0.0
        return float(tx_id) * 2.0, 0.0

    rx_x = 0.5 * np.arange(num_rx, dtype=np.float64)
    positions = []
    for tx_id in tx_order:
        base_x, base_z = tx_base(int(tx_id))
        for rx in range(num_rx):
            positions.append([base_x + rx_x[rx], base_z])
    return np.asarray(positions, dtype=np.float64)


def steering_matrix(
    positions: np.ndarray,
    az_deg: float | np.ndarray,
    el_deg: float | np.ndarray = 0.0,
) -> np.ndarray:
    az = np.radians(np.asarray(az_deg, dtype=np.float64))
    el = np.radians(np.asarray(el_deg, dtype=np.float64))
    az_b, el_b = np.broadcast_arrays(az, el)
    vx = np.sin(az_b).ravel() * np.cos(el_b).ravel()
    vz = np.sin(el_b).ravel()
    phase = positions[:, 0, np.newaxis] * vx[np.newaxis, :]
    phase += positions[:, 1, np.newaxis] * vz[np.newaxis, :]
    return np.exp(-1j * 2.0 * np.pi * phase)


@dataclass
class TIRadarConfig:
    # Antenna and TDM order.
    num_rx: int = 4
    num_tx: int = 3
    tx_order: tuple[int, ...] = (0, 1, 2)

    # Timing from profileCfg, in microseconds.
    idle_time: float = 100.0
    adc_start_time: float = 4.38
    ramp_end_time: float = 73.30

    # RF.
    start_freq: float = 60.0       # GHz
    freq_slope: float = 54.508     # MHz/us

    # Sampling.
    num_adc_samples: int = 512
    sample_rate: float = 7500.0    # ksps

    # Frame.
    num_chirps_total: int = 528
    frame_periodicity_ms: float = 100.0

    # FFT and search.
    rng_nfft: int = 0
    vel_nfft: int = 0
    use_range_bins: int = 0
    num_angle_bins: int = 360
    num_elev_bins: int = 181
    az_left: float = -60.0
    az_right: float = 60.0
    el_down: float = -60.0
    el_up: float = 60.0
    range_window: str = "cheb"
    velocity_window: str = "cheb"

    # Derived.
    num_chirps_per_tx: int = field(init=False)
    num_range_bins: int = field(init=False)
    num_doppler_bins: int = field(init=False)
    num_virtual_ant: int = field(init=False)
    use_range: int = field(init=False)
    fft2d_points: int = field(init=False)
    bandwidth: float = field(init=False)
    center_freq_hz: float = field(init=False)
    wavelength: float = field(init=False)
    range_resolution: float = field(init=False)
    doppler_resolution: float = field(init=False)
    max_range: float = field(init=False)
    max_velocity: float = field(init=False)
    frame_size_i16: int = field(init=False)
    frame_bytes: int = field(init=False)
    virtual_ant_pos: np.ndarray = field(init=False)
    az_ant_idx: np.ndarray = field(init=False)
    az_grid: np.ndarray = field(init=False)
    el_grid: np.ndarray = field(init=False)
    az_steering: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        self.tx_order = tuple(int(t) for t in self.tx_order)
        if len(self.tx_order) == 0:
            self.tx_order = tuple(range(int(self.num_tx)))
        self.num_tx = len(self.tx_order)
        if self.num_chirps_total % self.num_tx != 0:
            raise ValueError(
                f"num_chirps_total ({self.num_chirps_total}) is not divisible by "
                f"num_tx ({self.num_tx}). Check frameCfg/chirpCfg."
            )

        self.num_chirps_per_tx = self.num_chirps_total // self.num_tx
        self.rng_nfft = int(self.rng_nfft or self.num_adc_samples)
        self.vel_nfft = int(self.vel_nfft or self.num_chirps_per_tx)
        self.num_range_bins = self.rng_nfft
        self.num_doppler_bins = self.vel_nfft
        self.fft2d_points = self.vel_nfft
        self.use_range = int(self.use_range_bins or self.num_range_bins)
        self.use_range = max(1, min(self.use_range, self.num_range_bins))
        self.num_virtual_ant = self.num_rx * self.num_tx

        fs_hz = self.sample_rate * 1e3
        slope_hz_s = self.freq_slope * 1e12
        adc_duration_s = self.num_adc_samples / fs_hz
        self.bandwidth = slope_hz_s * adc_duration_s
        self.range_resolution = C_LIGHT * fs_hz / (2.0 * slope_hz_s * self.rng_nfft)
        self.max_range = self.range_resolution * self.use_range

        self.center_freq_hz = (
            self.start_freq * 1e9
            + self.adc_start_time * self.freq_slope * 1e6
            + self.bandwidth / 2.0
        )
        self.wavelength = C_LIGHT / self.center_freq_hz
        chirp_interval = (self.ramp_end_time + self.idle_time) * 1e-6
        self.doppler_resolution = C_LIGHT / (
            2.0 * self.fft2d_points * self.num_tx * self.center_freq_hz * chirp_interval
        )
        self.max_velocity = self.doppler_resolution * self.fft2d_points / 2.0

        self.frame_size_i16 = (
            self.num_chirps_total * self.num_rx * self.num_adc_samples * 2
        )
        self.frame_bytes = self.frame_size_i16 * np.dtype(np.int16).itemsize

        self.virtual_ant_pos = build_ti_virtual_ant_pos(self.tx_order, self.num_rx)
        z_min = float(np.min(self.virtual_ant_pos[:, 1]))
        self.az_ant_idx = np.where(np.isclose(self.virtual_ant_pos[:, 1], z_min))[0]
        if self.az_ant_idx.size < self.num_rx:
            self.az_ant_idx = np.arange(self.num_virtual_ant)

        self.az_grid = make_angle_grid(self.az_left, self.az_right, self.num_angle_bins)
        self.el_grid = make_angle_grid(self.el_down, self.el_up, self.num_elev_bins)
        self.az_steering = steering_matrix(self.virtual_ant_pos[self.az_ant_idx], self.az_grid, 0.0)

    @classmethod
    def from_cfg(
        cls,
        cfg_path: str | os.PathLike[str],
        *,
        rng_nfft: int | None = None,
        vel_nfft: int | None = None,
        use_range_bins: int | None = None,
        num_angle_bins: int = 360,
        num_elev_bins: int = 181,
        az_left: float = -60.0,
        az_right: float = 60.0,
        el_down: float = -60.0,
        el_up: float = 60.0,
        range_window: str = "cheb",
        velocity_window: str = "cheb",
    ) -> "TIRadarConfig":
        raw = parse_cfg_file(cfg_path)
        kwargs: dict[str, Any] = {}
        tx_ids = [0, 1, 2]

        if "channelCfg" in raw:
            ch = raw["channelCfg"][0]
            rx_ids = bit_indices(int(ch[0]), 8)
            tx_ids = bit_indices(int(ch[1]), 4)
            kwargs["num_rx"] = len(rx_ids) if rx_ids else 4
            kwargs["num_tx"] = len(tx_ids) if tx_ids else 3

        if "profileCfg" in raw:
            p = raw["profileCfg"][0]
            kwargs["start_freq"] = float(p[1])
            kwargs["idle_time"] = float(p[2])
            kwargs["adc_start_time"] = float(p[3])
            kwargs["ramp_end_time"] = float(p[4])
            kwargs["freq_slope"] = float(p[7])
            kwargs["num_adc_samples"] = int(p[9])
            kwargs["sample_rate"] = float(p[10])

        frame_start = 0
        frame_end = max(len(tx_ids) - 1, 0)
        loops = 1
        if "frameCfg" in raw:
            f = raw["frameCfg"][0]
            frame_start = int(f[0])
            frame_end = int(f[1])
            loops = int(f[2])
            kwargs["frame_periodicity_ms"] = float(f[4]) if len(f) > 4 else 100.0
            kwargs["num_chirps_total"] = (frame_end - frame_start + 1) * loops

        chirp_by_idx: dict[int, list[Any]] = {}
        for c in raw.get("chirpCfg", []):
            if len(c) >= 8:
                chirp_by_idx[int(c[0])] = c

        tx_order = []
        for chirp_idx in range(frame_start, frame_end + 1):
            chirp = chirp_by_idx.get(chirp_idx)
            if chirp is None:
                if tx_ids:
                    tx_order.append(tx_ids[(chirp_idx - frame_start) % len(tx_ids)])
                continue
            enabled = bit_indices(int(chirp[7]), 4)
            tx_order.append(enabled[0] if enabled else 0)

        if tx_order:
            kwargs["tx_order"] = tuple(tx_order)
            kwargs["num_tx"] = len(tx_order)
        elif tx_ids:
            kwargs["tx_order"] = tuple(tx_ids)
            kwargs["num_tx"] = len(tx_ids)

        return cls(
            **kwargs,
            rng_nfft=int(rng_nfft or 0),
            vel_nfft=int(vel_nfft or 0),
            use_range_bins=int(use_range_bins or 0),
            num_angle_bins=num_angle_bins,
            num_elev_bins=num_elev_bins,
            az_left=az_left,
            az_right=az_right,
            el_down=el_down,
            el_up=el_up,
            range_window=range_window,
            velocity_window=velocity_window,
        )

    def summary(self) -> str:
        fs_hz = self.sample_rate * 1e3
        lines = [
            "TI IWR6843 Config:",
            f"  Antennas      : {self.num_tx}TX {self.num_rx}RX -> {self.num_virtual_ant} VA",
            f"  TDM order     : {list(self.tx_order)} ({self.num_chirps_per_tx} chirps/TX)",
            f"  Frame layout  : {self.num_chirps_total} chirps x {self.num_rx} RX x "
            f"{self.num_adc_samples} samples x IQ int16 = {self.frame_bytes:,} B",
            f"  RF            : start={self.start_freq:.3f} GHz, center={self.center_freq_hz/1e9:.3f} GHz, "
            f"slope={self.freq_slope:.3f} MHz/us",
            f"  Timing        : idle={self.idle_time:.2f} us, ramp={self.ramp_end_time:.2f} us, "
            f"adc_start={self.adc_start_time:.2f} us, frame={self.frame_periodicity_ms:.1f} ms",
            f"  ADC           : Fs={fs_hz/1e6:.3f} MHz, adc_samples={self.num_adc_samples}",
            f"  Range         : res={self.range_resolution:.4f} m/bin, max={self.max_range:.2f} m, "
            f"rng_nfft={self.rng_nfft}, use_range={self.use_range}",
            f"  Doppler       : res={self.doppler_resolution:.4f} m/s/bin, "
            f"max=+/-{self.max_velocity:.2f} m/s, FFT={self.fft2d_points}",
            f"  Angle         : az=[{self.az_left:.0f},{self.az_right:.0f}]/{self.num_angle_bins}, "
            f"el=[{self.el_down:.0f},{self.el_up:.0f}]/{self.num_elev_bins}, "
            f"az_ant={self.az_ant_idx.tolist()}",
            f"  Windows       : range={self.range_window}, doppler={self.velocity_window}",
        ]
        return "\n".join(lines)


def collect_bin_files(bin_path: Path) -> list[Path]:
    if not bin_path.name.startswith("adc_data_Raw_"):
        return [bin_path]
    files = sorted(
        bin_path.parent.glob("adc_data_Raw_*.bin"),
        key=lambda p: int(p.stem.split("_")[-1]),
    )
    return files if files else [bin_path]


def resolve_capture_paths(
    input_dir: str | None,
    cfg_path: str | None,
    bin_path: str | None,
) -> tuple[Path, Path, Path]:
    if input_dir:
        input_path = Path(input_dir).expanduser().resolve()
    elif bin_path:
        input_path = Path(bin_path).expanduser().resolve().parent
    elif cfg_path:
        input_path = Path(cfg_path).expanduser().resolve().parent
    else:
        raise FileNotFoundError("Provide --input-dir, or both --cfg/--bin.")

    if not input_path.exists() or not input_path.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_path}")

    cfg = Path(cfg_path).expanduser().resolve() if cfg_path else input_path / "radar.cfg"
    if bin_path:
        adc_bin = Path(bin_path).expanduser().resolve()
    else:
        candidates = sorted(
            input_path.glob("adc_data_Raw_*.bin"),
            key=lambda p: int(p.stem.split("_")[-1]),
        )
        adc_bin = candidates[0] if candidates else input_path / "adc_data_Raw_0.bin"

    if not cfg.exists():
        raise FileNotFoundError(f"TI cfg not found: {cfg}")
    if not adc_bin.exists():
        raise FileNotFoundError(f"TI raw bin not found: {adc_bin}")
    return input_path, cfg, adc_bin


def read_raw_i16(bin_path: Path) -> np.ndarray:
    parts = collect_bin_files(bin_path)
    if len(parts) == 1:
        return np.fromfile(str(parts[0]), dtype=np.int16)
    chunks = [np.fromfile(str(p), dtype=np.int16) for p in parts]
    print(f"  concatenated {len(parts)} adc_data_Raw_*.bin files")
    return np.concatenate(chunks)


def validate_raw(raw: np.ndarray, label: str = "") -> bool:
    if raw.size == 0:
        print(f"[ERROR]{label} raw file is empty")
        return False
    s = raw[: min(raw.size, 200_000)].astype(np.float64)
    if float(np.std(s)) < 1.0:
        print(f"[ERROR]{label} raw data variance is too low")
        return False
    return True


def split_frames(raw: np.ndarray, cfg: TIRadarConfig) -> tuple[list[np.ndarray], int, int]:
    n_full = raw.size // cfg.frame_size_i16
    remainder = raw.size % cfg.frame_size_i16
    frames = [
        raw[i * cfg.frame_size_i16:(i + 1) * cfg.frame_size_i16]
        for i in range(n_full)
    ]
    return frames, n_full, remainder


def organize_frame_qfirst(
    raw_frame: np.ndarray,
    cfg: TIRadarConfig,
    iq_order: str = "qfirst",
) -> np.ndarray:
    """Convert DCA1000 complex ADC int16 data to (chirp, rx, sample)."""
    ret = np.empty(raw_frame.size // 2, dtype=np.complex64)
    if iq_order == "ifirst":
        ret[0::2] = raw_frame[0::4].astype(np.float32) + 1j * raw_frame[2::4].astype(np.float32)
        ret[1::2] = raw_frame[1::4].astype(np.float32) + 1j * raw_frame[3::4].astype(np.float32)
    else:
        ret[0::2] = raw_frame[2::4].astype(np.float32) + 1j * raw_frame[0::4].astype(np.float32)
        ret[1::2] = raw_frame[3::4].astype(np.float32) + 1j * raw_frame[1::4].astype(np.float32)
    return ret.reshape((cfg.num_chirps_total, cfg.num_rx, cfg.num_adc_samples))


def raw_frame_to_fft1d(
    frame_raw: np.ndarray,
    cfg: TIRadarConfig,
    *,
    iq_order: str = "qfirst",
    dc_remove: bool = True,
) -> np.ndarray:
    """Raw frame -> (chirps_per_tx, virtual_ant, range_bin) range FFT cube."""
    adc = organize_frame_qfirst(frame_raw, cfg, iq_order=iq_order).astype(np.complex128, copy=False)
    if dc_remove:
        adc = adc - adc.mean(axis=2, keepdims=True)

    win = make_window(cfg.range_window, cfg.num_adc_samples, (80.0,))
    adc_win = adc * win[np.newaxis, np.newaxis, :]
    rfft = np.fft.fft(adc_win, n=cfg.rng_nfft, axis=2)
    rfft = rfft[:, :, :cfg.use_range]

    cpt = cfg.num_chirps_per_tx
    ntx = cfg.num_tx
    nrx = cfg.num_rx
    nr = cfg.use_range
    tdm = rfft.reshape(cpt, ntx, nrx, nr)

    fft1d = np.empty((cpt, cfg.num_virtual_ant, nr), dtype=np.complex128)
    for tx_slot in range(ntx):
        for rx in range(nrx):
            va = tx_slot * nrx + rx
            fft1d[:, va, :] = tdm[:, tx_slot, rx, :]
    return fft1d


def doppler_fft_hw(
    fft1d: np.ndarray,
    cfg: TIRadarConfig,
    mean_sub: np.ndarray | None = None,
) -> np.ndarray:
    """Doppler FFT. Returns (virtual_ant, range, doppler_bin)."""
    nc = cfg.num_chirps_per_tx
    nfft = cfg.fft2d_points
    win = make_window(cfg.velocity_window, nc, (80.0,))
    rd_maps = np.zeros((cfg.num_virtual_ant, cfg.use_range, nfft), dtype=np.complex128)
    for va in range(cfg.num_virtual_ant):
        data = fft1d[:, va, :].copy()
        if mean_sub is not None:
            data -= mean_sub[va]
        data *= win[:, np.newaxis]
        rd_maps[va] = np.fft.fft(data, n=nfft, axis=0).T / nc
    return rd_maps


def abs_sum_hw(rd_maps: np.ndarray, cfg: TIRadarConfig) -> np.ndarray:
    return np.sum(np.abs(rd_maps), axis=0) / float(cfg.fft2d_points)


def cfar_1d_so_mult(signal: np.ndarray, guard: int, search: int, mul_fac: float) -> list[int]:
    n = len(signal)
    total = guard + search
    detections: list[int] = []
    for i in range(n):
        left_sum = 0.0
        right_sum = 0.0
        for j in range(search):
            left_sum += float(signal[(i - total + j) % n])
            right_sum += float(signal[(i + guard + 1 + j) % n])
        noise = min(left_sum / search, right_sum / search)
        if float(signal[i]) > noise * mul_fac:
            detections.append(i)
    return detections


def cfar_detect_hw(
    rd_abs_sum: np.ndarray,
    cfg: TIRadarConfig,
    *,
    skip_bins: int = 5,
    cfar_guard: int = 2,
    cfar_search: int = 6,
    cfar_mul: float = 4.0,
    local_max: bool = True,
    zero_doppler_bins: int = 1,
    max_detections: int = 160,
    min_range_m: float | None = None,
    max_range_m: float | None = None,
) -> list[tuple[int, int, int, float]]:
    nr, nfft = rd_abs_sum.shape
    vel_map, _ = shifted_doppler_indices(nfft)
    nv = len(vel_map)
    valid_data = rd_abs_sum[:, vel_map]

    start_r = max(skip_bins, 0)
    if min_range_m is not None:
        start_r = max(start_r, int(np.floor(min_range_m / cfg.range_resolution)))
    end_r = nr
    if max_range_m is not None:
        end_r = min(end_r, int(np.ceil(max_range_m / cfg.range_resolution)) + 1)

    cfar_mask = np.zeros((nr, nv), dtype=bool)
    center_v = nv // 2
    for r in range(start_r, end_r):
        det_v = cfar_1d_so_mult(valid_data[r], cfar_guard, cfar_search, cfar_mul)
        for v in det_v:
            if zero_doppler_bins > 0 and abs(int(v) - center_v) < zero_doppler_bins:
                continue
            cfar_mask[r, v] = True

    if local_max:
        local_peak = np.ones_like(cfar_mask, dtype=bool)
        for dr in (-1, 0, 1):
            shifted_r = np.roll(valid_data, dr, axis=0)
            if dr < 0:
                shifted_r[dr:, :] = -np.inf
            elif dr > 0:
                shifted_r[:dr, :] = -np.inf
            for dv in (-1, 0, 1):
                if dr == 0 and dv == 0:
                    continue
                shifted = np.roll(shifted_r, dv, axis=1)
                local_peak &= valid_data >= shifted
        cfar_mask &= local_peak

    detections: list[tuple[int, int, int, float]] = []
    for r in range(start_r, end_r):
        for v in range(nv):
            if cfar_mask[r, v]:
                detections.append((r, int(vel_map[v]), int(v), float(valid_data[r, v])))

    if max_detections is not None and max_detections > 0 and len(detections) > max_detections:
        detections.sort(key=lambda d: d[3], reverse=True)
        detections = detections[:max_detections]
    return detections


def doppler_compensation(vel_signed: int, tx_slot: int, cfg: TIRadarConfig, sign: float = 1.0) -> complex:
    phase = sign * 2.0 * np.pi * vel_signed * tx_slot / (cfg.fft2d_points * cfg.num_tx)
    return np.exp(1j * phase)


def build_sinc_buf(num_angle: int = 128, n_ant_h: int = 8) -> np.ndarray:
    sinc = np.zeros(num_angle, dtype=np.complex128)
    for k in range(num_angle):
        for n in range(n_ant_h):
            sinc[k] += np.exp(-1j * 2.0 * np.pi * n * k / num_angle)
        sinc[k] /= max(n_ant_h, 1)
    return sinc


def estimate_azimuth_dpk(
    point_data: np.ndarray,
    cfg: TIRadarConfig,
    sinc_buf: np.ndarray | None = None,
    *,
    dpk_times: int = 2,
    dpk_threshold: float = 3.0,
) -> list[tuple[int, float]]:
    del sinc_buf
    spectrum = np.abs(point_data[cfg.az_ant_idx] @ cfg.az_steering)
    residual = float(np.mean(spectrum) + 1e-12)
    is_peak = np.ones_like(spectrum, dtype=bool)
    is_peak[1:] &= spectrum[1:] >= spectrum[:-1]
    is_peak[:-1] &= spectrum[:-1] >= spectrum[1:]
    peak_idx = np.where(is_peak)[0]
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(spectrum))])

    order = peak_idx[np.argsort(spectrum[peak_idx])[::-1]]
    valid_peaks: list[tuple[int, float]] = []
    for az_idx in order[: max(1, int(dpk_times))]:
        cut_pow = float(spectrum[az_idx])
        if cut_pow > residual * dpk_threshold:
            valid_peaks.append((int(az_idx), cut_pow))
    return valid_peaks


def estimate_elevation(
    point_data: np.ndarray,
    az_deg: float,
    cfg: TIRadarConfig,
) -> float:
    if cfg.num_tx < 2:
        return 0.0

    row_data = []
    row_z = []
    for tx_slot in range(cfg.num_tx):
        va_idx = tx_slot * cfg.num_rx + np.arange(cfg.num_rx)
        positions = cfg.virtual_ant_pos[va_idx]
        az_comp = steering_matrix(positions, az_deg, 0.0).ravel()
        sig = np.sum(point_data[va_idx] * az_comp)
        if np.abs(sig) > 1e-12:
            row_data.append(sig)
            row_z.append(float(np.mean(positions[:, 1])))

    if len(row_data) < 2:
        return 0.0

    row_data_arr = np.asarray(row_data, dtype=np.complex128)
    row_z_arr = np.asarray(row_z, dtype=np.float64)
    sin_limit = float(np.max(np.abs(np.sin(np.radians(cfg.el_grid)))))
    estimates = []
    weights = []
    for i in range(len(row_data_arr)):
        for j in range(i + 1, len(row_data_arr)):
            dz = row_z_arr[i] - row_z_arr[j]
            if abs(dz) < 1e-9:
                continue
            phase = np.angle(row_data_arr[i] * np.conj(row_data_arr[j]))
            sin_el = phase / (2.0 * np.pi * dz)
            estimates.append(np.clip(sin_el, -sin_limit, sin_limit))
            weights.append(np.abs(row_data_arr[i]) * np.abs(row_data_arr[j]) * abs(dz))

    if estimates and float(np.sum(weights)) > 1e-12:
        sin_el = float(np.average(np.asarray(estimates), weights=np.asarray(weights)))
        return float(np.degrees(np.arcsin(np.clip(sin_el, -sin_limit, sin_limit))))

    el_steering = steering_matrix(cfg.virtual_ant_pos, az_deg, cfg.el_grid)
    spectrum = np.abs(point_data @ el_steering)
    return float(cfg.el_grid[int(np.argmax(spectrum))])


def estimate_calibration(
    first_fft1d: np.ndarray,
    cfg: TIRadarConfig,
    *,
    calib_file: str | None = None,
    auto_calib: bool = True,
    skip_bins: int = 5,
) -> np.ndarray | None:
    if calib_file:
        path = Path(calib_file).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        calib = np.load(str(path))
        if calib.shape[0] != cfg.num_virtual_ant:
            raise ValueError(
                f"Calibration length {calib.shape[0]} != virtual antennas {cfg.num_virtual_ant}"
            )
        print(f"  loaded calibration: {path} ({calib.shape})")
        return calib.astype(np.complex128)

    if not auto_calib:
        return None

    rd_maps = doppler_fft_hw(first_fft1d, cfg, mean_sub=None)
    vel_map, _ = shifted_doppler_indices(cfg.fft2d_points)
    zero_bin = int(vel_map[cfg.fft2d_points // 2])
    start = min(max(skip_bins, 0), cfg.use_range - 1)
    rp_v0 = np.sum(np.abs(rd_maps[:, :, zero_bin]), axis=0)
    peak_r = int(np.argmax(rp_v0[start:])) + start
    calib_ref = np.asarray([rd_maps[va, peak_r, zero_bin] for va in range(cfg.num_virtual_ant)])
    calib_phase = np.conj(calib_ref / (np.abs(calib_ref) + 1e-30))
    print(
        f"  auto calibration: range_bin={peak_r} "
        f"({peak_r * cfg.range_resolution:.2f} m), mean_amp={np.mean(np.abs(calib_ref)):.1f}"
    )
    return calib_phase


def process_frame_hw(
    fft1d: np.ndarray,
    cfg: TIRadarConfig,
    *,
    mean_sub: np.ndarray | None = None,
    cfar_guard: int = 2,
    cfar_search: int = 6,
    cfar_mul: float = 4.0,
    dpk_times: int = 2,
    dpk_threshold: float = 3.0,
    skip_bins: int = 5,
    max_angle: float = 60.0,
    cfar_local_max: bool = True,
    zero_doppler_bins: int = 1,
    max_detections: int = 160,
    min_range_m: float | None = None,
    max_range_m: float | None = None,
    sinc_buf: np.ndarray | None = None,
    calib_phase: np.ndarray | None = None,
    enable_elevation: bool = True,
    doppler_comp_sign: float = 1.0,
    mount_rotation: np.ndarray | None = None,
    z_clip: float | None = None,
) -> tuple[np.ndarray, dict[str, int], np.ndarray]:
    rd_maps = doppler_fft_hw(fft1d, cfg, mean_sub=mean_sub)
    rd_abs_sum = abs_sum_hw(rd_maps, cfg)
    detections = cfar_detect_hw(
        rd_abs_sum,
        cfg,
        skip_bins=skip_bins,
        cfar_guard=cfar_guard,
        cfar_search=cfar_search,
        cfar_mul=cfar_mul,
        local_max=cfar_local_max,
        zero_doppler_bins=zero_doppler_bins,
        max_detections=max_detections,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
    )

    _, signed_bins = shifted_doppler_indices(cfg.fft2d_points)
    points_list: list[tuple[float, float, float, float, float, float, float, float]] = []
    dpk_total = 0
    dpk_pass = 0

    for r_idx, vbin, v_idx, _power in detections:
        vel_signed = int(signed_bins[v_idx])
        point_data = np.zeros(cfg.num_virtual_ant, dtype=np.complex128)
        for tx_slot in range(cfg.num_tx):
            comp = doppler_compensation(vel_signed, tx_slot, cfg, sign=doppler_comp_sign)
            for rx in range(cfg.num_rx):
                va = tx_slot * cfg.num_rx + rx
                val = rd_maps[va, r_idx, vbin]
                if calib_phase is not None:
                    val *= calib_phase[va]
                point_data[va] = val * comp

        az_peaks = estimate_azimuth_dpk(
            point_data,
            cfg,
            sinc_buf,
            dpk_times=dpk_times,
            dpk_threshold=dpk_threshold,
        )
        dpk_total += max(1, int(dpk_times))
        dpk_pass += len(az_peaks)

        for az_idx, cut_pow in az_peaks:
            range_val = r_idx * cfg.range_resolution
            vel_val = vel_signed * cfg.doppler_resolution
            az_deg = float(cfg.az_grid[az_idx])
            if max_angle > 0 and abs(az_deg) > max_angle:
                continue

            el_deg = estimate_elevation(point_data, az_deg, cfg) if enable_elevation else 0.0
            if max_angle > 0 and abs(el_deg) > max_angle:
                continue

            az_rad = np.radians(az_deg)
            el_rad = np.radians(el_deg)
            x = range_val * np.cos(el_rad) * np.sin(az_rad)
            y = range_val * np.cos(el_rad) * np.cos(az_rad)
            z = range_val * np.sin(el_rad)

            if mount_rotation is not None:
                xyz = mount_rotation @ np.asarray([x, y, z], dtype=np.float64)
                x, y, z = float(xyz[0]), float(xyz[1]), float(xyz[2])
                rr = max(range_val, 1e-10)
                az_deg = float(np.degrees(np.arctan2(x, y)))
                el_deg = float(np.degrees(np.arcsin(np.clip(z / rr, -1.0, 1.0))))

            if z_clip is not None and z_clip > 0:
                z = float(np.clip(z, -z_clip, z_clip))

            points_list.append((x, y, z, range_val, vel_val, az_deg, el_deg, float(cut_pow)))

    pts = np.zeros(len(points_list), dtype=PT_DTYPE)
    for i, point in enumerate(points_list):
        pts[i] = point

    stats = {
        "cfar_detections": len(detections),
        "dpk_candidates": dpk_total,
        "dpk_passed": dpk_pass,
        "final_points": len(points_list),
    }
    return pts, stats, rd_abs_sum


def save_npz(frames: np.ndarray, points: np.ndarray, filepath: Path, meta: dict[str, Any]) -> None:
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(filepath),
        frames=frames,
        points=points,
        meta=np.array([json.dumps(meta, ensure_ascii=False)]),
    )
    print(f"saved point cloud: {filepath} ({len(points)} points, {len(frames)} frames)")


def save_rdmap(rd_abs_sum: np.ndarray, frame_id: int, output_dir: Path, cfg: TIRadarConfig) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    vel_bins, _ = shifted_doppler_indices(cfg.fft2d_points)
    valid = rd_abs_sum[:, vel_bins]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        20.0 * np.log10(np.maximum(valid, 1e-10)),
        aspect="auto",
        origin="lower",
        cmap="jet",
    )
    ax.set_xlabel(f"Velocity bin (shifted, 0-{cfg.fft2d_points - 1})")
    ax.set_ylabel(f"Range bin (0-{cfg.use_range - 1})")
    ax.set_title(f"TI Range-Doppler Map - Frame {frame_id}")
    fig.colorbar(im, ax=ax, label="dB")
    fig.tight_layout()
    fig.savefig(output_dir / f"rdmap_{frame_id:04d}.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def visualize_pointcloud(points: np.ndarray, title: str, filepath: Path) -> None:
    if len(points) == 0:
        return
    x = points["x"]
    y = points["y"]
    z = points["z"]
    pw = points["power"]
    vel = points["velocity"]
    rng = points["range"]
    pw_n = (pw - pw.min()) / (pw.max() - pw.min() + 1e-10)

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    sc1 = ax1.scatter(x, y, z, c=pw_n, cmap="jet", s=8, alpha=0.75)
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")
    ax1.set_zlabel("Z")
    ax1.set_title(f"{title}\n{len(points)} pts")
    fig.colorbar(sc1, ax=ax1, shrink=0.6, label="Power")

    ax2 = fig.add_subplot(132)
    sc2 = ax2.scatter(x, y, c=vel, cmap="coolwarm", s=8, alpha=0.75)
    ax2.set_xlabel("X (m)")
    ax2.set_ylabel("Y (m)")
    ax2.set_title("Top View")
    ax2.grid(True, alpha=0.3)
    ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, shrink=0.8, label="Velocity (m/s)")

    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(rng, vel, c=pw_n, cmap="hot", s=8, alpha=0.75)
    ax3.set_xlabel("Range (m)")
    ax3.set_ylabel("Velocity (m/s)")
    ax3.set_title("Range-Velocity")
    ax3.grid(True, alpha=0.3)
    fig.colorbar(sc3, ax=ax3, shrink=0.8, label="Power")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


def get_frame_points(points: np.ndarray, frame_meta: np.void) -> np.ndarray:
    start = int(frame_meta["offset"])
    end = start + int(frame_meta["num_points"])
    return points[start:end]


def write_statistics(
    path: Path,
    *,
    input_dir: Path,
    cfg_path: Path,
    adc_bin_path: Path,
    output_dir: Path,
    args: argparse.Namespace,
    cfg: TIRadarConfig,
    frames: np.ndarray,
    points: np.ndarray,
    frame_stats: list[dict[str, int]],
    elapsed: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("TI IWR6843 point cloud statistics\n")
        f.write("=" * 70 + "\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Input dir: {input_dir}\n")
        f.write(f"Config: {cfg_path}\n")
        f.write(f"Raw bin: {adc_bin_path}\n")
        f.write(f"Output: {output_dir}\n\n")
        f.write(cfg.summary() + "\n\n")
        f.write("Processing\n")
        f.write("-" * 70 + "\n")
        f.write(f"Frames: [{args.start_frame}, {args.end_frame if args.end_frame is not None else 'END'})\n")
        f.write(
            f"CFAR: guard={args.cfar_guard}, search={args.cfar_search}, mul={args.cfar_mul}, "
            f"local_max={not args.no_local_max}, zero_doppler_bins={args.zero_doppler_bins}, "
            f"max_detections={args.max_detections}\n"
        )
        f.write(
            f"Range filter: skip_bins={args.skip_bins}, min={args.min_range}, max={args.max_range}\n"
        )
        f.write(
            f"DPK: times={args.dpk_times}, threshold={args.dpk_threshold}, max_angle={args.max_angle}\n"
        )
        f.write(
            f"Elevation: enabled={not args.no_elevation}, z_clip={args.z_clip}, mount={args.mount}\n"
        )
        f.write(
            f"Calibration: calib={args.calib}, auto={not args.no_auto_calib}\n\n"
        )
        f.write("Summary\n")
        f.write("-" * 70 + "\n")
        f.write(f"Total frames: {len(frames)}\n")
        f.write(f"Total points: {len(points)}\n")
        f.write(f"Elapsed: {elapsed:.2f} s\n")
        if len(frames) > 0:
            f.write(f"Average points/frame: {len(points) / len(frames):.2f}\n")
            f.write(f"Frames with points: {int(np.sum(frames['num_points'] > 0))}\n")
        if len(points) > 0:
            f.write(
                f"XYZ range: X[{points['x'].min():.2f},{points['x'].max():.2f}] "
                f"Y[{points['y'].min():.2f},{points['y'].max():.2f}] "
                f"Z[{points['z'].min():.2f},{points['z'].max():.2f}]\n"
            )
            f.write(
                f"Velocity range: [{points['velocity'].min():.2f},{points['velocity'].max():.2f}] m/s\n"
            )

        f.write("\nPer-frame\n")
        f.write("-" * 70 + "\n")
        f.write(f"{'frame':>6} {'CFAR':>6} {'DPK_cand':>9} {'DPK_pass':>9} {'points':>7} {'offset':>8}\n")
        for fr, st in zip(frames, frame_stats):
            f.write(
                f"{int(fr['frame_id']):6d} {st['cfar_detections']:6d} "
                f"{st['dpk_candidates']:9d} {st['dpk_passed']:9d} "
                f"{int(fr['num_points']):7d} {int(fr['offset']):8d}\n"
            )


def should_save_rdmap(args: argparse.Namespace, fi: int, n_frames: int) -> bool:
    if not args.save_rdmap:
        return False
    every = int(args.rdmap_every)
    if every <= 1:
        return True
    return fi == 0 or fi == n_frames - 1 or (fi % every == 0)


def run(args: argparse.Namespace) -> None:
    input_dir, cfg_path, adc_bin_path = resolve_capture_paths(args.input_dir, args.cfg, args.bin)
    cfg = TIRadarConfig.from_cfg(
        cfg_path,
        rng_nfft=args.range_nfft,
        vel_nfft=args.vel_nfft,
        use_range_bins=args.use_range_bins,
        num_angle_bins=args.angle_bins,
        num_elev_bins=args.elev_bins,
        az_left=-abs(args.max_angle),
        az_right=abs(args.max_angle),
        el_down=args.el_down,
        el_up=args.el_up,
        range_window=args.range_window,
        velocity_window=args.doppler_window,
    )
    print(cfg.summary())

    output_dir = (
        Path(args.output).expanduser().resolve()
        if args.output
        else input_dir.parent / f"{input_dir.name}_ti_pointcloud"
    )
    rdmap_dir = output_dir / "rdmaps"
    pc_dir = output_dir / "pointclouds"
    stats_dir = output_dir / "stats"
    for d in (rdmap_dir, pc_dir, stats_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading raw ADC: {adc_bin_path}")
    raw = read_raw_i16(adc_bin_path)
    print(f"  raw int16 count: {raw.size:,}; expected/frame: {cfg.frame_size_i16:,}")
    if not validate_raw(raw, adc_bin_path.name):
        sys.exit(2)

    frame_chunks, n_full, remainder = split_frames(raw, cfg)
    if remainder:
        pct = 100.0 * remainder / cfg.frame_size_i16
        print(f"  discarded incomplete tail: {remainder:,} int16 ({pct:.1f}% of one frame)")
    print(f"  complete frames: {n_full}")

    start_fi = max(0, int(args.start_frame))
    end_fi = n_full if args.end_frame is None else min(n_full, int(args.end_frame))
    frame_chunks = frame_chunks[start_fi:end_fi]
    n_frames = len(frame_chunks)
    if n_frames == 0:
        raise ValueError("Selected frame range is empty.")
    print(f"  processing frames [{start_fi}, {end_fi}) -> {n_frames} frames")

    print("\nPreparing first frame and calibration...")
    first_fft1d = raw_frame_to_fft1d(
        frame_chunks[0],
        cfg,
        iq_order=args.iq_order,
        dc_remove=not args.no_adc_dc_remove,
    )
    calib_phase = estimate_calibration(
        first_fft1d,
        cfg,
        calib_file=args.calib,
        auto_calib=not args.no_auto_calib,
        skip_bins=args.skip_bins,
    )

    sinc_buf = build_sinc_buf(cfg.num_angle_bins, len(cfg.az_ant_idx))
    mount_rotation = MOUNT_PRESETS[args.mount]

    print("\n" + "=" * 72)
    print("TI point cloud generation (Calterah-style standalone DSP)")
    print(
        f"CFAR: SO-1D G={args.cfar_guard}, S={args.cfar_search}, mul={args.cfar_mul}, "
        f"local_max={not args.no_local_max}, zero_bins={args.zero_doppler_bins}, "
        f"max_det={args.max_detections}"
    )
    print(
        f"DPK : times={args.dpk_times}, threshold={args.dpk_threshold}, "
        f"max_angle={args.max_angle} deg"
    )
    print(
        f"Elev: enabled={not args.no_elevation}, fov=[{args.el_down},{args.el_up}], "
        f"mount={args.mount}"
    )
    print("=" * 72)

    all_pt_list: list[np.ndarray] = []
    all_fr_list: list[tuple[int, float, int, int]] = []
    all_stats: list[dict[str, int]] = []
    offset = 0
    mean_prev: np.ndarray | None = None

    t0 = time.time()
    for fi, frame_raw in enumerate(frame_chunks):
        fft1d = first_fft1d if fi == 0 else raw_frame_to_fft1d(
            frame_raw,
            cfg,
            iq_order=args.iq_order,
            dc_remove=not args.no_adc_dc_remove,
        )

        mean_cur = fft1d.mean(axis=0)
        if args.mean_mode == "none":
            mean_sub = None
        elif args.mean_mode == "current":
            mean_sub = mean_cur
        else:
            mean_sub = mean_prev if mean_prev is not None else mean_cur
        mean_prev = mean_cur

        pts, stats, rd_abs = process_frame_hw(
            fft1d,
            cfg,
            mean_sub=mean_sub,
            cfar_guard=args.cfar_guard,
            cfar_search=args.cfar_search,
            cfar_mul=args.cfar_mul,
            dpk_times=args.dpk_times,
            dpk_threshold=args.dpk_threshold,
            skip_bins=args.skip_bins,
            max_angle=args.max_angle,
            cfar_local_max=not args.no_local_max,
            zero_doppler_bins=args.zero_doppler_bins,
            max_detections=args.max_detections,
            min_range_m=args.min_range,
            max_range_m=args.max_range,
            sinc_buf=sinc_buf,
            calib_phase=calib_phase,
            enable_elevation=not args.no_elevation,
            doppler_comp_sign=args.doppler_comp_sign,
            mount_rotation=mount_rotation,
            z_clip=args.z_clip,
        )

        frame_id = start_fi + fi
        all_pt_list.append(pts)
        all_fr_list.append((frame_id, frame_id * cfg.frame_periodicity_ms / 1000.0, len(pts), offset))
        offset += len(pts)
        all_stats.append(stats)

        if should_save_rdmap(args, fi, n_frames):
            save_rdmap(rd_abs, frame_id, rdmap_dir, cfg)

        if fi % 5 == 0 or fi == n_frames - 1:
            print(
                f"  frame {frame_id:4d}: CFAR={stats['cfar_detections']:4d} "
                f"DPK={stats['dpk_candidates']:4d}/{stats['dpk_passed']:4d} "
                f"points={stats['final_points']:4d}"
            )

    elapsed = time.time() - t0
    total_pts = sum(len(p) for p in all_pt_list)
    all_points = (
        np.concatenate(all_pt_list)
        if total_pts > 0 else np.zeros(0, dtype=PT_DTYPE)
    )
    all_frames = np.asarray(all_fr_list, dtype=FR_DTYPE)
    print(f"\nDone: {n_frames} frames, {total_pts} points, {elapsed:.1f}s")

    meta = {
        "source": str(adc_bin_path),
        "input_dir": str(input_dir),
        "config": str(cfg_path),
        "start_frame": start_fi,
        "end_frame": end_fi,
        "radar": {
            "num_tx": cfg.num_tx,
            "num_rx": cfg.num_rx,
            "tx_order": list(cfg.tx_order),
            "num_virtual_ant": cfg.num_virtual_ant,
            "range_resolution": cfg.range_resolution,
            "doppler_resolution": cfg.doppler_resolution,
            "num_range_bins": cfg.use_range,
            "num_doppler_bins": cfg.fft2d_points,
        },
        "processing": {
            "cfar_guard": args.cfar_guard,
            "cfar_search": args.cfar_search,
            "cfar_mul": args.cfar_mul,
            "dpk_times": args.dpk_times,
            "dpk_threshold": args.dpk_threshold,
            "max_angle": args.max_angle,
            "mean_mode": args.mean_mode,
            "auto_calib": not args.no_auto_calib,
            "calib": args.calib,
            "mount": args.mount,
        },
    }
    save_npz(all_frames, all_points, pc_dir / "ti_iwr6843_pointcloud.npz", meta)

    if total_pts > 0:
        non_empty = np.flatnonzero(all_frames["num_points"] > 0)
        if non_empty.size > 0:
            first_fr = all_frames[int(non_empty[0])]
            first_pts = get_frame_points(all_points, first_fr)
            visualize_pointcloud(
                first_pts,
                f"Frame {int(first_fr['frame_id'])}",
                pc_dir / f"frame_{int(first_fr['frame_id']):04d}.png",
            )

            last_fr = all_frames[int(non_empty[-1])]
            if int(last_fr["frame_id"]) != int(first_fr["frame_id"]):
                last_pts = get_frame_points(all_points, last_fr)
                visualize_pointcloud(
                    last_pts,
                    f"Frame {int(last_fr['frame_id'])}",
                    pc_dir / f"frame_{int(last_fr['frame_id']):04d}.png",
                )

            visualize_pointcloud(all_points, "All Frames", pc_dir / "all_frames.png")

    write_statistics(
        stats_dir / "statistics.txt",
        input_dir=input_dir,
        cfg_path=cfg_path,
        adc_bin_path=adc_bin_path,
        output_dir=output_dir,
        args=args,
        cfg=cfg,
        frames=all_frames,
        points=all_points,
        frame_stats=all_stats,
        elapsed=elapsed,
    )
    print(f"\nOutput: {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Standalone TI IWR6843 raw ADC to point cloud",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--input-dir", default=None, help="Directory containing radar.cfg and adc_data_Raw_0.bin")
    p.add_argument("--cfg", default=None, help="Path to radar.cfg")
    p.add_argument("--bin", default=None, help="Path to adc_data_Raw_0.bin")
    p.add_argument("--output", "-o", default=None, help="Output directory")
    p.add_argument("--calib", default=None, help="Optional complex calibration .npy vector")
    p.add_argument("--no-auto-calib", action="store_true", help="Disable first-frame phase calibration")

    p.add_argument("--iq-order", choices=["qfirst", "ifirst"], default="qfirst", help="DCA1000 IQ ordering")
    p.add_argument("--no-adc-dc-remove", action="store_true", help="Disable per-chirp ADC sample DC removal")
    p.add_argument("--range-nfft", type=int, default=None, help="Range FFT length; default = ADC samples")
    p.add_argument("--vel-nfft", type=int, default=None, help="Doppler FFT length; default = chirps per TX")
    p.add_argument("--use-range-bins", type=int, default=None, help="Number of range bins to keep")
    p.add_argument("--range-window", choices=["cheb", "hann", "hanning", "hamming", "blackman", "none"], default="cheb")
    p.add_argument("--doppler-window", choices=["cheb", "hann", "hanning", "hamming", "blackman", "none"], default="cheb")

    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)

    p.add_argument("--mean-mode", choices=["prev", "current", "none"], default="prev",
                   help="Slow-time clutter mean subtraction mode")
    p.add_argument("--cfar-guard", type=int, default=2)
    p.add_argument("--cfar-search", type=int, default=6)
    p.add_argument("--cfar-mul", type=float, default=4.0)
    p.add_argument("--no-local-max", action="store_true", help="Disable 3x3 local-maximum pruning")
    p.add_argument("--zero-doppler-bins", type=int, default=1,
                   help="Suppress center Doppler bins; 1 removes only zero Doppler")
    p.add_argument("--max-detections", type=int, default=160,
                   help="Maximum CFAR detections per frame; 0 means unlimited")
    p.add_argument("--skip-bins", type=int, default=5)
    p.add_argument("--min-range", type=float, default=None)
    p.add_argument("--max-range", type=float, default=6.0)

    p.add_argument("--dpk-times", type=int, default=2)
    p.add_argument("--dpk-threshold", type=float, default=3.0)
    p.add_argument("--angle-bins", type=int, default=360)
    p.add_argument("--elev-bins", type=int, default=181)
    p.add_argument("--max-angle", type=float, default=60.0)
    p.add_argument("--el-down", type=float, default=-60.0)
    p.add_argument("--el-up", type=float, default=60.0)
    p.add_argument("--no-elevation", action="store_true")
    p.add_argument("--z-clip", type=float, default=None, help="Optional absolute Z clipping in meters")
    p.add_argument("--doppler-comp-sign", type=float, choices=[-1.0, 1.0], default=1.0,
                   help="TI normally uses +1; use -1 only for debugging")
    p.add_argument("--mount", choices=list(MOUNT_PRESETS.keys()), default="none")

    p.add_argument("--save-rdmap", action="store_true", default=True)
    p.add_argument("--no-rdmap", dest="save_rdmap", action="store_false")
    p.add_argument("--rdmap-every", type=int, default=1,
                   help="Save RDMap every N frames plus first/last; 1 saves all")
    return p


def main() -> None:
    args = build_parser().parse_args()
    try:
        run(args)
    except Exception as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
