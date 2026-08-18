"""
Calterah 4T4R 60GHz 雷达原始 bin → 点云 (Sky 金标准算法移植)

输入: 单个 .bin/.dat 文件，包含若干个连续帧 (每帧 2 MB)
输出: 完全与 sky32b750/generate_pointcloud.py 一致的目录结构
  <out>/rdmaps/rdmap_XXXX.png
  <out>/pointclouds/calterah_pointcloud.npz
  <out>/pointclouds/frame_XXXX.png
  <out>/pointclouds/all_frames.png
  <out>/stats/statistics.txt

算法链 (与 sky32b750/generate_pointcloud.py 完全一致):
  Step 1: Range FFT (512pt, Hanning)
  Step 2: P2 Mean (chirp 维均值，meanEn=2 用前一帧)
  Step 3: P2 Minus + Doppler FFT (TDM-MIMO, vel_nfft=128)
  Step 4: ABS SUM (sum(|X|) / 128)
  Step 5: 1D SO-CFAR (仅 Doppler 维)
  Step 6: 方位角/俯仰角 2D beamforming (使用实物天线排布)
  Step 7: 局部峰值筛选 + 多峰检测
  Step 8: 坐标变换  x = R cos(el) sin(az), y = R cos(el) cos(az), z = R sin(el)
  Step 9: 坐标变换  x = R cos(el) sin(az), y = R cos(el) cos(az), z = R sin(el)

硬件参数 (来自 sensor.json 实际烧录值):
  fc = 60 GHz, BW = 4 GHz, chirp_rampup = 48.0 us, chirp_period = 70.0 us
  nchirp = 512, adc_freq = 20 MHz, dec_factor = 2, samples/chirp = 512
  adc_sample_start = 2 us, adc_sample_end = 46 us, rng_nfft = 512
  4TX × 4RX = 16 VA，RX3/RX2/RX1/RX0 横向排列，TX1/TX2/TX3 纵向排列
  TDM 顺序: TX0 → TX1 → TX2 → TX3 (循环 128 次 → 512 chirps)
  固件派生值: rng_delta = 0.035 m, vel_delta = 0.070 m/s

数据格式 (LVDS, 与 Sky 完全不同):
  每帧 2 MB = 512 chirps × 512 samples × 4 RX × int16(2B)
  布局 (chirp, rx, sample) 按顺序紧排，单字节 int16 (默认 little-endian)
  无 OSPI 字节重排，无 PREP 硬件去交织 (按 chirp 索引 % 4 = TX 索引自己拆分)

用法:
  python generate_pointcloud.py --bin D:/data/_sampling0.dat
  python generate_pointcloud.py --bin ... --output out_dir --cfar-mul 4.0 --skip-bins 5
  python generate_pointcloud.py --bin ... --dtype ">i2"   # 若数据是 big-endian
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=RuntimeWarning)

C_LIGHT = 299_792_458.0

# ---------------------------------------------------------------------------
# 输出 dtype (与 sky32b750 / mmwave_dsp.point_cloud 完全一致)
# ---------------------------------------------------------------------------
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


def make_window(kind: str, n: int, params: tuple[float, ...] = (80.0,)) -> np.ndarray:
    """Create a DSP window.  Falls back gracefully if SciPy is unavailable."""
    kind = (kind or "hann").lower()
    if kind in {"cheb", "chebwin", "chebyshev"}:
        try:
            from scipy.signal.windows import chebwin
            return chebwin(n, at=float(params[0]), sym=False).astype(np.float64)
        except Exception:
            return np.hanning(n).astype(np.float64)
    if kind in {"hann", "hanning"}:
        return np.hanning(n).astype(np.float64)
    return np.ones(n, dtype=np.float64)


def make_angle_grid(left_deg: float, right_deg: float, n: int) -> np.ndarray:
    """Beamforming grid with uniform spacing in sin(theta), like firmware steering."""
    if n <= 1:
        return np.array([(left_deg + right_deg) * 0.5], dtype=np.float64)
    s0 = np.sin(np.radians(left_deg))
    s1 = np.sin(np.radians(right_deg))
    sin_grid = np.linspace(s0, s1, n)
    return np.degrees(np.arcsin(np.clip(sin_grid, -1.0, 1.0)))


def build_default_ant_pos() -> np.ndarray:
    """Virtual antenna coordinates in wavelength units from the board layout.

    RX order in raw channels is RX0, RX1, RX2, RX3; physically the top row is
    RX3, RX2, RX1, RX0 from left to right.  TX0 is lower-left, while TX1/2/3
    share the right column from top to bottom.  The 0.5 lambda spacing gives a
    compact 2D virtual array and keeps TX0+TX3 as an 8-element azimuth row.
    """
    rx_pos = np.array([
        [1.5, 0.0],  # RX0 rightmost
        [1.0, 0.0],  # RX1
        [0.5, 0.0],  # RX2
        [0.0, 0.0],  # RX3 leftmost
    ], dtype=np.float64)
    tx_pos = np.array([
        [0.0, 0.0],  # TX0 lower-left
        [2.0, 1.0],  # TX1 upper-right
        [2.0, 0.5],  # TX2 middle-right
        [2.0, 0.0],  # TX3 lower-right
    ], dtype=np.float64)
    return np.vstack([tx + rx_pos for tx in tx_pos])


def shifted_doppler_indices(nfft: int) -> tuple[np.ndarray, np.ndarray]:
    """Return FFT indices in negative-to-positive order and signed bin numbers."""
    half = nfft // 2
    idx = np.concatenate([np.arange(nfft - half, nfft), np.arange(0, half)])
    signed = np.concatenate([np.arange(-half, 0), np.arange(0, half)])
    return idx.astype(int), signed.astype(int)


def velocity_axis(cfg: "CalterahRadarConfig") -> np.ndarray:
    _, signed = shifted_doppler_indices(cfg.fft2d_points)
    return signed * cfg.doppler_resolution


def doppler_compensation(vel_signed: int, tx: int, cfg: "CalterahRadarConfig") -> complex:
    phase = 2 * np.pi * vel_signed * tx / (cfg.fft2d_points * cfg.num_tx)
    return np.exp(-1j * phase)


def steering_vector(
    positions: np.ndarray,
    az_deg: float,
    el_deg: float = 0.0,
) -> np.ndarray:
    az = np.radians(az_deg)
    el = np.radians(el_deg)
    vx = np.sin(az) * np.cos(el)
    vy = np.sin(el)
    phase = positions[:, 0] * vx + positions[:, 1] * vy
    return np.exp(-1j * 2 * np.pi * phase)


def steering_matrix(
    positions: np.ndarray,
    az_deg,
    el_deg=0.0,
) -> np.ndarray:
    """Steering vectors as columns for one or many az/el grid points."""
    az = np.radians(np.asarray(az_deg, dtype=np.float64))
    el = np.radians(np.asarray(el_deg, dtype=np.float64))
    az_b, el_b = np.broadcast_arrays(az, el)
    vx = np.sin(az_b).ravel() * np.cos(el_b).ravel()
    vy = np.sin(el_b).ravel()
    phase = positions[:, 0, np.newaxis] * vx[np.newaxis, :]
    phase += positions[:, 1, np.newaxis] * vy[np.newaxis, :]
    return np.exp(-1j * 2 * np.pi * phase)


def beamform_power(
    point_data: np.ndarray,
    positions: np.ndarray,
    az_deg: float,
    el_deg: float = 0.0,
) -> float:
    sv = steering_vector(positions, az_deg, el_deg)
    return float(np.abs(np.sum(point_data * sv)))


# ===========================================================================
# Calterah 雷达配置
# ===========================================================================
@dataclass
class CalterahRadarConfig:
    # RF (来自 sensor.json 实际烧录值)
    freq_start: float = 60.0e9       # Hz, chirp 起扫频率 (fc - BW/2 = 62 - 2)
    bandwidth: float = 4.0e9         # Hz (fmcw_bandwidth = 4000 MHz)
    chirp_rampup: float = 48.0e-6    # s, Tu (sensor.json: 48.0 us)
    chirp_period: float = 70.0e-6    # s, Tr (sensor.json: 70.0 us)
    sample_start: float = 2.0e-6     # s, ADC 采样起点 (sensor.json: 2.0 us)
    sample_end: float = 46.0e-6      # s, ADC 采样终点 (sensor.json: 46.0 us)
    adc_rate: float = 20e6           # Hz (sensor.json: 20 MHz)
    dec_factor: int = 2              # DFE 抽取因子 (sensor.json: 2)

    # 天线 (4T4R, 单芯片 TDM)
    num_tx: int = 4
    num_rx: int = 4

    # 帧
    num_chirps_total: int = 512      # 整帧 chirp 数 (4 TX × 128)
    samples_per_chirp: int = 512     # 每 chirp 每 RX 的 ADC 样点数 (= rng_nfft)

    # FFT
    rng_nfft: int = 512              # range FFT 点数 (sensor.json: 512)
    vel_nfft: int = 128              # Doppler FFT 点数 (sensor.json: 128)
    num_angle_bins: int = 360        # 方位搜索点数 (sensor.json doa_npoint[0])
    num_elev_bins: int = 360         # 俯仰搜索点数 (sensor.json doa_npoint[1])

    # Beamforming 搜索范围 (sensor.json)
    az_left: float = -60.0
    az_right: float = 60.0
    el_down: float = -90.0
    el_up: float = 90.0

    # I/O
    sample_dtype: str = "<i2"        # raw int16, little-endian (ARC EM6 原生)

    # 派生量 (init=False)
    num_ant: int = field(init=False)
    num_chirps: int = field(init=False)        # chirps per TX
    fft2d_points: int = field(init=False)
    use_range: int = field(init=False)
    fc: float = field(init=False)
    lambda_: float = field(init=False)
    slope: float = field(init=False)
    range_resolution: float = field(init=False)
    max_range: float = field(init=False)
    doppler_resolution: float = field(init=False)
    max_velocity: float = field(init=False)
    frame_bytes: int = field(init=False)
    virtual_ant_pos: np.ndarray = field(init=False)
    az_ant_idx: np.ndarray = field(init=False)
    az_grid: np.ndarray = field(init=False)
    el_grid: np.ndarray = field(init=False)
    az_steering: np.ndarray = field(init=False)

    def __post_init__(self):
        self.num_ant = self.num_tx * self.num_rx          # 16 VA
        self.num_chirps = self.num_chirps_total // self.num_tx   # 128 chirps/TX
        self.fft2d_points = self.vel_nfft                        # 128
        self.use_range = self.rng_nfft // 2                       # 256

        # Calterah firmware reports carrier_freq=fmcw_startfreq.  Using 60 GHz
        # keeps vel_delta aligned with radar_params (about 0.070 m/s).
        self.fc = self.freq_start
        self.lambda_ = C_LIGHT / self.fc
        self.slope = self.bandwidth / self.chirp_rampup           # Hz/s

        # Range bin spacing = c * Fs_eff / (2 * slope * rng_nfft)
        # 与固件 radar_params 输出的 rng_delta 一致 (0.035 m)
        fs_eff = self.adc_rate / self.dec_factor
        self.range_resolution = C_LIGHT * fs_eff / (2.0 * self.slope * self.rng_nfft)
        self.max_range = self.range_resolution * self.use_range

        # Doppler: same-TX slow-time interval is num_tx * chirp_period.
        # 与固件 radar_params 输出的 vel_delta 一致 (0.070 m/s)
        tc = self.chirp_period * self.num_tx
        self.doppler_resolution = self.lambda_ / (
            2.0 * self.vel_nfft * tc
        )
        self.max_velocity = self.doppler_resolution * (self.vel_nfft / 2.0)

        self.frame_bytes = (self.num_chirps_total
                            * self.samples_per_chirp
                            * self.num_rx
                            * np.dtype(self.sample_dtype).itemsize)
        self.virtual_ant_pos = build_default_ant_pos()
        self.az_ant_idx = np.where(
            np.isclose(self.virtual_ant_pos[:, 1], 0.0)
        )[0].astype(int)
        self.az_grid = make_angle_grid(self.az_left, self.az_right, self.num_angle_bins)
        self.el_grid = make_angle_grid(self.el_down, self.el_up, self.num_elev_bins)
        self.az_steering = steering_matrix(
            self.virtual_ant_pos[self.az_ant_idx], self.az_grid, 0.0
        )

    def summary(self) -> str:
        fs_eff = self.adc_rate / self.dec_factor
        return (
            f"Calterah 4T4R Config:\n"
            f"  RF             : fc={self.fc/1e9:.2f} GHz, BW={self.bandwidth/1e9:.2f} GHz, "
            f"slope={self.slope/1e12:.2f} MHz/us\n"
            f"  Chirp          : rampup={self.chirp_rampup*1e6:.1f} us, "
            f"period={self.chirp_period*1e6:.1f} us, "
            f"ADC window=[{self.sample_start*1e6:.0f}, {self.sample_end*1e6:.0f}] us\n"
            f"  ADC            : Fs={self.adc_rate/1e6:.0f} MHz, dec={self.dec_factor}, "
            f"Fs_eff={fs_eff/1e6:.0f} MHz\n"
            f"  Antennas       : {self.num_tx}TX × {self.num_rx}RX → {self.num_ant} VA "
            f"(2D array from board layout, unit=λ)\n"
            f"  TDM order      : TX0→TX1→TX2→TX3 ({self.num_chirps} chirps/TX)\n"
            f"  Frame layout   : {self.num_chirps_total} chirps × {self.samples_per_chirp} samp × "
            f"{self.num_rx} RX × 2B = {self.frame_bytes:,} B ({self.frame_bytes/1024/1024:.1f} MB)\n"
            f"  Range          : res={self.range_resolution*100:.2f} cm/bin, "
            f"max={self.max_range:.2f} m, use_range={self.use_range}, "
            f"rng_nfft={self.rng_nfft}\n"
            f"  Doppler        : res={self.doppler_resolution:.4f} m/s/bin, "
            f"max=±{self.max_velocity:.2f} m/s, FFT={self.fft2d_points}-pt\n"
            f"  Angle          : az=[{self.az_left:.0f},{self.az_right:.0f}]/{self.num_angle_bins}, "
            f"el=[{self.el_down:.0f},{self.el_up:.0f}]/{self.num_elev_bins}\n"
            f"  Raw dtype      : {self.sample_dtype}\n"
        )


# ===========================================================================
# 数据加载 (raw bin → 单帧 fft1d)
# ===========================================================================
def read_raw_bin(path, cfg: CalterahRadarConfig) -> np.ndarray:
    """读取整 bin → int16 1D 数组。"""
    return np.fromfile(path, dtype=np.dtype(cfg.sample_dtype))


def validate_raw(raw: np.ndarray, label: str = "") -> bool:
    """LVDS idle / 全零检测，与 cal_pointcloud_sky 一致。"""
    s = raw[: min(raw.size, 100_000)]
    idle_ratio = float(np.sum(s == 3855)) / max(len(s), 1)
    if idle_ratio > 0.9:
        print(f"[ERROR]{label} 检出 >90% LVDS idle (0x0F0F)，数据无效")
        return False
    if float(s.astype(np.float64).std()) < 1.0:
        print(f"[ERROR]{label} 数据方差过低，可能损坏")
        return False
    return True


def split_frames(raw: np.ndarray, cfg: CalterahRadarConfig):
    """按完整雷达帧切分。旧单文件入口仍可使用。"""
    samples_per_frame = cfg.num_chirps_total * cfg.samples_per_chirp * cfg.num_rx
    n_full = raw.size // samples_per_frame
    out = []
    for i in range(n_full):
        out.append(raw[i * samples_per_frame : (i + 1) * samples_per_frame])
    return out, n_full


@dataclass
class RawFileSpec:
    path: Path
    start_frame_index: int
    frame_count: int


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_capture_sources(capture_dir: Path, cfg: CalterahRadarConfig):
    capture_dir = capture_dir.resolve()
    cal_dir = capture_dir / "Calterah"
    meta_path = cal_dir / "meta.json"
    if not meta_path.exists():
        print(f"[ERROR] 找不到 Calterah meta: {meta_path}")
        sys.exit(1)

    meta = _load_json(meta_path)
    specs: list[RawFileSpec] = []

    if meta.get("output_files"):
        for item in meta["output_files"]:
            raw_path = cal_dir / item["file"]
            if not raw_path.is_file():
                print(f"[ERROR] 找不到分片文件: {raw_path}")
                sys.exit(1)
            size_frames = raw_path.stat().st_size // cfg.frame_bytes
            frame_count = int(item.get("frames") or size_frames)
            frame_count = min(frame_count, int(size_frames))
            specs.append(RawFileSpec(
                path=raw_path,
                start_frame_index=int(item.get("start_frame_index", 0)),
                frame_count=frame_count,
            ))
    else:
        raw_name = meta.get("output_file") or "adc_data.bin"
        if "*" in raw_name:
            raw_files = sorted(cal_dir.glob(raw_name))
        else:
            raw_files = [cal_dir / raw_name]
        if not raw_files or not raw_files[0].exists():
            raw_files = sorted(cal_dir.glob("adc_data_*.bin"))
        if not raw_files:
            raw_files = [cal_dir / "adc_data.bin"]

        next_start = 0
        for raw_path in raw_files:
            if not raw_path.is_file():
                print(f"[ERROR] 找不到 raw 文件: {raw_path}")
                sys.exit(1)
            frame_count = raw_path.stat().st_size // cfg.frame_bytes
            specs.append(RawFileSpec(raw_path, next_start, int(frame_count)))
            next_start += int(frame_count)

    if not specs:
        print(f"[ERROR] {cal_dir} 下没有可处理的 Calterah raw 文件")
        sys.exit(1)

    return specs, meta, cal_dir


def _resolve_sources(args, cfg: CalterahRadarConfig):
    if args.capture_dir:
        specs, meta, cal_dir = _resolve_capture_sources(Path(args.capture_dir), cfg)
        default_output = cal_dir / "pointcloud_artifacts"
        default_npz = cal_dir / "pointcloud.npz"
        return specs, meta, default_output, default_npz

    if not args.bin:
        print("[ERROR] 需要提供 --capture-dir 或 --bin")
        sys.exit(1)

    bin_path = Path(args.bin).resolve()
    if not bin_path.is_file():
        print(f"[ERROR] 找不到 bin 文件: {bin_path}")
        sys.exit(1)

    frame_count = bin_path.stat().st_size // cfg.frame_bytes
    specs = [RawFileSpec(bin_path, 0, int(frame_count))]
    output = bin_path.parent / (bin_path.stem + "_calterah_pointcloud")
    return specs, {}, output, output / "pointclouds" / "calterah_pointcloud.npz"


def _selected_frame_count(
    specs: list[RawFileSpec],
    start_frame: int,
    end_frame: int | None,
) -> int:
    count = 0
    for spec in specs:
        lo = max(start_frame, spec.start_frame_index)
        hi = spec.start_frame_index + spec.frame_count
        if end_frame is not None:
            hi = min(hi, end_frame)
        count += max(0, hi - lo)
    return count


def iter_raw_frames(
    specs: list[RawFileSpec],
    cfg: CalterahRadarConfig,
    start_frame: int,
    end_frame: int | None,
    timestamps: list[float] | None = None,
):
    dtype = np.dtype(cfg.sample_dtype)
    samples_per_frame = cfg.frame_bytes // dtype.itemsize

    for spec in specs:
        if spec.frame_count <= 0:
            continue

        raw = np.memmap(str(spec.path), dtype=dtype, mode="r")
        actual_frames = min(spec.frame_count, raw.size // samples_per_frame)
        if actual_frames <= 0:
            print(f"[WARN] {spec.path.name} 没有完整帧，跳过")
            continue

        if not validate_raw(raw[: min(raw.size, 100_000)], spec.path.name):
            sys.exit(2)

        local_start = max(0, start_frame - spec.start_frame_index)
        local_end = actual_frames
        if end_frame is not None:
            local_end = min(local_end, max(0, end_frame - spec.start_frame_index))

        for local_i in range(local_start, local_end):
            global_fid = spec.start_frame_index + local_i
            begin = local_i * samples_per_frame
            end = begin + samples_per_frame
            ts = (
                float(timestamps[global_fid])
                if timestamps is not None and 0 <= global_fid < len(timestamps)
                else float(global_fid) * 0.1
            )
            yield global_fid, ts, np.asarray(raw[begin:end])


def raw_frame_to_fft1d(
    frame_raw: np.ndarray,
    cfg: CalterahRadarConfig,
    dc_remove: bool = True,
) -> np.ndarray:
    """单帧 raw int16 → Sky 风格 fft1d (chirps_per_tx, num_va, use_range) complex。

    步骤:
      1. reshape raw layout (nchirp=512, rx=4, samp=512) -> (chirp, sample, rx)
      2. DC offset removal (每 chirp 每 RX 减均值)
      3. TDM 去交织: chirp_idx % 4 = tx 索引, // 4 = chirp_in_tx
      4. Chebyshev 80dB 窗 + Range FFT (512 → use_range=256 正频率)
      5. 重排为 (chirp_in_tx, va=tx*4+rx, range_bin)
    """
    nchirp = cfg.num_chirps_total
    samp = cfg.samples_per_chirp
    nrx = cfg.num_rx
    ntx = cfg.num_tx
    cpt = cfg.num_chirps             # chirps per TX = 128
    nfft = cfg.rng_nfft
    nr = cfg.use_range

    # adc = frame_raw.reshape(nchirp, samp, nrx).astype(np.float64)
    adc = frame_raw.reshape(nchirp, nrx, samp).transpose(0, 2, 1).astype(np.float64)

    if dc_remove:
        adc -= adc.mean(axis=1, keepdims=True)  # 每 chirp 每 RX 自身 DC

    # TDM 去交织: chirp 0,1,2,3 = TX0,TX1,TX2,TX3, 然后循环 128 次
    # reshape 为 (chirps_per_tx, num_tx, samp, n_rx)
    adc_tdm = adc.reshape(cpt, ntx, samp, nrx)

    # Sensor config uses Chebyshev 80 dB range window.
    win = make_window("cheb", samp, (80.0,))
    adc_win = adc_tdm * win[np.newaxis, np.newaxis, :, np.newaxis]
    rfft_full = np.fft.fft(adc_win, n=nfft, axis=2)
    rfft_pos = rfft_full[:, :, :nr, :]            # (cpt, ntx, nr, nrx)

    # 重组为 Sky 布局 (cpt, num_va, nr); va = tx*nrx + rx
    fft1d = np.zeros((cpt, ntx * nrx, nr), dtype=np.complex128)
    for tx in range(ntx):
        for rx in range(nrx):
            va = tx * nrx + rx
            fft1d[:, va, :] = rfft_pos[:, tx, :, rx]
    return fft1d


# ===========================================================================
# Sky 算法 - Doppler FFT (硬件对齐)
# ===========================================================================
def doppler_fft_hw(
    fft1d: np.ndarray,
    cfg: CalterahRadarConfig,
    mean_sub: np.ndarray | None = None,
) -> np.ndarray:
    """Doppler FFT.  Returns (num_va, use_range, vel_nfft)."""
    nc = cfg.num_chirps        # 128
    na = cfg.num_ant           # 16
    nr = cfg.use_range         # 128
    ntx = cfg.num_tx
    nrx = cfg.num_rx
    nfft = cfg.fft2d_points    # 128

    # Sensor config uses Chebyshev 80 dB velocity window.
    win = make_window("cheb", nc, (80.0,))
    rd_maps = np.zeros((na, nr, nfft), dtype=np.complex128)

    for tx in range(ntx):
        for rx in range(nrx):
            ant = tx * nrx + rx
            data = fft1d[:, ant, :].copy()
            if mean_sub is not None:
                data -= mean_sub[ant]
            data *= win[:, np.newaxis]
            rd_maps[ant] = np.fft.fft(data, n=nfft, axis=0).T / nc
    return rd_maps


def abs_sum_hw(rd_maps: np.ndarray) -> np.ndarray:
    """sum(|X|) / 128 - 与 Sky 一致。"""
    return np.sum(np.abs(rd_maps), axis=0) / 128.0


# ===========================================================================
# Sky 算法 - CFAR
# ===========================================================================
def cfar_1d_so_mult(signal, guard, search, mul_fac):
    """1D SO-CFAR (wrap-around)，与 Sky 完全一致。"""
    n = len(signal)
    total = guard + search
    detections = []
    for i in range(n):
        left_sum = 0.0
        for j in range(search):
            left_sum += signal[(i - total + j) % n]
        right_sum = 0.0
        for j in range(search):
            right_sum += signal[(i + guard + 1 + j) % n]
        left_mean = left_sum / search
        right_mean = right_sum / search
        noise = min(left_mean, right_mean)
        if signal[i] > noise * mul_fac:
            detections.append(i)
    return detections


def cfar_detect_hw(
    rd_abs_sum: np.ndarray,
    cfg: CalterahRadarConfig,
    skip_bins: int = 5,
    cfar_guard: int = 2,
    cfar_search: int = 6,
    cfar_mul: float = 4.0,
    local_max: bool = True,
    zero_doppler_bins: int = 0,
    max_detections: int = 160,
    min_range_m: float | None = None,
    max_range_m: float | None = None,
):
    """Doppler CFAR with 2D peak pruning for point-cloud generation."""
    nr, nfft = rd_abs_sum.shape
    vel_map, vel_signed = shifted_doppler_indices(nfft)
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
        row = valid_data[r]
        det_v = cfar_1d_so_mult(row, cfar_guard, cfar_search, cfar_mul)
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

    detections = []
    for r in range(start_r, end_r):
        for v in range(nv):
            if cfar_mask[r, v]:
                detections.append((r, int(vel_map[v]), int(v), valid_data[r, v]))

    if max_detections is not None and len(detections) > max_detections:
        detections.sort(key=lambda d: d[3], reverse=True)
        detections = detections[:max_detections]
    return detections


# ===========================================================================
# Sky 算法 - 方位角 DPK (16 元素 ULA)
# ===========================================================================
def build_sinc_buf(num_angle: int = 128, n_ant_h: int = 16) -> np.ndarray:
    """构建 DPK sinc 插值核。Calterah 用 16 元素 (Sky 用 8)。"""
    sinc = np.zeros(num_angle, dtype=np.complex128)
    for k in range(num_angle):
        for n in range(n_ant_h):
            sinc[k] += np.exp(-1j * 2 * np.pi * n * k / num_angle)
        sinc[k] /= n_ant_h
    return sinc


def estimate_azimuth_dpk(
    point_data: np.ndarray,
    cfg: CalterahRadarConfig,
    sinc_buf: np.ndarray,
    dpk_times: int = 2,
    dpk_threshold: float = 8.0,
):
    """Azimuth DPK on the y=0 horizontal virtual ULA."""
    del sinc_buf  # Kept in the signature for compatibility with older callers.
    spectrum = np.abs(point_data[cfg.az_ant_idx] @ cfg.az_steering)
    residual = float(np.mean(spectrum) + 1e-12)
    is_peak = np.ones_like(spectrum, dtype=bool)
    is_peak[1:] &= spectrum[1:] >= spectrum[:-1]
    is_peak[:-1] &= spectrum[:-1] >= spectrum[1:]
    peak_idx = np.where(is_peak)[0]
    if peak_idx.size == 0:
        peak_idx = np.array([int(np.argmax(spectrum))])

    order = peak_idx[np.argsort(spectrum[peak_idx])[::-1]]
    valid_peaks = []
    for az_idx in order[:dpk_times]:
        cut_pow = float(spectrum[az_idx])
        if cut_pow > residual * dpk_threshold:
            valid_peaks.append((int(az_idx), cut_pow))
    return valid_peaks


def idx_to_angle(idx: int, num_angle: int) -> float:
    """Azimuth grid index → angle (deg), defaulting to the -60~60deg FOV."""
    return float(make_angle_grid(-60.0, 60.0, num_angle)[idx])


def idx_to_azimuth(idx: int, cfg: CalterahRadarConfig) -> float:
    return float(cfg.az_grid[idx])


def idx_to_elevation(idx: int, cfg: CalterahRadarConfig) -> float:
    return float(cfg.el_grid[idx])


# ===========================================================================
# Sky 算法 - 俯仰 (steering-vector beamforming + 短 FFT)
# ===========================================================================
def build_steering_vec(num_angle: int = 128, num_rx: int = 4) -> np.ndarray:
    """方位 steering 矢量 (与 sky32b750 build_steering_vec 一致)。

    sv[a, rx] = exp(-1j * 2π * rx * a / num_angle)
    """
    sv = np.zeros((num_angle, num_rx), dtype=np.complex128)
    for a in range(num_angle):
        for rx in range(num_rx):
            sv[a, rx] = np.exp(-1j * 2 * np.pi * rx * a / num_angle)
    return sv


def default_elevation_tx_list(cfg: CalterahRadarConfig, n_elev: int = 3) -> list[int]:
    """TX1/TX2/TX3 form the vertical aperture on this board layout."""
    candidates = [tx for tx in (1, 2, 3) if tx < cfg.num_tx]
    if not candidates:
        candidates = list(range(cfg.num_tx))
    n = max(1, min(int(n_elev), len(candidates)))
    return candidates[:n]


# def estimate_elevation(
#     point_data: np.ndarray,
#     az_idx: int,
#     cfg: CalterahRadarConfig,
#     steering_vec: np.ndarray,
#     n_elev: int = 3,
# ) -> int:
#     """俯仰角估计 (与 sky32b750 estimate_elevation 一致)。

#     步骤:
#       1. 对前 n_elev 个 TX，把它们各自的 4 RX 用 steering_vec 沿方位
#          beamforming → 得到 n_elev 个 "elevation 行" 复数样本。
#       2. 把 n_elev 个样本零填充到 num_angle 点做 FFT，取最大模对应索引。

#     前提 (Sky 假设): TX0/TX1/TX2 中至少一个有 y 方向物理偏移，否则 FFT
#     输出无俯仰信息。CAL60S244 AiP 几何未公开，若实测俯仰栏不准，可通过
#     --n-elev 参数调整 (例如改成 2 取相邻 TX 的相位差)。
#     """
#     num_angle = cfg.num_angle_bins
#     nrx = cfg.num_rx

#     el_data = np.zeros(n_elev, dtype=np.complex128)
#     for tx in range(n_elev):
#         acc = 0j
#         for rx in range(nrx):
#             va = tx * nrx + rx
#             val = point_data[va]
#             acc += val * steering_vec[az_idx, rx]
#         el_data[tx] = acc

#     padded = np.zeros(num_angle, dtype=np.complex128)
#     padded[:n_elev] = el_data
#     el_fft = np.fft.fft(padded)
#     return int(np.argmax(np.abs(el_fft)))

def estimate_elevation(
    point_data: np.ndarray,
    az_idx: int,
    cfg: CalterahRadarConfig,
    steering_vec: np.ndarray,
    tx_list: list[int] | None = None,
) -> int:
    """Elevation from the vertical TX1/TX2/TX3 phase slope."""
    del steering_vec
    if tx_list is None:
        tx_list = default_elevation_tx_list(cfg, 3)
    tx_list = [int(tx) for tx in tx_list if 0 <= int(tx) < cfg.num_tx]
    if len(tx_list) < 2:
        return int(np.argmin(np.abs(cfg.el_grid)))

    az_deg = float(cfg.az_grid[az_idx])

    row_data = []
    row_y = []
    for tx in tx_list:
        va_idx = tx * cfg.num_rx + np.arange(cfg.num_rx)
        positions = cfg.virtual_ant_pos[va_idx]
        az_comp = steering_matrix(positions, az_deg, 0.0).ravel()
        row_data.append(np.sum(point_data[va_idx] * az_comp))
        row_y.append(float(np.mean(positions[:, 1])))

    row_data = np.asarray(row_data, dtype=np.complex128)
    row_y = np.asarray(row_y, dtype=np.float64)
    row_y = row_y - np.mean(row_y)

    sin_limit = float(np.max(np.abs(np.sin(np.radians(cfg.el_grid)))))
    sin_est = []
    weights = []
    for i in range(len(row_data)):
        for j in range(i + 1, len(row_data)):
            dy = row_y[i] - row_y[j]
            if abs(dy) < 1e-9:
                continue
            phase = np.angle(row_data[i] * np.conj(row_data[j]))
            est = phase / (2 * np.pi * dy)
            sin_est.append(np.clip(est, -sin_limit, sin_limit))
            weights.append(abs(row_data[i]) * abs(row_data[j]) * abs(dy))

    if weights and float(np.sum(weights)) > 1e-12:
        sin_el = float(np.average(np.asarray(sin_est), weights=np.asarray(weights)))
        el_deg = np.degrees(np.arcsin(np.clip(sin_el, -sin_limit, sin_limit)))
        return int(np.argmin(np.abs(cfg.el_grid - el_deg)))

    el_rad = np.radians(cfg.el_grid)
    el_steering = np.exp(-1j * 2 * np.pi * row_y[:, None] * np.sin(el_rad)[None, :])
    spectrum = np.abs(row_data @ el_steering)
    return int(np.argmax(spectrum))
# ===========================================================================
# 校准 (自动从首帧最强目标提取 16-VA 相位)
# ===========================================================================
def estimate_calibration(
    frames_data: list[np.ndarray],
    cfg: CalterahRadarConfig,
    calib_file: str | None = None,
) -> np.ndarray:
    """与 Sky estimate_calibration 完全一致 (改为 16 VA)。"""
    if calib_file is not None and os.path.isfile(calib_file):
        calib_phase = np.load(calib_file)
        print(f"  加载校准: {calib_file} ({calib_phase.shape})")
        return calib_phase

    fft1d = frames_data[0]
    rd_maps = doppler_fft_hw(fft1d, cfg, mean_sub=None)

    nfft = cfg.fft2d_points
    vel_map, _ = shifted_doppler_indices(nfft)
    v256_zero = vel_map[nfft // 2]   # 速度 0 对应的 fft bin

    rp_v0 = np.sum(np.abs(rd_maps[:, :, v256_zero]), axis=0)
    peak_r = int(np.argmax(rp_v0[5:])) + 5

    calib_ref = np.array([rd_maps[va, peak_r, v256_zero] for va in range(cfg.num_ant)])
    # calib_phase = calib_ref / (np.abs(calib_ref) + 1e-30)
#  相位校准应该取共轭
    calib_phase = np.conj(calib_ref / (np.abs(calib_ref) + 1e-30))

    print(
        f"  自校准: 参考目标 r={peak_r} "
        f"({peak_r * cfg.range_resolution:.2f}m), "
        f"平均幅度={np.mean(np.abs(calib_ref)):.1f}"
    )
    return calib_phase


# ===========================================================================
# Sky 算法 - 单帧处理 (Calterah 版: 16 VA ULA + n_elev-TX 俯仰)
# ===========================================================================
def process_frame_hw(
    fft1d: np.ndarray,
    cfg: CalterahRadarConfig,
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
    steering_vec: np.ndarray | None = None,
    calib_phase: np.ndarray | None = None,
    n_elev: int = 3,
    enable_elevation: bool = False,
):
    """单帧完整处理 (硬件对齐)，结构与 sky32b750 process_frame_hw 一致。

    与 Sky 的差异:
      - 方位角用 16 元素 ULA (Sky 用 8 元素 VA[8:16])
      - 俯仰用前 n_elev (默认 3) 个 TX × 4 RX 的 steering 矢量 beamforming
        (与 Sky 完全相同的算法)
      - enable_elevation=False 时退化为 z=0 (老版本行为)
    """
    num_angle = cfg.num_angle_bins
    if sinc_buf is None:
        sinc_buf = build_sinc_buf(num_angle, cfg.num_ant)
    if steering_vec is None:
        steering_vec = build_steering_vec(num_angle, cfg.num_rx)

    rd_maps = doppler_fft_hw(fft1d, cfg, mean_sub=mean_sub)
    rd_abs_sum = abs_sum_hw(rd_maps)
    detections = cfar_detect_hw(
        rd_abs_sum, cfg, skip_bins=skip_bins,
        cfar_guard=cfar_guard, cfar_search=cfar_search, cfar_mul=cfar_mul,
        local_max=cfar_local_max,
        zero_doppler_bins=zero_doppler_bins,
        max_detections=max_detections,
        min_range_m=min_range_m,
        max_range_m=max_range_m,
    )

    points_list = []
    dpk_total = 0
    dpk_pass = 0
    elev_tx_list = default_elevation_tx_list(cfg, n_elev)

    _, signed_bins = shifted_doppler_indices(cfg.fft2d_points)

    for r_idx, vbin, v_idx, power in detections:
        vel_signed = int(signed_bins[v_idx])
        point_data = np.zeros(cfg.num_ant, dtype=np.complex128)

        # Doppler 补偿 (TDM)
        for tx in range(cfg.num_tx):
            doppler_comp = doppler_compensation(vel_signed, tx, cfg)
            for rx in range(cfg.num_rx):
                va = tx * cfg.num_rx + rx
                val = rd_maps[va, r_idx, vbin]
                if calib_phase is not None:
                    val = val * calib_phase[va]
                val = val * doppler_comp
                point_data[va] = val

        # DPK 方位估计 (16 元素 ULA)
        az_peaks = estimate_azimuth_dpk(
            point_data, cfg, sinc_buf,
            dpk_times=dpk_times, dpk_threshold=dpk_threshold,
        )
        dpk_total += dpk_times
        dpk_pass += len(az_peaks)

        for az_idx, cut_pow in az_peaks:
            range_val = r_idx * cfg.range_resolution
            vel_val = vel_signed * cfg.doppler_resolution
            az_deg = idx_to_azimuth(az_idx, cfg)

            if enable_elevation:
                # el_idx = estimate_elevation(
                #     point_data, az_idx, cfg, steering_vec, n_elev=n_elev,
                # )
                el_idx = estimate_elevation(
                    point_data, az_idx, cfg, steering_vec, tx_list=elev_tx_list,
                )

                el_deg = idx_to_elevation(el_idx, cfg)
            else:
                el_deg = 0.0

            if abs(az_deg) > max_angle:
                continue

            az_rad = np.radians(az_deg)
            el_rad = np.radians(el_deg)
            x = range_val * np.cos(el_rad) * np.sin(az_rad)
            y = range_val * np.cos(el_rad) * np.cos(az_rad)
            z = range_val * np.sin(el_rad)

            points_list.append((
                x, y, z, range_val, vel_val,
                az_deg, el_deg, float(cut_pow),
            ))

    n = len(points_list)
    pts = np.zeros(n, dtype=PT_DTYPE)
    for i, p in enumerate(points_list):
        pts[i] = p

    stats = {
        "cfar_detections": len(detections),
        "dpk_candidates": dpk_total,
        "dpk_passed": dpk_pass,
        "final_points": n,
    }
    return pts, stats, rd_abs_sum


# ===========================================================================
# 输出: NPZ / RDMap / 点云图 (与 sky32b750 完全一致)
# ===========================================================================
def save_npz(all_frames, all_points, filepath):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    np.savez(filepath, points=all_points, frames=all_frames)
    print(f"点云已保存: {filepath} ({len(all_points)} 点, {len(all_frames)} 帧)")


def save_rdmap(rd_abs_sum, frame_id, output_dir, cfg: CalterahRadarConfig):
    nfft = cfg.fft2d_points
    vel_bins, _ = shifted_doppler_indices(nfft)
    valid = rd_abs_sum[:, vel_bins]

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(
        20 * np.log10(np.maximum(valid, 1e-10)),
        aspect="auto", origin="lower", cmap="jet",
    )
    ax.set_xlabel(f"Velocity bin (0-{nfft - 1}, shifted)")
    ax.set_ylabel(f"Range bin (0-{cfg.use_range - 1})")
    ax.set_title(f"Range-Doppler Map - Frame {frame_id}")
    plt.colorbar(im, ax=ax, label="dB")
    plt.savefig(
        os.path.join(output_dir, f"rdmap_{frame_id:04d}.png"),
        dpi=100, bbox_inches="tight",
    )
    plt.close(fig)


def visualize_pointcloud(points_arr, title, filepath):
    """与 sky32b750 visualize_pointcloud 一致 (3 panel)。"""
    if len(points_arr) == 0:
        return
    x = points_arr["x"]; y = points_arr["y"]; z = points_arr["z"]
    pw = points_arr["power"]; vel = points_arr["velocity"]; rng = points_arr["range"]
    pw_n = (pw - pw.min()) / (pw.max() - pw.min() + 1e-10)

    fig = plt.figure(figsize=(14, 5))
    ax1 = fig.add_subplot(131, projection="3d")
    sc1 = ax1.scatter(x, y, z, c=pw_n, cmap="jet", s=8, alpha=0.75)
    ax1.set_xlabel("X"); ax1.set_ylabel("Y"); ax1.set_zlabel("Z")
    ax1.set_title(f"{title}\n{len(points_arr)} pts")
    fig.colorbar(sc1, ax=ax1, shrink=0.6, label="Power")

    ax2 = fig.add_subplot(132)
    sc2 = ax2.scatter(x, y, c=vel, cmap="coolwarm", s=8, alpha=0.75)
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)"); ax2.set_title("Top View")
    ax2.grid(True, alpha=0.3); ax2.set_aspect("equal")
    fig.colorbar(sc2, ax=ax2, shrink=0.8, label="Velocity (m/s)")

    ax3 = fig.add_subplot(133)
    sc3 = ax3.scatter(rng, vel, c=pw_n, cmap="hot", s=8, alpha=0.75)
    ax3.set_xlabel("Range (m)"); ax3.set_ylabel("Velocity (m/s)")
    ax3.set_title("Range-Velocity"); ax3.grid(True, alpha=0.3)
    fig.colorbar(sc3, ax=ax3, shrink=0.8, label="Power")

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ===========================================================================
# 主流程
# ===========================================================================
def run(args):
    cfg = CalterahRadarConfig(
        sample_dtype=args.dtype,
        dec_factor=args.dec_factor,
        samples_per_chirp=args.samples_per_chirp,
        vel_nfft=args.vel_nfft,
        el_down=args.el_down,
        el_up=args.el_up,
    )
    print(cfg.summary())

    specs, meta, default_output, default_npz = _resolve_sources(args, cfg)
    output_dir = Path(args.output).resolve() if args.output else Path(default_output).resolve()
    npz_output = Path(args.npz_output).resolve() if args.npz_output else Path(default_npz).resolve()

    rdmap_dir = output_dir / "rdmaps"
    pc_dir = output_dir / "pointclouds"
    stats_dir = output_dir / "stats"
    stats_dir.mkdir(parents=True, exist_ok=True)
    if args.save_rdmap:
        rdmap_dir.mkdir(parents=True, exist_ok=True)
    if args.save_preview:
        pc_dir.mkdir(parents=True, exist_ok=True)

    print("\n输入 raw 文件:")
    total_full = 0
    for spec in specs:
        print(
            f"  {spec.path.name}: start={spec.start_frame_index}, "
            f"frames={spec.frame_count}, size={spec.path.stat().st_size / 1024 / 1024:.1f} MB"
        )
        total_full += spec.frame_count
    print(f"  总完整帧数: {total_full}")

    start_fi = max(0, args.start_frame)
    end_fi = args.end_frame
    n_frames = _selected_frame_count(specs, start_fi, end_fi)
    if n_frames == 0:
        print("[ERROR] 选定范围内无完整帧")
        sys.exit(3)
    print(f"  待处理: 帧 [{start_fi}, {end_fi if end_fi is not None else 'end'}) → {n_frames} 帧")

    timestamps = meta.get("timestamps") if isinstance(meta.get("timestamps"), list) else None
    frame_iter = iter_raw_frames(specs, cfg, start_fi, end_fi, timestamps=timestamps)
    try:
        first_fid, first_ts, first_raw = next(frame_iter)
    except StopIteration:
        print("[ERROR] 选定范围内无可迭代完整帧")
        sys.exit(3)

    print("\n首帧 Range FFT / 天线校准...")
    first_fft1d = raw_frame_to_fft1d(first_raw, cfg)
    print(f"  first fft1d shape: {first_fft1d.shape} (chirps/TX, VA, range)")

    calib_phase = estimate_calibration([first_fft1d], cfg, calib_file=args.calib)

    # ---- 预计算 DPK sinc 核 + 方位 steering 矢量 ----
    sinc_buf = build_sinc_buf(cfg.num_angle_bins, cfg.num_ant)
    steering_vec = build_steering_vec(cfg.num_angle_bins, cfg.num_rx)

    print("\n" + "=" * 70)
    print("Calterah 点云生成 (Sky 算法移植)")
    print(
        f"CFAR: 1D SO, G={args.cfar_guard}, S={args.cfar_search}, "
        f"mul={args.cfar_mul}, local_max={not args.no_local_max}, "
        f"zero_bins={args.zero_doppler_bins}, max_det={args.max_detections}"
    )
    print(f"DPK : times={args.dpk_times}, threshold={args.dpk_threshold}, "
          f"skip_bins={args.skip_bins}, max_az_angle={args.max_angle}°")
    enable_elevation = args.enable_elevation and not args.no_elevation
    print(
        f"Elev: enabled={enable_elevation}, "
        f"vertical_tx={default_elevation_tx_list(cfg, args.n_elev)}, "
        f"fov=[{args.el_down},{args.el_up}]"
    )
    print("=" * 70)

    # ---- 6. 逐帧处理 ----
    all_pt_list = []
    all_fr_list = []
    all_stats = []
    offset = 0
    mean_prev = None

    def process_one(global_fid: int, frame_ts: float, fft1d: np.ndarray):
        nonlocal offset, mean_prev

        mean_cur = fft1d.mean(axis=0)
        mean_sub = mean_prev if mean_prev is not None else mean_cur
        mean_prev = mean_cur

        pts, stats, rd_abs = process_frame_hw(
            fft1d, cfg, mean_sub=mean_sub,
            cfar_guard=args.cfar_guard, cfar_search=args.cfar_search,
            cfar_mul=args.cfar_mul,
            dpk_times=args.dpk_times, dpk_threshold=args.dpk_threshold,
            skip_bins=args.skip_bins, max_angle=args.max_angle,
            cfar_local_max=not args.no_local_max,
            zero_doppler_bins=args.zero_doppler_bins,
            max_detections=args.max_detections,
            min_range_m=args.min_range,
            max_range_m=args.max_range,
            sinc_buf=sinc_buf, steering_vec=steering_vec,
            calib_phase=calib_phase,
            n_elev=args.n_elev,
            enable_elevation=enable_elevation,
        )

        all_pt_list.append(pts)
        all_fr_list.append((global_fid, frame_ts, len(pts), offset))
        offset += len(pts)
        all_stats.append((global_fid, stats))

        if args.save_rdmap:
            save_rdmap(rd_abs, global_fid, str(rdmap_dir), cfg)

        processed = len(all_fr_list)
        if processed % args.log_every == 0 or processed == 1 or processed == n_frames:
            print(
                f"  [{processed:5d}/{n_frames}] 帧{global_fid:6d}: "
                f"CFAR={stats['cfar_detections']:3d} "
                f"DPK={stats['dpk_candidates']:3d}/{stats['dpk_passed']:3d} "
                f"最终={stats['final_points']:3d}"
            )

    t0 = time.time()
    process_one(first_fid, first_ts, first_fft1d)

    for global_fid, frame_ts, frame_raw in frame_iter:
        fft1d = raw_frame_to_fft1d(frame_raw, cfg)
        process_one(global_fid, frame_ts, fft1d)

    elapsed = time.time() - t0
    total_pts = sum(len(p) for p in all_pt_list)
    print(f"\n完成: {n_frames}帧, {total_pts}点, {elapsed:.1f}s")

    all_points = (np.concatenate(all_pt_list)
                  if total_pts > 0 else np.zeros(0, dtype=PT_DTYPE))
    all_frames = np.array(all_fr_list, dtype=FR_DTYPE)

    # ---- 7. 写统计 ----
    sf = stats_dir / "statistics.txt"
    with open(sf, "w", encoding="utf-8") as f:
        f.write(f"生成时间: {datetime.now()}\n")
        f.write("输入文件:\n")
        for spec in specs:
            f.write(f"  {spec.path} start={spec.start_frame_index} frames={spec.frame_count}\n")
        if args.capture_dir:
            f.write(f"capture_dir: {Path(args.capture_dir).resolve()}\n")
        f.write(f"总帧数: {n_frames}, 总点数: {total_pts}\n")
        f.write(f"输出: {npz_output}\n")
        f.write(f"CFAR: 1D SO, G={args.cfar_guard}, S={args.cfar_search}, "
                f"mul={args.cfar_mul}\n")
        f.write(
            f"CFAR post: local_max={not args.no_local_max}, "
            f"zero_doppler_bins={args.zero_doppler_bins}, "
            f"max_detections={args.max_detections}, "
            f"range=[{args.min_range}, {args.max_range}]\n"
        )
        f.write(f"DPK: times={args.dpk_times}, threshold={args.dpk_threshold}\n")
        f.write(
            f"Elevation: enabled={enable_elevation}, "
            f"vertical_tx={default_elevation_tx_list(cfg, args.n_elev)}, "
            f"fov=[{args.el_down}, {args.el_up}], "
            f"elevation_not_filtered_by_max_angle=True\n\n"
        )
        f.write(f"{'帧':>4} {'CFAR':>6} {'DPK候选':>8} {'DPK通过':>8} {'最终':>6}\n")
        f.write("-" * 40 + "\n")
        for frame_id, s in all_stats:
            f.write(
                f"{frame_id:4d} {s['cfar_detections']:6d} "
                f"{s['dpk_candidates']:8d} {s['dpk_passed']:8d} "
                f"{s['final_points']:6d}\n"
            )

    # ---- 8. 保存 NPZ ----
    save_npz(all_frames, all_points, npz_output)

    # ---- 9. 点云图 ----
    if args.save_preview and total_pts > 0:
        f0 = all_frames[0]
        pts0 = all_points[f0["offset"]:f0["offset"] + f0["num_points"]]
        if len(pts0) > 0:
            visualize_pointcloud(
                pts0, f"Frame {f0['frame_id']}",
                pc_dir / f"frame_{f0['frame_id']:04d}.png",
            )
        if len(all_frames) > 1:
            fl = all_frames[-1]
            ptsl = all_points[fl["offset"]:fl["offset"] + fl["num_points"]]
            if len(ptsl) > 0:
                visualize_pointcloud(
                    ptsl, f"Frame {fl['frame_id']}",
                    pc_dir / f"frame_{fl['frame_id']:04d}.png",
                )
        visualize_pointcloud(
            all_points, "All Frames",
            pc_dir / "all_frames.png",
        )

    print(f"\n输出: {npz_output}")
    print(f"统计: {sf}")


def main():
    p = argparse.ArgumentParser(
        description="Calterah 4T4R raw → 点云 (支持单 bin 和 capture_dir 分片)",
    )
    p.add_argument("--capture-dir", default=None,
                   help="采集目录，自动读取 Calterah/meta.json 和 adc_data_*.bin")
    p.add_argument("--bin", default=None, help="原始 bin/dat 文件 (旧单文件入口)")
    p.add_argument("--output", default=None,
                   help="统计/可视化输出目录 (capture 默认 Calterah/pointcloud_artifacts)")
    p.add_argument("--npz-output", default=None,
                   help="点云 NPZ 输出路径 (capture 默认 Calterah/pointcloud.npz)")
    p.add_argument("--calib", default=None,
                   help="校准 .npy (16,) complex；不提供则首帧自校准")

    # 数据格式
    p.add_argument("--dtype", default="<i2",
                   help="raw 数据 dtype，默认 <i2 (little-endian int16)；如乱序可试 >i2")
    p.add_argument("--samples-per-chirp", type=int, default=512,
                   help="每 chirp 每 RX 样点数 (= rng_nfft, 默认 512 → 2MB/帧)")
    p.add_argument("--dec-factor", type=int, default=2,
                   help="DFE 抽取因子 (sensor.json: 2)")
    p.add_argument("--vel-nfft", type=int, default=128,
                   help="Doppler FFT 点数 (sensor.json: 128)")
    p.add_argument("--el-down", type=float, default=-90.0,
                   help="俯仰搜索下界 (deg)，默认 -90")
    p.add_argument("--el-up", type=float, default=90.0,
                   help="俯仰搜索上界 (deg)，默认 90")

    # 帧选择
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--log-every", type=int, default=50,
                   help="每处理多少帧打印一次进度")

    # CFAR (默认匹配实际采集后处理命令)
    p.add_argument("--cfar-guard", type=int, default=2)
    p.add_argument("--cfar-search", type=int, default=6)
    p.add_argument("--cfar-mul", type=float, default=2.0)
    p.add_argument("--no-local-max", action="store_true",
                   help="关闭 3x3 local-maximum 筛选")
    p.add_argument("--local-max", dest="no_local_max", action="store_false",
                   help="启用 3x3 local-maximum 筛选")
    p.set_defaults(no_local_max=True)
    p.add_argument("--zero-doppler-bins", type=int, default=1,
                   help="抑制中心零多普勒附近 bin 数；人体动作默认 1，角反射器调试可设 0")
    p.add_argument("--max-detections", type=int, default=400,
                   help="每帧最多保留 CFAR 检测数")
    p.add_argument("--min-range", type=float, default=None,
                   help="最小保留距离 (m)")
    p.add_argument("--max-range", type=float, default=6.0,
                   help="最大保留距离 (m)")

    # DPK
    p.add_argument("--dpk-times", type=int, default=2)
    p.add_argument("--dpk-threshold", type=float, default=3.0)
    p.add_argument("--skip-bins", type=int, default=5)
    p.add_argument("--max-angle", type=float, default=60.0)

    # 俯仰默认开启；可用 --no-elevation 强制 z=0。
    p.add_argument("--n-elev", type=int, default=3,
                   help="俯仰 beamforming 用的垂直 TX 数 (默认 3: TX1/TX2/TX3)")
    p.add_argument("--enable-elevation", action="store_true", default=True,
                   help="启用俯仰估计并输出 z/elevation (默认开启)")
    p.add_argument("--no-elevation", action="store_true",
                   help="兼容旧命令: 禁用俯仰估计，强制 z=0")

    # 输出
    p.add_argument("--save-rdmap", action="store_true", default=False,
                   help="保存每帧 RDMap 图片")
    p.add_argument("--no-rdmap", dest="save_rdmap", action="store_false")
    p.add_argument("--save-preview", action="store_true", default=False,
                   help="保存首帧/尾帧/全局点云预览图")

    args = p.parse_args()
    if args.log_every <= 0:
        args.log_every = 50
    run(args)


if __name__ == "__main__":
    main()
