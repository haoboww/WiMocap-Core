# WiMocap-Core 后处理

这个目录是当前三雷达采集数据的可迁移后处理闭环，支持 Calterah、TI IWR6843 和 BGT60 4T4R：

1. Calterah/TI 从分片 raw data 生成点云；BGT60 直接转换采集时生成的点云，不重做雷达 DSP。
2. 以当前雷达自己的时间戳匹配 `cam0 cam2 cam4 cam6`。
3. 使用软链接组织相机帧，不复制原图。
4. 运行 EasyMocap，输出 `keypoints3d` 和 SMPL label。

不包含 `export_dataset`，训练直接使用对齐后的 `pointcloud.npz` 和 `output/smpl/`。

## 目录要求

原始 session 应保持采集时的结构：

```text
debug0801/
└── capture_20260803_115639/
    ├── Calterah/
    │   ├── adc_data_000.bin
    │   ├── adc_data_001.bin
    │   └── meta.json
    ├── TI_IWR6843/
    │   ├── adc_data_Raw_0.bin
    │   ├── adc_data_Raw_1.bin
    │   ├── radar.cfg
    │   └── meta.json
    ├── BGT60_4T4R/
    │   ├── pointcloud/index.csv
    │   ├── pointcloud/pcd_000.bin
    │   ├── fft1d/usb_fft_*.bin
    │   └── meta.json
    ├── cam0/
    ├── cam2/
    ├── cam4/
    └── cam6/
```

原始盘可以只读挂载。输出目录必须放在支持软链接的 Linux 文件系统上，推荐 ext4 或 XFS，不要放在 exFAT/NTFS 原始盘上。

## 新机器环境

以下版本已经在当前机器验证：Python 3.9、PyTorch 2.8.0、CUDA 12.8、torchvision 0.23.0。

```bash
git clone git@github.com:haoboww/WiMocap-Core.git
cd WiMocap-Core

conda create -n mocap python=3.9 -y
conda activate mocap

sudo apt-get update
sudo apt-get install -y libgl1 libegl1 libglib2.0-0

python -m pip install --upgrade pip
python -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip install -e third_party/EasyMocapFork

python scripts/check_environment.py
```

如果新机器驱动不支持 CUDA 12.8，只替换 PyTorch/torchvision 安装命令；其余依赖保持不变。处理前必须确认 `python scripts/check_environment.py` 显示 `CUDA available: True`。

YOLOv5 和 SMPL 权重包含在 Git 仓库中。HRNet 权重约 243 MiB，超过 GitHub 普通 Git 的单文件限制，需要单独放置：

```text
third_party/EasyMocapFork/data/models/pose_hrnet_w48_384x288.pth
```

可以从已配置好的机器复制，或者用仓库依赖的 `gdown` 下载到外部模型目录后建立软链接：

```bash
mkdir -p /path/to/shared/models/wimocap
gdown 'https://drive.google.com/uc?id=11ezQ6a_MxIRtj26WqhH3V3-xPI3XqYAw' \
  -O /path/to/shared/models/wimocap/pose_hrnet_w48_384x288.pth

mkdir -p third_party/EasyMocapFork/data/models
ln -s /path/to/shared/models/wimocap/pose_hrnet_w48_384x288.pth \
  third_party/EasyMocapFork/data/models/pose_hrnet_w48_384x288.pth

sha256sum third_party/EasyMocapFork/data/models/pose_hrnet_w48_384x288.pth
```

正确的 SHA256 是：

```text
95e0fec3194826d5e3f806ea89be68bbb84517b114c3a32b3058c56610b5ef61
```

可以同时检查一组原始数据：

```bash
python scripts/check_environment.py \
  --session /media/wais/SANDISK1/debug0801/capture_20260803_115639
```

`check_environment.py` 会检查模型大小、HRNet SHA256、GMM 姿态先验的数值和梯度，任一项失败都不要开始批处理。

## 处理单个 Session

先进入仓库并激活环境：

```bash
cd /path/to/WiMocap-Core
conda activate mocap
```

Calterah：

```bash
nice -n 8 ionice -c2 -n7 \
python data_pipeline/process_radar_session.py \
  --data-root /media/wais/SANDISK1/debug0801/capture_20260803_115639 \
  --radar calterah \
  --output-root /data/processed_calterah \
  --threads 4
```

TI IWR6843：

```bash
nice -n 8 ionice -c2 -n7 \
python data_pipeline/process_radar_session.py \
  --data-root /media/wais/SANDISK1/debug0801/capture_20260803_115639 \
  --radar ti \
  --output-root /data/processed_ti \
  --threads 4
```

BGT60 4T4R：

```bash
nice -n 8 ionice -c2 -n7 \
python data_pipeline/process_radar_session.py \
  --data-root /media/wais/SANDISK1/debug0801/capture_20260803_115639 \
  --radar bgt60 \
  --output-root /data/processed_bgt60 \
  --threads 4
```

BGT60 命令不会读取 `fft1d/usb_fft_*.bin`。它只读取 `pointcloud/index.csv` 和 `pointcloud/pcd_*.bin`，转换后以 BGT60 自己的 3057 帧点云时间戳匹配相机，再生成自己的一套 SMPL。

三条命令依次执行即可，默认不会并行。Calterah/TI 的默认点云参数与当前验证流程一致：

```text
--max-range 6.0
--max-detections 400
--no-local-max
--cfar-mul 2
--dpk-threshold 3
```

Calterah/TI 默认都启用 elevation。TI 默认不保存 RDMap，避免产生大量图片；确实需要时增加 `--save-rdmap`。这些 DSP 参数不作用于 BGT60，因为 BGT60 使用固件已经生成的点云。

## 批量处理

`--data-root` 也可以指向包含多个 `capture_*` 的目录。脚本会递归发现符合当前结构的 session，并严格串行处理：

```bash
nice -n 8 ionice -c2 -n7 \
python data_pipeline/process_radar_session.py \
  --data-root /media/wais/SANDISK1/debug0801 \
  --radar calterah \
  --output-root /data/processed_calterah \
  --threads 4 \
  --stop-on-error
```

然后依次将 `--radar` 和输出目录改为 `ti`、`bgt60`，各执行一次。

8 卡服务器建议使用可恢复的任务调度器。每张 GPU 同时只运行一个 session/radar，单任务失败不会中断剩余队列：

```bash
python scripts/run_batch_8gpu.py \
  --data-roots \
    /mnt/SANDISK/debug0731 \
    /mnt/SANDISK/debug0801 \
    /mnt/SANDISK/debug0803 \
    /mnt/SANDISK_backup/debug0804 \
  --output-base /data/wimocap_processed \
  --run-dir /data/wimocap_processed/_batch_runs/run_01 \
  --gpus 0 1 2 3 4 5 6 7 \
  --radars bgt60 ti calterah \
  --threads 4 \
  --python "$CONDA_PREFIX/bin/python" \
  --exclude capture_20260803_115639 capture_20260803_120503
```

任务清单、实时状态和失败记录分别保存在 `run-dir` 的 `manifest.json`、`summary.json` 和 `events.jsonl`。重复执行时底层 pipeline 会跳过已经完整的步骤；不要让两台机器同时写同一个输出目录。

## 输出结构

假设输出根目录是 `/data/processed_ti`，结果为：

```text
/data/processed_ti/debug0801/capture_20260803_115639/
├── TI_IWR6843/
│   ├── pointclouds/ti_iwr6843_pointcloud.npz  # 完整点云
│   ├── pointclouds/*.png
│   └── stats/statistics.txt
├── cam0/ cam2/ cam4/ cam6/                    # 原图软链接
├── alignment.json                             # 雷达帧到相机帧映射
├── matches.csv
├── pointcloud.npz                             # 训练用对齐点云
├── output/
│   ├── keypoints2d/
│   ├── keypoints3d/
│   └── smpl/                                  # 训练 label
├── logs/
└── processing_status.json
```

Calterah 结构相同，但完整点云位于 `Calterah/pointcloud.npz`；BGT60 完整点云位于 `BGT60_4T4R/pointcloud.npz`。

训练只使用同一个结果目录中的：

```text
pointcloud.npz
output/smpl/
```

不要使用雷达子目录中的“完整点云”直接配 SMPL。完整 raw 点云帧数可能多于可靠时间戳数量，例如 TI raw 可解出 866 帧但只有 665 个可靠硬件时间戳；根目录的训练点云会自动只保留有可靠相机匹配的 665 帧，与 SMPL 严格一致。

BGT60 的根目录训练点云以 `BGT60_4T4R/pointcloud/index.csv` 为唯一时间轴，保留固件输出的动态点和静态历史 TLV。其坐标统一为 `x=横向、y=前向、z=向上`；`power` 是固件发送的压缩 8 位强度，无法还原为未压缩功率。

BGT60 转换默认在遇到损坏的串口点云包时保留该帧时间戳并写空点云，同时把帧号和原因写入完整点云 NPZ 的 `meta.malformed_frames`，后续帧不会错位。协议诊断时可直接运行 `data_pipeline/bgt60/convert_pointcloud.py` 并增加 `--strict`，让首个坏包立即报错退出。

## 进度、恢复和重跑

每一步日志在结果目录下：

```bash
tail -f /data/processed_ti/debug0801/capture_20260803_115639/logs/pointcloud.log
tail -f /data/processed_ti/debug0801/capture_20260803_115639/logs/easymocap.log
cat /data/processed_ti/debug0801/capture_20260803_115639/processing_status.json
```

默认会跳过已经完整存在的步骤。只继续匹配和 EasyMocap：

```bash
python data_pipeline/process_radar_session.py \
  --data-root /path/to/capture_session \
  --radar ti \
  --output-root /data/processed_ti \
  --steps match,easymocap
```

确实需要覆盖重跑时增加 `--force`。先检查命令和输出位置但不执行时增加 `--dry-run`。

EasyMocap 的当前 SMPL 配置包含二阶时序平滑。手动使用 `run_easymocap.py --end N` 做小样本测试时，建议至少使用 10 帧；3 帧之类的极短序列会在平滑优化阶段产生 `NaN`，完整 session 不受影响。

## 搬迁检查

搬到新机器后先执行：

```bash
cd /path/to/WiMocap-Core
du -sh .
sha256sum -c MANIFEST.sha256
python scripts/check_environment.py --session /path/to/one/capture_session
python data_pipeline/process_radar_session.py \
  --data-root /path/to/one/capture_session \
  --radar calterah \
  --output-root /path/to/ext4/processed_calterah \
  --dry-run
```

确认原始盘路径、GPU、模型、四路标定和输出盘文件系统无误后，再去掉 `--dry-run`。
