#!/usr/bin/env python3
"""
Run EasyMocap pipeline on captured data.

Usage:
    python -m data_pipeline.run_easymocap --input logs/captures/20241230_123456 --subs cam0 cam2 cam4 cam6 cam12
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EASYMOCAP_ROOT = PROJECT_ROOT / "third_party" / "EasyMocapFork"


def write_opencv_matrix(f, name, data, rows, cols):
    """Write matrix in OpenCV YAML format"""
    f.write(f'{name}: !!opencv-matrix\n')
    f.write(f'   rows: {rows}\n')
    f.write(f'   cols: {cols}\n')
    f.write(f'   dt: d\n')
    flat = np.array(data).flatten().tolist()
    f.write(f'   data: {flat}\n')


def convert_calibration(intri_dir, extri_file, output_dir, cameras):
    """Convert WiMocap calibration to EasyMocap format (intri.yml, extri.yml)"""
    intri_dir = Path(intri_dir)
    output_dir = Path(output_dir)
    
    # Load extrinsics
    with open(extri_file) as f:
        extri_data = json.load(f)
    cam_extri = extri_data['camera_extrinsics']
    
    # Write intri.yml
    with open(output_dir / 'intri.yml', 'w') as f:
        f.write('%YAML:1.0\n---\nnames:\n')
        for cam in cameras:
            f.write(f'   - {cam}\n')
        
        for cam in cameras:
            intri_file = intri_dir / f'{cam}.json'
            if not intri_file.exists():
                print(f"Warning: intrinsics not found for {cam}")
                continue
            
            with open(intri_file) as jf:
                data = json.load(jf)['camera']
            
            K = np.array(data['camera_matrix'])
            dist = np.array(data['dist_coeffs'])
            if len(dist) < 5:
                dist = np.pad(dist, (0, 5 - len(dist)))
            
            write_opencv_matrix(f, f'K_{cam}', K, 3, 3)
            write_opencv_matrix(f, f'dist_{cam}', dist[:5], 1, 5)
    
    # Write extri.yml
    with open(output_dir / 'extri.yml', 'w') as f:
        f.write('%YAML:1.0\n---\nnames:\n')
        for cam in cameras:
            f.write(f'   - {cam}\n')
        
        for cam in cameras:
            if cam not in cam_extri:
                print(f"Warning: extrinsics not found for {cam}")
                continue
            
            ext = cam_extri[cam]
            R = np.array(ext['R'])
            T = np.array(ext['T'])
            Rvec = np.array(ext.get('Rvec', [0, 0, 0]))
            
            write_opencv_matrix(f, f'R_{cam}', Rvec, 3, 1)
            write_opencv_matrix(f, f'Rot_{cam}', R, 3, 3)
            write_opencv_matrix(f, f'T_{cam}', T, 3, 1)
    
    print(f"Calibration saved to {output_dir}")


def has_direct_image_layout(input_dir, cameras):
    """Return True when input_dir already contains EasyMocap-readable cam folders."""
    input_dir = Path(input_dir)
    for cam in cameras:
        cam_dir = input_dir / cam
        if not cam_dir.is_dir() or not any(cam_dir.glob('*.jpg')):
            return False
    return True


def is_duplicate_image_dir(prepared_dir, source_dir):
    """Check whether images/<cam> is a generated mirror of <cam>."""
    if prepared_dir.is_symlink():
        return True
    if not prepared_dir.is_dir():
        return False

    for item in prepared_dir.iterdir():
        if item.is_dir() or item.suffix.lower() != '.jpg':
            return False

        source = source_dir / item.name
        if not source.exists():
            return False

        try:
            if item.samefile(source):
                continue
        except OSError:
            pass

        if item.is_symlink():
            continue

        try:
            item_stat = item.stat()
            source_stat = source.stat()
        except OSError:
            return False

        if item_stat.st_size != source_stat.st_size:
            return False
        if abs(item_stat.st_mtime - source_stat.st_mtime) > 1e-6:
            return False

    return True


def cleanup_duplicate_images_dir(input_dir, cameras):
    """Remove old generated images/<cam> mirrors when direct cam folders are used."""
    input_dir = Path(input_dir)
    images_dir = input_dir / 'images'
    if not images_dir.exists():
        return

    removed = []
    for cam in cameras:
        prepared_dir = images_dir / cam
        source_dir = input_dir / cam
        if not prepared_dir.exists() and not prepared_dir.is_symlink():
            continue
        if not is_duplicate_image_dir(prepared_dir, source_dir):
            continue

        if prepared_dir.is_symlink():
            prepared_dir.unlink()
        else:
            shutil.rmtree(prepared_dir)
        removed.append(cam)

    try:
        images_dir.rmdir()
    except OSError:
        pass

    if removed:
        cams = ', '.join(removed)
        print(f"Removed duplicate prepared image dirs: images/{{{cams}}}")


def count_smpl_results(output_dir):
    smpl_dir = Path(output_dir) / 'smpl'
    if not smpl_dir.exists():
        return 0
    return len(list(smpl_dir.glob('*.json')))


def link_or_copy_image(src, dst):
    """Create a lightweight image entry, copying only if links are unavailable."""
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        dst.symlink_to(src.resolve())
        return 'symlink'
    except OSError:
        try:
            os.link(src.resolve(), dst)
            return 'hardlink'
        except OSError:
            shutil.copy2(src, dst)
            return 'copy'


def prepare_images(input_dir, output_dir, cameras, session_ts=None):
    """Prepare images in EasyMocap format"""
    input_dir = Path(input_dir)
    images_dir = output_dir / 'images'

    if input_dir == Path(output_dir) and has_direct_image_layout(input_dir, cameras):
        cleanup_duplicate_images_dir(input_dir, cameras)
        print("Using existing camera folders directly; no images/ mirror created.")
        return ''
    
    # Mapping from EasyMocap camera name to possible source names
    # cam12 is RealSense RGB in our setup
    cam_aliases = {
        'cam12': ['rs_color', 'rs_color_' + (session_ts.split('_')[0] if session_ts else '')],
    }
    
    for cam in cameras:
        # Get possible source names
        source_names = [cam] + cam_aliases.get(cam, [])
        
        src_dir = None
        for src_name in source_names:
            # Source patterns to try
            patterns = [
                input_dir / f'{src_name}_{session_ts}_raw' if session_ts else None,  # camera_driver format
                input_dir / 'aligned' / src_name,  # multimodal_capture aligned format
                input_dir / src_name,  # direct format
            ]
            # Also try partial timestamp match for rs_color_20251230 format
            if src_name.startswith('rs_color'):
                for d in input_dir.glob('rs_color_*_raw'):
                    patterns.append(d)
                for d in (input_dir / 'aligned').glob('rs_color_*'):
                    if d.is_dir():
                        patterns.append(d)
            
            for p in patterns:
                if p and p.exists() and p.is_dir():
                    src_dir = p
                    break
            if src_dir:
                break
        
        if not src_dir:
            print(f"Warning: no images found for {cam} (tried: {source_names})")
            continue
        
        dst_dir = images_dir / cam
        dst_dir.mkdir(parents=True, exist_ok=True)
        
        # Link images when possible; fall back to hardlinks before real copies.
        src_images = sorted(src_dir.glob('*.jpg'))
        modes = {'symlink': 0, 'hardlink': 0, 'copy': 0}
        for i, src in enumerate(src_images):
            dst = dst_dir / f'{i:06d}.jpg'
            modes[link_or_copy_image(src, dst)] += 1
        
        mode_summary = ', '.join(f"{count} {mode}" for mode, count in modes.items() if count)
        print(f"  {cam}: {len(src_images)} images from {src_dir.name} ({mode_summary})")
    
    return 'images'


def run_emc(data_root, cameras, exp_config='config/mv1p/detect_triangulate_fitSMPL.yml', 
            frame_range=None, output_dir=None, skip_vis=False, image_root='images'):
    """Run EasyMocap command"""
    data_root = Path(data_root).resolve()
    
    # Default output to data_root/output
    if output_dir is None:
        output_dir = data_root / 'output'
    else:
        output_dir = Path(output_dir).resolve()
    
    # Build command - use python module instead of emc command
    cmd = [
        sys.executable, '-m', 'apps.mocap.run',
        '--data', 'config/datasets/mvimage.yml',
        '--exp', exp_config,
        '--root', str(data_root),
        '--subs', *cameras,
        '--subs_vis', *cameras,
        '--out', str(output_dir),  # Always specify output dir
    ]

    if image_root != 'images':
        cmd.extend([
            '--opt_data',
            'args.reader.images.root', image_root,
            'args.reader.image_shape.root', image_root,
        ])
    
    if frame_range:
        cmd.extend(['--ranges', str(frame_range[0]), str(frame_range[1]), '1'])
    
    if skip_vis:
        cmd.extend(['--skip_vis', '--skip_vis_final'])
    
    print(f"\nRunning: {' '.join(cmd)}")
    print(f"Working directory: {EASYMOCAP_ROOT}")
    
    # Suppress warnings and enable headless rendering
    env = os.environ.copy()
    env['PYTHONWARNINGS'] = 'ignore::FutureWarning,ignore::RuntimeWarning'
    # NVIDIA servers provide headless OpenGL through EGL.  Forcing OSMesa here
    # makes even skipped visualization stages fail while importing pyrender on
    # machines without libOSMesa.
    env['PYOPENGL_PLATFORM'] = 'egl'
    
    # Run from EasyMocap directory
    result = subprocess.run(cmd, cwd=EASYMOCAP_ROOT, env=env)
    return result.returncode == 0


def main():
    p = argparse.ArgumentParser(description="Run EasyMocap on captured data")
    
    # Input
    p.add_argument("--input", required=True, help="Input directory (capture session)")
    # p.add_argument("--subs", nargs="+", default=['cam0', 'cam2', 'cam4', 'cam6', 'cam12'],
    #                help="Camera names to use")
    p.add_argument("--subs", nargs="+", default=['cam0', 'cam2', 'cam4', 'cam6'],
                   help="Camera names to use")
    # Calibration
    p.add_argument("--intri", default="configs/camera_intrinsics",
                   help="Intrinsics directory")
    p.add_argument("--extri", default="configs/camera_extrinsics/unified.json",
                   help="Extrinsics file")
    
    # EasyMocap
    p.add_argument("--exp", default=str(PROJECT_ROOT / "configs/easymocap/detect_triangulate_fitSMPL.yml"),
                   help="Experiment config (default: WiMocap simplified config)")
    p.add_argument("--start", type=int, default=0, help="Start frame")
    p.add_argument("--end", type=int, default=None, help="End frame")
    p.add_argument("--output-dir", default=None, help="Output directory (default: input/output)")
    
    # Options
    p.add_argument("--skip-prepare", action="store_true", help="Skip data preparation")
    p.add_argument("--skip-run", action="store_true", help="Skip running EasyMocap")
    p.add_argument("--skip-vis", dest="skip_vis", action="store_true", help="Skip visualization")
    p.add_argument("--vis", dest="skip_vis", action="store_false", help="Enable visualization")
    p.add_argument("--skip-existing", action="store_true",
                   help="Skip EasyMocap when output/smpl already has enough frames")
    p.set_defaults(skip_vis=True)
    
    args = p.parse_args()
    requested_end = args.end
    
    input_dir = Path(args.input).resolve()
    intri_dir = (PROJECT_ROOT / args.intri).resolve()
    extri_file = (PROJECT_ROOT / args.extri).resolve()
    
    # Detect session timestamp from directory structure
    session_ts = None
    for item in input_dir.iterdir():
        if item.is_dir() and '_raw' in item.name:
            parts = item.name.split('_')
            if len(parts) >= 3:
                session_ts = f"{parts[-3]}_{parts[-2]}"
                break
    
    image_root = '' if has_direct_image_layout(input_dir, args.subs) else 'images'

    # Prepare data
    if not args.skip_prepare:
        print("=" * 50)
        print("Preparing data for EasyMocap")
        print("=" * 50)
        
        # Convert calibration
        convert_calibration(intri_dir, extri_file, input_dir, args.subs)
        
        # Prepare images
        image_root = prepare_images(input_dir, input_dir, args.subs, session_ts)
    
    # Count frames
    images_dir = input_dir / image_root if image_root else input_dir
    if images_dir.exists():
        first_cam_dir = images_dir / args.subs[0]
        if first_cam_dir.exists():
            n_frames = len(list(first_cam_dir.glob('*.jpg')))
            print(f"\nTotal frames: {n_frames}")
            
            if args.end is None:
                args.end = n_frames
        else:
            n_frames = 0
    else:
        n_frames = 0
    
    # Run EasyMocap
    if not args.skip_run:
        print("\n" + "=" * 50)
        print("Running EasyMocap")
        print("=" * 50)
        
        # Output directory: default to input/output, or user-specified
        output_dir = Path(args.output_dir) if args.output_dir else input_dir / 'output'

        if args.skip_existing and n_frames > 0:
            existing = count_smpl_results(output_dir)
            expected = max(0, min(args.end or n_frames, n_frames) - args.start)
            if existing >= expected:
                print(f"\nSkipping EasyMocap: found {existing} SMPL files in {output_dir}/smpl")
                return 0

        # Only use frame_range when the caller requested a subset.
        frame_range = (
            (args.start, args.end)
            if args.start > 0 or requested_end is not None
            else None
        )
        
        success = run_emc(
            input_dir,
            args.subs,
            args.exp,
            frame_range,
            output_dir,
            args.skip_vis,
            image_root,
        )
        
        if success:
            print("\n✓ EasyMocap completed successfully!")
            print(f"  Output: {output_dir}")
            print(f"  SMPL results: {output_dir}/smpl/")
        else:
            print("\n✗ EasyMocap failed")
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
