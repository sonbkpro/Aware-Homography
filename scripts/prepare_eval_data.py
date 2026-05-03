"""
scripts/prepare_eval_data.py
-----------------------------
Converts a directory of raw images into the paper's frame_a / frame_b
evaluation format, optionally applying a known homography to create
ground-truth warped pairs for quantitative testing.

Usage:
    # From a video: extract consecutive pairs at a given stride
    python scripts/prepare_eval_data.py \\
        --video data/raw/my_video.mp4 \\
        --out_dir data/eval/MY_CATEGORY \\
        --stride 5 \\
        --num_pairs 200

    # From a directory of images (already extracted frames)
    python scripts/prepare_eval_data.py \\
        --frames data/raw/frames/ \\
        --out_dir data/eval/MY_CATEGORY \\
        --stride 3
"""

import argparse
import os
import glob
import shutil
from pathlib import Path

import cv2
import numpy as np


def parse_args():
    p = argparse.ArgumentParser(description="Prepare evaluation data")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  help="Input video file")
    src.add_argument("--frames", help="Directory of pre-extracted frame images")
    p.add_argument("--out_dir",  required=True, help="Output directory")
    p.add_argument("--stride",   type=int, default=1, help="Frame stride for pairs")
    p.add_argument("--num_pairs",type=int, default=None, help="Max number of pairs")
    p.add_argument("--resize",   type=str, default=None,
                   help="Resize frames before saving, e.g. '640x360'")
    return p.parse_args()


def extract_frames_from_video(video_path: str, stride: int = 1):
    cap = cv2.VideoCapture(video_path)
    frames, idx = [], 0
    while True:
        ret, f = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(f)
        idx += 1
    cap.release()
    return frames


def load_frames_from_dir(frames_dir: str, stride: int = 1):
    exts   = ("*.png", "*.jpg", "*.jpeg")
    files  = sorted(sum([glob.glob(os.path.join(frames_dir, e)) for e in exts], []))
    return [cv2.imread(f) for f in files[::stride]]


def main():
    args = parse_args()

    if args.video:
        print(f"Extracting frames from {args.video}...")
        frames = extract_frames_from_video(args.video, stride=1)
    else:
        print(f"Loading frames from {args.frames}...")
        frames = load_frames_from_dir(args.frames, stride=1)

    print(f"  {len(frames)} frames loaded.")

    if args.resize:
        w_str, h_str = args.resize.split("x")
        W, H = int(w_str), int(h_str)
        frames = [cv2.resize(f, (W, H)) for f in frames]

    # Build consecutive pairs at stride
    stride = args.stride
    pairs  = [(frames[i], frames[i + stride])
              for i in range(0, len(frames) - stride, stride)]
    if args.num_pairs:
        pairs = pairs[: args.num_pairs]

    print(f"  {len(pairs)} pairs at stride={stride}")

    # Create output directories
    out = Path(args.out_dir)
    dir_a = out / "frame_a"
    dir_b = out / "frame_b"
    dir_a.mkdir(parents=True, exist_ok=True)
    dir_b.mkdir(parents=True, exist_ok=True)

    for i, (fa, fb) in enumerate(pairs):
        cv2.imwrite(str(dir_a / f"{i:05d}.png"), fa)
        cv2.imwrite(str(dir_b / f"{i:05d}.png"), fb)
        if (i + 1) % 50 == 0:
            print(f"  Saved {i+1}/{len(pairs)} pairs...")

    print(f"\nDone! Data saved to:")
    print(f"  {dir_a}")
    print(f"  {dir_b}")
    print(f"\nTo evaluate:")
    print(f"  python evaluate.py --config configs/default.yaml \\")
    print(f"      --checkpoint checkpoints/best_model.pth \\")
    print(f"      --data_root {out}")


if __name__ == "__main__":
    main()
