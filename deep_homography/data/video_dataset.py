"""
deep_homography/data/video_dataset.py
--------------------------------------
Dataset and DataLoader for unsupervised homography estimation.

Faithfully mirrors the data protocol of Zhang et al. (ECCV 2020):
  - Input:  consecutive frame pairs (I_a, I_b) from real video.
  - Patches: random crop of size 256×256.
  - Labels: NONE (fully unsupervised).

Extension for triangle consistency loss:
  - Yields TRIPLETS (I_a, I_b, I_c) = frames at t, t+gap, t+2*gap.

Split is done at the VIDEO level, not frame level, to prevent leakage.
"""

import os
import glob
import random
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from deep_homography.data.augmentations import EvalTransform, TrainTransform

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


def _is_video(path: str) -> bool:
    return Path(path).suffix.lower() in VIDEO_EXTENSIONS


def _load_video_frames(video_path: str, stride: int = 1) -> List[np.ndarray]:
    """
    Reads all frames from a video file at a given stride.

    Args:
        video_path: Path to video file.
        stride:     Read every `stride`-th frame.

    Returns:
        list of BGR np.ndarray (H, W, 3) uint8.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames, idx = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride == 0:
            frames.append(frame)
        idx += 1
    cap.release()
    return frames


# ---------------------------------------------------------------------------
# Primary dataset: video files
# ---------------------------------------------------------------------------

class VideoHomographyDataset(Dataset):
    """
    Unsupervised dataset that yields frame triplets from video files.

    Each sample:   (I_a, I_b, I_c) = frames at t, t+gap, t+2×gap.
    For pair-only training simply ignore I_c.

    Args:
        video_root (str):   Root directory with video files (recursive search).
        transform:          Augmentation / normalisation callable.
        frame_gap (int):    Temporal gap between frames (paper: 1).
        video_stride (int): Sub-sample video frames to reduce dataset size.
        split (str):        "train" | "val" | "all".
        train_ratio (float): Fraction of videos for training.
        seed (int):         RNG seed for reproducible split.
        max_frames_per_video (int): Cap frames per video (None = no cap).
    """

    def __init__(
        self,
        video_root: str,
        transform: Optional[Callable] = None,
        frame_gap: int = 1,
        video_stride: int = 1,
        split: str = "train",
        train_ratio: float = 0.9,
        seed: int = 42,
        max_frames_per_video: Optional[int] = None,
    ):
        super().__init__()
        self.transform  = transform
        self.frame_gap  = frame_gap
        self.max_frames = max_frames_per_video

        all_videos = self._discover_videos(video_root)
        if not all_videos:
            raise FileNotFoundError(
                f"No video files found under '{video_root}'. "
                f"Supported: {VIDEO_EXTENSIONS}"
            )

        rng = random.Random(seed)
        rng.shuffle(all_videos)
        n_train = max(1, int(len(all_videos) * train_ratio))
        selected = (
            all_videos[:n_train] if split == "train"
            else all_videos[n_train:] if split == "val"
            else all_videos
        )

        self.frame_lists: List[List[np.ndarray]] = []
        self.index: List[Tuple[int, int]] = []

        print(f"[VideoHomographyDataset] Loading {len(selected)} videos (split={split})...")
        for vid_path in selected:
            try:
                frames = _load_video_frames(vid_path, stride=video_stride)
            except IOError as e:
                print(f"  WARNING: {e}")
                continue
            if self.max_frames:
                frames = frames[: self.max_frames]
            min_len = 2 * frame_gap + 1
            if len(frames) < min_len:
                continue
            vid_idx = len(self.frame_lists)
            self.frame_lists.append(frames)
            for t in range(len(frames) - 2 * frame_gap):
                self.index.append((vid_idx, t))

        print(f"[VideoHomographyDataset] {len(self.index):,} triplets ready.")

    @staticmethod
    def _discover_videos(root: str) -> List[str]:
        paths = []
        for ext in VIDEO_EXTENSIONS:
            paths.extend(glob.glob(os.path.join(root, "**", f"*{ext}"), recursive=True))
        return sorted(paths)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> dict:
        vid_idx, t = self.index[idx]
        frames = self.frame_lists[vid_idx]
        g = self.frame_gap

        triplet = [frames[t], frames[t + g], frames[t + 2 * g]]

        if self.transform is not None:
            img_a, img_b, img_c = self.transform(triplet)
        else:
            img_a = torch.from_numpy(triplet[0].transpose(2, 0, 1)).float() / 255.0
            img_b = torch.from_numpy(triplet[1].transpose(2, 0, 1)).float() / 255.0
            img_c = torch.from_numpy(triplet[2].transpose(2, 0, 1)).float() / 255.0

        return {"img_a": img_a, "img_b": img_b, "img_c": img_c, "index": idx}


# ---------------------------------------------------------------------------
# Evaluation dataset: pre-extracted image pairs (paper's RE/LT/LL/SF/LF)
# ---------------------------------------------------------------------------

class ImagePairDataset(Dataset):
    """
    Loads pre-extracted image pairs for quantitative evaluation.

    Expected layout:
        root/
          frame_a/  0001.png  0002.png  ...
          frame_b/  0001.png  0002.png  ...
          [frame_c/ ...]    optional, for triplet eval
          [gt_points/ *.npy]  optional, shape (N, 2, 2): [src_xy, dst_xy]

    Args:
        root (str):          Directory with frame_a/, frame_b/ sub-dirs.
        transform:           Augmentation callable.
        gt_points_dir (str): Optional path to .npy GT correspondence files.
    """

    IMG_EXTS = ("*.png", "*.jpg", "*.jpeg")

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
        gt_points_dir: Optional[str] = None,
    ):
        self.transform = transform

        def _find(d: str) -> List[str]:
            return sorted(sum([glob.glob(os.path.join(d, e)) for e in self.IMG_EXTS], []))

        dir_a = os.path.join(root, "frame_a")
        dir_b = os.path.join(root, "frame_b")
        dir_c = os.path.join(root, "frame_c")

        self.files_a = _find(dir_a)
        self.files_b = _find(dir_b)
        self.files_c = _find(dir_c) if os.path.isdir(dir_c) else None

        assert len(self.files_a) == len(self.files_b) and self.files_a, (
            f"Mismatch or empty: frame_a={len(self.files_a)}, frame_b={len(self.files_b)}"
        )

        self.gt_points = None
        if gt_points_dir and os.path.isdir(gt_points_dir):
            gt_files = sorted(glob.glob(os.path.join(gt_points_dir, "*.npy")))
            if len(gt_files) == len(self.files_a):
                self.gt_points = [np.load(f) for f in gt_files]

        print(f"[ImagePairDataset] {len(self.files_a)} pairs from '{root}'.")

    def __len__(self) -> int:
        return len(self.files_a)

    def __getitem__(self, idx: int) -> dict:
        fa = cv2.imread(self.files_a[idx])
        fb = cv2.imread(self.files_b[idx])
        fc = cv2.imread(self.files_c[idx]) if self.files_c else fb.copy()

        if self.transform:
            img_a, img_b, img_c = self.transform([fa, fb, fc])
        else:
            img_a = torch.from_numpy(fa.transpose(2, 0, 1)).float() / 255.0
            img_b = torch.from_numpy(fb.transpose(2, 0, 1)).float() / 255.0
            img_c = torch.from_numpy(fc.transpose(2, 0, 1)).float() / 255.0

        sample = {"img_a": img_a, "img_b": img_b, "img_c": img_c, "index": idx}
        if self.gt_points is not None:
            sample["gt_points"] = torch.from_numpy(self.gt_points[idx].astype(np.float32))
        return sample


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def build_train_loader(cfg: dict) -> DataLoader:
    """Constructs the training DataLoader from a parsed config dict."""
    d, t = cfg["data"], cfg["train"]
    transform = TrainTransform(
        crop_h=d["patch_height"], crop_w=d["patch_width"],
        grayscale=d["grayscale"], mean=d["normalize_mean"], std=d["normalize_std"],
    )
    dataset = VideoHomographyDataset(
        video_root=d["video_path"], transform=transform,
        frame_gap=d["triplet_gap"], split="train", train_ratio=d["train_split"],
    )
    return DataLoader(
        dataset, batch_size=t["batch_size"], shuffle=True,
        num_workers=d["num_workers"], pin_memory=d["pin_memory"], drop_last=True,
    )


def build_val_loader(cfg: dict) -> DataLoader:
    """Constructs the validation DataLoader (no random augmentation)."""
    d, e = cfg["data"], cfg["eval"]
    transform = EvalTransform(
        crop_h=d["patch_height"], crop_w=d["patch_width"],
        grayscale=d["grayscale"], mean=d["normalize_mean"], std=d["normalize_std"],
    )
    dataset = VideoHomographyDataset(
        video_root=d["video_path"], transform=transform,
        frame_gap=d["triplet_gap"], split="val", train_ratio=d["train_split"],
    )
    return DataLoader(
        dataset, batch_size=e["batch_size"], shuffle=False,
        num_workers=d["num_workers"], pin_memory=d["pin_memory"],
    )


def build_eval_loader(root: str, cfg: dict, gt_points_dir: str = None) -> DataLoader:
    """Constructs a DataLoader from a pre-extracted image-pair directory."""
    d, e = cfg["data"], cfg["eval"]
    transform = EvalTransform(
        crop_h=d["patch_height"], crop_w=d["patch_width"],
        grayscale=d["grayscale"], mean=d["normalize_mean"], std=d["normalize_std"],
    )
    dataset = ImagePairDataset(root=root, transform=transform, gt_points_dir=gt_points_dir)
    return DataLoader(
        dataset, batch_size=e["batch_size"], shuffle=False,
        num_workers=d["num_workers"], pin_memory=d["pin_memory"],
    )
