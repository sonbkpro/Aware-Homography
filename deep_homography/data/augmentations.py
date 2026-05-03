"""
deep_homography/data/augmentations.py
--------------------------------------
Geometric and photometric augmentations for unsupervised homography training.

Key design principle:
  - Spatial transforms (crop, flip) → applied IDENTICALLY to all frames in
    a triplet so the ground-truth geometric relationship is preserved.
  - Photometric transforms (blur, brightness, noise) → applied INDEPENDENTLY
    per frame to stress-test luminance invariance, the documented failure mode
    of pixel-intensity baselines.
"""

import random
import numpy as np
import cv2
import torch


# ---------------------------------------------------------------------------
# Spatial augmentations (applied identically to every frame in a triplet)
# ---------------------------------------------------------------------------

class RandomCropPair:
    """
    Crops the same random region from every frame in a list.

    Mirrors the paper's protocol: sample patches of size 315×560 from the
    original frames.  The same (top, left) offset is used for all frames so
    the inter-frame homography remains valid.

    Args:
        crop_h (int): Output patch height (paper: 315).
        crop_w (int): Output patch width  (paper: 560).
    """

    def __init__(self, crop_h: int = 315, crop_w: int = 560):
        self.crop_h = crop_h
        self.crop_w = crop_w

    def __call__(self, frames: list) -> list:
        h, w = frames[0].shape[:2]
        if h < self.crop_h or w < self.crop_w:
            # Resize up so we can always crop
            scale = max(self.crop_h / h, self.crop_w / w)
            new_h, new_w = int(h * scale) + 1, int(w * scale) + 1
            frames = [cv2.resize(f, (new_w, new_h)) for f in frames]
            h, w = new_h, new_w

        top  = random.randint(0, h - self.crop_h)
        left = random.randint(0, w - self.crop_w)
        return [f[top: top + self.crop_h, left: left + self.crop_w] for f in frames]


class RandomHorizontalFlipPair:
    """Horizontally flips all frames in a list with probability p."""

    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, frames: list) -> list:
        if random.random() < self.p:
            return [np.fliplr(f).copy() for f in frames]
        return frames


# ---------------------------------------------------------------------------
# Photometric augmentations (applied independently per frame)
# ---------------------------------------------------------------------------

class IndependentGaussianBlur:
    """Gaussian blur per frame with probability p."""

    def __init__(self, p: float = 0.3):
        self.p = p

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            ksize = random.choice([3, 5, 7])
            sigma = random.uniform(0.5, 2.0)
            frame = cv2.GaussianBlur(frame, (ksize, ksize), sigma)
        return frame


class IndependentBrightnessContrast:
    """
    Independently shifts brightness and contrast per frame.

    Simulates exposure / flash / illumination differences — the critical
    failure mode for pixel-intensity baselines like Nguyen et al. (2018).
    """

    def __init__(
        self,
        brightness_range: tuple = (-30, 30),
        contrast_range: tuple = (0.7, 1.3),
        p: float = 0.5,
    ):
        self.brightness_range = brightness_range
        self.contrast_range   = contrast_range
        self.p = p

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            alpha = random.uniform(*self.contrast_range)
            beta  = random.uniform(*self.brightness_range)
            frame = np.clip(
                alpha * frame.astype(np.float32) + beta, 0, 255
            ).astype(np.uint8)
        return frame


class IndependentGammaNoise:
    """Additive Gaussian noise to simulate low-light sensor noise."""

    def __init__(self, sigma_range: tuple = (0, 15), p: float = 0.3):
        self.sigma_range = sigma_range
        self.p = p

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if random.random() < self.p:
            sigma = random.uniform(*self.sigma_range)
            noise = np.random.randn(*frame.shape).astype(np.float32) * sigma
            frame = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        return frame


# ---------------------------------------------------------------------------
# Composed pipelines
# ---------------------------------------------------------------------------

class TrainTransform:
    """
    Full training augmentation pipeline.

    Args:
        crop_h, crop_w: Patch size (paper: 315 × 560).
        grayscale:      Convert to single-channel (matches paper).
        mean, std:      Normalisation statistics.
    """

    def __init__(
        self,
        crop_h: int = 315,
        crop_w: int = 560,
        grayscale: bool = True,
        mean: float = 0.485,
        std: float = 0.229,
    ):
        self.spatial = [
            RandomCropPair(crop_h, crop_w),
            RandomHorizontalFlipPair(p=0.5),
        ]
        self.photometric = [
            IndependentGaussianBlur(p=0.3),
            IndependentBrightnessContrast(p=0.5),
            IndependentGammaNoise(p=0.3),
        ]
        self.grayscale = grayscale
        self.mean = mean
        self.std  = std

    def _to_gray(self, frame: np.ndarray) -> np.ndarray:
        if self.grayscale and frame.ndim == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return frame

    def _to_tensor(self, frame: np.ndarray) -> torch.Tensor:
        frame = frame.astype(np.float32) / 255.0
        if frame.ndim == 2:
            frame = frame[np.newaxis]        # (1, H, W)
        else:
            frame = frame.transpose(2, 0, 1) # (C, H, W)
        t = torch.from_numpy(frame)
        return (t - self.mean) / self.std

    def __call__(self, frames: list) -> list:
        frames = [self._to_gray(f) for f in frames]
        for t in self.spatial:
            frames = t(frames)
        frames = [self._apply_photometric(f) for f in frames]
        return [self._to_tensor(f) for f in frames]

    def _apply_photometric(self, frame: np.ndarray) -> np.ndarray:
        for t in self.photometric:
            frame = t(frame)
        return frame


class EvalTransform:
    """
    Evaluation transform: center crop + grayscale + normalise.
    No random augmentation.
    """

    def __init__(
        self,
        crop_h: int = 315,
        crop_w: int = 560,
        grayscale: bool = True,
        mean: float = 0.485,
        std: float = 0.229,
    ):
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.grayscale = grayscale
        self.mean = mean
        self.std  = std

    def __call__(self, frames: list) -> list:
        out = []
        for f in frames:
            h, w = f.shape[:2]
            if self.grayscale and f.ndim == 3:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            # Resize if frame is too small
            if h < self.crop_h or w < self.crop_w:
                scale = max(self.crop_h / h, self.crop_w / w)
                f = cv2.resize(f, (int(w * scale) + 1, int(h * scale) + 1))
                h, w = f.shape[:2]
            top  = (h - self.crop_h) // 2
            left = (w - self.crop_w) // 2
            f = f[top: top + self.crop_h, left: left + self.crop_w]
            f = f.astype(np.float32) / 255.0
            if f.ndim == 2:
                f = f[np.newaxis]
            else:
                f = f.transpose(2, 0, 1)
            t = torch.from_numpy(f)
            out.append((t - self.mean) / self.std)
        return out
