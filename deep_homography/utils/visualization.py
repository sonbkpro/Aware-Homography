"""
utils/visualization.py
----------------------
Visualization utilities for monitoring homography estimation.

Provides:
  - overlay_channels: Red/green ghost visualization (exactly as in the paper).
  - visualise_masks:  Colour-coded K-plane mask visualisation.
  - visualise_flow:   HSV flow visualisation (like RAFT).
  - log_to_tensorboard: Batch logger for TensorBoard.
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Optional, List

from deep_homography.utils.homography_utils import HomographySTN


# ---------------------------------------------------------------------------
# Paper's ghost overlay visualization
# ---------------------------------------------------------------------------

def overlay_channels(
    img_warped: torch.Tensor,   # (1, H, W) or (3, H, W) in [0, 1]
    img_target: torch.Tensor,   # (1, H, W) or (3, H, W) in [0, 1]
) -> np.ndarray:
    """
    Creates the red/green ghost overlay used in the paper (Fig.1, 6, 7).

    Mix: R channel = target, G+B channels = warped source.
    Misaligned pixels appear as red (target-only) or green (source-only) ghosts.

    Args:
        img_warped: Tensor in [0, 1].
        img_target: Tensor in [0, 1].

    Returns:
        RGB numpy array (H, W, 3) uint8.
    """
    def to_gray(t):
        if t.shape[0] == 3:
            t = 0.299 * t[0] + 0.587 * t[1] + 0.114 * t[2]
        else:
            t = t[0]
        return t.clamp(0, 1)

    w = np.nan_to_num(to_gray(img_warped).cpu().numpy(), nan=0.0)  # (H, W)
    t = np.nan_to_num(to_gray(img_target).cpu().numpy(), nan=0.0)  # (H, W)

    # R = target, G = warped, B = warped
    out = np.stack([t, w, w], axis=-1)     # (H, W, 3)
    out = (out * 255).clip(0, 255).astype(np.uint8)
    return out


# ---------------------------------------------------------------------------
# Mask visualisation
# ---------------------------------------------------------------------------

def visualise_masks(
    masks: torch.Tensor,   # (K, H, W) soft masks in [0, 1]
    img:   Optional[torch.Tensor] = None,  # (1, H, W) background image
) -> np.ndarray:
    """
    Colour-coded K-plane mask visualisation.

    Assigns a unique hue to each plane and blends with the background image.

    Args:
        masks: (K, H, W) partition-of-unity masks.
        img:   Optional background image (grayscale tensor).

    Returns:
        (H, W, 3) RGB uint8 image.
    """
    K, H, W = masks.shape
    palette = [
        (255,   0,   0),   # red    (plane 0)
        (  0, 200,   0),   # green  (plane 1)
        (  0,   0, 255),   # blue   (plane 2)
        (255, 165,   0),   # orange (plane 3)
    ]

    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for k in range(K):
        m = np.nan_to_num(masks[k].cpu().numpy(), nan=0.0)[:, :, np.newaxis]  # (H, W, 1)
        colour = np.array(palette[k % len(palette)], dtype=np.float32) / 255.0
        canvas += m * colour

    canvas = np.nan_to_num(canvas, nan=0.0)
    canvas = (canvas * 255).clip(0, 255).astype(np.uint8)

    # Blend with background
    if img is not None:
        bg = img[0].cpu().numpy()
        bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
        bg = (np.stack([bg, bg, bg], axis=-1) * 255).astype(np.uint8)
        canvas = cv2.addWeighted(bg, 0.4, canvas, 0.6, 0)

    return canvas


# ---------------------------------------------------------------------------
# Flow visualisation (HSV wheel)
# ---------------------------------------------------------------------------

def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Convert optical flow (H, W, 2) to HSV colour-wheel image (H, W, 3) uint8.
    Hue = direction, Saturation = 1, Value = magnitude (normalised).
    """
    u, v = flow[..., 0], flow[..., 1]
    magnitude = np.sqrt(u ** 2 + v ** 2)
    angle = np.arctan2(v, u)  # [-π, π]

    hue = np.nan_to_num((angle + np.pi) / (2 * np.pi) * 180, nan=0.0).astype(np.float32)
    sat = np.ones_like(hue) * 255
    val = np.nan_to_num(magnitude / (magnitude.max() + 1e-8) * 255, nan=0.0).astype(np.float32)

    hsv = np.stack([hue, sat, val], axis=-1).astype(np.uint8)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb


def visualise_flow(flow: torch.Tensor) -> np.ndarray:
    """
    Args:
        flow: (2, H, W) flow tensor.
    Returns:
        (H, W, 3) uint8 RGB flow image.
    """
    flow_np = flow.detach().cpu().permute(1, 2, 0).numpy()  # (H, W, 2)
    return flow_to_rgb(flow_np)


# ---------------------------------------------------------------------------
# TensorBoard logger
# ---------------------------------------------------------------------------

def log_to_tensorboard(
    writer,
    step:          int,
    losses:        dict,
    img_a:         torch.Tensor,       # (B, 1, H, W)
    img_b:         torch.Tensor,       # (B, 1, H, W)
    H_final:       torch.Tensor,       # (B, K, 3, 3)
    masks:         torch.Tensor,       # (B, K, H_f, W_f)
    flow_init:     torch.Tensor,       # (B, 2, H_f, W_f)
    num_vis:       int = 4,
    prefix:        str = "train",
):
    """
    Logs scalars and images to TensorBoard.

    Args:
        writer:    SummaryWriter instance.
        step:      Global training step.
        losses:    Dict of loss values.
        img_a/b:   Batch images.
        H_final:   Predicted homographies.
        masks:     Plane masks.
        flow_init: Initial soft-argmax flow.
        num_vis:   Number of samples to visualise.
        prefix:    'train' or 'val'.
    """
    # Scalars
    for name, val in losses.items():
        v = val.item() if isinstance(val, torch.Tensor) else float(val)
        writer.add_scalar(f"{prefix}/loss_{name}", v, step)

    # Images (first `num_vis` samples in batch)
    stn = HomographySTN()
    B = min(num_vis, img_a.shape[0])
    K = H_final.shape[1]

    # Select dominant-plane H for visualisation
    mask_support = masks.flatten(2).sum(dim=-1)   # (B, K)
    dom_idx = mask_support.argmax(dim=1)           # (B,)

    vis_overlays = []
    vis_masks    = []
    vis_flows    = []

    with torch.no_grad():
        for b in range(B):
            k = dom_idx[b].item()
            H_k = H_final[b : b + 1, k]                # (1, 3, 3)
            warped = stn(img_a[b : b + 1], H_k)[0]     # (1, H, W)

            # Denormalise images to [0, 1]
            img_a_vis = img_a[b].clamp(0, 1)
            img_b_vis = img_b[b].clamp(0, 1)
            warped_vis = warped.clamp(0, 1)

            overlay = overlay_channels(warped_vis, img_b_vis)
            vis_overlays.append(overlay)

            # Mask visualisation (upsample masks to image resolution)
            m_up = F.interpolate(
                masks[b : b + 1], size=img_a.shape[-2:], mode="bilinear", align_corners=False
            )[0]  # (K, H, W)
            vis_masks.append(visualise_masks(m_up, img_a[b]))

            # Flow
            f = F.interpolate(
                flow_init[b : b + 1], size=img_a.shape[-2:], mode="bilinear", align_corners=False
            )[0]  # (2, H, W)
            vis_flows.append(visualise_flow(f))

    # Stack and log as image grids
    def stack_to_tensor(imgs):
        arr = np.stack(imgs, axis=0)           # (B, H, W, 3)
        return torch.from_numpy(arr).permute(0, 3, 1, 2).float() / 255.0

    writer.add_images(f"{prefix}/overlay",   stack_to_tensor(vis_overlays), step)
    writer.add_images(f"{prefix}/masks",     stack_to_tensor(vis_masks),    step)
    writer.add_images(f"{prefix}/flow_init", stack_to_tensor(vis_flows),    step)
