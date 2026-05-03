"""
utils/metrics.py
----------------
Evaluation metrics for homography estimation, matching the quantitative
protocol of Zhang et al. (ECCV 2020):

  1. Corner L2 error (mean Euclidean distance between warped corners and GT).
  2. Point L2 error (mean distance over 6-8 manually labeled correspondences).
  3. Inlier percentage (fraction of matched points within threshold T pixels).

All metrics are computed per-sample and per-category if category labels given.
"""

import torch
import numpy as np
from typing import Optional, List, Tuple
from deep_homography.utils.homography_utils import warp_points, H_to_4corners


# ---------------------------------------------------------------------------
# Corner-point L2 error (cheap proxy for full homography error)
# ---------------------------------------------------------------------------

def corner_l2_error(
    H_pred: torch.Tensor,      # (B, 3, 3) predicted homography
    H_gt:   torch.Tensor,      # (B, 3, 3) ground-truth homography
    patch_h: int = 315,
    patch_w: int = 560,
) -> torch.Tensor:
    """
    Mean L2 distance between predicted and GT corner displacements.

    Error = (1/4) Σ_{corners} ||H_pred(corner) - H_gt(corner)||_2

    Args:
        H_pred, H_gt: (B, 3, 3) homographies.
        patch_h, patch_w: Patch dimensions.

    Returns:
        (B,) per-sample corner L2 errors.
    """
    delta_pred = H_to_4corners(H_pred, patch_h, patch_w)  # (B, 4, 2)
    delta_gt   = H_to_4corners(H_gt,   patch_h, patch_w)  # (B, 4, 2)
    dist = (delta_pred - delta_gt).norm(dim=-1)            # (B, 4)
    return dist.mean(dim=1)                                # (B,)


# ---------------------------------------------------------------------------
# Point correspondence L2 error (paper's primary metric)
# ---------------------------------------------------------------------------

def point_l2_error(
    H_pred:   torch.Tensor,    # (B, 3, 3)
    gt_points: torch.Tensor,   # (B, N, 2, 2) → [src_xy, dst_xy]
) -> torch.Tensor:
    """
    Mean L2 distance between predicted warped source points and labeled GT points.

    For each of the N labeled correspondences (src_i, dst_i):
        error_i = ||H_pred(src_i) - dst_i||_2

    This exactly matches the evaluation protocol of Zhang et al. (2020).

    Args:
        H_pred:    (B, 3, 3) predicted homography.
        gt_points: (B, N, 2, 2) tensor where:
                   gt_points[:, :, 0, :] = source (x, y) in pixel coords
                   gt_points[:, :, 1, :] = target (x, y) in pixel coords

    Returns:
        (B,) per-sample mean point L2 errors.
    """
    src_pts = gt_points[:, :, 0, :]   # (B, N, 2)
    dst_pts = gt_points[:, :, 1, :]   # (B, N, 2)

    # Warp source points under predicted H
    warped = warp_points(src_pts, H_pred)    # (B, N, 2)
    dist   = (warped - dst_pts).norm(dim=-1) # (B, N)
    return dist.mean(dim=1)                  # (B,)


# ---------------------------------------------------------------------------
# Inlier percentage (paper's robustness metric)
# ---------------------------------------------------------------------------

def inlier_percentage(
    H_pred:    torch.Tensor,   # (B, 3, 3)
    gt_points: torch.Tensor,   # (B, N, 2, 2)
    threshold: float = 3.0,    # Pixel threshold (paper: 3 pixels)
) -> torch.Tensor:
    """
    Percentage of GT correspondences warped within `threshold` pixels.

    Args:
        H_pred:    (B, 3, 3)
        gt_points: (B, N, 2, 2)
        threshold: Inlier distance threshold in pixels.

    Returns:
        (B,) per-sample inlier percentages in [0, 100].
    """
    src_pts = gt_points[:, :, 0, :]
    dst_pts = gt_points[:, :, 1, :]
    warped  = warp_points(src_pts, H_pred)
    dist    = (warped - dst_pts).norm(dim=-1)       # (B, N)
    inliers = (dist < threshold).float().mean(dim=1) * 100.0  # (B,)
    return inliers


# ---------------------------------------------------------------------------
# Multi-plane: select best plane homography per sample
# ---------------------------------------------------------------------------

def select_best_plane_H(
    H_all:     torch.Tensor,    # (B, K, 3, 3)
    gt_points: torch.Tensor,    # (B, N, 2, 2)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    For each sample, select the plane k* that minimises point L2 error.

    This is the oracle upper bound for multi-plane performance.
    In practice, we use the dominant plane (highest total mask weight).

    Returns:
        H_best:   (B, 3, 3) best homography per sample.
        best_idx: (B,) plane indices.
    """
    B, K = H_all.shape[:2]
    errors = []
    for k in range(K):
        e = point_l2_error(H_all[:, k], gt_points)  # (B,)
        errors.append(e)
    errors = torch.stack(errors, dim=1)              # (B, K)
    best_idx = errors.argmin(dim=1)                  # (B,)
    H_best = H_all[torch.arange(B, device=H_all.device), best_idx]  # (B, 3, 3)
    return H_best, best_idx


def select_dominant_plane_H(
    H_all:   torch.Tensor,    # (B, K, 3, 3)
    masks:   torch.Tensor,    # (B, K, H_f, W_f)
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Select plane k* = argmax_k (Σ_pixels M^k) — the plane with the most
    mask support.  This is the inference-time plane selection strategy.

    Returns:
        H_dominant: (B, 3, 3) dominant plane homography.
        dom_idx:    (B,) dominant plane indices.
    """
    B, K = H_all.shape[:2]
    mask_support = masks.flatten(2).sum(dim=-1)   # (B, K)
    dom_idx = mask_support.argmax(dim=1)           # (B,)
    H_dominant = H_all[torch.arange(B, device=H_all.device), dom_idx]
    return H_dominant, dom_idx


# ---------------------------------------------------------------------------
# Batch evaluation loop (called from evaluate.py)
# ---------------------------------------------------------------------------

class HomographyEvaluator:
    """
    Accumulates metrics over the evaluation dataset and computes summaries.

    Usage:
        evaluator = HomographyEvaluator(threshold=3.0)
        for batch in val_loader:
            evaluator.update(H_pred, batch.get("gt_points"))
        results = evaluator.summary()
    """

    def __init__(self, threshold: float = 3.0):
        self.threshold = threshold
        self._errors:   List[float] = []
        self._inliers:  List[float] = []

    def reset(self):
        self._errors.clear()
        self._inliers.clear()

    def update(
        self,
        H_pred:    torch.Tensor,              # (B, 3, 3) or (B, K, 3, 3)
        masks:     Optional[torch.Tensor],    # (B, K, H, W) if K>1
        gt_points: Optional[torch.Tensor],    # (B, N, 2, 2) or None
        patch_h:   int = 315,
        patch_w:   int = 560,
    ):
        """Update metrics with one batch."""
        # If multi-plane, select dominant plane for evaluation
        if H_pred.dim() == 4:
            assert masks is not None, "masks required for multi-plane evaluation"
            H_pred, _ = select_dominant_plane_H(H_pred, masks)

        if gt_points is not None:
            err = point_l2_error(H_pred, gt_points)      # (B,)
            inl = inlier_percentage(H_pred, gt_points, self.threshold)  # (B,)
            self._errors.extend(err.detach().cpu().tolist())
            self._inliers.extend(inl.detach().cpu().tolist())

    def summary(self) -> dict:
        """Return summary statistics."""
        if not self._errors:
            return {}
        errors  = np.array(self._errors)
        inliers = np.array(self._inliers)
        return {
            "mean_error":          float(errors.mean()),
            "median_error":        float(np.median(errors)),
            "mean_inlier_pct":     float(inliers.mean()),
            "num_samples":         len(errors),
        }
