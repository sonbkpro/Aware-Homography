"""
homography_utils.py
-------------------
Core geometric utilities for homography estimation.

Includes:
  - HomographySTN:    Differentiable image warping via Spatial Transformer Network.
  - warp_image:       Convenience wrapper.
  - H_inverse:        Batched matrix inversion with gradient.
  - H_compose:        Batched matrix multiplication (H2 ∘ H1).
  - H_to_4corners:    Convert H to 4-corner displacement (for evaluation).
  - corners_to_H:     Convert 4-corner correspondence to H (DLT).
  - normalise_H:      Canonical scale normalisation.
  - matrix_log_approx: Padé-approximant matrix logarithm for geodesic loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple


# ---------------------------------------------------------------------------
# Spatial Transformer Network (STN) warping
# ---------------------------------------------------------------------------

class HomographySTN(nn.Module):
    """
    Differentiable image/feature-map warping via a 3×3 homography matrix.

    Uses F.grid_sample with bilinear interpolation and zero-padding for
    out-of-bounds regions.  The warping is fully differentiable w.r.t. H.

    This is the same mechanism as used in the original paper (Jaderberg 2015),
    but here we support both image and feature-map warping.

    Usage:
        stn = HomographySTN()
        I_warped = stn(I_a, H_ab)   # warp I_a to align with I_b
    """

    def __init__(self, mode: str = "bilinear", padding_mode: str = "zeros"):
        super().__init__()
        self.mode = mode
        self.padding_mode = padding_mode

    def forward(
        self,
        x: torch.Tensor,     # (B, C, H, W) image or feature map
        H: torch.Tensor,     # (B, 3, 3) homography
    ) -> torch.Tensor:
        """
        Warp x by H: each output pixel q samples from H^{-1}(q) in x.

        Note: We warp in the INVERSE direction (backward warp) which avoids
        holes and is compatible with F.grid_sample.

        Args:
            x: (B, C, H, W) source image/features.
            H: (B, 3, 3) homography mapping source → target.

        Returns:
            (B, C, H, W) warped output.
        """
        B, C, height, width = x.shape
        device, dtype = x.device, x.dtype

        # Build normalised target pixel grid in range [-1, 1]
        grid = self._make_grid(B, height, width, device, dtype)  # (B, H*W, 3)

        # Invert H for backward warp: target pixel q → source pixel H^{-1} q
        H_inv = torch.linalg.inv(H)  # (B, 3, 3)

        # Apply H_inv to grid: (B, 3, H*W)
        # First convert normalised grid to pixel coordinates
        grid_px = self._norm_to_pixel(grid, height, width)  # (B, H*W, 3)
        src_px = torch.bmm(grid_px, H_inv.permute(0, 2, 1))  # (B, H*W, 3)

        # Perspective divide
        src_px = src_px / src_px[..., 2:3].clamp(min=1e-8)

        # Convert back to normalised [-1, 1] for grid_sample
        src_norm = self._pixel_to_norm(src_px[..., :2], height, width)  # (B, H*W, 2)
        sample_grid = src_norm.reshape(B, height, width, 2)              # (B, H, W, 2)

        return F.grid_sample(
            x, sample_grid,
            mode=self.mode,
            padding_mode=self.padding_mode,
            align_corners=True,
        )

    @staticmethod
    def _make_grid(
        B: int, H: int, W: int,
        device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        """
        Builds a homogeneous normalised grid of shape (B, H*W, 3).
        Coordinates run from (-1,-1) to (1,1).
        """
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        ones  = torch.ones_like(grid_x)
        grid  = torch.stack([grid_x, grid_y, ones], dim=-1)  # (H, W, 3)
        grid  = grid.reshape(1, H * W, 3).expand(B, -1, -1)
        return grid

    @staticmethod
    def _norm_to_pixel(
        pts_norm: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Convert normalised [-1,1] coords (x, y, 1) to pixel coords.
        pts_norm: (B, N, 3)  →  return (B, N, 3)
        """
        pts = pts_norm.clone()
        pts[..., 0] = (pts[..., 0] + 1.0) * (W - 1) / 2.0
        pts[..., 1] = (pts[..., 1] + 1.0) * (H - 1) / 2.0
        return pts

    @staticmethod
    def _pixel_to_norm(
        pts_px: torch.Tensor, H: int, W: int
    ) -> torch.Tensor:
        """
        Convert pixel coords (x, y) to normalised [-1, 1].
        pts_px: (B, N, 2)  →  return (B, N, 2)
        """
        pts = pts_px.clone()
        pts[..., 0] = 2.0 * pts[..., 0] / (W - 1) - 1.0
        pts[..., 1] = 2.0 * pts[..., 1] / (H - 1) - 1.0
        return pts


def warp_image(
    x: torch.Tensor,
    H: torch.Tensor,
    mode: str = "bilinear",
    padding_mode: str = "zeros",
) -> torch.Tensor:
    """
    Convenience wrapper for HomographySTN warping.

    Args:
        x: (B, C, H, W) source image or feature map.
        H: (B, 3, 3) homography.
    Returns:
        Warped (B, C, H, W).
    """
    stn = HomographySTN(mode=mode, padding_mode=padding_mode)
    return stn(x, H)


# ---------------------------------------------------------------------------
# Homography arithmetic
# ---------------------------------------------------------------------------

def H_inverse(H: torch.Tensor) -> torch.Tensor:
    """
    Batched differentiable matrix inversion.

    Args:
        H: (B, 3, 3)
    Returns:
        H_inv: (B, 3, 3)
    """
    return torch.linalg.inv(H)


def H_compose(H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
    """
    Compose two homographies: H2 ∘ H1  (apply H1 first, then H2).

    For the triangle consistency: H_{ac} = H_{bc} ∘ H_{ab}.

    Args:
        H1, H2: (B, 3, 3)
    Returns:
        (B, 3, 3)
    """
    H = torch.bmm(H2, H1)
    # Canonical normalisation
    H = H / H[:, 2:3, 2:3].clamp(min=1e-8)
    return H


def normalise_H(H: torch.Tensor) -> torch.Tensor:
    """
    Normalise homography to canonical scale: H[2,2] = 1.

    Args:
        H: (B, 3, 3)
    Returns:
        (B, 3, 3)
    """
    return H / H[:, 2:3, 2:3].clamp(min=1e-8)


# ---------------------------------------------------------------------------
# 4-corner parameterisation (evaluation / visualisation)
# ---------------------------------------------------------------------------

def H_to_4corners(
    H: torch.Tensor,
    patch_h: int = 315,
    patch_w: int = 560,
) -> torch.Tensor:
    """
    Compute displacements of 4 patch corners under homography H.

    Used to compute the corner-point L2 error for quantitative evaluation,
    exactly as in the original paper.

    Args:
        H:        (B, 3, 3) homography.
        patch_h, patch_w: Patch dimensions.

    Returns:
        delta: (B, 4, 2)  displacement vectors for TL, TR, BR, BL.
               delta[i] = H(corner_i) - corner_i.
    """
    B = H.shape[0]
    h, w = patch_h, patch_w

    # 4 corners in homogeneous coordinates: (4, 3)
    corners = torch.tensor(
        [[0, 0, 1], [w - 1, 0, 1], [w - 1, h - 1, 1], [0, h - 1, 1]],
        dtype=H.dtype, device=H.device,
    ).float()  # (4, 3)

    corners = corners.unsqueeze(0).expand(B, -1, -1)  # (B, 4, 3)

    # Apply H: (B, 4, 3) @ (B, 3, 3)^T = (B, 4, 3)
    warped = torch.bmm(corners, H.permute(0, 2, 1))   # (B, 4, 3)

    # Perspective divide
    warped = warped / warped[..., 2:3].clamp(min=1e-8)

    # Displacement
    delta = warped[..., :2] - corners[..., :2]         # (B, 4, 2)
    return delta


def warp_points(
    pts: torch.Tensor,   # (B, N, 2) in pixel (x, y)
    H: torch.Tensor,     # (B, 3, 3)
) -> torch.Tensor:
    """
    Apply homography H to a set of 2D points.

    Args:
        pts: (B, N, 2) source pixel coordinates.
        H:   (B, 3, 3) homography.

    Returns:
        (B, N, 2) warped pixel coordinates.
    """
    B, N, _ = pts.shape
    ones = torch.ones(B, N, 1, device=pts.device, dtype=pts.dtype)
    pts_h = torch.cat([pts, ones], dim=-1)              # (B, N, 3)
    warped = torch.bmm(pts_h, H.permute(0, 2, 1))       # (B, N, 3)
    warped = warped / warped[..., 2:3].clamp(min=1e-8)
    return warped[..., :2]


# ---------------------------------------------------------------------------
# Matrix logarithm (for geodesic consistency loss)
# ---------------------------------------------------------------------------

def matrix_log_approx(M: torch.Tensor, num_terms: int = 10) -> torch.Tensor:
    """
    Approximate matrix logarithm via Taylor series:
        log(M) ≈ log(I + (M - I)) = Σ_{n=1}^{N} (-1)^{n+1} (M-I)^n / n

    The Taylor series converges only when ||M - I||_F < 1.  For larger
    deviations (common at the start of training) the series diverges to
    Inf/NaN.  We therefore fall back to the first-order approximation
    (M - I) — equivalent to the Frobenius consistency metric — whenever
    ||M - I||_F >= 0.5.  This fallback is always finite and differentiable.

    Args:
        M:         (B, 3, 3) matrix (should be close to identity for log).
        num_terms: Number of Taylor terms (used only in the convergent regime).

    Returns:
        (B, 3, 3) approximate log(M).
    """
    I = torch.eye(3, device=M.device, dtype=M.dtype).unsqueeze(0)
    X = M - I                           # (B, 3, 3)

    result = torch.zeros_like(M)
    power = X.clone()
    for n in range(1, num_terms + 1):
        result = result + ((-1) ** (n + 1)) / n * power
        power = torch.bmm(power, X)

    # Sanitise any Inf/NaN produced by diverged power iterations before blending.
    result = torch.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    # Per-sample fallback: when ||X||_F >= 0.5 the series is unreliable;
    # use the first-order approximation X (always finite, zero-grad through mask).
    norm_X = X.detach().flatten(1).norm(dim=-1)          # (B,)
    large  = (norm_X >= 0.5).float().view(-1, 1, 1)      # (B, 1, 1)
    return large * X + (1.0 - large) * result


def geodesic_distance(H1: torch.Tensor, H2: torch.Tensor) -> torch.Tensor:
    """
    Approximate geodesic distance between two homographies on GL(3).

    d(H1, H2) = ||log(H1^{-1} H2)||_F

    In practice for inverse consistency: H2 = I, H1 = H_ab H_ba,
    so d = ||log(H_ab H_ba)||_F.

    Args:
        H1, H2: (B, 3, 3)
    Returns:
        (B,) geodesic distances.
    """
    H1_inv = torch.linalg.inv(H1)
    M = torch.bmm(H1_inv, H2)          # (B, 3, 3)
    # Normalise
    M = normalise_H(M)
    log_M = matrix_log_approx(M)
    return log_M.flatten(1).norm(dim=1) # (B,)
