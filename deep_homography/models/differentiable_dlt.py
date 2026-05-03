"""
differentiable_dlt.py
---------------------
Differentiable Weighted Direct Linear Transformation (DLT) homography solver.

This is the geometric heart of our architectural improvement over Zhang et al.
(2020). Instead of regressing 8 numbers from a GlobalAveragePool→FC head,
we:
  1. Extract dense correspondences from the correlation volume (soft-argmax).
  2. Weight each correspondence by the plane mask M^k.
  3. Solve the 8-DOF homography via a differentiable SVD.

Because torch.linalg.svd is fully differentiable, gradients flow back through
the DLT solve into both the mask (M^k) and the feature extractor (through
the soft-argmax correspondences).

Mathematical derivation:
    For N correspondences (x_i, y_i) → (x_i', y_i') with weights w_i,
    the homography H satisfies:

        [x_i']   [h1 h2 h3] [x_i]
        [y_i'] ~ [h4 h5 h6] [y_i]
        [ 1  ]   [h7 h8 h9] [ 1 ]

    In homogeneous coordinates: x' × (H x) = 0 yields two equations per
    correspondence (the third is redundant):

        [-x  -y  -1   0   0   0  x'x  x'y  x'] h = 0
        [ 0   0   0  -x  -y  -1  y'x  y'y  y'] h = 0

    The weighted system is M = Σ_i w_i A_i^T A_i,  where A_i is the 2×9
    matrix above.  H = reshape(V[-1]), where V is from SVD(M).

    Tikhonov regularisation: M ← M + ε I   prevents rank deficiency when
    weights collapse to zero in early training.

Reference:
    DeTone et al. (2016) -- DLT parameterisation
    Brachmann & Rother (2019) -- Differentiable RANSAC / DLT
    Ranftl & Koltun (2020)    -- Deep Fundamental Matrix Estimation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Coordinate normalisation (improves DLT numerical conditioning)
# ---------------------------------------------------------------------------

def normalize_points(pts: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Hartley normalisation: translate centroid to origin, scale so mean
    distance to origin is sqrt(2).  Critical for numerical stability of DLT.

    Args:
        pts: (B, N, 2) in pixel coordinates.
    Returns:
        pts_norm: (B, N, 2) normalised points.
        T:        (B, 3, 3) normalisation transform such that pts_norm = T pts.
    """
    B, N, _ = pts.shape
    mean = pts.mean(dim=1, keepdim=True)                          # (B, 1, 2)
    centered = pts - mean                                          # (B, N, 2)
    scale = centered.norm(dim=-1).mean(dim=1) / math.sqrt(2)      # (B,)
    scale = scale.clamp(min=1e-8)

    pts_norm = centered / scale.view(B, 1, 1)

    # Compose into 3×3 transform T
    T = torch.zeros(B, 3, 3, device=pts.device, dtype=pts.dtype)
    T[:, 0, 0] = 1.0 / scale
    T[:, 1, 1] = 1.0 / scale
    T[:, 0, 2] = -mean[:, 0, 0] / scale
    T[:, 1, 2] = -mean[:, 0, 1] / scale
    T[:, 2, 2] = 1.0

    return pts_norm, T


def build_dlt_matrix(
    src: torch.Tensor,    # (B, N, 2) normalised
    dst: torch.Tensor,    # (B, N, 2) normalised
    weights: torch.Tensor # (B, N)    soft weights in [0,1]
) -> torch.Tensor:
    """
    Builds the weighted DLT coefficient matrix M = Σ_i w_i A_i^T A_i.

    Returns: M of shape (B, 9, 9).
    """
    B, N, _ = src.shape
    x,  y  = src[..., 0], src[..., 1]    # (B, N)
    xp, yp = dst[..., 0], dst[..., 1]    # (B, N)
    ones  = torch.ones_like(x)
    zeros = torch.zeros_like(x)

    # Row 1 of A_i: [-x, -y, -1,  0,  0,  0, x'x, x'y, x']
    row1 = torch.stack(
        [-x, -y, -ones, zeros, zeros, zeros, xp * x, xp * y, xp],
        dim=-1,
    )  # (B, N, 9)

    # Row 2 of A_i: [ 0,  0,  0, -x, -y, -1, y'x, y'y, y']
    row2 = torch.stack(
        [zeros, zeros, zeros, -x, -y, -ones, yp * x, yp * y, yp],
        dim=-1,
    )  # (B, N, 9)

    # Weight rows: w_i applied to BOTH rows of correspondence i
    w = weights.unsqueeze(-1)          # (B, N, 1)
    row1_w = row1 * w                  # (B, N, 9)
    row2_w = row2 * w                  # (B, N, 9)

    # Stack: A_w = [row1_w; row2_w] → (B, 2N, 9)
    A_w = torch.cat([row1_w, row2_w], dim=1)  # (B, 2N, 9)

    # M = A_w^T A_w → (B, 9, 9)
    M = torch.bmm(A_w.permute(0, 2, 1), A_w)
    return M


def solve_dlt(M: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Solves M h = 0 for the homography vector h via SVD.

    The solution is the right singular vector corresponding to the smallest
    singular value (i.e., the last column of V in M = U Σ V^T).

    Tikhonov regularisation M ← M + ε I is applied to prevent rank collapse
    in early training when weights are near-uniform.

    Args:
        M:   (B, 9, 9) DLT matrix.
        eps: Regularisation strength.
    Returns:
        H:   (B, 3, 3) homography matrix (unnormalised).
    """
    # Tikhonov regularisation
    M = M + eps * torch.eye(9, device=M.device, dtype=M.dtype).unsqueeze(0)

    # SVD: M = U Σ V^T  →  h = V[:, -1] (column of smallest singular value)
    try:
        _, _, Vt = torch.linalg.svd(M)  # Vt: (B, 9, 9)
    except RuntimeError:
        # Fallback: add stronger regularisation
        M = M + 1e-3 * torch.eye(9, device=M.device, dtype=M.dtype).unsqueeze(0)
        _, _, Vt = torch.linalg.svd(M)

    # Last row of Vt = last right singular vector
    h = Vt[:, -1, :]    # (B, 9)  – note: torch returns V^T, so rows are V^T cols
    H = h.reshape(-1, 3, 3)
    return H


class DifferentiableDLT(nn.Module):
    """
    Full differentiable weighted DLT pipeline:

        1. Sample N correspondences from the flow field on a regular grid.
        2. Extract mask weights at the sampled locations.
        3. Hartley-normalise source and target points.
        4. Build and solve the 9×9 weighted DLT system via SVD.
        5. Denormalise: H = T_dst^{-1} H_norm T_src.
        6. Divide by H[2,2] so H is in canonical scale.

    Args:
        num_points (int): Number of sampled correspondences N.
                          Must be ≥ 4 for a solvable system.
                          512 gives good coverage at 315×560.
        eps (float):      Tikhonov regularisation for SVD stability.
    """

    def __init__(self, num_points: int = 512, eps: float = 1e-6):
        super().__init__()
        self.num_points = num_points
        self.eps = eps

    def _sample_grid_points(
        self,
        H_patch: int,
        W_patch: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Creates a regular sub-sampled grid of source coordinates.
        We use a fixed grid (not random) so the DLT matrix has
        good spatial coverage and is deterministic at test time.

        Returns: (N, 2) tensor of (x, y) pixel coordinates.
        """
        n = int(self.num_points ** 0.5)
        # Evenly spaced along height and width (avoid borders by 5%)
        margin_h = int(H_patch * 0.05)
        margin_w = int(W_patch * 0.05)
        ys = torch.linspace(margin_h, H_patch - margin_h - 1, n, device=device, dtype=dtype)
        xs = torch.linspace(margin_w, W_patch - margin_w - 1, n, device=device, dtype=dtype)
        # Meshgrid → (N, 2): (x, y) order
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        pts = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)  # (n², 2)
        return pts  # shape (n², 2)

    def _sample_at_coords(
        self,
        field: torch.Tensor,   # (B, C, H, W)
        coords: torch.Tensor,  # (N, 2) absolute pixel (x, y)
    ) -> torch.Tensor:
        """
        Bilinear sample from field at given pixel coordinates.

        Returns: (B, N, C)
        """
        B, C, H, W = field.shape
        N = coords.shape[0]

        # Normalise to [-1, 1]
        norm_x = 2.0 * coords[:, 0] / (W - 1) - 1.0
        norm_y = 2.0 * coords[:, 1] / (H - 1) - 1.0
        grid = torch.stack([norm_x, norm_y], dim=-1)         # (N, 2)
        grid = grid.unsqueeze(0).unsqueeze(0).expand(B, 1, N, 2)  # (B, 1, N, 2)

        sampled = F.grid_sample(field, grid, align_corners=True, mode="bilinear")
        # sampled: (B, C, 1, N) → (B, N, C)
        return sampled.squeeze(2).permute(0, 2, 1)

    def forward(
        self,
        flow: torch.Tensor,   # (B, 2, H, W) current flow (src → dst)
        mask: torch.Tensor,   # (B, 1, H, W) plane mask weights ∈ [0,1]
        img_h: int,
        img_w: int,
    ) -> torch.Tensor:
        """
        Args:
            flow:  (B, 2, H, W) soft-argmax flow from CorrVolume.
            mask:  (B, 1, H, W) plane mask for this plane k.
            img_h, img_w: Original image dimensions (for coordinate scaling).

        Returns:
            H: (B, 3, 3) estimated homography matrix.
        """
        B = flow.shape[0]
        _, _, Hf, Wf = flow.shape  # feature map resolution (may differ from img)

        # ---- Sample grid points in feature space ----
        src_pts_feat = self._sample_grid_points(Hf, Wf, flow.device, flow.dtype)  # (N, 2)
        N = src_pts_feat.shape[0]

        # ---- Sample flow at grid points → get dst positions in feature space ----
        flow_at_pts = self._sample_at_coords(flow, src_pts_feat)           # (B, N, 2)
        src_pts = src_pts_feat.unsqueeze(0).expand(B, -1, -1)              # (B, N, 2)
        dst_pts = src_pts + flow_at_pts                                     # (B, N, 2)

        # ---- Scale to image coordinates ----
        scale_x = img_w / Wf
        scale_y = img_h / Hf
        scale = torch.tensor([scale_x, scale_y], device=flow.device, dtype=flow.dtype)
        src_img = src_pts * scale.unsqueeze(0).unsqueeze(0)  # (B, N, 2)
        dst_img = dst_pts * scale.unsqueeze(0).unsqueeze(0)  # (B, N, 2)

        # ---- Sample mask weights at grid points ----
        weights = self._sample_at_coords(mask, src_pts_feat)  # (B, N, 1)
        weights = weights.squeeze(-1)                          # (B, N)
        weights = F.softmax(weights * 10.0, dim=1)            # sharpen + normalise to sum=1

        # ---- Hartley normalisation ----
        src_norm, T_src = normalize_points(src_img)  # (B, N, 2), (B, 3, 3)
        dst_norm, T_dst = normalize_points(dst_img)  # (B, N, 2), (B, 3, 3)

        # ---- Build and solve DLT ----
        M = build_dlt_matrix(src_norm, dst_norm, weights)   # (B, 9, 9)
        H_norm = solve_dlt(M, eps=self.eps)                 # (B, 3, 3)

        # ---- Denormalise: H = T_dst^{-1} H_norm T_src ----
        T_dst_inv = torch.linalg.inv(T_dst)                 # (B, 3, 3)
        H = torch.bmm(T_dst_inv, torch.bmm(H_norm, T_src)) # (B, 3, 3)

        # ---- Canonical scale: divide by H[2,2] ----
        scale_factor = H[:, 2, 2].unsqueeze(-1).unsqueeze(-1).clamp(min=1e-8)
        H = H / scale_factor

        return H


class FourPointDLT(nn.Module):
    """
    Lightweight DLT from exactly 4 corner point displacements.
    This exactly reproduces the parameterisation in DeTone et al. (2016)
    and Zhang et al. (2020) -- kept as a baseline module.

    The 4 corners of the patch are displaced by learnable offsets to yield
    the homography.  No flow field is required.

    Args:
        patch_h, patch_w: Patch dimensions.
    """

    def __init__(self, patch_h: int = 315, patch_w: int = 560):
        super().__init__()
        self.patch_h = patch_h
        self.patch_w = patch_w

    def forward(self, delta_corners: torch.Tensor) -> torch.Tensor:
        """
        Args:
            delta_corners: (B, 8) corner displacement offsets (u1,v1,...,u4,v4).
        Returns:
            H: (B, 3, 3) homography matrix.
        """
        B = delta_corners.shape[0]
        h, w = self.patch_h, self.patch_w

        # Source corners: TL, TR, BR, BL (pixel coordinates)
        src = torch.tensor(
            [[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]],
            dtype=delta_corners.dtype, device=delta_corners.device,
        ).unsqueeze(0).expand(B, -1, -1).float()  # (B, 4, 2)

        dst = src + delta_corners.reshape(B, 4, 2)  # (B, 4, 2)

        # Equal weights (no masking in 4-point mode)
        weights = torch.ones(B, 4, device=delta_corners.device, dtype=delta_corners.dtype)

        src_norm, T_src = normalize_points(src)
        dst_norm, T_dst = normalize_points(dst)
        M = build_dlt_matrix(src_norm, dst_norm, weights)
        H_norm = solve_dlt(M)
        T_dst_inv = torch.linalg.inv(T_dst)
        H = torch.bmm(T_dst_inv, torch.bmm(H_norm, T_src))
        H = H / H[:, 2:3, 2:3].clamp(min=1e-8)
        return H
