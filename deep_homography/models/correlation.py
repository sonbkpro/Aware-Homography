"""
correlation.py
--------------
All-pairs feature correlation volume, following RAFT (Teed & Deng, ECCV 2020)
but adapted for homography estimation.

Key idea: Instead of regressing 8 numbers directly from concatenated features
(the original paper's h(·)), we build a 4D cost volume that captures the
similarity of every source pixel to every target pixel. This gives the
network explicit, geometrically meaningful information about correspondences
-- the root cause of the original paper's failure on large-baseline scenes.

Architecture:
    CorrVolume: All-pairs dot product → 4D tensor (B, H_a, W_a, H_b, W_b).
    CorrPyramid: Pool CorrVolume at multiple scales to aggregate context.
    CorrLookup:  For a given (u, v) flow field, sample the pyramid at a
                 local radius to produce a per-pixel correlation feature.

The CorrPyramid + CorrLookup pattern exactly mirrors RAFT's approach and
enables the GRU-based iterative refiner in iterative_refiner.py to operate
on a fixed-size local correlation neighbourhood at each update step.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class CorrVolume:
    """
    All-pairs feature correlation.

    Computes the inner-product similarity between every pair of feature
    vectors (one from source, one from target) and stores the result as a
    4D tensor.  Stored as a plain Python object (not nn.Module) because it
    is re-computed every forward pass and holds no learnable parameters.

    Shape convention:
        Input features:  (B, C, H, W)
        Output corr:     (B, H, W, H, W)   (flattened target HW stored last)
                         → reshaped to (B*H*W, 1, H, W) for pooling.

    Memory note: at resolution H/4 × W/4 (e.g., 64×64 for a 256×256 patch)
    this is B × 64 × 64 × 64 × 64 ≈ B × 16M elements at float32 → ~64 MB
    for B=8.  We therefore operate at H/8 × W/8 for the coarsest volume and
    use the H/4 level only for the final DLT correspondence extraction.
    """

    def __init__(self, feat_a: torch.Tensor, feat_b: torch.Tensor):
        """
        Args:
            feat_a, feat_b: (B, C, H, W) L2-normalized feature maps.
        """
        B, C, H, W = feat_a.shape
        # Flatten spatial dimensions for batch matrix multiply
        fa = feat_a.view(B, C, H * W)                  # (B, C, N)
        fb = feat_b.view(B, C, H * W)                  # (B, C, N)
        # All-pairs dot product; divide by sqrt(C) for stable softmax
        corr = torch.bmm(fa.permute(0, 2, 1), fb) / (C ** 0.5)  # (B, N, N)
        # Reshape to spatial: (B, H, W, H, W)
        self.corr = corr.view(B, H, W, H, W)
        self.H, self.W = H, W

    def __getitem__(self, level: int) -> torch.Tensor:
        """
        Returns the correlation volume at a given pooling level.
        Level 0 = full resolution; level k = 2^k average-pooled target.

        Returns: (B * H_src * W_src, 1, H_tgt >> level, W_tgt >> level)
        """
        corr = self.corr
        B, H, W, Ht, Wt = corr.shape
        # Reshape target dims for pooling
        corr_flat = corr.reshape(B * H * W, 1, Ht, Wt)
        if level > 0:
            corr_flat = F.avg_pool2d(corr_flat, kernel_size=2 ** level, stride=2 ** level)
        return corr_flat


class CorrPyramid:
    """
    Multi-scale correlation pyramid.

    Stores `num_levels` average-pooled versions of the correlation volume
    so that CorrLookup can retrieve multi-scale local windows efficiently.

    Args:
        feat_a, feat_b: (B, C, H, W) feature maps.
        num_levels (int): Number of pyramid levels (paper: 4).
    """

    def __init__(
        self, feat_a: torch.Tensor, feat_b: torch.Tensor, num_levels: int = 4
    ):
        self.num_levels = num_levels
        # L2-normalize features for numerically stable dot products
        fa_norm = F.normalize(feat_a, dim=1)
        fb_norm = F.normalize(feat_b, dim=1)
        self.volume = CorrVolume(fa_norm, fb_norm)
        self.B, _, self.H, self.W = feat_a.shape
        # Pre-cache all levels
        self._levels: List[torch.Tensor] = [
            self.volume[lvl] for lvl in range(num_levels)
        ]

    def lookup(self, flow: torch.Tensor, radius: int = 4) -> torch.Tensor:
        """
        Look up a local correlation neighbourhood around the current flow
        estimate at each source pixel.

        Args:
            flow: (B, 2, H, W) current flow estimate (in pixel units).
            radius: Half-window size for neighbourhood lookup.

        Returns:
            corr_feats: (B, num_levels * (2r+1)^2, H, W) flattened
                        per-pixel correlation features.
        """
        B, _, H, W = flow.shape
        out_feats = []

        for lvl, corr_lvl in enumerate(self._levels):
            # corr_lvl: (B*H*W, 1, Ht, Wt)
            scale = 2 ** lvl
            Ht = self.H // scale
            Wt = self.W // scale

            # Scaled flow
            flow_lvl = flow / scale  # (B, 2, H, W)

            # Build sampling grid: for each source pixel (i, j), sample
            # a (2r+1)×(2r+1) target window centred at
            # (j + flow_u, i + flow_v).  The base source coordinate is
            # essential: without it every pixel samples around the same
            # top-left target neighbourhood, which makes the update nearly
            # independent of the actual image pair.
            r = radius
            offset_y, offset_x = torch.meshgrid(
                torch.arange(-r, r + 1, device=flow.device, dtype=flow.dtype),
                torch.arange(-r, r + 1, device=flow.device, dtype=flow.dtype),
                indexing="ij",
            )
            grid_offsets = torch.stack([offset_x, offset_y], dim=-1)  # x/y order

            ys, xs = torch.meshgrid(
                torch.arange(H, device=flow.device, dtype=flow.dtype),
                torch.arange(W, device=flow.device, dtype=flow.dtype),
                indexing="ij",
            )
            base = torch.stack([xs, ys], dim=-1)  # (H, W, 2), x/y order

            # flow_lvl: (B, 2, H, W) → (B, H, W, 2)
            flow_perm = flow_lvl.permute(0, 2, 3, 1)  # (B, H, W, 2)
            coords = base.unsqueeze(0) / scale + flow_perm
            coords_flat = coords.reshape(B * H * W, 1, 1, 2)

            # Offsets broadcast to (B*H*W, 2r+1, 2r+1, 2)
            offsets = grid_offsets.unsqueeze(0)  # (1, 2r+1, 2r+1, 2)
            sample_pts = coords_flat + offsets  # (B*H*W, 2r+1, 2r+1, 2)

            # Normalise to [-1, 1]
            sample_pts[..., 0] = 2.0 * sample_pts[..., 0] / (Wt - 1) - 1.0
            sample_pts[..., 1] = 2.0 * sample_pts[..., 1] / (Ht - 1) - 1.0
            sample_pts = sample_pts.clamp(-1.2, 1.2)

            # Sample: (B*H*W, 1, 2r+1, 2r+1)
            sampled = F.grid_sample(
                corr_lvl, sample_pts, align_corners=True, mode="bilinear"
            )
            # Reshape: (B, H, W, (2r+1)^2)
            sampled = sampled.squeeze(1).reshape(B, H, W, (2 * r + 1) ** 2)
            out_feats.append(sampled)

        # Concatenate all levels: (B, H, W, num_levels*(2r+1)^2)
        corr_feats = torch.cat(out_feats, dim=-1)
        # → (B, num_levels*(2r+1)^2, H, W) for Conv2d input
        return corr_feats.permute(0, 3, 1, 2).contiguous()


class SoftArgmaxCorrespondence(nn.Module):
    """
    Extracts dense soft-argmax correspondences from a correlation volume.

    For each source pixel p_i, computes the expected target position:
        p_i' = Σ_{j} softmax(corr[i, :]) · pos_j

    This gives a differentiable, dense flow field (u, v) for every source
    pixel, which is then fed to the differentiable DLT solver.

    Args:
        temperature (float): Softmax temperature.  Lower = harder assignment.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(
        self, feat_a: torch.Tensor, feat_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_a: (B, C, H, W) source features.
            feat_b: (B, C, H, W) target features.
        Returns:
            flow: (B, 2, H, W) flow field in pixel units (u=horizontal, v=vertical).
        """
        B, C, H, W = feat_a.shape
        # L2 normalize
        fa = F.normalize(feat_a, dim=1).view(B, C, H * W)  # (B, C, N)
        fb = F.normalize(feat_b, dim=1).view(B, C, H * W)  # (B, C, N)

        # All-pairs similarity: (B, N_src, N_tgt)
        corr = torch.bmm(fa.permute(0, 2, 1), fb) / (C ** 0.5)

        # Softmax over target positions (each source pixel → distribution over targets)
        attn = F.softmax(corr / self.temperature, dim=-1)  # (B, N_src, N_tgt)

        # Build target coordinate grid
        ys, xs = torch.meshgrid(
            torch.arange(H, dtype=torch.float32, device=feat_a.device),
            torch.arange(W, dtype=torch.float32, device=feat_a.device),
            indexing="ij",
        )
        # pos_tgt: (N_tgt, 2) → (x, y)
        pos_tgt = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1)  # (N, 2)
        pos_tgt = pos_tgt.unsqueeze(0).expand(B, -1, -1)  # (B, N, 2)

        # Expected target positions: (B, N_src, 2)
        expected = torch.bmm(attn, pos_tgt)

        # Build source coordinate grid
        pos_src = pos_tgt.clone()  # (B, N_src, 2)

        # Flow = expected_target - source_position
        flow = (expected - pos_src).permute(0, 2, 1)  # (B, 2, N_src)
        flow = flow.reshape(B, 2, H, W)

        return flow
