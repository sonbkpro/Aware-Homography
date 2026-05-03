"""
iterative_refiner.py
--------------------
RAFT-inspired iterative GRU-based homography refinement.

The core idea: instead of a single-shot regression (the original paper),
we maintain a hidden state and iteratively update the estimated homography
using local correlation features evaluated at the CURRENT estimate.

At each iteration:
  1. Warp source features by the current H estimate.
  2. Look up correlation pyramid at the residual flow.
  3. Feed correlation + context features into a ConvGRU.
  4. GRU outputs a residual flow delta Δf.
  5. Update the flow: f ← f + Δf.
  6. Recompute H from the updated flow via weighted DLT.

This iterative refinement gives two key advantages over the original paper:
  - It handles LARGER BASELINES by progressively reducing residual error.
  - The GRU's hidden state accumulates information across iterations,
    functioning like a learned Lucas-Kanade optimizer.

References:
    RAFT: Recurrent All-Pairs Field Transforms (Teed & Deng, ECCV 2020)
    GMA: Global Motion Aggregation (Jiang et al., ICCV 2021)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional

from deep_homography.models.correlation import CorrPyramid
from deep_homography.utils.homography_utils import warp_image, HomographySTN


# ---------------------------------------------------------------------------
# Context encoder -- encodes I_a for GRU hidden state initialisation
# ---------------------------------------------------------------------------

class ContextEncoder(nn.Module):
    """
    Encodes the source image into a context feature map used to:
      (a) Initialise the GRU hidden state.
      (b) Concatenate with correlation features at each update step.

    A lightweight ResNet-style encoder at stride 8 (same resolution as the
    correlation pyramid's working level).

    Args:
        in_channels (int):      Input feature channels from FPN level3.
        hidden_dim (int):       GRU hidden state channels.
        context_dim (int):      Context feature channels fed each GRU step.
    """

    def __init__(
        self,
        in_channels: int = 128,
        hidden_dim: int = 128,
        context_dim: int = 128,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.context_dim = context_dim

        # Shared backbone (processes level3 features)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
        )
        # Head 1: initial hidden state
        self.hidden_head = nn.Conv2d(128, hidden_dim, 1)
        # Head 2: per-step context
        self.context_head = nn.Conv2d(128, context_dim, 1)

    def forward(
        self, feat_a: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_a: (B, C, H, W) level3 features of source image.
        Returns:
            hidden_init: (B, hidden_dim,  H, W) initial GRU hidden state.
            context:     (B, context_dim, H, W) static context per step.
        """
        x = self.body(feat_a)
        hidden_init = torch.tanh(self.hidden_head(x))
        context = F.relu(self.context_head(x))
        return hidden_init, context


# ---------------------------------------------------------------------------
# Separable ConvGRU update unit
# ---------------------------------------------------------------------------

class ConvGRU(nn.Module):
    """
    Convolutional GRU that processes 2D spatial feature maps.

    At each iteration it receives:
        h:   (B, hidden_dim, H, W) hidden state from previous iter.
        inp: (B, input_dim, H, W)  = [correlation_features | context]

    And outputs:
        h':  (B, hidden_dim, H, W) updated hidden state.
        δf:  (B, 2, H, W)          residual flow correction.

    Architecture: Separable (row then column) GRU gates for efficiency,
    as in RAFT's GRU but with a smaller kernel for 2D spatial maps.

    Args:
        hidden_dim (int): Hidden state channels.
        input_dim (int):  Input feature channels (corr + context).
    """

    def __init__(self, hidden_dim: int = 128, input_dim: int = 256):
        super().__init__()
        combined = hidden_dim + input_dim

        # Standard ConvGRU gates
        self.convz = nn.Conv2d(combined, hidden_dim, 3, padding=1)  # reset gate
        self.convr = nn.Conv2d(combined, hidden_dim, 3, padding=1)  # update gate
        self.convq = nn.Conv2d(combined, hidden_dim, 3, padding=1)  # candidate

        # Flow head: hidden_dim → 2-channel residual flow
        self.flow_head = nn.Sequential(
            nn.Conv2d(hidden_dim, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 2, 3, padding=1),
        )

        # Zero-init flow head so it starts as identity at iter 0
        nn.init.zeros_(self.flow_head[-1].weight)
        nn.init.zeros_(self.flow_head[-1].bias)

    def forward(
        self,
        h: torch.Tensor,    # (B, hidden_dim, H, W)
        x: torch.Tensor,    # (B, input_dim, H, W) = corr_feats + context
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            h_new: (B, hidden_dim, H, W) updated hidden state.
            delta_flow: (B, 2, H, W) residual flow.
        """
        hx = torch.cat([h, x], dim=1)    # (B, hidden+input, H, W)

        z = torch.sigmoid(self.convz(hx))          # update gate
        r = torch.sigmoid(self.convr(hx))          # reset gate
        q = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))  # candidate

        h_new = (1 - z) * h + z * q               # new hidden state
        delta_flow = self.flow_head(h_new)         # residual (B, 2, H, W)

        return h_new, delta_flow


# ---------------------------------------------------------------------------
# Motion encoder -- compresses correlation + flow into compact input for GRU
# ---------------------------------------------------------------------------

class MotionEncoder(nn.Module):
    """
    Compresses correlation lookup features and current flow into a
    fixed-size feature vector per spatial position.

    Args:
        corr_channels (int): Channels from CorrPyramid.lookup().
        flow_channels (int): 2 (u, v).
        out_channels (int):  Output channel count for GRU input.
    """

    def __init__(
        self,
        corr_channels: int,
        flow_channels: int = 2,
        out_channels: int = 128,
    ):
        super().__init__()
        # Correlation compression
        self.corr_enc = nn.Sequential(
            nn.Conv2d(corr_channels, 256, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 192, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        # Flow compression
        self.flow_enc = nn.Sequential(
            nn.Conv2d(flow_channels, 64, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 1),
            nn.ReLU(inplace=True),
        )
        self.out_conv = nn.Sequential(
            nn.Conv2d(192 + 32, out_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(
        self,
        corr_feats: torch.Tensor,   # (B, C_corr, H, W)
        flow: torch.Tensor,          # (B, 2, H, W)
    ) -> torch.Tensor:
        corr_out = self.corr_enc(corr_feats)
        flow_out = self.flow_enc(flow)
        return self.out_conv(torch.cat([corr_out, flow_out], dim=1))


# ---------------------------------------------------------------------------
# Main Iterative Refiner
# ---------------------------------------------------------------------------

class IterativeRefiner(nn.Module):
    """
    RAFT-style iterative homography refiner.

    Given initial features and an optional starting flow, performs `num_iters`
    GRU updates to progressively refine the homography estimate.

    The refiner works at 1/8 resolution (FPN level3) for efficiency.
    The final H is computed via DifferentiableDLT at 1/4 resolution (level2).

    Args:
        feat_dim (int):     Feature channels from FPN.
        hidden_dim (int):   GRU hidden state channels.
        context_dim (int):  Context encoder output channels.
        corr_levels (int):  Correlation pyramid levels.
        corr_radius (int):  Correlation lookup radius.
    """

    def __init__(
        self,
        feat_dim: int = 128,
        hidden_dim: int = 128,
        context_dim: int = 128,
        corr_levels: int = 4,
        corr_radius: int = 4,
    ):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.context_dim = context_dim
        self.corr_levels = corr_levels
        self.corr_radius = corr_radius

        # Correlation channels per level: (2r+1)^2
        corr_ch_per_level = (2 * corr_radius + 1) ** 2
        total_corr_ch = corr_levels * corr_ch_per_level  # e.g., 4 * 81 = 324

        # Sub-modules
        self.context_enc = ContextEncoder(feat_dim, hidden_dim, context_dim)
        self.motion_enc  = MotionEncoder(total_corr_ch, 2, 128)
        self.gru         = ConvGRU(hidden_dim, 128 + context_dim)

        # Warp utility
        self.stn = HomographySTN()

    def _init_flow(self, B: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Returns zero flow (identity initialisation)."""
        return torch.zeros(B, 2, H, W, device=device)

    def _flow_to_H(
        self,
        flow: torch.Tensor,   # (B, 2, H_feat, W_feat)
        mask: torch.Tensor,   # (B, 1, H_feat, W_feat)
        img_h: int,
        img_w: int,
        dlt_solver,           # DifferentiableDLT instance
    ) -> torch.Tensor:
        """Converts a flow field + mask to homography via DifferentiableDLT."""
        H = dlt_solver(flow, mask, img_h, img_w)   # (B, 3, 3)
        return H

    def forward(
        self,
        feat_a_coarse: torch.Tensor,  # (B, C, H/8,  W/8) for GRU
        feat_b_coarse: torch.Tensor,  # (B, C, H/8,  W/8)
        feat_a_fine:   torch.Tensor,  # (B, C, H/4,  W/4) for DLT
        feat_b_fine:   torch.Tensor,  # (B, C, H/4,  W/4)
        masks_k:       torch.Tensor,  # (B, 1, H/4, W/4)  plane k mask
        dlt_solver,                    # DifferentiableDLT instance
        num_iters:     int = 12,
        img_h:         int = 315,
        img_w:         int = 560,
    ) -> List[torch.Tensor]:
        """
        Performs iterative refinement for plane k.

        Args:
            feat_a_coarse, feat_b_coarse: Level3 (1/8) features for GRU.
            feat_a_fine, feat_b_fine:     Level2 (1/4) features for DLT.
            masks_k:   Plane k's mask at fine resolution.
            dlt_solver: DifferentiableDLT module.
            num_iters: Number of GRU update steps.
            img_h, img_w: Original patch dimensions.

        Returns:
            predictions: List of (B, 3, 3) homography matrices, one per
                         iteration. The last element is the final estimate.
                         All are used in the loss with exponential weighting.
        """
        B, C, Hc, Wc = feat_a_coarse.shape

        # ---- Initialise ----
        hidden, context = self.context_enc(feat_a_coarse)   # (B, D, Hc, Wc)
        flow = self._init_flow(B, Hc, Wc, feat_a_coarse.device)  # (B, 2, Hc, Wc)

        # ---- Build correlation pyramid (level3 resolution) ----
        corr_pyramid = CorrPyramid(feat_a_coarse, feat_b_coarse, self.corr_levels)

        predictions: List[torch.Tensor] = []

        for _ in range(num_iters):
            flow = flow.detach()  # stop gradient through flow for stability

            # ---- Look up correlation at current flow ----
            corr_feats = corr_pyramid.lookup(flow, self.corr_radius)  # (B, C_corr, Hc, Wc)

            # ---- Motion encoding ----
            motion = self.motion_enc(corr_feats, flow)  # (B, 128, Hc, Wc)
            inp = torch.cat([motion, context], dim=1)   # (B, 128+context_dim, Hc, Wc)

            # ---- GRU update ----
            hidden, delta_flow = self.gru(hidden, inp)   # delta_flow: (B, 2, Hc, Wc)
            flow = flow + delta_flow

            # ---- Upsample flow to fine resolution for DLT ----
            flow_fine = F.interpolate(flow, size=feat_a_fine.shape[-2:], mode="bilinear", align_corners=False)
            scale_x = feat_a_fine.shape[-1] / Wc
            scale_y = feat_a_fine.shape[-2] / Hc
            flow_fine[:, 0] *= scale_x
            flow_fine[:, 1] *= scale_y

            # ---- Downsample mask to match coarse flow if needed ----
            mask_fine = F.interpolate(masks_k, size=feat_a_fine.shape[-2:], mode="bilinear", align_corners=False)

            # ---- DLT solve ----
            H_k = self._flow_to_H(flow_fine, mask_fine, img_h, img_w, dlt_solver)
            predictions.append(H_k)

        return predictions  # list of (B, 3, 3), length = num_iters
