"""
plane_mask_head.py
------------------
Transformer-based K-plane mask predictor.

This module is the key architectural innovation over the original paper's
single binary mask m(·).  Instead of one mask that RANSAC-rejects or
attends, we predict K soft masks {M^k}_{k=1..K} that:

  1. Form a PARTITION OF UNITY: Σ_k M^k(p) = 1 at every pixel p.
     (Achieved by softmax over the K-dim at each spatial location.)

  2. Are SPATIALLY SMOOTH: penalised by a Total-Variation (TV) prior.
     This prevents degenerate per-pixel checkerboard assignments.

  3. Are MUTUALLY EXCLUSIVE (entropy-encouraged): each pixel should be
     "owned" by one plane, not spread uniformly.  An entropy regulariser
     pushes the softmax toward one-hot assignments.

  4. Each M^k is used to weight the DLT solve for homography H^k, so
     H^k is estimated primarily from the pixels that "belong" to plane k.

Architecture:
    Input:  Concatenated FPN features of both images + correlation summary.
    Layers: 2-layer Cross-Attention Transformer (lightweight).
    Output: K soft masks (B, K, H, W) — softmax over K at each pixel.

The transformer processes flattened spatial tokens, allowing global
context (e.g., "the sky is one plane") to inform mask assignments,
which ConvNet-only approaches cannot easily do.

Design choice for K:
    K=2 is the minimum useful setting (dominant plane + outliers).
    K=3 handles multi-plane scenes (façade + ground + moving object).
    K=1 reduces to the original paper's single-mask baseline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


# ---------------------------------------------------------------------------
# Positional encoding
# ---------------------------------------------------------------------------

class SinusoidalPositionalEncoding2D(nn.Module):
    """
    2D sinusoidal positional encoding for spatial transformer tokens.
    Adds a unique (x, y) encoding to each spatial position.

    Args:
        embed_dim (int): Must be divisible by 4.
        max_h, max_w:   Maximum feature map dimensions.
    """

    def __init__(self, embed_dim: int, max_h: int = 100, max_w: int = 200):
        super().__init__()
        assert embed_dim % 4 == 0, "embed_dim must be divisible by 4"
        half = embed_dim // 2

        # Y encoding (rows)
        pe_y = torch.zeros(max_h, half)
        pos_y = torch.arange(max_h).unsqueeze(1).float()
        div = torch.exp(
            torch.arange(0, half, 2).float() * (-math.log(10000.0) / half)
        )
        pe_y[:, 0::2] = torch.sin(pos_y * div)
        pe_y[:, 1::2] = torch.cos(pos_y * div)

        # X encoding (cols)
        pe_x = torch.zeros(max_w, half)
        pos_x = torch.arange(max_w).unsqueeze(1).float()
        pe_x[:, 0::2] = torch.sin(pos_x * div)
        pe_x[:, 1::2] = torch.cos(pos_x * div)

        self.register_buffer("pe_y", pe_y)  # (max_h, half)
        self.register_buffer("pe_x", pe_x)  # (max_w, half)
        self.embed_dim = embed_dim

    def forward(self, B: int, H: int, W: int) -> torch.Tensor:
        """
        Returns: (B, H*W, embed_dim) positional encodings.
        """
        # (H, half) repeated W times + (W, half) repeated H times → (H, W, embed_dim)
        enc_y = self.pe_y[:H].unsqueeze(1).expand(-1, W, -1)   # (H, W, half)
        enc_x = self.pe_x[:W].unsqueeze(0).expand(H, -1, -1)   # (H, W, half)
        enc = torch.cat([enc_y, enc_x], dim=-1)                 # (H, W, embed_dim)
        enc = enc.reshape(1, H * W, self.embed_dim).expand(B, -1, -1)
        return enc


# ---------------------------------------------------------------------------
# Lightweight Cross-Attention Block
# ---------------------------------------------------------------------------

class CrossAttentionBlock(nn.Module):
    """
    One block of: Self-Attention → Cross-Attention → FFN
    with pre-norm (LayerNorm before each sub-layer).

    Used to let query tokens (from img_a) attend to key/value tokens (from
    img_b's correlation features) to learn alignment-aware masks.

    Args:
        embed_dim (int): Token dimension.
        num_heads (int): Number of attention heads.
        mlp_ratio (float): FFN hidden dim multiplier.
        dropout (float): Dropout rate.
    """

    def __init__(
        self,
        embed_dim: int = 128,
        num_heads: int = 4,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        # Self-attention on source features
        self.norm1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # Cross-attention: source queries attend to correlation context
        self.norm2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim, num_heads, dropout=dropout, batch_first=True
        )
        # FFN
        self.norm3 = nn.LayerNorm(embed_dim)
        hidden = int(embed_dim * mlp_ratio)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        q: torch.Tensor,       # (B, N_q, D) source tokens
        kv: torch.Tensor,      # (B, N_kv, D) correlation context tokens
    ) -> torch.Tensor:
        # Self-attention
        x = q
        x2 = self.norm1(x)
        sa, _ = self.self_attn(x2, x2, x2)
        x = x + sa

        # Cross-attention
        x2 = self.norm2(x)
        ca, _ = self.cross_attn(x2, kv, kv)
        x = x + ca

        # FFN
        x = x + self.ffn(self.norm3(x))
        return x


# ---------------------------------------------------------------------------
# Main Plane Mask Head
# ---------------------------------------------------------------------------

class PlaneMaskHead(nn.Module):
    """
    K-plane partition-of-unity mask predictor.

    Inputs:
        feat_a: (B, C, H, W) features from image a (FPN level2 or level3).
        feat_b: (B, C, H, W) features from image b.
        corr_summary: Optional (B, C_corr, H, W) summary of correlation volume.

    Output:
        masks: (B, K, H, W) soft masks, softmax over K (partition of unity).

    Pipeline:
        1. Project feat_a + feat_b + corr_summary to embed_dim.
        2. Add sinusoidal 2D positional encodings.
        3. Two CrossAttentionBlock layers.
        4. Project to K channels + softmax.

    Args:
        in_channels (int):   Feature channels (C) from FPN.
        embed_dim (int):     Transformer hidden dim.
        num_heads (int):     Attention heads.
        num_layers (int):    Number of CrossAttentionBlock layers.
        num_planes (int):    K — number of planes.
        corr_channels (int): Channels in corr_summary (0 = not used).
    """

    def __init__(
        self,
        in_channels: int = 128,
        embed_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        num_planes: int = 2,
        corr_channels: int = 0,
    ):
        super().__init__()
        self.num_planes = num_planes
        self.embed_dim = embed_dim

        # Input projection: concatenate feat_a + feat_b [+ corr_summary]
        in_ch = in_channels * 2 + corr_channels
        self.input_proj = nn.Sequential(
            nn.Conv2d(in_ch, embed_dim, 1),
            nn.GroupNorm(8, embed_dim),
            nn.GELU(),
        )

        # Separate projection for correlation context (cross-attention KV)
        self.corr_proj = nn.Conv2d(
            max(corr_channels, in_channels), embed_dim, 1
        ) if corr_channels > 0 else None

        # Positional encoding
        self.pos_enc = SinusoidalPositionalEncoding2D(embed_dim)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            CrossAttentionBlock(embed_dim, num_heads)
            for _ in range(num_layers)
        ])

        # Output head: embed_dim → K plane logits
        self.out_head = nn.Linear(embed_dim, num_planes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Init output logits near zero → uniform mask at start of training
        nn.init.zeros_(self.out_head.weight)
        nn.init.zeros_(self.out_head.bias)

    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor,
        corr_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            feat_a:       (B, C, H, W)
            feat_b:       (B, C, H, W)
            corr_summary: (B, C_corr, H, W) or None

        Returns:
            masks: (B, K, H, W) in [0, 1], summing to 1 over K.
        """
        B, C, H, W = feat_a.shape

        # ---- Build input tokens (query) ----
        if corr_summary is not None:
            inp = torch.cat([feat_a, feat_b, corr_summary], dim=1)
        else:
            inp = torch.cat([feat_a, feat_b], dim=1)

        tokens = self.input_proj(inp)          # (B, embed_dim, H, W)
        tokens = tokens.flatten(2).permute(0, 2, 1)  # (B, H*W, embed_dim)

        # ---- Build context tokens (key/value) from correlation ----
        if corr_summary is not None and self.corr_proj is not None:
            ctx = self.corr_proj(corr_summary)           # (B, embed_dim, H, W)
        else:
            # Fall back to feat_b as context
            ctx = self.input_proj(
                torch.cat([feat_b, feat_b], dim=1)
                if corr_summary is None else inp
            )
        ctx = ctx.flatten(2).permute(0, 2, 1)            # (B, H*W, embed_dim)

        # ---- Add positional encodings ----
        pe = self.pos_enc(B, H, W).to(tokens.device)     # (B, H*W, embed_dim)
        tokens = tokens + pe
        ctx    = ctx    + pe

        # ---- Transformer blocks ----
        for block in self.blocks:
            tokens = block(tokens, ctx)

        # ---- Output logits → spatial masks ----
        logits = self.out_head(tokens)                    # (B, H*W, K)
        logits = logits.permute(0, 2, 1).reshape(B, self.num_planes, H, W)

        # Softmax over K: ensures partition of unity at every pixel
        masks = F.softmax(logits, dim=1)                  # (B, K, H, W)
        return masks


# ---------------------------------------------------------------------------
# Mask regularisation losses (used in losses/total_loss.py)
# ---------------------------------------------------------------------------

def mask_tv_loss(masks: torch.Tensor) -> torch.Tensor:
    """
    Total Variation smoothness loss on mask assignments.
    Encourages spatially coherent plane regions (penalises noisy boundaries).

    Args:
        masks: (B, K, H, W)
    Returns:
        Scalar TV loss.
    """
    diff_h = (masks[:, :, 1:, :] - masks[:, :, :-1, :]).abs().mean()
    diff_w = (masks[:, :, :, 1:] - masks[:, :, :, :-1]).abs().mean()
    return diff_h + diff_w


def mask_entropy_loss(masks: torch.Tensor) -> torch.Tensor:
    """
    Entropy regularisation: encourages "hard" (one-hot) assignments.
    Minimising entropy pushes each pixel to be fully assigned to one plane.

    H(masks) = -Σ_k M^k log(M^k)   averaged over pixels and batch.

    Note: we MINIMISE this, so it discourages uniform (maximum entropy) masks.

    Args:
        masks: (B, K, H, W) -- already softmax probabilities.
    Returns:
        Scalar mean entropy.
    """
    eps = 1e-8
    entropy = -(masks * (masks + eps).log()).sum(dim=1)   # (B, H, W)
    return entropy.mean()
