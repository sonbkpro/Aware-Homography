"""
losses/total_loss.py
--------------------
All loss components for multi-plane unsupervised deep homography estimation.

Loss taxonomy (referencing the paper review):

  L_recon    : Per-plane feature-space reconstruction (extends Eq.4 of original paper).
               Uses learned features (not pixel intensity) — same as original paper.
               But extends to K planes with partition-of-unity mask weighting.

  L_triplet  : Contrastive triplet loss (Eq.5-6 of original paper, extended to K planes).
               Minimises ||M^k ⊙ (F_a_warped - F_b)||_1 while maximising ||F_a - F_b||_1.

  L_geo      : Geodesic inverse consistency on SL(3).
               Replaces the Frobenius-norm ||H_ab H_ba - I||^2_F (Eq.6, μ term) with
               geodesic distance ||log(H_ab H_ba)||_F — more principled on the manifold.

  L_triangle : Triangle consistency across 3 frames.
               ||log(H_ac  (H_bc ∘ H_ab)^{-1})||_F  for each plane k.
               This term is entirely NEW and exploits temporal structure ignored by the
               original paper.

  L_mask_tv  : Total variation smoothness on plane masks.
  L_mask_ent : Entropy regularisation (encourage hard plane assignments).

  RAFT-style sequence loss: Each plane's homography is predicted at every GRU
  iteration. We apply exponentially increasing weights γ^{N-i} to earlier iterates
  and γ^0=1 to the last, following RAFT's training recipe.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from deep_homography.utils.homography_utils import (
    warp_image, geodesic_distance, H_compose, H_inverse, normalise_H
)
from deep_homography.models.plane_mask_head import mask_tv_loss, mask_entropy_loss


# ---------------------------------------------------------------------------
# Helper: masked feature L1 reconstruction
# ---------------------------------------------------------------------------

def masked_feature_l1(
    feat_warped: torch.Tensor,   # (B, C, H, W)
    feat_target: torch.Tensor,   # (B, C, H, W)
    mask_w:      torch.Tensor,   # (B, 1, H, W) mask of warped source
    mask_t:      torch.Tensor,   # (B, 1, H, W) mask of target
) -> torch.Tensor:
    """
    Normalised masked L1 feature loss (Eq.4 generalised):

        Ln = Σ_i (M_w · M_t) · ||F_w - F_t||_1  /  Σ_i (M_w · M_t)

    The joint mask M_w · M_t ensures we only compare pixels that BOTH the
    warped source and the target agree are reliable (inlier).
    """
    joint_mask = mask_w * mask_t              # (B, 1, H, W)
    diff = (feat_warped - feat_target).abs()  # (B, C, H, W)
    weighted = joint_mask * diff              # (B, C, H, W)
    denom = joint_mask.sum() + 1e-8
    return weighted.sum() / denom


def masked_feature_l1_parts(
    feat_warped: torch.Tensor,
    feat_target: torch.Tensor,
    mask_w: torch.Tensor,
    mask_t: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return numerator and denominator for globally normalised mask L1."""
    joint_mask = mask_w * mask_t
    diff = (feat_warped - feat_target).abs()
    return (joint_mask * diff).sum(), joint_mask.sum()


# ---------------------------------------------------------------------------
# 1. Per-plane reconstruction loss
# ---------------------------------------------------------------------------

def reconstruction_loss(
    img_a:    torch.Tensor,        # (B, 1, H, W) source image
    img_b:    torch.Tensor,        # (B, 1, H, W) target image
    feats_a:  torch.Tensor,        # (B, C, H_f, W_f) source features
    feats_b:  torch.Tensor,        # (B, C, H_f, W_f) target features
    masks:    torch.Tensor,        # (B, K, H_f, W_f) plane masks
    H_final:  torch.Tensor,        # (B, K, 3, 3) final homographies per plane
    feat_extractor: nn.Module,     # Feature extractor (shared, frozen during loss eval)
    stn:      nn.Module,           # HomographySTN
) -> torch.Tensor:
    """
    Per-plane masked feature reconstruction loss.

    For each plane k:
      1. Warp I_a by H^k.
      2. Extract features of warped I_a.
      3. Compute masked L1 between warped features and target features.
      4. Weight by plane k's mask.

    The total loss is globally normalised across K planes.  This avoids a
    degenerate shortcut where a zero-support plane contributes zero loss and
    lowers the average simply by collapsing.
    If K=1 this reduces exactly to Eq.4 of the original paper.
    """
    B, K, _, _ = masks.shape
    numerator = masks.new_tensor(0.0)
    denominator = masks.new_tensor(0.0)

    for k in range(K):
        H_k = H_final[:, k]           # (B, 3, 3)
        mask_k = masks[:, k : k + 1]  # (B, 1, H_f, W_f)

        # Warp source image (full resolution for feature extraction)
        img_a_warped = stn(img_a, H_k)   # (B, 1, H, W)

        # Extract features of warped source (reuse the shared extractor)
        feats_a_warped = feat_extractor(img_a_warped)["level2"]  # (B, C, H_f, W_f)

        # Warp source mask to get M'_a
        mask_k_up = F.interpolate(mask_k, size=img_a.shape[-2:], mode="bilinear", align_corners=False)
        mask_k_warped = stn(mask_k_up, H_k)                          # (B, 1, H, W)
        mask_k_warped_feat = F.interpolate(
            mask_k_warped, size=feats_a_warped.shape[-2:], mode="bilinear", align_corners=False
        )  # (B, 1, H_f, W_f)

        # Forward reconstruction: warped source vs target
        num_k, den_k = masked_feature_l1_parts(
            feats_a_warped, feats_b, mask_k_warped_feat, mask_k
        )
        numerator = numerator + num_k
        denominator = denominator + den_k

    return numerator / (denominator + 1e-8)


# ---------------------------------------------------------------------------
# 2. Triplet contrastive loss (generalised to K planes)
# ---------------------------------------------------------------------------

def triplet_contrastive_loss(
    feats_a: torch.Tensor,   # (B, C, H_f, W_f) un-warped source features
    feats_b: torch.Tensor,   # (B, C, H_f, W_f) target features
    lambda_: float = 2.0,
) -> torch.Tensor:
    """
    Contrastive term: maximise ||F_a - F_b||_1 (before warping).

    This prevents the feature extractor from collapsing to all-zero maps
    (the trivial solution). Identical to Eq.5 of the original paper, but
    now applied to features that are richer (multi-channel, deeper extractor).

    The caller minimises (-lambda_ * L_contrast), i.e. maximises L_contrast.
    We return +L_contrast; the negative sign is handled in total_loss().
    """
    return (feats_a - feats_b).abs().mean()


# ---------------------------------------------------------------------------
# 3. Geodesic inverse consistency loss (replaces Frobenius ||H H^{-1} - I||²)
# ---------------------------------------------------------------------------

def geodesic_inverse_consistency(
    H_ab: torch.Tensor,    # (B, 3, 3)
    H_ba: torch.Tensor,    # (B, 3, 3)
) -> torch.Tensor:
    """
    Geodesic inverse consistency:

        L_geo = ||log(H_ab H_ba)||_F

    In the original paper Eq.6, this term was the Frobenius ||H_ab H_ba - I||²_F.
    The geodesic version is more principled because it respects the Riemannian
    metric on GL(3) and is scale-invariant.

    For K planes, we compute per-plane and average.
    """
    # H_ab H_ba should equal I if consistent
    product = torch.bmm(H_ab, H_ba)     # (B, 3, 3)
    product = normalise_H(product)
    # geodesic_distance(product, I) = ||log(I^{-1} product)||_F = ||log(product)||_F
    I = torch.eye(3, device=H_ab.device, dtype=H_ab.dtype).unsqueeze(0).expand_as(product)
    dist = geodesic_distance(I, product)      # (B,)
    return dist.mean()


def multi_plane_inverse_consistency(
    H_ab_all: torch.Tensor,   # (B, K, 3, 3)
    H_ba_all: torch.Tensor,   # (B, K, 3, 3)
) -> torch.Tensor:
    """
    Per-plane geodesic inverse consistency, averaged over K planes.
    """
    K = H_ab_all.shape[1]
    losses = [geodesic_inverse_consistency(H_ab_all[:, k], H_ba_all[:, k]) for k in range(K)]
    return sum(losses) / K


# ---------------------------------------------------------------------------
# 4. Triangle consistency loss (NEW -- exploits 3-frame temporal structure)
# ---------------------------------------------------------------------------

def triangle_consistency_loss(
    H_ab_all: torch.Tensor,   # (B, K, 3, 3)  a → b
    H_bc_all: torch.Tensor,   # (B, K, 3, 3)  b → c
    H_ac_all: torch.Tensor,   # (B, K, 3, 3)  a → c  (direct)
) -> torch.Tensor:
    """
    Triangle consistency: H_ac ≈ H_bc ∘ H_ab.

    Loss: ||log( H_ac^{-1}  (H_bc ∘ H_ab) )||_F for each plane k.

    This term is the key new regulariser that exploits temporal structure.
    It provides a free supervision signal: if H_ab and H_bc are estimated
    correctly, their composition should equal H_ac.  Violations indicate
    accumulated drift that the network must correct.

    For K planes, averaged over K.
    """
    B, K, _, _ = H_ab_all.shape
    I = torch.eye(3, device=H_ab_all.device, dtype=H_ab_all.dtype)
    I = I.unsqueeze(0).expand(B, -1, -1)

    plane_losses = []
    for k in range(K):
        H_ab = H_ab_all[:, k]   # (B, 3, 3)
        H_bc = H_bc_all[:, k]   # (B, 3, 3)
        H_ac = H_ac_all[:, k]   # (B, 3, 3)

        # Composed: H_bc ∘ H_ab
        H_composed = H_compose(H_ab, H_bc)   # (B, 3, 3)

        # Triangle residual: H_ac^{-1} (H_bc ∘ H_ab) should equal I
        H_ac_inv = H_inverse(H_ac)
        residual = torch.bmm(H_ac_inv, H_composed)   # (B, 3, 3)
        residual = normalise_H(residual)

        dist = geodesic_distance(I, residual)         # (B,)
        plane_losses.append(dist.mean())

    return sum(plane_losses) / K


# ---------------------------------------------------------------------------
# 5. RAFT-style sequence loss (exponential weighting over iterates)
# ---------------------------------------------------------------------------

def sequence_loss(
    H_preds_all: List[List[torch.Tensor]],  # [K][num_iters] of (B, 3, 3)
    img_a:       torch.Tensor,              # (B, 1, H, W)
    img_b:       torch.Tensor,              # (B, 1, H, W)
    feats_a:     torch.Tensor,              # (B, C, H_f, W_f)
    feats_b:     torch.Tensor,              # (B, C, H_f, W_f)
    masks:       torch.Tensor,              # (B, K, H_f, W_f)
    feat_extractor: nn.Module,
    stn:         nn.Module,
    gamma:       float = 0.85,
) -> torch.Tensor:
    """
    Exponentially weighted loss over GRU iterates, following RAFT.

    For iteration i out of N total:  weight = γ^{N-i}
    Early iterates get lower weight; final iterate gets weight 1.0.

    This provides learning signal at every step of the GRU rollout,
    which significantly stabilises training versus only supervising the last.
    """
    K = len(H_preds_all)
    num_iters = len(H_preds_all[0])
    weighted = []

    for i in range(num_iters):
        weight = gamma ** (num_iters - i - 1)
        H_iter = torch.stack([H_preds_all[k][i] for k in range(K)], dim=1)  # (B, K, 3, 3)
        loss_i = reconstruction_loss(
            img_a, img_b, feats_a, feats_b, masks, H_iter, feat_extractor, stn
        )
        weighted.append(weight * loss_i)

    return sum(weighted)


def mask_balance_loss(masks: torch.Tensor) -> torch.Tensor:
    """
    Penalise total collapse to a single plane.

    Entropy regularisation makes masks crisp, but by itself it also rewards the
    trivial all-pixels-one-plane solution.  A small batch-level balance term
    keeps every plane alive long enough to specialise.
    """
    B, K, _, _ = masks.shape
    if K == 1:
        return masks.new_tensor(0.0)
    support = masks.flatten(2).mean(dim=-1)       # (B, K), sums to 1 per sample
    mean_support = support.mean(dim=0)            # (K,)
    target = torch.full_like(mean_support, 1.0 / K)
    return (mean_support - target).pow(2).mean()


# ---------------------------------------------------------------------------
# Master loss function
# ---------------------------------------------------------------------------

class TotalLoss(nn.Module):
    """
    Weighted combination of all loss terms.

    Hyperparameters (from config):
        lambda_recon:      Weight for reconstruction loss.
        lambda_triplet:    Weight for contrastive term (paper's λ=2.0).
        lambda_geo:        Weight for geodesic consistency.
        lambda_triangle:   Weight for triangle consistency.
        lambda_mask_smooth: TV weight for mask smoothness.
        lambda_mask_entropy: Entropy weight for hard assignments.
        gamma:             RAFT exponential sequence weight.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        lc = cfg["loss"]
        self.lambda_recon         = lc["lambda_recon"]
        self.lambda_triplet       = lc["lambda_triplet"]
        self.lambda_geo           = lc["lambda_geo"]
        self.lambda_triangle      = lc["lambda_triangle"]
        self.lambda_mask_smooth   = lc["lambda_mask_smooth"]
        self.lambda_mask_entropy  = lc["lambda_mask_entropy"]
        self.lambda_mask_balance  = lc.get("lambda_mask_balance", 0.0)
        self.gamma                = lc["gamma"]

        self.stn = None   # set by caller after HomographySTN is created

    def forward(
        self,
        triplet_out:    Dict,        # from model.forward_triplet()
        img_a:          torch.Tensor,
        img_b:          torch.Tensor,
        img_c:          torch.Tensor,
        feat_extractor: nn.Module,
        stn:            nn.Module,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses given the full triplet model output.

        Args:
            triplet_out: Output dict from MultiPlaneHomographyNet.forward_triplet().
            img_a/b/c:   Raw image tensors.
            feat_extractor: The model's feature extractor module.
            stn:         HomographySTN warp module.

        Returns:
            dict with individual loss values + 'total' key.
        """
        out_ab = triplet_out["ab"]
        out_ba = triplet_out["ba"]
        out_bc = triplet_out["bc"]
        out_ac = triplet_out["ac"]

        losses = {}

        # ---- 1. Sequence reconstruction loss (a→b and b→a) ----
        l_recon_ab = sequence_loss(
            out_ab["H_preds"], img_a, img_b,
            out_ab["feat_a_fine"], out_ab["feat_b_fine"],
            out_ab["masks"], feat_extractor, stn, self.gamma,
        )
        l_recon_ba = sequence_loss(
            out_ba["H_preds"], img_b, img_a,
            out_ba["feat_a_fine"], out_ba["feat_b_fine"],
            out_ba["masks"], feat_extractor, stn, self.gamma,
        )
        losses["recon"] = self.lambda_recon * (l_recon_ab + l_recon_ba)

        # ---- 2. Contrastive triplet loss ----
        l_contrast_ab = triplet_contrastive_loss(
            out_ab["feat_a_fine"], out_ab["feat_b_fine"]
        )
        l_contrast_ba = triplet_contrastive_loss(
            out_ba["feat_a_fine"], out_ba["feat_b_fine"]
        )
        # We MAXIMISE these (subtract in total), per paper Eq.6 (−λL term)
        losses["triplet"] = -self.lambda_triplet * (l_contrast_ab + l_contrast_ba) / 2.0

        # ---- 3. Geodesic inverse consistency ----
        H_ab_final = out_ab["H_final"]   # (B, K, 3, 3)
        H_ba_final = out_ba["H_final"]
        losses["geo"] = self.lambda_geo * multi_plane_inverse_consistency(
            H_ab_final, H_ba_final
        )

        # ---- 4. Triangle consistency ----
        H_bc_final = out_bc["H_final"]
        H_ac_final = out_ac["H_final"]
        losses["triangle"] = self.lambda_triangle * triangle_consistency_loss(
            H_ab_final, H_bc_final, H_ac_final
        )

        # ---- 5. Mask regularisation (on a→b masks only for efficiency) ----
        masks_ab = out_ab["masks"]   # (B, K, H_f, W_f)
        losses["mask_tv"]  = self.lambda_mask_smooth  * mask_tv_loss(masks_ab)
        losses["mask_ent"] = self.lambda_mask_entropy * mask_entropy_loss(masks_ab)
        losses["mask_balance"] = self.lambda_mask_balance * mask_balance_loss(masks_ab)

        # ---- Total ----
        losses["total"] = sum(v for v in losses.values())

        return losses
