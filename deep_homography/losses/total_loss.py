"""
losses/total_loss.py
--------------------
All loss components for multi-plane unsupervised deep homography estimation.

Loss taxonomy (referencing the paper review):

  L_recon    : Robust image-space reconstruction.
               Warps image A with each plane H, blends planes by the predicted
               target-space masks, and compares to image B with Charbonnier
               intensity + gradient penalties over valid warped pixels.

  L_triplet  : Legacy contrastive feature term. Disabled by default because
               maximising feature distance fights correspondence matching unless
               feature scale is otherwise constrained.

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
    geodesic_distance, H_compose, H_inverse, normalise_H
)
from deep_homography.models.plane_mask_head import mask_tv_loss, mask_entropy_loss


# ---------------------------------------------------------------------------
# Helper: robust image-space reconstruction
# ---------------------------------------------------------------------------

def denormalise_image(img: torch.Tensor, mean: float, std: float) -> torch.Tensor:
    """Map normalised image tensors back to [0, 1] for photometric losses."""
    return (img * std + mean).clamp(0.0, 1.0)


def charbonnier(x: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """Smooth L1-like penalty with stable gradients around zero."""
    return torch.sqrt(x * x + eps * eps)


def image_gradients(img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward finite differences in x and y."""
    dx = img[:, :, :, 1:] - img[:, :, :, :-1]
    dy = img[:, :, 1:, :] - img[:, :, :-1, :]
    return dx, dy


def masked_mean(value: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Average a per-pixel/per-channel value under a single-channel mask."""
    if mask.shape[1] == 1 and value.shape[1] != 1:
        mask = mask.expand(-1, value.shape[1], *mask.shape[2:])
    return (value * mask).sum() / (mask.sum() + eps)


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
    mean:     float = 0.485,
    std:      float = 0.229,
    gradient_weight: float = 0.25,
) -> torch.Tensor:
    """
    Multi-plane image reconstruction loss.

    Earlier revisions used features from the trainable matcher itself as the
    reconstruction target.  That is a weak anchor: the feature extractor can
    change its scale/semantics while the homography stays near identity.  This
    loss compares denormalised images directly, blends the K warped planes by
    their target-space masks, and ignores pixels that warp outside the source.
    """
    B, K, _, _ = masks.shape
    _, C, H, W = img_a.shape
    img_a_raw = denormalise_image(img_a, mean, std)
    img_b_raw = denormalise_image(img_b, mean, std)
    masks_full = F.interpolate(masks, size=(H, W), mode="bilinear", align_corners=False)

    warped_sum = torch.zeros_like(img_a_raw)
    weight_sum = torch.zeros(B, 1, H, W, device=img_a.device, dtype=img_a.dtype)
    ones = torch.ones(B, 1, H, W, device=img_a.device, dtype=img_a.dtype)

    for k in range(K):
        H_k = H_final[:, k]           # (B, 3, 3)
        mask_k = masks_full[:, k : k + 1]
        valid_k = stn(ones, H_k).clamp(0.0, 1.0)
        weight_k = mask_k * valid_k
        warped_sum = warped_sum + weight_k * stn(img_a_raw, H_k)
        weight_sum = weight_sum + weight_k

    warped_blend = warped_sum / weight_sum.clamp_min(1e-6)
    valid = (weight_sum > 0.05).to(img_a.dtype)

    photo = masked_mean(charbonnier(warped_blend - img_b_raw), valid)
    if gradient_weight <= 0:
        return photo

    wx = valid[:, :, :, 1:] * valid[:, :, :, :-1]
    wy = valid[:, :, 1:, :] * valid[:, :, :-1, :]
    warped_dx, warped_dy = image_gradients(warped_blend)
    target_dx, target_dy = image_gradients(img_b_raw)
    grad_x = masked_mean(charbonnier(warped_dx - target_dx), wx)
    grad_y = masked_mean(charbonnier(warped_dy - target_dy), wy)
    return photo + gradient_weight * (grad_x + grad_y)


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
    mean:        float = 0.485,
    std:         float = 0.229,
    gradient_weight: float = 0.25,
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
    weight_sum = 0.0

    for i in range(num_iters):
        weight = gamma ** (num_iters - i - 1)
        H_iter = torch.stack([H_preds_all[k][i] for k in range(K)], dim=1)  # (B, K, 3, 3)
        loss_i = reconstruction_loss(
            img_a, img_b, feats_a, feats_b, masks, H_iter, feat_extractor, stn,
            mean=mean, std=std, gradient_weight=gradient_weight,
        )
        weighted.append(weight * loss_i)
        weight_sum += weight

    return sum(weighted) / max(weight_sum, 1e-8)


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
        self.photometric_gradient_weight = lc.get("photometric_gradient_weight", 0.25)
        dc = cfg["data"]
        self.normalize_mean = dc.get("normalize_mean", 0.485)
        self.normalize_std = dc.get("normalize_std", 0.229)

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
            mean=self.normalize_mean, std=self.normalize_std,
            gradient_weight=self.photometric_gradient_weight,
        )
        l_recon_ba = sequence_loss(
            out_ba["H_preds"], img_b, img_a,
            out_ba["feat_a_fine"], out_ba["feat_b_fine"],
            out_ba["masks"], feat_extractor, stn, self.gamma,
            mean=self.normalize_mean, std=self.normalize_std,
            gradient_weight=self.photometric_gradient_weight,
        )
        losses["recon"] = self.lambda_recon * (l_recon_ab + l_recon_ba)

        # ---- 2. Contrastive triplet loss ----
        if self.lambda_triplet:
            l_contrast_ab = triplet_contrastive_loss(
                out_ab["feat_a_fine"], out_ab["feat_b_fine"]
            )
            l_contrast_ba = triplet_contrastive_loss(
                out_ba["feat_a_fine"], out_ba["feat_b_fine"]
            )
            # Disabled by default: maximising feature distance harms matching
            # unless the feature scale is tightly controlled.
            losses["triplet"] = -self.lambda_triplet * (l_contrast_ab + l_contrast_ba) / 2.0
        else:
            losses["triplet"] = img_a.new_tensor(0.0)

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
