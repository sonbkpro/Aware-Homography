"""
homography_net.py
-----------------
MultiPlaneHomographyNet -- the full model integrating all sub-modules.

This class orchestrates:
  1. Feature extraction (weight-shared for I_a and I_b).
  2. K-plane mask prediction.
  3. Soft-argmax correspondence extraction.
  4. Differentiable DLT per plane.
  5. Iterative GRU refinement per plane.

It exposes a clean forward() that returns all intermediate quantities
needed by the loss functions in losses/total_loss.py.

Two-stage training protocol (reproducing the original paper's strategy):
  Stage 1: 'warmup_mode=True'   → attention mask role disabled (G_β = F_β).
           Train for ~60k iterations so DLT + GRU learn a stable baseline.
  Stage 2: 'warmup_mode=False'  → enable attention mask weighting of features
           for the homography estimator (G^k_β = F_β ⊙ M^k_β).

This staged approach prevents early mask collapse (a well-known pathology
where the mask learns to zero out everything to minimise reconstruction loss).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from deep_homography.models.feature_extractor  import MultiScaleFeatureExtractor
from deep_homography.models.correlation         import CorrPyramid, SoftArgmaxCorrespondence
from deep_homography.models.plane_mask_head     import PlaneMaskHead
from deep_homography.models.differentiable_dlt  import DifferentiableDLT
from deep_homography.models.iterative_refiner   import IterativeRefiner
from deep_homography.utils.homography_utils     import HomographySTN, warp_image


class MultiPlaneHomographyNet(nn.Module):
    """
    Full multi-plane unsupervised deep homography estimation network.

    Args:
        backbone (str):       Backbone architecture for feature extractor.
        feat_dim (int):       FPN output channels.
        use_dino (bool):      Whether to fuse DINOv2 features.
        num_planes (int):     K — number of planes.
        hidden_dim (int):     GRU hidden state channels.
        context_dim (int):    Context encoder channels.
        corr_levels (int):    Correlation pyramid levels.
        corr_radius (int):    Correlation lookup radius.
        mask_embed_dim (int): Transformer hidden dim for mask head.
        mask_num_heads (int): Attention heads in mask head.
        mask_num_layers (int): Transformer depth for mask head.
        num_points_dlt (int): Grid points for DLT solve.
        dlt_eps (float):      DLT Tikhonov regularisation.
        patch_h, patch_w (int): Input patch dimensions.
    """

    def __init__(
        self,
        backbone:        str   = "resnet34",
        feat_dim:        int   = 128,
        use_dino:        bool  = False,
        num_planes:      int   = 2,
        hidden_dim:      int   = 128,
        context_dim:     int   = 128,
        corr_levels:     int   = 4,
        corr_radius:     int   = 4,
        mask_embed_dim:  int   = 128,
        mask_num_heads:  int   = 4,
        mask_num_layers: int   = 2,
        num_points_dlt:  int   = 512,
        dlt_eps:         float = 1e-6,
        patch_h:         int   = 315,
        patch_w:         int   = 560,
    ):
        super().__init__()
        self.num_planes = num_planes
        self.patch_h    = patch_h
        self.patch_w    = patch_w

        # ---- Feature extractor (SHARED weights for I_a and I_b) ----
        self.feature_extractor = MultiScaleFeatureExtractor(
            backbone=backbone,
            feat_dim=feat_dim,
            use_dino=use_dino,
            pretrained=True,
        )

        # ---- Soft-argmax for dense correspondences ----
        self.soft_argmax = SoftArgmaxCorrespondence(temperature=0.1)

        # ---- K-plane mask head ----
        # Receives level2 features (finest scale) for detailed spatial masks
        self.mask_head = PlaneMaskHead(
            in_channels=feat_dim,
            embed_dim=mask_embed_dim,
            num_heads=mask_num_heads,
            num_layers=mask_num_layers,
            num_planes=num_planes,
            corr_channels=0,   # No corr summary for mask head (kept lightweight)
        )

        # ---- Differentiable DLT solver ----
        self.dlt_solver = DifferentiableDLT(
            num_points=num_points_dlt,
            eps=dlt_eps,
        )

        # ---- Iterative GRU refiner (one shared refiner for all planes) ----
        self.refiner = IterativeRefiner(
            feat_dim=feat_dim,
            hidden_dim=hidden_dim,
            context_dim=context_dim,
            corr_levels=corr_levels,
            corr_radius=corr_radius,
        )

        # ---- Warping ----
        self.stn = HomographySTN()

        # ---- Training mode flag ----
        self.warmup_mode = True   # set to False after warmup_epochs

    def set_warmup_mode(self, enabled: bool):
        """Toggle two-stage training protocol."""
        self.warmup_mode = enabled

    def _extract_features(
        self, img: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Extract multi-scale features from one image."""
        return self.feature_extractor(img)

    def _get_initial_flow(
        self,
        feat_a_fine: torch.Tensor,
        feat_b_fine: torch.Tensor,
    ) -> torch.Tensor:
        """
        Soft-argmax flow at finest scale (level2) for DLT initialisation.
        This gives a dense correspondence field before any GRU refinement.
        """
        return self.soft_argmax(feat_a_fine, feat_b_fine)

    def _compute_masks(
        self,
        feat_a_fine: torch.Tensor,
        feat_b_fine: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict K plane masks from level2 features.

        Returns:
            masks: (B, K, H_fine, W_fine) partition-of-unity masks.
        """
        return self.mask_head(feat_a_fine, feat_b_fine)

    def _apply_attention_mask(
        self,
        feat: torch.Tensor,    # (B, C, H, W)
        mask_k: torch.Tensor,  # (B, 1, H, W)
    ) -> torch.Tensor:
        """
        In Stage 2: weight features by plane mask (attention role).
        In Stage 1 (warmup): pass features through unchanged.
        """
        if self.warmup_mode:
            return feat
        return feat * mask_k

    def forward(
        self,
        img_a: torch.Tensor,   # (B, 1, H, W) source image
        img_b: torch.Tensor,   # (B, 1, H, W) target image
        num_iters: int = 12,
    ) -> Dict:
        """
        Forward pass for one pair (I_a → I_b).

        Args:
            img_a, img_b: (B, 1, H, W) grayscale normalised tensors.
            num_iters:    Number of GRU refinement iterations.

        Returns:
            dict with keys:
              'H_preds':     List[List[(B,3,3)]]  shape [K][num_iters]
                             Homography predictions for each plane at each iter.
              'H_final':     (B, K, 3, 3) final homography per plane.
              'masks':       (B, K, H_fine, W_fine) plane masks.
              'feat_a_fine': (B, C, H/4, W/4) level2 features of img_a.
              'feat_b_fine': (B, C, H/4, W/4) level2 features of img_b.
              'flow_init':   (B, 2, H/4, W/4) initial soft-argmax flow.
        """
        B = img_a.shape[0]
        H_patch, W_patch = self.patch_h, self.patch_w

        # ================================================================
        # 1. Feature extraction (shared weights, applied to both images)
        # ================================================================
        feats_a = self._extract_features(img_a)  # {level2, level3, level4}
        feats_b = self._extract_features(img_b)

        feat_a_fine   = feats_a["level2"]  # (B, C, H/4,  W/4)
        feat_b_fine   = feats_b["level2"]
        feat_a_coarse = feats_a["level3"]  # (B, C, H/8,  W/8)
        feat_b_coarse = feats_b["level3"]

        # ================================================================
        # 2. Soft-argmax initial flow (level2 resolution)
        # ================================================================
        flow_init = self._get_initial_flow(feat_a_fine, feat_b_fine)  # (B,2,H/4,W/4)

        # ================================================================
        # 3. Predict K plane masks
        # ================================================================
        masks = self._compute_masks(feat_a_fine, feat_b_fine)  # (B, K, H/4, W/4)

        # ================================================================
        # 4. Per-plane iterative refinement + DLT
        # ================================================================
        H_preds_all: List[List[torch.Tensor]] = []  # [K][num_iters]

        for k in range(self.num_planes):
            mask_k = masks[:, k : k + 1, :, :]  # (B, 1, H/4, W/4)

            # Optionally weight features by mask (Stage 2)
            fa_fine_k   = self._apply_attention_mask(feat_a_fine,   mask_k)
            fb_fine_k   = self._apply_attention_mask(feat_b_fine,   mask_k)
            fa_coarse_k = F.interpolate(mask_k, size=feat_a_coarse.shape[-2:],
                                         mode="bilinear", align_corners=False)
            fa_coarse_k = self._apply_attention_mask(feat_a_coarse, fa_coarse_k)
            fb_coarse_k = F.interpolate(mask_k, size=feat_b_coarse.shape[-2:],
                                         mode="bilinear", align_corners=False)
            fb_coarse_k = self._apply_attention_mask(feat_b_coarse, fb_coarse_k)

            # GRU iterative refinement → list of H estimates (one per iter)
            H_preds_k = self.refiner(
                feat_a_coarse=fa_coarse_k,
                feat_b_coarse=fb_coarse_k,
                feat_a_fine=fa_fine_k,
                feat_b_fine=fb_fine_k,
                masks_k=mask_k,
                dlt_solver=self.dlt_solver,
                num_iters=num_iters,
                img_h=H_patch,
                img_w=W_patch,
            )  # List[(B, 3, 3)], length = num_iters

            H_preds_all.append(H_preds_k)

        # Final H per plane: last iterate of each plane's refiner
        H_final = torch.stack(
            [H_preds_all[k][-1] for k in range(self.num_planes)], dim=1
        )  # (B, K, 3, 3)

        return {
            "H_preds":      H_preds_all,    # [K][num_iters] list of (B,3,3)
            "H_final":      H_final,        # (B, K, 3, 3)
            "masks":        masks,          # (B, K, H/4, W/4)
            "feat_a_fine":  feat_a_fine,    # (B, C, H/4, W/4)
            "feat_b_fine":  feat_b_fine,
            "feat_a_coarse": feat_a_coarse, # (B, C, H/8, W/8)
            "feat_b_coarse": feat_b_coarse,
            "flow_init":    flow_init,      # (B, 2, H/4, W/4)
        }

    def forward_triplet(
        self,
        img_a: torch.Tensor,
        img_b: torch.Tensor,
        img_c: torch.Tensor,
        num_iters: int = 12,
    ) -> Dict:
        """
        Forward pass for triplet (I_a, I_b, I_c) — used with triangle loss.

        Runs forward() for all three ordered pairs:
            a→b: H_ab,  b→a: H_ba  (inverse pair for consistency loss)
            b→c: H_bc,  a→c: H_ac  (for triangle consistency: H_ac ≈ H_bc ∘ H_ab)

        Returns:
            dict containing out_ab, out_ba, out_bc, out_ac sub-dicts.
        """
        out_ab = self.forward(img_a, img_b, num_iters)   # a → b
        out_ba = self.forward(img_b, img_a, num_iters)   # b → a (inverse check)
        out_bc = self.forward(img_b, img_c, num_iters)   # b → c
        out_ac = self.forward(img_a, img_c, num_iters)   # a → c (triangle target)
        return {
            "ab": out_ab,
            "ba": out_ba,
            "bc": out_bc,
            "ac": out_ac,
        }


def build_model(cfg: dict) -> MultiPlaneHomographyNet:
    """
    Instantiate the model from config dict.

    Args:
        cfg: Parsed YAML config.
    Returns:
        MultiPlaneHomographyNet instance.
    """
    m = cfg["model"]
    return MultiPlaneHomographyNet(
        backbone        = m["backbone"],
        feat_dim        = m["feature_dim"],
        use_dino        = m["use_dino"],
        num_planes      = m["num_planes"],
        hidden_dim      = m["hidden_dim"],
        context_dim     = m["context_dim"],
        corr_levels     = m["corr_levels"],
        corr_radius     = m["corr_radius"],
        mask_embed_dim  = m["mask_hidden_dim"],
        mask_num_heads  = m["mask_num_heads"],
        mask_num_layers = m["mask_num_layers"],
        num_points_dlt  = m["num_corr_points"],
        dlt_eps         = m["dlt_eps"],
        patch_h         = cfg["data"]["patch_height"],
        patch_w         = cfg["data"]["patch_width"],
    )
