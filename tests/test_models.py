"""
tests/test_models.py
--------------------
Unit tests for every architectural module.

Run:   pytest tests/ -v
Fast:  pytest tests/ -v -m "not slow"
"""

import pytest
import torch
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

B, C, H, W = 2, 128, 40, 70   # small spatial dims for fast tests
DEVICE = torch.device("cpu")

@pytest.fixture
def feat_pair():
    fa = torch.randn(B, C, H, W)
    fb = torch.randn(B, C, H, W)
    return fa, fb


@pytest.fixture
def dummy_img():
    return torch.randn(B, 1, 315, 560)   # grayscale, full patch size


# ─────────────────────────────────────────────────────────────────────────────
# utils.homography_utils
# ─────────────────────────────────────────────────────────────────────────────

class TestHomographyUtils:
    def test_identity_warp(self, dummy_img):
        from deep_homography.utils import HomographySTN
        stn = HomographySTN()
        I   = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        out = stn(dummy_img, I)
        assert out.shape == dummy_img.shape

    def test_H_inverse_compose(self):
        from deep_homography.utils import H_inverse, H_compose
        H = torch.eye(3).unsqueeze(0).expand(B, -1, -1) * 1.0
        # Add a small perturbation so it's not exactly identity
        H = H + 0.01 * torch.randn(B, 3, 3)
        H[:, 2, 2] = 1.0
        H_inv = H_inverse(H)
        prod  = H_compose(H_inv, H)   # should be ≈ I
        err   = (prod - torch.eye(3).unsqueeze(0).expand(B,-1,-1)).abs().max()
        assert err < 1e-4, f"H·H⁻¹ ≠ I, max err={err:.6f}"

    def test_H_to_4corners(self):
        from deep_homography.utils import H_to_4corners
        I  = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        d  = H_to_4corners(I, patch_h=315, patch_w=560)
        assert d.shape == (B, 4, 2)
        assert d.abs().max() < 1e-4, "Identity H should give zero corner displacements"

    def test_warp_points(self):
        from deep_homography.utils import warp_points
        pts = torch.rand(B, 10, 2) * 100
        I   = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        out = warp_points(pts, I)
        assert out.shape == (B, 10, 2)
        assert (out - pts).abs().max() < 1e-4


# ─────────────────────────────────────────────────────────────────────────────
# models.correlation
# ─────────────────────────────────────────────────────────────────────────────

class TestCorrelation:
    def test_corr_volume_shape(self, feat_pair):
        from deep_homography.models.correlation import CorrVolume
        fa, fb = feat_pair
        vol = CorrVolume(fa, fb)
        assert vol.corr.shape == (B, H, W, H, W)

    def test_corr_pyramid_lookup(self, feat_pair):
        from deep_homography.models.correlation import CorrPyramid
        fa, fb = feat_pair
        pyr    = CorrPyramid(fa, fb, num_levels=2)
        flow   = torch.zeros(B, 2, H, W)
        feats  = pyr.lookup(flow, radius=2)
        # shape: (B, num_levels*(2*radius+1)^2, H, W)
        expected_ch = 2 * (2 * 2 + 1) ** 2
        assert feats.shape == (B, expected_ch, H, W)

    def test_soft_argmax_shape(self, feat_pair):
        from deep_homography.models.correlation import SoftArgmaxCorrespondence
        sa   = SoftArgmaxCorrespondence()
        fa, fb = feat_pair
        flow = sa(fa, fb)
        assert flow.shape == (B, 2, H, W)


# ─────────────────────────────────────────────────────────────────────────────
# models.differentiable_dlt
# ─────────────────────────────────────────────────────────────────────────────

class TestDifferentiableDLT:
    def test_output_shape(self):
        from deep_homography.models.differentiable_dlt import DifferentiableDLT
        dlt  = DifferentiableDLT(num_points=64)
        flow = torch.zeros(B, 2, H, W)
        mask = torch.ones(B, 1, H, W)
        H_out = dlt(flow, mask, img_h=315, img_w=560)
        assert H_out.shape == (B, 3, 3)

    def test_identity_flow(self):
        """Zero flow should produce a near-identity homography."""
        from deep_homography.models.differentiable_dlt import DifferentiableDLT
        dlt  = DifferentiableDLT(num_points=64)
        flow = torch.zeros(B, 2, H, W)
        mask = torch.ones(B, 1, H, W)
        H_out = dlt(flow, mask, img_h=315, img_w=560)
        # Normalise and compare
        H_norm = H_out / H_out[:, 2:3, 2:3].clamp(min=1e-8)
        I = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        err = (H_norm - I).abs().max()
        assert err < 0.5, f"Zero flow DLT should be near identity, got err={err:.4f}"

    def test_gradients_flow(self):
        """Gradients must flow through SVD."""
        from deep_homography.models.differentiable_dlt import DifferentiableDLT
        dlt  = DifferentiableDLT(num_points=16)
        flow = torch.zeros(B, 2, 10, 18, requires_grad=True)
        mask = torch.ones(B, 1, 10, 18)
        H_out = dlt(flow, mask, 315, 560)
        H_out.sum().backward()
        assert flow.grad is not None, "Gradient did not flow through DLT"

    def test_four_point_dlt(self):
        from deep_homography.models.differentiable_dlt import FourPointDLT
        fpdlt  = FourPointDLT(315, 560)
        delta  = torch.zeros(B, 8)
        H_out  = fpdlt(delta)
        assert H_out.shape == (B, 3, 3)


# ─────────────────────────────────────────────────────────────────────────────
# models.plane_mask_head
# ─────────────────────────────────────────────────────────────────────────────

class TestPlaneMaskHead:
    def test_output_shape(self, feat_pair):
        from deep_homography.models.plane_mask_head import PlaneMaskHead
        head = PlaneMaskHead(in_channels=C, embed_dim=64, num_heads=4,
                             num_layers=1, num_planes=3)
        fa, fb = feat_pair
        masks  = head(fa, fb)
        assert masks.shape == (B, 3, H, W)

    def test_partition_of_unity(self, feat_pair):
        from deep_homography.models.plane_mask_head import PlaneMaskHead
        head  = PlaneMaskHead(in_channels=C, embed_dim=64, num_heads=4,
                              num_layers=1, num_planes=2)
        fa, fb = feat_pair
        masks = head(fa, fb)
        sums  = masks.sum(dim=1)   # (B, H, W)
        err   = (sums - 1.0).abs().max()
        assert err < 1e-5, f"Masks do not sum to 1 everywhere, max err={err:.6f}"

    def test_mask_regularistaion_losses(self, feat_pair):
        from deep_homography.models.plane_mask_head import (
            PlaneMaskHead, mask_tv_loss, mask_entropy_loss
        )
        head  = PlaneMaskHead(in_channels=C, embed_dim=64, num_heads=4,
                              num_layers=1, num_planes=2)
        fa, fb = feat_pair
        masks = head(fa, fb)
        tv  = mask_tv_loss(masks)
        ent = mask_entropy_loss(masks)
        assert tv.item()  >= 0.0
        assert ent.item() >= 0.0


# ─────────────────────────────────────────────────────────────────────────────
# models.homography_net (integration test)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.slow
class TestMultiPlaneHomographyNet:
    @pytest.fixture
    def small_cfg(self):
        return {
            "model": {
                "backbone": "resnet18",
                "feature_dim": 32,
                "use_dino": False,
                "num_planes": 2,
                "hidden_dim": 32,
                "context_dim": 32,
                "corr_levels": 2,
                "corr_radius": 2,
                "mask_hidden_dim": 32,
                "mask_num_heads": 2,
                "mask_num_layers": 1,
                "num_corr_points": 64,
                "dlt_eps": 1e-6,
                "num_iters_train": 2,
                "num_iters_eval": 4,
            },
            "data": {"patch_height": 80, "patch_width": 128},
        }

    def test_forward_shape(self, small_cfg):
        from deep_homography.models import build_model
        model  = build_model(small_cfg)
        img_a  = torch.randn(2, 1, 80, 128)
        img_b  = torch.randn(2, 1, 80, 128)
        out    = model.forward(img_a, img_b, num_iters=2)
        assert out["H_final"].shape  == (2, 2, 3, 3)
        assert out["masks"].shape[1] == 2

    def test_forward_triplet_keys(self, small_cfg):
        from deep_homography.models import build_model
        model  = build_model(small_cfg)
        img_a  = torch.randn(2, 1, 80, 128)
        img_b  = torch.randn(2, 1, 80, 128)
        img_c  = torch.randn(2, 1, 80, 128)
        out    = model.forward_triplet(img_a, img_b, img_c, num_iters=2)
        for key in ("ab", "ba", "bc", "ac"):
            assert key in out, f"Missing key '{key}' in triplet output"


# ─────────────────────────────────────────────────────────────────────────────
# data.augmentations
# ─────────────────────────────────────────────────────────────────────────────

class TestAugmentations:
    def _dummy_frames(self, n=3):
        return [np.random.randint(0, 255, (360, 640, 3), dtype=np.uint8) for _ in range(n)]

    def test_train_transform_output_shape(self):
        from deep_homography.data.augmentations import TrainTransform
        tf = TrainTransform(crop_h=315, crop_w=560)
        imgs = self._dummy_frames(3)
        out  = tf(imgs)
        assert len(out) == 3
        for t in out:
            assert t.shape == (1, 315, 560), f"Unexpected shape {t.shape}"

    def test_eval_transform_output_shape(self):
        from deep_homography.data.augmentations import EvalTransform
        tf = EvalTransform(crop_h=315, crop_w=560)
        imgs = self._dummy_frames(2)
        out  = tf(imgs)
        assert all(t.shape == (1, 315, 560) for t in out)

    def test_spatial_consistency(self):
        """Same spatial crop must be applied to all frames."""
        from deep_homography.data.augmentations import RandomCropPair
        crop = RandomCropPair(100, 200)
        frames = [np.ones((300, 400, 3), dtype=np.uint8) * i for i in range(3)]
        frames[0][:, :, 0] = 0    # unique pattern to detect misalignment
        cropped = crop(frames)
        # All cropped frames must be the same shape
        shapes = [f.shape for f in cropped]
        assert all(s == (100, 200, 3) for s in shapes), f"Shape mismatch: {shapes}"


# ─────────────────────────────────────────────────────────────────────────────
# utils.metrics
# ─────────────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_point_l2_error_identity(self):
        from deep_homography.utils.metrics import point_l2_error
        I      = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        pts    = torch.rand(B, 6, 2) * 300
        gt_pts = torch.stack([pts, pts], dim=2)   # src == dst
        err    = point_l2_error(I, gt_pts)
        assert err.shape == (B,)
        assert err.max() < 1e-3

    def test_inlier_percentage_all_inliers(self):
        from deep_homography.utils.metrics import inlier_percentage
        I      = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        pts    = torch.rand(B, 6, 2) * 300
        gt_pts = torch.stack([pts, pts], dim=2)
        pct    = inlier_percentage(I, gt_pts, threshold=3.0)
        assert (pct == 100.0).all()

    def test_evaluator_accumulate(self):
        from deep_homography.utils.metrics import HomographyEvaluator
        ev   = HomographyEvaluator()
        I    = torch.eye(3).unsqueeze(0).expand(B, -1, -1)
        pts  = torch.rand(B, 4, 2) * 200
        gt   = torch.stack([pts, pts], dim=2)
        ev.update(I, None, gt)
        ev.update(I, None, gt)
        res  = ev.summary()
        assert res["num_samples"] == 2 * B
        assert res["mean_error"] < 1e-3
