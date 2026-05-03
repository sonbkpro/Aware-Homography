"""
scripts/check_install.py
------------------------
Quick sanity check after installation.
Run this first to verify everything is importable and GPU is visible.

Usage:
    python scripts/check_install.py
"""

import sys


def check(label, fn):
    try:
        result = fn()
        print(f"  ✓  {label}" + (f": {result}" if result else ""))
    except Exception as e:
        print(f"  ✗  {label}: {e}")
        return False
    return True


def main():
    print("\n── deep-homography installation check ──────────────────────\n")
    ok = True

    ok &= check("Python version",     lambda: sys.version.split()[0])
    ok &= check("PyTorch",            lambda: __import__("torch").__version__)
    ok &= check("CUDA available",     lambda: __import__("torch").cuda.is_available())
    ok &= check("torchvision",        lambda: __import__("torchvision").__version__)
    ok &= check("OpenCV",             lambda: __import__("cv2").__version__)
    ok &= check("numpy",              lambda: __import__("numpy").__version__)
    ok &= check("kornia",             lambda: __import__("kornia").__version__)
    ok &= check("einops",             lambda: __import__("einops").__version__)
    ok &= check("PyYAML",             lambda: __import__("yaml").__version__)
    ok &= check("tqdm",               lambda: __import__("tqdm").__version__)

    print()
    ok &= check("deep_homography package",   lambda: __import__("deep_homography").__version__)
    ok &= check("data module",               lambda: __import__("deep_homography.data") and "ok")
    ok &= check("models module",             lambda: __import__("deep_homography.models") and "ok")
    ok &= check("losses module",             lambda: __import__("deep_homography.losses") and "ok")
    ok &= check("utils module",              lambda: __import__("deep_homography.utils") and "ok")

    print()
    import torch
    ok &= check("Feature extractor (CPU)",  _test_feature_extractor)
    ok &= check("Correlation volume (CPU)", _test_correlation)
    ok &= check("DifferentiableDLT (CPU)", _test_dlt)
    ok &= check("Mask head (CPU)",         _test_mask_head)

    print()
    if ok:
        print("✅  All checks passed. You are ready to train!\n")
        print("    python train.py --config configs/default.yaml\n")
    else:
        print("⚠️   Some checks failed. Review errors above.\n")
    return 0 if ok else 1


def _test_feature_extractor():
    import torch
    from deep_homography.models import MultiScaleFeatureExtractor
    model = MultiScaleFeatureExtractor(backbone="resnet18", feat_dim=32, pretrained=False)
    x     = torch.randn(1, 1, 80, 128)
    out   = model(x)
    assert set(out.keys()) == {"level2", "level3", "level4"}
    return f"level2={tuple(out['level2'].shape)}"


def _test_correlation():
    import torch
    from deep_homography.models.correlation import CorrPyramid, SoftArgmaxCorrespondence
    B, C, H, W = 1, 32, 10, 18
    fa  = torch.randn(B, C, H, W)
    fb  = torch.randn(B, C, H, W)
    pyr = CorrPyramid(fa, fb, num_levels=2)
    f   = pyr.lookup(torch.zeros(B, 2, H, W), radius=2)
    sa  = SoftArgmaxCorrespondence()(fa, fb)
    return f"corr={tuple(f.shape)}, flow={tuple(sa.shape)}"


def _test_dlt():
    import torch
    from deep_homography.models.differentiable_dlt import DifferentiableDLT
    dlt  = DifferentiableDLT(num_points=16)
    flow = torch.zeros(1, 2, 10, 18)
    mask = torch.ones(1, 1, 10, 18)
    H    = dlt(flow, mask, 80, 128)
    return f"H.shape={tuple(H.shape)}"


def _test_mask_head():
    import torch
    from deep_homography.models.plane_mask_head import PlaneMaskHead
    head   = PlaneMaskHead(in_channels=32, embed_dim=32, num_heads=2,
                           num_layers=1, num_planes=2)
    fa, fb = torch.randn(1, 32, 10, 18), torch.randn(1, 32, 10, 18)
    masks  = head(fa, fb)
    return f"masks={tuple(masks.shape)}, sum≈{masks.sum(1).mean().item():.2f}"


if __name__ == "__main__":
    sys.exit(main())
