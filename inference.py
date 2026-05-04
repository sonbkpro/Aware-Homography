"""
inference.py
============
Complete inference pipeline for MultiPlaneHomographyNet.

This module answers the key question:
    "Given K=2 homography matrices at inference time, how do we warp
     image A to align with image B?"

Answer: The PlaneMaskHead runs on every forward pass and produces the
soft plane masks M^1...M^K alongside H^1...H^K. We use those masks
as per-pixel weights to combine the K warped images.

Three strategies are provided, in order of sophistication:

  Strategy 1 — Dominant plane  (simplest, one H for entire image)
  Strategy 2 — Hard argmax     (per-pixel hard assignment, clean boundaries)
  Strategy 3 — Soft blend      (per-pixel weighted average, smooth, differentiable)

For most video stabilization / image alignment tasks, Strategy 3 is
the correct choice.  Strategies 1 and 2 are provided for ablation and
special cases (e.g. when one plane truly dominates the scene).
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pathlib import Path
from typing import Tuple, Optional

from deep_homography.models      import build_model, MultiPlaneHomographyNet
from deep_homography.utils       import HomographySTN
from deep_homography.data.augmentations import EvalTransform


# ─────────────────────────────────────────────────────────────────────────────
# Core warping strategies
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def warp_multiplane(
    img_a:   torch.Tensor,    # (B, C, H, W)  source image, normalised
    H_all:   torch.Tensor,    # (B, K, 3, 3)  one homography per plane
    masks:   torch.Tensor,    # (B, K, Hf, Wf) plane masks (sum-to-1 over K)
    strategy: str = "soft",   # "dominant" | "argmax" | "soft"
    stn:     Optional[HomographySTN] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp image A using K homographies weighted by K plane masks.

    Args:
        img_a:    (B, C, H, W)  source image tensor (normalised).
        H_all:    (B, K, 3, 3)  per-plane homography matrices.
        masks:    (B, K, Hf, Wf) soft plane masks, softmax over K dimension.
        strategy: Which blending strategy to use (see module docstring).
        stn:      Optional pre-created HomographySTN (avoids re-creation).

    Returns:
        warped:  (B, C, H, W)  image A warped to align with B.
        weights: (B, K, H, W)  per-pixel plane weights used (for visualisation).
    """
    if stn is None:
        stn = HomographySTN()

    B, K, _, _ = H_all.shape
    _, C, H, W = img_a.shape

    # Upsample masks to full image resolution
    masks_full = F.interpolate(
        masks, size=(H, W), mode="bilinear", align_corners=False
    )  # (B, K, H, W) — still sums to 1 over K dim

    if strategy == "dominant":
        return _strategy_dominant(img_a, H_all, masks_full, stn)
    elif strategy == "argmax":
        return _strategy_argmax(img_a, H_all, masks_full, stn)
    elif strategy == "soft":
        return _strategy_soft_blend(img_a, H_all, masks_full, stn)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: dominant | argmax | soft")


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 1: Dominant plane
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_dominant(img_a, H_all, masks_full, stn):
    """
    Select one global homography: the plane k* with the most mask support.

        k* = argmax_k  Σ_{pixels} M^k(p)

    Then warp the entire image with H^{k*}.

    When to use:
        - Scene has one overwhelmingly dominant plane (textbook homography case).
        - You need a single 3×3 matrix as output (e.g. for downstream stitching).
        - Fast inference where simplicity matters.

    Limitation:
        Ignores all information from the other K-1 planes. If there is a
        foreground object on a different plane (e.g. a person walking in front
        of a building), that region will be misaligned.
    """
    B, K, H, W = masks_full.shape

    # Total spatial support per plane: (B, K)
    support = masks_full.flatten(2).sum(dim=-1)
    dominant_k = support.argmax(dim=1)   # (B,)

    # Select the dominant H for each sample in the batch
    H_dominant = H_all[torch.arange(B, device=H_all.device), dominant_k]  # (B, 3, 3)
    warped = stn(img_a, H_dominant)  # (B, C, H, W)

    # Build weight map: 1 at dominant plane pixels, 0 elsewhere (for vis)
    weights = torch.zeros_like(masks_full)
    for b in range(B):
        weights[b, dominant_k[b]] = 1.0

    return warped, weights


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 2: Hard argmax
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_argmax(img_a, H_all, masks_full, stn):
    """
    At every pixel p, choose the plane with the highest mask weight:

        k*(p) = argmax_k  M^k(p)

    Then composite: output(p) = warp(A, H^{k*(p)})[p]

    This is equivalent to partitioning the image into K disjoint regions,
    each warped by its own homography. Boundaries are sharp.

    When to use:
        - The planes are physically distinct (e.g. sky vs ground).
        - You want clean region boundaries without ghosting.
        - You are doing region-based compositing downstream.

    Limitation:
        Hard boundaries can cause visible seams if mask edges are noisy.
        Masking with argmax is not differentiable (can't be used in training loss).
    """
    B, K, H, W = masks_full.shape

    # Warp image separately for each plane
    warped_planes = []
    for k in range(K):
        H_k = H_all[:, k]               # (B, 3, 3)
        warped_k = stn(img_a, H_k)      # (B, C, H, W)
        warped_planes.append(warped_k)
    warped_stack = torch.stack(warped_planes, dim=1)  # (B, K, C, H, W)

    # Hard assignment: (B, H, W) with values in {0, ..., K-1}
    assignment = masks_full.argmax(dim=1)  # (B, H, W)

    # Build one-hot weight map: (B, K, H, W)
    weights = F.one_hot(assignment, num_classes=K).float()   # (B, H, W, K)
    weights = weights.permute(0, 3, 1, 2)                    # (B, K, H, W)

    # Composite: sum over K (only one plane is 1 at each pixel)
    # weights: (B, K, H, W) → unsqueeze channel dim → (B, K, 1, H, W)
    w = weights.unsqueeze(2)                      # (B, K, 1, H, W)
    warped = (warped_stack * w).sum(dim=1)        # (B, C, H, W)

    return warped, weights


# ─────────────────────────────────────────────────────────────────────────────
# Strategy 3: Soft blend  ← RECOMMENDED
# ─────────────────────────────────────────────────────────────────────────────

def _strategy_soft_blend(img_a, H_all, masks_full, stn):
    """
    At every pixel p, take the weighted average of all K warped images:

        output(p) = Σ_k  M^k(p) · warp(A, H^k)[p]

    Because masks sum to 1 (partition of unity), this is a proper convex
    combination: the output pixel is a weighted average of the K candidate
    aligned pixels. No pixel is discarded.

    When to use:
        - General case (recommended default).
        - The planes overlap spatially (common in real scenes).
        - Smooth transitions between plane regions are desired.
        - The result will be used in a differentiable downstream step.

    Mathematical guarantee:
        If M^k(p) ≈ 1 for some k (sharp mask), this reduces exactly to
        Strategy 2 at that pixel. If M^k(p) = 1/K everywhere (uniform mask),
        it produces a ghostly average — this only happens if the mask head
        has not trained properly.

    Ghosting concern:
        At boundary pixels where M¹ ≈ M² ≈ 0.5, the two warped images
        are averaged equally, which can appear as a half-transparent double
        exposure. This is the cost of smoothness. The entropy loss during
        training pushes masks toward hard assignments precisely to prevent this.
        If ghosting persists, increase lambda_mask_entropy in config.
    """
    B, K, H, W = masks_full.shape

    # Accumulate the weighted sum
    warped_blend = torch.zeros_like(img_a)  # (B, C, H, W)

    for k in range(K):
        H_k      = H_all[:, k]                          # (B, 3, 3)
        warped_k = stn(img_a, H_k)                      # (B, C, H, W)
        mask_k   = masks_full[:, k].unsqueeze(1)        # (B, 1, H, W)
        warped_blend = warped_blend + mask_k * warped_k

    return warped_blend, masks_full


# ─────────────────────────────────────────────────────────────────────────────
# High-level inference function (use this in practice)
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_pair(
    model:      MultiPlaneHomographyNet,
    img_a_np:   np.ndarray,          # BGR uint8, any resolution
    img_b_np:   np.ndarray,          # BGR uint8 (used only for visual comparison)
    transform:  EvalTransform,
    device:     torch.device,
    num_iters:  int = 24,
    strategy:   str = "soft",
) -> dict:
    """
    Full inference on a single image pair.

    Args:
        model:      Trained MultiPlaneHomographyNet.
        img_a_np:   Source image (BGR uint8).
        img_b_np:   Target image (BGR uint8).
        transform:  EvalTransform (crops + normalises).
        device:     Torch device.
        num_iters:  GRU refinement iterations (more = slower but better).
        strategy:   Blending strategy: "soft" | "argmax" | "dominant".

    Returns dict with:
        'warped_np'    : np.ndarray (H, W, 3) BGR uint8 — A warped to B
        'H_all'        : (K, 3, 3) numpy — all K homography matrices
        'masks_np'     : (K, H, W) numpy — plane masks at full resolution
        'weights_np'   : (K, H, W) numpy — actual per-pixel blend weights used
        'dominant_k'   : int — index of dominant plane
    """
    model.eval()
    stn = HomographySTN().to(device)

    # Preprocess: EvalTransform returns a list of tensors
    imgs = transform([img_a_np, img_b_np, img_b_np])   # 3rd is dummy for triplet API
    img_a_t = imgs[0].unsqueeze(0).to(device)           # (1, 1, H, W)
    img_b_t = imgs[1].unsqueeze(0).to(device)

    # ── Forward pass ─────────────────────────────────────────────────────────
    out = model.forward(img_a_t, img_b_t, num_iters=num_iters)
    #
    # out["H_final"]:   (1, K, 3, 3)  per-plane homographies
    # out["masks"]:     (1, K, Hf, Wf) plane masks (softmax over K)
    # out["flow_init"]: (1, 2, Hf, Wf) initial soft-argmax flow (diagnostic)
    #
    H_all = out["H_final"]   # (1, K, 3, 3)
    masks = out["masks"]     # (1, K, Hf, Wf)

    # ── Warp using chosen strategy ───────────────────────────────────────────
    warped, weights = warp_multiplane(
        img_a_t, H_all, masks, strategy=strategy, stn=stn
    )  # warped: (1, 1, H, W),  weights: (1, K, H, W)

    # ── Convert back to uint8 BGR numpy ──────────────────────────────────────
    warped_np = _tensor_to_bgr_uint8(warped[0], transform)

    # Dominant plane (by total mask support)
    support   = masks[0].flatten(1).sum(dim=-1)   # (K,)
    dominant_k = support.argmax().item()

    return {
        "warped_np":    warped_np,                                   # (H, W, 3) BGR uint8
        "H_all":        H_all[0].cpu().numpy(),                      # (K, 3, 3)
        "masks_np":     _upsample_masks(masks[0], img_a_np),        # (K, H, W) float
        "weights_np":   _upsample_masks(weights[0], img_a_np),      # (K, H, W) float
        "dominant_k":   dominant_k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tensor_to_bgr_uint8(
    tensor:    torch.Tensor,   # (C, H, W) normalised
    transform: EvalTransform,
) -> np.ndarray:
    """Denormalise and convert to BGR uint8."""
    t = tensor.cpu().float()
    t = t * transform.std + transform.mean   # undo normalisation
    t = t.clamp(0, 1)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)               # grayscale → 3-channel
    np_img = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(np_img, cv2.COLOR_RGB2BGR)


def _upsample_masks(
    masks: torch.Tensor,    # (K, Hf, Wf)
    ref_img: np.ndarray,    # BGR reference for target resolution
) -> np.ndarray:
    H, W = ref_img.shape[:2]
    m = F.interpolate(
        masks.unsqueeze(0), size=(H, W), mode="bilinear", align_corners=False
    ).squeeze(0)            # (K, H, W)
    return m.cpu().numpy()


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_comparison_grid(
    img_a_np:   np.ndarray,   # BGR uint8
    img_b_np:   np.ndarray,   # BGR uint8
    warped_np:  np.ndarray,   # BGR uint8
    masks_np:   np.ndarray,   # (K, H, W) float
) -> np.ndarray:
    """
    Builds a side-by-side visualisation:

        [ Image A | Image B | Warped A | Ghost overlay | Plane masks ]

    Args:
        img_a_np:  Source image.
        img_b_np:  Target image.
        warped_np: Warped source image.
        masks_np:  (K, H, W) plane masks.

    Returns:
        np.ndarray BGR uint8 canvas.
    """
    H, W = img_a_np.shape[:2]

    # Ghost overlay: R = target, G+B = warped (misalignment shows as colour fringe)
    def _ghost(warped, target):
        w = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(float) / 255
        t = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY).astype(float) / 255
        out = np.stack([t, w, w], axis=-1)
        return (out * 255).clip(0, 255).astype(np.uint8)

    ghost = _ghost(warped_np, img_b_np)

    # Colour-coded K-plane mask
    palette = [
        (255,  50,  50),   # red   plane 0
        ( 50, 200,  50),   # green plane 1
        ( 50,  50, 255),   # blue  plane 2
        (255, 165,   0),   # amber plane 3
    ]
    K = masks_np.shape[0]
    mask_vis = np.zeros((H, W, 3), dtype=np.float32)
    for k in range(K):
        c = np.array(palette[k % len(palette)], dtype=np.float32) / 255
        mask_vis += masks_np[k, :, :, np.newaxis] * c
    mask_vis = (mask_vis * 255).clip(0, 255).astype(np.uint8)

    # Combine: resize all to same height (crop may have changed size)
    def _resize(img):
        return cv2.resize(img, (W, H))

    row = np.concatenate([
        _resize(img_a_np),
        _resize(img_b_np),
        _resize(warped_np),
        _resize(ghost),
        _resize(mask_vis),
    ], axis=1)

    # Add labels at top
    labels = ["Source A", "Target B", "Warped A", "Ghost overlay", "Plane masks"]
    label_row = np.zeros((28, row.shape[1], 3), dtype=np.uint8)
    for i, lbl in enumerate(labels):
        x = i * W + W // 2
        cv2.putText(label_row, lbl, (x - 55, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    return np.vstack([label_row, row])


# ─────────────────────────────────────────────────────────────────────────────
# CLI demo entry point
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import argparse
    import yaml

    p = argparse.ArgumentParser(description="Multi-plane homography inference")
    p.add_argument("--img_a",       required=True,  help="Source image path")
    p.add_argument("--img_b",       required=True,  help="Target image path")
    p.add_argument("--checkpoint",  required=True,  help="Model checkpoint path")
    p.add_argument("--config",      default="configs/default.yaml")
    p.add_argument("--strategy",    default="soft",
                   choices=["soft", "argmax", "dominant"],
                   help="Blending strategy (default: soft)")
    p.add_argument("--num_iters",   type=int, default=24)
    p.add_argument("--out",         default="warped_output.png")
    p.add_argument("--grid",        default="comparison_grid.png",
                   help="Side-by-side comparison image")
    p.add_argument("--gpu",         type=int, default=0)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    # Load model
    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    print(f"Loaded checkpoint: {args.checkpoint}")

    # Load images
    img_a = cv2.imread(args.img_a)
    img_b = cv2.imread(args.img_b)
    assert img_a is not None and img_b is not None, "Could not read images."

    transform = EvalTransform(
        crop_h=cfg["data"]["patch_height"],
        crop_w=cfg["data"]["patch_width"],
        grayscale=cfg["data"]["grayscale"],
        mean=cfg["data"]["normalize_mean"],
        std=cfg["data"]["normalize_std"],
    )

    # Run inference
    result = infer_pair(
        model, img_a, img_b, transform, device,
        num_iters=args.num_iters,
        strategy=args.strategy,
    )

    # Save warped image
    cv2.imwrite(args.out, result["warped_np"])
    print(f"Warped image saved → {args.out}")

    # Print homographies
    K = result["H_all"].shape[0]
    print(f"\nDominant plane: k={result['dominant_k']}")
    for k in range(K):
        print(f"\nH^{k} =\n{np.round(result['H_all'][k], 4)}")

    # Save comparison grid
    grid = make_comparison_grid(img_a, img_b, result["warped_np"], result["masks_np"])
    cv2.imwrite(args.grid, grid)
    print(f"Comparison grid saved → {args.grid}")
    print("\nColumns: [Source A | Target B | Warped A | Ghost overlay | Plane masks]")


if __name__ == "__main__":
    main()
