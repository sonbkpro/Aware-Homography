"""
inference.py
============
Complete inference pipeline for MultiPlaneHomographyNet.

KEY DESIGN DECISIONS (answers to the four questions):

Q1 — What is warped_output.png?
    Image A transformed by the estimated homography so its viewpoint matches
    image B. Features (edges, corners, textures) in the warped output should
    align spatially with those in image B. This is the primary product of
    homography estimation — the input to panorama stitching, video
    stabilisation, and visual odometry.

Q2 — Is the input cropped during inference?
    YES. EvalTransform center-crops both images to (patch_height × patch_width),
    e.g. 315×560, before passing them to the network. The warped output lives
    in that cropped coordinate system. All visualisations therefore use the
    CROPPED images as the reference, not the original full-resolution frames.
    The original images are kept separately only for display context.
    This file stores img_a_crop and img_b_crop explicitly to prevent
    the shape mismatch bug that occurs when mixing original and cropped arrays.

Q3 — What is comparison_grid.png?
    A horizontal 6-panel strip at uniform resolution:
      [Cropped A | Cropped B | A warped to B | Ghost overlay | Plane weights | Mask+boundaries]
    - Ghost overlay: R=target, G+B=warped. Perfect alignment -> grey.
      Misalignment -> red (target-only) or cyan (source-only) fringe.
    - Plane masks: colour-coded by plane index (red=plane0, green=plane1).
      Shows which plane owns each pixel at inference time.

Q4 — Mask visualisation is integrated directly (see visualise_masks and
     visualise_masks_on_image functions below).

Q5 — Bug fix: np.stack shape mismatch.
    Root cause: img_b_np (original resolution e.g. 1080x1920) vs
    warped_np (cropped resolution 315x560). The ghost function received
    arrays with different spatial dimensions. Fixed by always passing
    the cropped versions of img_a and img_b to all visualisation functions.

Three blending strategies:
  "dominant" - one global H (argmax by total mask support)
  "argmax"   - per-pixel hard assignment to highest-weight plane
  "soft"     - per-pixel weighted average (RECOMMENDED)
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from deep_homography.data.augmentations import EvalTransform
from deep_homography.models import MultiPlaneHomographyNet, build_model
from deep_homography.utils import HomographySTN


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing: crop exactly as EvalTransform does, keep cropped numpy arrays
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_pair(
    img_a_np:  np.ndarray,
    img_b_np:  np.ndarray,
    transform: EvalTransform,
    device:    torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray]:
    """
    Apply EvalTransform to both images and also return the CROPPED uint8
    numpy arrays so that all downstream visualisation works in the same
    coordinate system as the model output.

    Returns:
        img_a_t:    (1, C, H_crop, W_crop) tensor, normalised
        img_b_t:    (1, C, H_crop, W_crop) tensor, normalised
        img_a_crop: (H_crop, W_crop, 3)    BGR uint8 cropped source
        img_b_crop: (H_crop, W_crop, 3)    BGR uint8 cropped target
    """
    crop_h = transform.crop_h
    crop_w = transform.crop_w

    def _center_crop_bgr(img: np.ndarray) -> np.ndarray:
        h, w = img.shape[:2]
        if h < crop_h or w < crop_w:
            scale = max(crop_h / h, crop_w / w)
            img = cv2.resize(img, (int(w * scale) + 1, int(h * scale) + 1))
            h, w = img.shape[:2]
        top  = (h - crop_h) // 2
        left = (w - crop_w) // 2
        return img[top: top + crop_h, left: left + crop_w].copy()

    img_a_crop = _center_crop_bgr(img_a_np)
    img_b_crop = _center_crop_bgr(img_b_np)

    # EvalTransform expects BGR uint8 frames as a list; third frame is a dummy
    tensors = transform([img_a_np, img_b_np, img_b_np])
    img_a_t = tensors[0].unsqueeze(0).to(device)
    img_b_t = tensors[1].unsqueeze(0).to(device)

    return img_a_t, img_b_t, img_a_crop, img_b_crop


def tensor_to_bgr(
    tensor: torch.Tensor,
    mean:   float = 0.485,
    std:    float = 0.229,
) -> np.ndarray:
    """Denormalise a model-output tensor and convert to BGR uint8 numpy."""
    t = tensor.detach().cpu().float()
    t = (t * std + mean).clamp(0.0, 1.0)
    if t.shape[0] == 1:
        t = t.repeat(3, 1, 1)
    np_rgb = (t.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    return cv2.cvtColor(np_rgb, cv2.COLOR_RGB2BGR)


# ─────────────────────────────────────────────────────────────────────────────
# Warping strategies
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def warp_multiplane(
    img_a_t:  torch.Tensor,
    H_all:    torch.Tensor,
    masks:    torch.Tensor,
    strategy: str = "soft",
    stn:      Optional[HomographySTN] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Warp image A using K homographies blended by K plane masks.

    Strategy "soft" (RECOMMENDED):
        output(p) = sum_k  M^k(p) * warp(A, H^k)[p]
        Convex combination at every pixel. Differentiable. Smooth transitions.

    Strategy "argmax":
        k*(p) = argmax_k M^k(p); output(p) = warp(A, H^{k*})[p]
        Hard per-pixel region selection. Clean boundaries, no ghosting at
        edges. Not differentiable.

    Strategy "dominant":
        k* = argmax_k sum_p M^k(p); uses H^{k*} for the whole image.
        One global homography. Fastest. Use when one plane dominates.

    Returns:
        warped:  (B, C, H, W) warped image tensor
        weights: (B, K, H, W) per-pixel plane weights actually applied
    """
    if stn is None:
        stn = HomographySTN()

    B, K, _, _ = H_all.shape
    _, C, H, W = img_a_t.shape

    masks_full = F.interpolate(
        masks, size=(H, W), mode="bilinear", align_corners=False
    )  # (B, K, H, W), sums to 1 over dim=1

    if strategy == "dominant":
        return _warp_dominant(img_a_t, H_all, masks_full, stn)
    elif strategy == "argmax":
        return _warp_argmax(img_a_t, H_all, masks_full, stn)
    elif strategy == "soft":
        return _warp_soft(img_a_t, H_all, masks_full, stn)
    else:
        raise ValueError(f"Unknown strategy '{strategy}'. Choose: soft | argmax | dominant")


def _warp_dominant(img_a_t, H_all, masks_full, stn):
    B = img_a_t.shape[0]
    support    = masks_full.flatten(2).sum(dim=-1)
    dominant_k = support.argmax(dim=1)
    H_dom  = H_all[torch.arange(B, device=H_all.device), dominant_k]
    warped = stn(img_a_t, H_dom)
    weights = torch.zeros_like(masks_full)
    for b in range(B):
        weights[b, dominant_k[b]] = 1.0
    return warped, weights


def _warp_argmax(img_a_t, H_all, masks_full, stn):
    B, K, H, W = masks_full.shape
    warped_planes = torch.stack(
        [stn(img_a_t, H_all[:, k]) for k in range(K)], dim=1
    )  # (B, K, C, H, W)
    assignment = masks_full.argmax(dim=1)
    weights    = F.one_hot(assignment, K).float().permute(0, 3, 1, 2)
    warped     = (warped_planes * weights.unsqueeze(2)).sum(dim=1)
    return warped, weights


def _warp_soft(img_a_t, H_all, masks_full, stn):
    warped = torch.zeros_like(img_a_t)
    for k in range(H_all.shape[1]):
        w_k    = masks_full[:, k].unsqueeze(1)
        warped = warped + w_k * stn(img_a_t, H_all[:, k])
    return warped, masks_full


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation  (all functions operate on CROPPED BGR arrays)
# ─────────────────────────────────────────────────────────────────────────────

_PLANE_COLOURS_BGR = [
    ( 50,  50, 255),   # red    plane 0
    ( 50, 200,  50),   # green  plane 1
    (255,  50,  50),   # blue   plane 2
    ( 30, 165, 255),   # amber  plane 3
]


def ghost_overlay(warped_bgr: np.ndarray, target_bgr: np.ndarray) -> np.ndarray:
    """
    Red/green ghost overlay.

    Both inputs MUST be the same spatial size — always pass cropped images.

    R = target,  G = B = warped.
    Perfect alignment -> grey. Misalignment -> red/cyan fringe.
    """
    assert warped_bgr.shape == target_bgr.shape, (
        f"ghost_overlay: shape mismatch warped={warped_bgr.shape} "
        f"vs target={target_bgr.shape}. Always pass the center-cropped images."
    )
    w = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    t = cv2.cvtColor(target_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    rgb = np.stack([t, w, w], axis=-1)          # (H, W, 3) in RGB order
    rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def visualise_masks(
    masks_np:   np.ndarray,
    background: Optional[np.ndarray] = None,
    alpha:      float = 0.6,
) -> np.ndarray:
    """
    Colour-coded K-plane mask visualisation.

    Each plane gets a unique colour. The colour at pixel p is the weighted
    sum of plane colours by M^k(p), so boundaries blend smoothly.

    Args:
        masks_np:   (K, H, W) float in [0, 1].
        background: (H, W, 3) BGR uint8 to blend underneath (optional).
        alpha:      Opacity of the colour layer.

    Returns:
        (H, W, 3) BGR uint8.
    """
    K, H, W = masks_np.shape
    canvas = np.zeros((H, W, 3), dtype=np.float32)
    for k in range(K):
        colour = np.array(_PLANE_COLOURS_BGR[k % len(_PLANE_COLOURS_BGR)],
                          dtype=np.float32) / 255.0
        canvas += masks_np[k, :, :, np.newaxis] * colour

    mask_vis = (canvas * 255).clip(0, 255).astype(np.uint8)

    if background is not None:
        bg = cv2.resize(background, (W, H)).astype(np.uint8)
        mask_vis = cv2.addWeighted(bg, 1.0 - alpha, mask_vis, alpha, 0)

    return mask_vis


def visualise_masks_on_image(
    img_bgr:   np.ndarray,
    masks_np:  np.ndarray,
    alpha:     float = 0.55,
    border_px: int = 2,
) -> np.ndarray:
    """
    Overlay plane-mask colours on the source image WITH per-plane region
    boundaries drawn at argmax decision transitions.

    Args:
        img_bgr:   (H, W, 3) BGR uint8 — the CROPPED source image.
        masks_np:  (K, H, W) float plane masks.
        alpha:     Opacity of the colour overlay.
        border_px: Width of the region-boundary contour.

    Returns:
        (H, W, 3) BGR uint8.
    """
    K, H, W = masks_np.shape
    overlay = visualise_masks(masks_np, background=None, alpha=1.0)
    bg = cv2.resize(img_bgr, (W, H)).astype(np.uint8)
    blended = cv2.addWeighted(bg, 1 - alpha, overlay, alpha, 0)

    assignment = masks_np.argmax(axis=0).astype(np.uint8)
    for k in range(K):
        region_k = (assignment == k).astype(np.uint8) * 255
        contours, _ = cv2.findContours(region_k, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(blended, contours, -1,
                         _PLANE_COLOURS_BGR[k % len(_PLANE_COLOURS_BGR)],
                         border_px)
    return blended


def make_comparison_grid(
    img_a_crop:  np.ndarray,
    img_b_crop:  np.ndarray,
    warped_bgr:  np.ndarray,
    masks_np:    np.ndarray,
    weights_np:  np.ndarray,
) -> np.ndarray:
    """
    Build a labelled 6-panel comparison strip. All inputs must be the same
    spatial resolution (H_crop x W_crop) — pass the center-cropped images.

    Panels:
        1. Cropped source A
        2. Cropped target B
        3. A warped to B  (warped_output.png content)
        4. Ghost overlay  (alignment quality indicator)
        5. Soft plane weights on source
        6. Mask overlay with region boundaries

    Returns:
        BGR uint8 canvas with header labels.
    """
    H, W = warped_bgr.shape[:2]

    def _to_bgr3(img: np.ndarray) -> np.ndarray:
        img = cv2.resize(img.astype(np.uint8), (W, H))
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 1:
            img = cv2.cvtColor(img[:, :, 0], cv2.COLOR_GRAY2BGR)
        return img

    a = _to_bgr3(img_a_crop)
    b = _to_bgr3(img_b_crop)
    w = _to_bgr3(warped_bgr)

    # Ghost: both w and b are now guaranteed (H, W, 3)
    gh = ghost_overlay(w, b)

    # Soft mask colours blended over source
    mp = _to_bgr3(visualise_masks(weights_np, background=a, alpha=0.6))

    # Mask + boundary contours on source
    bp = _to_bgr3(visualise_masks_on_image(a, masks_np))

    row = np.concatenate([a, b, w, gh, mp, bp], axis=1)

    labels = [
        "Source A (cropped)",
        "Target B (cropped)",
        "A warped to B",
        "Ghost overlay",
        "Plane weights",
        "Mask + boundaries",
    ]
    hdr_h  = 32
    header = np.zeros((hdr_h, row.shape[1], 3), dtype=np.uint8)
    sep    = (80, 80, 80)
    for i, lbl in enumerate(labels):
        tx = max(4, i * W + W // 2 - len(lbl) * 4)
        cv2.putText(header, lbl, (tx, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (220, 220, 220), 1, cv2.LINE_AA)
        if i > 0:
            cv2.line(header, (i * W, 0), (i * W, hdr_h), sep, 1)
            cv2.line(row,    (i * W, 0), (i * W, H),     sep, 1)

    return np.vstack([header, row])


# ─────────────────────────────────────────────────────────────────────────────
# High-level inference function
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def infer_pair(
    model:      MultiPlaneHomographyNet,
    img_a_np:   np.ndarray,
    img_b_np:   np.ndarray,
    transform:  EvalTransform,
    device:     torch.device,
    num_iters:  int = 24,
    strategy:   str = "soft",
) -> dict:
    """
    Full inference on a single image pair.

    Args:
        model:      Trained MultiPlaneHomographyNet (model.eval() called inside).
        img_a_np:   Source image, BGR uint8, any resolution.
        img_b_np:   Target image, BGR uint8, any resolution.
        transform:  EvalTransform (handles crop + normalisation).
        device:     Torch device.
        num_iters:  GRU refinement iterations.
        strategy:   "soft" | "argmax" | "dominant"

    Returns dict with keys:
        "warped_bgr"  : (H_crop, W_crop, 3) BGR uint8 — A warped to B
        "img_a_crop"  : (H_crop, W_crop, 3) BGR uint8 — cropped source
        "img_b_crop"  : (H_crop, W_crop, 3) BGR uint8 — cropped target
        "H_all"       : (K, 3, 3) float64 numpy — all K homography matrices
        "masks_np"    : (K, H_crop, W_crop) float32 — soft masks at image res
        "weights_np"  : (K, H_crop, W_crop) float32 — actual blend weights used
        "dominant_k"  : int — index of the plane with most spatial support
    """
    model.eval()
    stn = HomographySTN().to(device)

    img_a_t, img_b_t, img_a_crop, img_b_crop = preprocess_pair(
        img_a_np, img_b_np, transform, device
    )
    _, _, H_crop, W_crop = img_a_t.shape

    out   = model.forward(img_a_t, img_b_t, num_iters=num_iters)
    H_all = out["H_final"]   # (1, K, 3, 3)
    masks = out["masks"]     # (1, K, Hf, Wf)

    warped_t, weights_t = warp_multiplane(
        img_a_t, H_all, masks, strategy=strategy, stn=stn
    )

    warped_bgr = tensor_to_bgr(warped_t[0], transform.mean, transform.std)

    def _up(t):
        return F.interpolate(
            t, size=(H_crop, W_crop), mode="bilinear", align_corners=False
        )[0].cpu().numpy()

    masks_np   = _up(masks)
    weights_np = _up(weights_t)

    support    = masks[0].flatten(1).sum(dim=-1)
    dominant_k = int(support.argmax().item())

    return {
        "warped_bgr":  warped_bgr,
        "img_a_crop":  img_a_crop,
        "img_b_crop":  img_b_crop,
        "H_all":       H_all[0].cpu().numpy(),
        "masks_np":    masks_np,
        "weights_np":  weights_np,
        "dominant_k":  dominant_k,
    }


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Multi-plane homography inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Strategies:\n"
            "  soft     — per-pixel weighted average (recommended)\n"
            "  argmax   — hard per-pixel plane assignment\n"
            "  dominant — one global H for the whole image\n"
        ),
    )
    p.add_argument("--img_a",      required=True)
    p.add_argument("--img_b",      required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--config",     default="configs/default.yaml")
    p.add_argument("--strategy",   default="soft",
                   choices=["soft", "argmax", "dominant"])
    p.add_argument("--num_iters",  type=int, default=24)
    p.add_argument("--out_dir",    default="inference_output")
    p.add_argument("--gpu",        type=int, default=0)
    return p.parse_args(argv)


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.eval()
    print(f"Loaded: {args.checkpoint}")

    transform = EvalTransform(
        crop_h=cfg["data"]["patch_height"],
        crop_w=cfg["data"]["patch_width"],
        grayscale=cfg["data"]["grayscale"],
        mean=cfg["data"]["normalize_mean"],
        std=cfg["data"]["normalize_std"],
    )

    img_a = cv2.imread(args.img_a)
    img_b = cv2.imread(args.img_b)
    if img_a is None:
        sys.exit(f"Cannot read: {args.img_a}")
    if img_b is None:
        sys.exit(f"Cannot read: {args.img_b}")

    print(f"Source: {img_a.shape}  ->  crop to "
          f"{cfg['data']['patch_height']}x{cfg['data']['patch_width']}")

    result = infer_pair(
        model, img_a, img_b, transform, device,
        num_iters=args.num_iters,
        strategy=args.strategy,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cv2.imwrite(str(out_dir / "warped_output.png"),   result["warped_bgr"])
    cv2.imwrite(str(out_dir / "plane_masks.png"),
                visualise_masks(result["masks_np"], background=result["img_a_crop"]))
    cv2.imwrite(str(out_dir / "masks_on_source.png"),
                visualise_masks_on_image(result["img_a_crop"], result["masks_np"]))
    cv2.imwrite(str(out_dir / "comparison_grid.png"),
                make_comparison_grid(
                    result["img_a_crop"], result["img_b_crop"],
                    result["warped_bgr"], result["masks_np"], result["weights_np"],
                ))

    K = result["H_all"].shape[0]
    print(f"\nStrategy : {args.strategy}")
    print("Plane support: " + ", ".join(
        f"k{k}={result['masks_np'][k].mean()*100:.1f}%"
        for k in range(K)
    ))
    for k in range(K):
        print(f"\nH^{k} =\n{np.round(result['H_all'][k], 5)}")

    print(f"\nOutputs -> '{out_dir}/':")
    print("  warped_output.png   — Image A warped to align with B")
    print("  plane_masks.png     — Soft plane mask colours on source")
    print("  masks_on_source.png — Mask overlay with region boundaries")
    print("  comparison_grid.png — 6-panel side-by-side comparison")


if __name__ == "__main__":
    main()
