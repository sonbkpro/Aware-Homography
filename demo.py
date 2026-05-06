"""
demo.py
-------
Quick demonstration: run inference on a single video or image pair and
write a side-by-side ghost-overlay + mask visualisation to disk.

Usage:
    # From a video (uses consecutive frame pairs)
    python demo.py --video my_video.mp4 --checkpoint checkpoints/best_model.pth

    # From a pre-extracted image pair
    python demo.py --img_a frame_a.png --img_b frame_b.png \\
                   --checkpoint checkpoints/best_model.pth

    # Without a checkpoint (identity homography, useful to verify the pipeline)
    python demo.py --video my_video.mp4 --no_checkpoint

After `pip install -e .`:
    dh-demo --video my_video.mp4 --checkpoint checkpoints/best_model.pth
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml

from deep_homography.data.augmentations      import EvalTransform
from deep_homography.models                  import build_model, MultiPlaneHomographyNet
from deep_homography.utils                   import HomographySTN
from deep_homography.utils.metrics           import select_dominant_plane_H
from deep_homography.utils.visualization     import overlay_channels, visualise_masks, visualise_flow


DEFAULT_CONFIG = "configs/default.yaml"


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Demo: run homography on a single video / pair")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--video",  help="Video file path")
    src.add_argument("--img_a",  help="Source image path (use with --img_b)")
    p.add_argument("--img_b",   help="Target image path")
    p.add_argument("--config",  default=DEFAULT_CONFIG)
    p.add_argument("--checkpoint", default=None)
    p.add_argument("--no_checkpoint", action="store_true",
                   help="Run without weights (identity H, sanity check)")
    p.add_argument("--out_dir", default="demo_output", help="Output directory")
    p.add_argument("--num_pairs", type=int, default=10,
                   help="Number of frame pairs to process from video")
    p.add_argument("--frame_gap", type=int, default=1)
    p.add_argument("--gpu", type=int, default=0)
    return p.parse_args(argv)


def load_config(path: str) -> dict:
    try:
        with open(path) as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"[demo] Config '{path}' not found — using built-in defaults.")
        return {
            "data":  {"patch_height": 256, "patch_width": 256,
                      "grayscale": True, "normalize_mean": 0.485, "normalize_std": 0.229},
            "model": {"backbone": "resnet34", "feature_dim": 128, "use_dino": False,
                      "num_planes": 2, "hidden_dim": 128, "context_dim": 128,
                      "corr_levels": 4, "corr_radius": 4, "mask_hidden_dim": 128,
                      "mask_num_heads": 4, "mask_num_layers": 2,
                      "num_corr_points": 512, "dlt_eps": 1e-6,
                      "num_iters_train": 12, "num_iters_eval": 24},
            "eval":  {"num_iters": 12, "error_threshold": 3.0},
        }


# ─────────────────────────────────────────────────────────────────────────────
# Frame extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_frame_pairs_from_video(
    video_path: str, num_pairs: int, gap: int
) -> list:
    """Returns list of (frame_a, frame_b) BGR uint8 arrays."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    frames = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames.append(f)
    cap.release()

    pairs = []
    for i in range(0, min(num_pairs * gap, len(frames) - gap), gap):
        pairs.append((frames[i], frames[i + gap]))
    return pairs[:num_pairs]


# ─────────────────────────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def run_inference(
    model:      MultiPlaneHomographyNet,
    img_a_np:   np.ndarray,   # BGR uint8
    img_b_np:   np.ndarray,
    transform:  EvalTransform,
    stn:        HomographySTN,
    device:     torch.device,
    num_iters:  int = 12,
) -> dict:
    """
    Run model on one image pair and return visualisation arrays.

    Returns dict with keys:
        'ghost'      – red/green overlay of warped source on target
        'masks'      – colour-coded K-plane masks
        'flow'       – HSV flow colour wheel
        'H_dominant' – (3, 3) dominant-plane homography numpy array
    """
    img_a_t, img_b_t, _ = transform([img_a_np, img_b_np, img_b_np])
    img_a = img_a_t.unsqueeze(0).to(device)
    img_b = img_b_t.unsqueeze(0).to(device)

    out     = model.forward(img_a, img_b, num_iters)
    H_final = out["H_final"]     # (1, K, 3, 3)
    masks   = out["masks"]       # (1, K, Hf, Wf)

    H_dom, _ = select_dominant_plane_H(H_final, masks)   # (1, 3, 3)
    warped   = stn(img_a, H_dom)[0].cpu().clamp(0, 1)
    target   = img_b[0].cpu().clamp(0, 1)

    ghost = overlay_channels(warped, target)

    m_up = F.interpolate(masks[0:1], size=img_a.shape[-2:],
                         mode="bilinear", align_corners=False)[0].cpu()
    mask_vis = visualise_masks(m_up, img_a[0].cpu())

    f_up = F.interpolate(out["flow_init"][0:1], size=img_a.shape[-2:],
                         mode="bilinear", align_corners=False)[0].cpu()
    flow_vis = visualise_flow(f_up)

    return {
        "ghost":      ghost,
        "masks":      mask_vis,
        "flow":       flow_vis,
        "H_dominant": H_dom[0].cpu().numpy(),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(cfg).to(device)
    if args.checkpoint and not args.no_checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(ckpt.get("model", ckpt))
        print(f"[demo] Loaded checkpoint: {args.checkpoint}")
    else:
        print("[demo] No checkpoint loaded — running with random/default weights.")
    model.set_warmup_mode(False)
    model.eval()

    stn       = HomographySTN().to(device)
    transform = EvalTransform(
        crop_h=cfg["data"]["patch_height"],
        crop_w=cfg["data"]["patch_width"],
        grayscale=cfg["data"]["grayscale"],
        mean=cfg["data"]["normalize_mean"],
        std=cfg["data"]["normalize_std"],
    )
    num_iters = cfg["eval"]["num_iters"]
    out_dir   = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load frame pairs ────────────────────────────────────────────────────
    if args.video:
        pairs = load_frame_pairs_from_video(args.video, args.num_pairs, args.frame_gap)
        print(f"[demo] Loaded {len(pairs)} pairs from {args.video}")
    else:
        fa = cv2.imread(args.img_a)
        fb = cv2.imread(args.img_b)
        assert fa is not None and fb is not None, "Could not read image files."
        pairs = [(fa, fb)]

    # ── Run inference ────────────────────────────────────────────────────────
    for i, (fa, fb) in enumerate(pairs):
        result = run_inference(model, fa, fb, transform, stn, device, num_iters)

        # Side-by-side: [ghost | masks | flow]
        canvas  = np.concatenate([result["ghost"], result["masks"], result["flow"]], axis=1)
        out_img = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)
        path    = out_dir / f"pair_{i:04d}.png"
        cv2.imwrite(str(path), out_img)

        # Print H
        H = result["H_dominant"]
        print(f"[demo] Pair {i:04d}  →  {path.name}")
        print(f"       H =\n{np.round(H, 4)}\n")

    print(f"\n[demo] All outputs saved to '{out_dir}/'")
    print("       Columns: [ghost overlay | plane masks | optical flow]")


if __name__ == "__main__":
    main()
