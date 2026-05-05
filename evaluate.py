"""
evaluate.py
-----------
Quantitative evaluation matching Zhang et al. (ECCV 2020) Table 2 protocol.

Usage:
    # All 5 categories — prints LaTeX-ready table
    python evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint checkpoints/best_model.pth \\
        --eval_all data/eval

    # Single category
    python evaluate.py \\
        --config configs/default.yaml \\
        --checkpoint checkpoints/best_model.pth \\
        --data_root data/eval/RE \\
        --gt_dir data/eval/RE/gt_points

After `pip install -e .`:
    dh-evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pth \\
                --eval_all data/eval
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Optional, Dict

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from deep_homography.data        import build_eval_loader
from deep_homography.models      import build_model, MultiPlaneHomographyNet
from deep_homography.utils       import HomographySTN
from deep_homography.utils.metrics import (
    HomographyEvaluator, select_dominant_plane_H,
)
from deep_homography.utils.visualization import (
    overlay_channels, visualise_masks, visualise_flow,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

CATEGORIES = ["RE", "LT", "LL", "SF", "LF"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Evaluate MultiPlaneHomographyNet")
    p.add_argument("--config",      required=True)
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--data_root",   default=None,
                   help="Root with frame_a/, frame_b/ sub-dirs.")
    p.add_argument("--gt_dir",      default=None,
                   help="Directory with .npy GT point files.")
    p.add_argument("--category",    default="unknown")
    p.add_argument("--eval_all",    default=None,
                   help="Root containing RE/, LT/, LL/, SF/, LF/ sub-dirs.")
    p.add_argument("--vis_dir",     default="vis_results")
    p.add_argument("--gpu",         type=int, default=0)
    return p.parse_args(argv)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_category(
    model, data_root, gt_dir, cfg, device,
    vis_dir=None, category="unknown",
) -> dict:
    loader    = build_eval_loader(data_root, cfg, gt_dir)
    stn       = HomographySTN().to(device)
    evaluator = HomographyEvaluator(cfg["eval"]["error_threshold"])
    num_iters = cfg["eval"]["num_iters"]

    if vis_dir:
        Path(vis_dir).mkdir(parents=True, exist_ok=True)

    model.eval()

    for bidx, batch in enumerate(tqdm(loader, desc=f"  [{category}]")):
        img_a = batch["img_a"].to(device)
        img_b = batch["img_b"].to(device)

        out = model.forward(img_a, img_b, num_iters)
        H_final = out["H_final"]
        masks   = out["masks"]

        gt_pts = batch.get("gt_points")
        if gt_pts is not None:
            gt_pts = gt_pts.to(device)

        evaluator.update(H_final, masks, gt_pts,
                         cfg["data"]["patch_height"], cfg["data"]["patch_width"])

        # ── Visualisation (first 20 batches) ──────────────────────────────
        if vis_dir and bidx < 20:
            H_dom, _ = select_dominant_plane_H(H_final, masks)
            for b in range(min(img_a.shape[0], 2)):
                warped  = stn(img_a[b:b+1], H_dom[b:b+1])[0].cpu().clamp(0, 1)
                target  = img_b[b].cpu().clamp(0, 1)
                ghost   = overlay_channels(warped, target)

                m_up = F.interpolate(masks[b:b+1], size=img_a.shape[-2:],
                                     mode="bilinear", align_corners=False)[0].cpu()
                mask_vis = visualise_masks(m_up, img_a[b].cpu())

                f_up = F.interpolate(out["flow_init"][b:b+1], size=img_a.shape[-2:],
                                     mode="bilinear", align_corners=False)[0].cpu()
                flow_vis = visualise_flow(f_up)

                canvas = np.concatenate([ghost, mask_vis, flow_vis], axis=1)
                cv2.imwrite(
                    str(Path(vis_dir) / f"b{bidx:04d}_s{b}.png"),
                    cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR),
                )

    results = evaluator.summary()
    log.info(
        f"[{category}] mean_err={results.get('mean_error','N/A'):.3f}  "
        f"inlier%={results.get('mean_inlier_pct','N/A'):.1f}  "
        f"n={results.get('num_samples',0)}"
    )
    return results


def evaluate_all(model, eval_root, cfg, device, vis_dir=None):
    all_results: Dict[str, dict] = {}
    for cat in CATEGORIES:
        cat_dir = os.path.join(eval_root, cat)
        if not os.path.isdir(cat_dir):
            log.warning(f"Missing category dir: {cat_dir}")
            continue
        gt_dir  = os.path.join(cat_dir, "gt_points")
        gt_dir  = gt_dir if os.path.isdir(gt_dir) else None
        vis_cat = os.path.join(vis_dir, cat) if vis_dir else None
        all_results[cat] = evaluate_category(
            model, cat_dir, gt_dir, cfg, device, vis_cat, cat
        )

    # ── Print summary table ──────────────────────────────────────────────────
    log.info("\n" + "=" * 80)
    log.info(f"{'':30}" + "".join(f"{c:>10}" for c in CATEGORIES) + f"{'Avg':>10}")
    log.info("-" * 80)
    errors  = [all_results.get(c, {}).get("mean_error",      float("nan")) for c in CATEGORIES]
    inliers = [all_results.get(c, {}).get("mean_inlier_pct", float("nan")) for c in CATEGORIES]
    log.info(f"{'Ours (mean error)':30}" +
             "".join(f"{e:10.3f}" for e in errors) + f"{np.nanmean(errors):10.3f}")
    log.info(f"{'Ours (inlier %)':30}" +
             "".join(f"{i:10.1f}" for i in inliers) + f"{np.nanmean(inliers):10.1f}")
    log.info("=" * 80 + "\n")
    return all_results


def main():
    args   = parse_args()
    cfg    = load_config(args.config)
    device = torch.device(
        f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    )

    model = build_model(cfg).to(device)
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt.get("model", ckpt))
    model.set_warmup_mode(False)
    log.info(f"Loaded: {args.checkpoint}")

    vis_dir = args.vis_dir if cfg["eval"].get("visualize", True) else None

    if args.eval_all:
        evaluate_all(model, args.eval_all, cfg, device, vis_dir)
    elif args.data_root:
        evaluate_category(model, args.data_root, args.gt_dir, cfg, device,
                          vis_dir, args.category)
    else:
        log.error("Provide --data_root or --eval_all.")
        sys.exit(1)


if __name__ == "__main__":
    main()
