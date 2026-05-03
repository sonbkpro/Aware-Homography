"""
train.py
--------
Main training script for MultiPlaneHomographyNet.

Usage:
    python train.py --config configs/default.yaml
    python train.py --config configs/default.yaml --resume checkpoints/epoch_010.pth
    python train.py --config configs/my_config.yaml --gpus 0 1 2

After `pip install -e .`:
    dh-train --config configs/default.yaml
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from deep_homography.data        import build_train_loader, build_val_loader
from deep_homography.models      import build_model, MultiPlaneHomographyNet
from deep_homography.losses      import TotalLoss
from deep_homography.utils       import HomographySTN
from deep_homography.utils.visualization  import log_to_tensorboard
from deep_homography.utils.metrics        import HomographyEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train MultiPlaneHomographyNet")
    p.add_argument("--config",  default="configs/default.yaml", help="YAML config path")
    p.add_argument("--resume",  default=None, help="Checkpoint path to resume from")
    p.add_argument(
        "--gpus",
        type=int,
        nargs="+",
        default=[0],
        metavar="ID",
        help="GPU IDs to use for training (e.g. --gpus 0 1 2). Pass -1 to force CPU.",
    )
    p.add_argument("--workers", type=int, default=None, help="Override num_workers")
    return p.parse_args(argv)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint helpers
# ─────────────────────────────────────────────────────────────────────────────

def save_checkpoint(path, model, optimizer, scheduler, epoch, step, best):
    torch.save({
        "epoch": epoch, "global_step": step, "best_error": best,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }, path)
    log.info(f"Checkpoint → {path}")


def load_checkpoint(path, model, optimizer, scheduler) -> dict:
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    log.info(f"Resumed from {path} (epoch {ckpt['epoch']}, step {ckpt['global_step']})")
    return ckpt


# ─────────────────────────────────────────────────────────────────────────────
# Validation
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, val_loader, criterion, device, cfg, step, writer):
    model.eval()
    stn = HomographySTN().to(device)
    evaluator = HomographyEvaluator(cfg["eval"]["error_threshold"])
    total_loss = 0.0
    n_batches  = 0

    for batch in tqdm(val_loader, desc="  Validating", leave=False):
        img_a = batch["img_a"].to(device, non_blocking=True)
        img_b = batch["img_b"].to(device, non_blocking=True)
        img_c = batch["img_c"].to(device, non_blocking=True)

        out   = model.forward_triplet(img_a, img_b, img_c, cfg["eval"]["num_iters"])
        losses = criterion(out, img_a, img_b, img_c, model.feature_extractor, stn)
        total_loss += losses["total"].item()

        gt_pts = batch.get("gt_points")
        if gt_pts is not None:
            gt_pts = gt_pts.to(device)
        evaluator.update(
            out["ab"]["H_final"], out["ab"]["masks"], gt_pts,
            cfg["data"]["patch_height"], cfg["data"]["patch_width"],
        )
        n_batches += 1

    avg = total_loss / max(n_batches, 1)
    results = evaluator.summary()

    writer.add_scalar("val/loss_total", avg, step)
    if results:
        writer.add_scalar("val/mean_error",  results["mean_error"],      step)
        writer.add_scalar("val/inlier_pct",  results["mean_inlier_pct"], step)

    log.info(
        f"  [Val] loss={avg:.4f}  "
        + ("  ".join(f"{k}={v:.3f}" for k, v in results.items()) if results else "no GT")
    )
    model.train()
    return results.get("mean_error", avg)


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train(cfg, args):
    # ── GPU selection via CUDA_VISIBLE_DEVICES ─────────────────────────────────
    gpus = args.gpus
    use_cuda = torch.cuda.is_available() and not (len(gpus) == 1 and gpus[0] < 0)
    if use_cuda:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus))
        device = torch.device("cuda:0")
        log.info(f"Using GPU(s) {gpus}  →  CUDA_VISIBLE_DEVICES={os.environ['CUDA_VISIBLE_DEVICES']}")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")

    if args.workers is not None:
        cfg["data"]["num_workers"] = args.workers

    train_loader = build_train_loader(cfg)
    val_loader   = build_val_loader(cfg)
    log.info(f"Train: {len(train_loader.dataset):,} triplets | Val: {len(val_loader.dataset):,}")

    model = build_model(cfg).to(device)
    if use_cuda and len(gpus) > 1:
        model = nn.DataParallel(model, device_ids=list(range(len(gpus))))
        log.info(f"DataParallel: {len(gpus)} GPU(s) (device_ids={list(range(len(gpus)))})")
    # raw_model: unwrapped module — used for custom methods, optimizer params, and I/O
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    n_params = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
    log.info(f"Trainable parameters: {n_params:,}")

    criterion = TotalLoss(cfg)
    stn       = HomographySTN().to(device)

    optimizer = torch.optim.AdamW(
        raw_model.parameters(),
        lr=cfg["train"]["lr"],
        betas=(cfg["train"]["beta1"], cfg["train"]["beta2"]),
        eps=cfg["train"]["eps"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer,
        milestones=cfg["train"]["lr_step_epochs"],
        gamma=cfg["train"]["lr_decay"],
    )

    start_epoch, global_step, best_error = 0, 0, float("inf")
    if args.resume:
        ckpt = load_checkpoint(args.resume, raw_model, optimizer, scheduler)
        start_epoch  = ckpt["epoch"] + 1
        global_step  = ckpt["global_step"]
        best_error   = ckpt["best_error"]

    ckpt_dir = Path(cfg["train"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(ckpt_dir / "tb_logs"))

    max_epochs     = cfg["train"]["epochs"]
    warmup_epochs  = cfg["train"]["warmup_epochs"]
    log_interval   = cfg["train"]["log_interval"]
    save_interval  = cfg["train"]["save_interval"]
    clip_norm      = cfg["train"]["clip_grad_norm"]
    num_iters      = cfg["model"]["num_iters_train"]

    log.info(f"Training for {max_epochs} epochs (warmup={warmup_epochs})...")

    for epoch in range(start_epoch, max_epochs):
        raw_model.set_warmup_mode(epoch < warmup_epochs)
        if epoch < warmup_epochs:
            log.info(f"Epoch {epoch}: WARMUP stage (mask attention disabled)")

        model.train()
        running = {k: 0.0 for k in ["total", "recon", "triplet", "geo", "triangle"]}
        n_batches = 0
        t0 = time.time()

        for batch in tqdm(train_loader, desc=f"Epoch {epoch:3d}", leave=False):
            img_a = batch["img_a"].to(device, non_blocking=True)
            img_b = batch["img_b"].to(device, non_blocking=True)
            img_c = batch["img_c"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            out    = raw_model.forward_triplet(img_a, img_b, img_c, num_iters)
            losses = criterion(out, img_a, img_b, img_c, raw_model.feature_extractor, stn)
            losses["total"].backward()
            nn.utils.clip_grad_norm_(raw_model.parameters(), clip_norm)
            optimizer.step()

            for k in running:
                if k in losses:
                    v = losses[k]
                    running[k] += v.item() if isinstance(v, torch.Tensor) else float(v)
            n_batches   += 1
            global_step += 1

            if global_step % log_interval == 0:
                step_losses = {
                    k: v.item() if isinstance(v, torch.Tensor) else float(v)
                    for k, v in losses.items()
                }
                log_to_tensorboard(
                    writer=writer, step=global_step, losses=step_losses,
                    img_a=img_a.detach().cpu(), img_b=img_b.detach().cpu(),
                    H_final=out["ab"]["H_final"].detach().cpu(),
                    masks=out["ab"]["masks"].detach().cpu(),
                    flow_init=out["ab"]["flow_init"].detach().cpu(),
                    prefix="train",
                )

        elapsed = time.time() - t0
        avg = {k: v / max(n_batches, 1) for k, v in running.items()}
        lr_now = scheduler.get_last_lr()[0]
        log.info(
            f"Epoch {epoch:3d} [{elapsed:.0f}s] lr={lr_now:.2e}  "
            + "  ".join(f"{k}={v:.4f}" for k, v in avg.items())
        )
        writer.add_scalar("train/lr", lr_now, global_step)
        scheduler.step()

        val_error = validate(raw_model, val_loader, criterion, device, cfg, global_step, writer)

        if (epoch + 1) % save_interval == 0:
            save_checkpoint(
                str(ckpt_dir / f"epoch_{epoch:03d}.pth"),
                raw_model, optimizer, scheduler, epoch, global_step, best_error,
            )
        if val_error < best_error:
            best_error = val_error
            save_checkpoint(
                str(ckpt_dir / "best_model.pth"),
                raw_model, optimizer, scheduler, epoch, global_step, best_error,
            )
            log.info(f"  ↳ New best model (error={best_error:.4f})")

    writer.close()
    log.info("Training complete.")


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    train(cfg, args)


if __name__ == "__main__":
    main()
