# Multi-Plane Unsupervised Deep Homography Estimation

> **"Unsupervised Multi-Plane Deep Homography via Iterative Differentiable DLT and Plane-Aware Masks"**
>
> A rigorous extension of [Zhang et al., ECCV 2020 (Oral Top 2%)](https://link.springer.com/chapter/10.1007/978-3-030-58610-2_45).

---

## What this adds over the original paper

| Component | Zhang et al. 2020 | **This work** |
|-----------|-------------------|---------------|
| Feature extractor | 3-conv, 1 output channel | ResNet-34 + FPN, 128 channels |
| Homography solver | GAP → FC → 8 numbers | **Differentiable weighted DLT** (SVD) |
| Plane model | Single implicit plane | **K soft planes** (partition-of-unity) |
| Mask predictor | Single binary mask | **Transformer** with smoothness + entropy priors |
| Refinement | Single-shot | **RAFT-style GRU**, 6–24 iterations |
| Inverse consistency | Frobenius ‖HH⁻¹−I‖²_F | **Geodesic distance** ‖log(HH⁻¹)‖_F on GL(3) |
| Temporal structure | Ignored | **Triangle consistency** H_ac ≈ H_bc ∘ H_ab |
| Semantic prior | None | Optional **frozen DINOv2** branch |

Setting `num_planes: 1` in the config reduces this model exactly to the original paper's baseline.

---

## Quick Start

### 1 — Install
```bash
git clone <this-repo>
cd deep-homography

# GPU (recommended)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -e .

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -e .
```

### 2 — Verify installation
```bash
python scripts/check_install.py
```

### 3 — Add your videos
```
data/videos/
    scene1.mp4
    scene2.avi
    subdirectory/scene3.mp4
```

### 4 — Train
```bash
python train.py --config configs/default.yaml

# Resume from checkpoint
python train.py --config configs/default.yaml --resume checkpoints/epoch_010.pth

# Monitor
tensorboard --logdir checkpoints/tb_logs
```

### 5 — Run demo on a single video
```bash
python demo.py --video my_video.mp4 --checkpoint checkpoints/best_model.pth
# Output: demo_output/pair_0000.png  (ghost | mask | flow side-by-side)
```

### 6 — Evaluate
```bash
# Prepare eval data from a video
python scripts/prepare_eval_data.py \
    --video data/raw/test.mp4 \
    --out_dir data/eval/MY_CATEGORY \
    --stride 5 --num_pairs 200

# Run evaluation (prints LaTeX-style table)
python evaluate.py \
    --config configs/default.yaml \
    --checkpoint checkpoints/best_model.pth \
    --eval_all data/eval
```

---

## Repository Structure

```
deep-homography/
│
├── deep_homography/               ← Installable Python package
│   ├── __init__.py
│   ├── cli.py                     ← Console-script entry points
│   │
│   ├── data/
│   │   ├── video_dataset.py       ← VideoHomographyDataset + ImagePairDataset
│   │   └── augmentations.py       ← Spatial (shared) + photometric (independent)
│   │
│   ├── models/
│   │   ├── feature_extractor.py   ← ResNet-34 + FPN + optional DINOv2
│   │   ├── correlation.py         ← All-pairs correlation volume (RAFT-style)
│   │   ├── plane_mask_head.py     ← K-plane transformer mask predictor
│   │   ├── differentiable_dlt.py  ← Weighted DLT via SVD (differentiable)
│   │   ├── iterative_refiner.py   ← ConvGRU iterative update
│   │   └── homography_net.py      ← MultiPlaneHomographyNet (full model)
│   │
│   ├── losses/
│   │   └── total_loss.py          ← 6 loss terms + RAFT sequence loss
│   │
│   └── utils/
│       ├── homography_utils.py    ← STN warp, H arithmetic, matrix log
│       ├── metrics.py             ← L2 error, inlier %, evaluator class
│       └── visualization.py       ← Ghost overlay, mask colours, HSV flow
│
├── configs/
│   └── default.yaml               ← All hyperparameters
│
├── data/
│   ├── videos/                    ← Put training .mp4/.avi files here
│   └── eval/                      ← Evaluation image pairs (RE/LT/LL/SF/LF)
│
├── scripts/
│   ├── check_install.py           ← Post-install sanity check
│   └── prepare_eval_data.py       ← Convert raw video → eval format
│
├── tests/
│   └── test_models.py             ← Unit tests for every module
│
├── train.py                       ← Training entry point
├── evaluate.py                    ← Evaluation entry point
├── demo.py                        ← Single-video demo
├── Makefile                       ← Common commands
├── pyproject.toml                 ← Package metadata
└── requirements.txt
```

---

## Module Details

### `data/video_dataset.py`
`VideoHomographyDataset` scans a directory recursively for video files, reads frames with OpenCV, and yields **triplets (I_a, I_b, I_c)** sampled at `t, t+gap, t+2×gap`. The train/val split is performed at the **video level** (not frame level) to prevent temporal leakage — a common mistake in earlier work. `ImagePairDataset` handles pre-extracted `frame_a/` / `frame_b/` pairs matching the paper's evaluation layout.

### `data/augmentations.py`
Two pipelines — `TrainTransform` and `EvalTransform`. Spatial transforms (crop, horizontal flip) are applied **identically** to all frames so the geometric relationship is preserved. Photometric transforms (Gaussian blur, brightness/contrast jitter, additive noise) are applied **independently** per frame to stress-test luminance invariance.

### `models/feature_extractor.py`
`MultiScaleFeatureExtractor` wraps ResNet-34 (adapted for grayscale) with a lightweight FPN, producing three feature maps at strides 4/8/16, each with 128 channels. Weights are **shared** between the two input images, as in the original paper's `f(·)`. An optional frozen DINOv2-ViT-S/14 branch adds semantically-rich features at the finest scale via a learnable gate.

### `models/correlation.py`
`CorrPyramid` builds a 4D all-pairs correlation volume between source and target features, then provides efficient local lookups at multiple scales via `lookup(flow, radius)`. `SoftArgmaxCorrespondence` extracts a dense flow field by computing the expected target position under a softmax attention distribution — the starting point for the DLT solver.

### `models/differentiable_dlt.py`
`DifferentiableDLT` implements the full weighted DLT pipeline: sample N correspondences from the flow field on a regular grid → weight by plane mask → Hartley normalise → build 9×9 coefficient matrix `M = Σ w_i A_i^T A_i` → SVD (smallest right singular vector) → denormalise → canonical scale. Tikhonov regularisation `M ← M + εI` prevents SVD instability early in training. The entire pipeline is differentiable through `torch.linalg.svd`.

### `models/plane_mask_head.py`
`PlaneMaskHead` is a 2-layer cross-attention transformer that predicts K soft masks from concatenated source/target features. Key properties: (1) **partition of unity** — `softmax(logits, dim=K)` ensures `Σ_k M^k(p) = 1` at every pixel; (2) **zero-initialised output** — uniform masks at the start of training avoid early collapse; (3) **two regularisers** — TV smoothness (`mask_tv_loss`) and entropy (`mask_entropy_loss`) encourage spatially coherent, hard plane assignments.

### `models/iterative_refiner.py`
`IterativeRefiner` implements RAFT-style iterative homography refinement. At each step: warp source features by the current H → `CorrPyramid.lookup` at the residual flow → `MotionEncoder` compresses corr + flow → `ConvGRU` updates hidden state and produces Δflow → upsample flow → `DifferentiableDLT` → H^{k,i}. Returns the full list of iterates for use in the RAFT exponential sequence loss.

### `models/homography_net.py`
`MultiPlaneHomographyNet` orchestrates all modules. `forward(img_a, img_b)` returns per-plane homographies, masks, and flows. `forward_triplet(img_a, img_b, img_c)` runs all four pairs (ab, ba, bc, ac) needed for inverse and triangle consistency. `set_warmup_mode(True/False)` controls the two-stage training protocol.

### `losses/total_loss.py`
Six loss terms:

| Term | Formula | Default λ |
|------|---------|-----------|
| Reconstruction | Σ_k M^k · ‖warp(F_a, H^k) − F_b‖₁ | 1.0 |
| Contrastive (maximise) | −‖F_a − F_b‖₁ | 2.0 |
| Geodesic consistency | ‖log(H_ab H_ba)‖_F | 0.1 |
| Triangle consistency ★ | ‖log(H_ac⁻¹ (H_bc∘H_ab))‖_F | 0.1 |
| Mask TV | TV(M^k) | 0.01 |
| Mask entropy | Σ_k M^k log M^k | 0.01 |

★ New — exploits temporal structure that the original paper ignores.

RAFT exponential sequence loss weights each GRU iterate by γ^{N-i} (γ=0.85).

---

## Configuration

All hyperparameters live in `configs/default.yaml`. Key settings:

```yaml
model:
  num_planes: 2          # K=1 reduces to original paper baseline
  use_dino: false        # Enable for low-texture / low-light scenes
  num_iters_train: 12    # GRU iterations during training
  num_iters_eval: 24     # More iterations at test time

train:
  warmup_epochs: 10      # Stage 1: features unmasked (no attention role)
                         # Stage 2: features weighted by M^k
```

---

## Console Scripts (after `pip install -e .`)

```bash
dh-train    --config configs/default.yaml
dh-evaluate --config configs/default.yaml --checkpoint checkpoints/best_model.pth --eval_all data/eval
dh-demo     --video my_video.mp4 --checkpoint checkpoints/best_model.pth
```

---

## Tests

```bash
# All tests
pytest tests/ -v

# Fast (skip full-model integration tests)
pytest tests/ -v -m "not slow"
```

---

## Citation

If you build on this codebase, please cite the original paper:

```bibtex
@inproceedings{zhang2020content,
  title     = {Content-Aware Unsupervised Deep Homography Estimation},
  author    = {Zhang, Jirong and Wang, Chuan and Liu, Shuaicheng and
               Shi, Lanpeng and Jia, Jue and Yu, Jian and Ma, Jiayi},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2020},
  note      = {Oral Presentation, Top 2\%}
}
```
