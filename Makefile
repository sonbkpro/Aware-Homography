# ============================================================
# Makefile — Unsupervised Multi-Plane Deep Homography
# ============================================================
# Usage:
#   make install       Install the package in editable mode
#   make train         Start training with default config
#   make eval          Evaluate on all categories
#   make demo          Run single-pair demo
#   make test          Run unit tests
#   make clean         Remove build artefacts

PYTHON     ?= python
CONFIG     ?= configs/default.yaml
CHECKPOINT ?= checkpoints/best_model.pth
EVAL_ROOT  ?= data/eval
GPU        ?= 0

.PHONY: install train eval demo test clean lint format

# ---- Setup ------------------------------------------------------------------

install:
	$(PYTHON) -m pip install -e ".[dev]" --quiet
	@echo "✓ Package installed in editable mode"

install-cpu:
	$(PYTHON) -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -e . --quiet
	@echo "✓ CPU-only install complete"

# ---- Training ---------------------------------------------------------------

train:
	$(PYTHON) train.py --config $(CONFIG) --gpu $(GPU)

train-resume:
	$(PYTHON) train.py --config $(CONFIG) --resume $(CHECKPOINT) --gpu $(GPU)

# ---- Evaluation -------------------------------------------------------------

eval:
	$(PYTHON) evaluate.py \
		--config $(CONFIG) \
		--checkpoint $(CHECKPOINT) \
		--eval_all $(EVAL_ROOT) \
		--gpu $(GPU)

eval-category:
	$(PYTHON) evaluate.py \
		--config $(CONFIG) \
		--checkpoint $(CHECKPOINT) \
		--data_root $(EVAL_ROOT)/$(CAT) \
		--gt_dir $(EVAL_ROOT)/$(CAT)/gt_points \
		--category $(CAT) \
		--gpu $(GPU)

# ---- Demo -------------------------------------------------------------------

demo:
	$(PYTHON) demo.py --video $(VIDEO) --checkpoint $(CHECKPOINT) --gpu $(GPU)

# ---- Tests ------------------------------------------------------------------

test:
	$(PYTHON) -m pytest tests/ -v --tb=short

test-fast:
	$(PYTHON) -m pytest tests/ -v --tb=short -m "not slow"

# ---- Code quality -----------------------------------------------------------

lint:
	$(PYTHON) -m ruff check deep_homography/ train.py evaluate.py demo.py

format:
	$(PYTHON) -m ruff format deep_homography/ train.py evaluate.py demo.py

# ---- Cleanup ----------------------------------------------------------------

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf build/ dist/ *.egg-info/ .pytest_cache/ .ruff_cache/
	@echo "✓ Clean"
