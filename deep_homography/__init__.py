"""
deep_homography
===============
Unsupervised Multi-Plane Deep Homography Estimation.

"Unsupervised Multi-Plane Deep Homography via Iterative
 Differentiable DLT and Plane-Aware Masks"

Extends Zhang et al. (ECCV 2020) with:
  • K-plane partition-of-unity masks (transformer head)
  • Differentiable weighted DLT (no GAP→FC regression)
  • RAFT-style GRU iterative refinement
  • Geodesic SL(3) inverse consistency
  • Triangle consistency across video triplets
  • Optional DINOv2 semantic prior
"""

__version__ = "1.0.0"
__author__  = "Multi-Plane Homography Research"
