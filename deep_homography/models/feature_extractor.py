"""
feature_extractor.py
--------------------
Multi-scale feature extractor with:
  1. ResNet-34 backbone (matches original paper's h(·) backbone choice).
  2. Feature Pyramid Network (FPN) for 3 resolution levels: stride 4, 8, 16.
  3. Optional frozen DINOv2-ViT-S/14 side branch for semantic guidance in
     low-texture / low-light scenes (Direction G from the review).

The backbone is SHARED between the two input images -- weight-tied just as
in f(·) and m(·) in the original paper.

Output:
    A dict of feature maps at multiple scales:
      { 'level2': (B, feat_dim, H/4,  W/4 ),
        'level3': (B, feat_dim, H/8,  W/8 ),
        'level4': (B, feat_dim, H/16, W/16) }

Design note on channels:
    The FPN projects all levels to `feat_dim=128` channels.
    This is a deliberate divergence from the original paper's 3-conv
    extractor with only 1 output channel -- the richer channel depth
    allows the correlation volume (see correlation.py) to capture
    fine-grained feature interactions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from typing import Dict, Optional


# ---------------------------------------------------------------------------
# FPN neck -- projects ResNet feature pyramid to uniform channel width
# ---------------------------------------------------------------------------

class FPN(nn.Module):
    """
    Lightweight top-down Feature Pyramid Network.

    Takes a list of feature maps [c2, c3, c4] with increasing semantic
    content but decreasing spatial resolution and produces a list of maps
    [p2, p3, p4] all with `out_channels` channels.

    Architecture (per level):
        lateral 1×1 conv  → add top-down upsampled   → 3×3 conv
    """

    def __init__(self, in_channels: list, out_channels: int = 128):
        """
        Args:
            in_channels: [C_level2, C_level3, C_level4] from ResNet.
            out_channels: Uniform output channel width.
        """
        super().__init__()
        # Lateral 1×1 projections
        self.lat2 = nn.Conv2d(in_channels[0], out_channels, 1)
        self.lat3 = nn.Conv2d(in_channels[1], out_channels, 1)
        self.lat4 = nn.Conv2d(in_channels[2], out_channels, 1)

        # 3×3 output convolutions
        self.out2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.out4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, c2, c3, c4):
        """
        Args:
            c2: (B, C2, H/4,  W/4)
            c3: (B, C3, H/8,  W/8)
            c4: (B, C4, H/16, W/16)
        Returns:
            p2: (B, out_ch, H/4,  W/4)
            p3: (B, out_ch, H/8,  W/8)
            p4: (B, out_ch, H/16, W/16)
        """
        l4 = self.lat4(c4)
        l3 = self.lat3(c3) + F.interpolate(l4, size=c3.shape[-2:], mode="nearest")
        l2 = self.lat2(c2) + F.interpolate(l3, size=c2.shape[-2:], mode="nearest")

        p4 = self.out4(l4)
        p3 = self.out3(l3)
        p2 = self.out2(l2)
        return p2, p3, p4


# ---------------------------------------------------------------------------
# Optional DINOv2 side branch
# ---------------------------------------------------------------------------

class DINOv2Branch(nn.Module):
    """
    Frozen DINOv2-ViT-S/14 patch feature extractor.

    DINOv2 features are semantically rich, luminance-invariant, and work
    well in low-texture / low-light conditions -- directly addressing the
    weakness documented in the original paper.

    We freeze all weights (no gradients) and use the patch token grid as
    a spatial feature map, resized to match the FPN output.

    Args:
        out_channels: Project DINOv2 384-dim tokens to this width.
        target_level: Which FPN level to merge with ('level2'|'level3').
    """

    def __init__(self, out_channels: int = 128, target_level: str = "level2"):
        super().__init__()
        try:
            self.dino = torch.hub.load(
                "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
            )
        except Exception:
            # Graceful fallback if no internet / hub fails
            print("[DINOv2Branch] WARNING: Could not load DINOv2. Disabling branch.")
            self.dino = None
            self.project = None
            return

        # Freeze DINOv2
        for p in self.dino.parameters():
            p.requires_grad = False

        # Linear projection: 384 → out_channels
        self.project = nn.Conv2d(384, out_channels, 1)
        self.target_level = target_level

    def forward(self, x: torch.Tensor, target_h: int, target_w: int) -> Optional[torch.Tensor]:
        """
        Args:
            x: (B, 1 or 3, H, W) input image (values in original range).
            target_h, target_w: Spatial size to match FPN output.
        Returns:
            Projected DINOv2 feature map (B, out_channels, target_h, target_w)
            or None if DINOv2 unavailable.
        """
        if self.dino is None:
            return None

        # DINOv2 expects 3-channel normalized input
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # Resize to multiple of 14 (ViT-S/14 patch size)
        H, W = x.shape[-2:]
        new_H = (H // 14) * 14
        new_W = (W // 14) * 14
        x_resized = F.interpolate(x, (new_H, new_W), mode="bilinear", align_corners=False)

        with torch.no_grad():
            # Extract intermediate patch tokens (shape: B, N_patches, 384)
            feats = self.dino.forward_features(x_resized)
            patch_tokens = feats["x_norm_patchtokens"]  # (B, nH*nW, 384)

        nH = new_H // 14
        nW = new_W // 14
        B = patch_tokens.shape[0]
        # Reshape to spatial: (B, 384, nH, nW)
        spatial = patch_tokens.permute(0, 2, 1).reshape(B, 384, nH, nW)
        # Project channels
        projected = self.project(spatial)
        # Resize to target resolution
        projected = F.interpolate(
            projected, (target_h, target_w), mode="bilinear", align_corners=False
        )
        return projected


# ---------------------------------------------------------------------------
# Main feature extractor
# ---------------------------------------------------------------------------

class MultiScaleFeatureExtractor(nn.Module):
    """
    Weight-shared multi-scale feature extractor.

    Call it once with I_a to get {F_a_level2, F_a_level3, F_a_level4},
    and again with I_b to get {F_b_level2, F_b_level3, F_b_level4}.
    The SAME weights are used for both (as in the original paper's f(·)).

    Architecture:
        ResNet-34 stem + layer1/2/3  →  FPN  →  3 feature maps
        [optional] DINOv2 branch     →  additive fusion at level2

    Args:
        backbone (str):    'resnet18' | 'resnet34' | 'resnet50'.
        feat_dim (int):    Uniform output channel width (FPN output).
        use_dino (bool):   Whether to fuse frozen DINOv2 features.
        pretrained (bool): Use ImageNet-pretrained backbone weights.
    """

    _RESNET_CHANNELS = {
        "resnet18":  [64,  128, 256],   # layer1, layer2, layer3 out-channels
        "resnet34":  [64,  128, 256],
        "resnet50":  [256, 512, 1024],
    }

    def __init__(
        self,
        backbone: str = "resnet34",
        feat_dim: int = 128,
        use_dino: bool = False,
        pretrained: bool = True,
    ):
        super().__init__()
        assert backbone in self._RESNET_CHANNELS, f"Unknown backbone: {backbone}"

        # ---- Backbone ----
        weights = "IMAGENET1K_V1" if pretrained else None
        resnet = getattr(tvm, backbone)(weights=weights)

        # Adapt first conv for grayscale (1-channel) input
        old_conv = resnet.conv1
        resnet.conv1 = nn.Conv2d(
            1, old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False,
        )
        # Initialize with mean of RGB weights
        if pretrained:
            with torch.no_grad():
                resnet.conv1.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))

        # Split ResNet into stages (we don't use layer4 to preserve resolution)
        self.stem    = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1  = resnet.layer1  # stride 4
        self.layer2  = resnet.layer2  # stride 8
        self.layer3  = resnet.layer3  # stride 16

        # ---- FPN ----
        in_ch = self._RESNET_CHANNELS[backbone]
        self.fpn = FPN(in_channels=in_ch, out_channels=feat_dim)

        # ---- Optional DINOv2 ----
        self.use_dino = use_dino
        if use_dino:
            self.dino_branch = DINOv2Branch(out_channels=feat_dim, target_level="level2")
            # Learnable gate scalar to blend DINO into level2
            self.dino_gate = nn.Parameter(torch.tensor(0.1))
        else:
            self.dino_branch = None

        self.feat_dim = feat_dim

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, 1, H, W) grayscale image tensor, normalized.
        Returns:
            dict:
              'level2': (B, feat_dim, H//4,  W//4)
              'level3': (B, feat_dim, H//8,  W//8)
              'level4': (B, feat_dim, H//16, W//16)
        """
        # Backbone feature extraction
        x0 = self.stem(x)          # (B,  64, H/4,  W/4)
        c2 = self.layer1(x0)       # (B, C2,  H/4,  W/4)
        c3 = self.layer2(c2)       # (B, C3,  H/8,  W/8)
        c4 = self.layer3(c3)       # (B, C4,  H/16, W/16)

        # FPN fusion
        p2, p3, p4 = self.fpn(c2, c3, c4)

        # DINOv2 fusion (additive with learnable gate)
        if self.use_dino and self.dino_branch is not None:
            dino_feat = self.dino_branch(x, p2.shape[-2], p2.shape[-1])
            if dino_feat is not None:
                p2 = p2 + torch.sigmoid(self.dino_gate) * dino_feat

        return {
            "level2": p2,   # finest scale, used for DLT correspondences
            "level3": p3,
            "level4": p4,   # coarsest scale, used for GRU context
        }
