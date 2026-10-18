"""
src/models/multimodal_fusion.py
=================================
DFSAR + OHRC multimodal fusion models for IceShadowNet.

Two fusion variants are implemented:

1. EarlyFusionCNN
   Concatenates registered DFSAR (2-ch) + OHRC (1-ch) → 3-channel input.
   Feeds into a 3-channel CustomCNN variant.  Simple but assumes perfect
   spatial registration.

2. LateFusionResNet
   Separate feature extractors for DFSAR and OHRC, fused at the penultimate
   feature level by concatenation before the final classifier.  More robust
   to minor registration errors because each branch learns modality-specific
   features independently.

Fusion design tradeoff (viva-defensible):
   Early fusion is parameter-efficient but conflates radar and optical statistics
   from the very first convolution — problematic because DFSAR has coherent speckle
   while OHRC has optical shot noise, and both have completely different dynamic ranges.
   Late fusion allows each branch to normalise and extract features appropriate to
   its modality before combination, but doubles the parameter count.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.custom_cnn import CustomCNN
from src.models.resnet import IceResNet


class EarlyFusionCNN(nn.Module):
    """
    Early fusion: concatenate all input channels before the first conv.
    Input: (B, dfsar_channels + ohrc_channels, H, W)
    """
    def __init__(self, dfsar_channels: int = 2, ohrc_channels: int = 1, num_classes: int = 1):
        super().__init__()
        self.backbone = CustomCNN(
            in_channels=dfsar_channels + ohrc_channels,
            num_classes=num_classes,
        )

    def forward(self, dfsar: torch.Tensor, ohrc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            dfsar: (B, 2, H, W)
            ohrc:  (B, 1, H, W)
        Returns:
            logits (B, 1)
        """
        x = torch.cat([dfsar, ohrc], dim=1)  # (B, 3, H, W)
        return self.backbone(x)


class LateFusionResNet(nn.Module):
    """
    Late fusion: separate ResNet backbones for DFSAR and OHRC, merged before FC.
    """
    def __init__(
        self,
        dfsar_channels: int = 2,
        ohrc_channels:  int = 1,
        pretrained:     bool = False,
        num_classes:    int = 1,
    ):
        super().__init__()

        # ── DFSAR branch ──
        dfsar_resnet = IceResNet(in_channels=dfsar_channels, num_classes=512, pretrained=pretrained)
        # Strip the final FC to use as a feature extractor
        dfsar_resnet.model.fc = nn.Identity()
        self.dfsar_branch = dfsar_resnet

        # ── OHRC branch ──
        ohrc_resnet = IceResNet(in_channels=ohrc_channels, num_classes=512, pretrained=pretrained)
        ohrc_resnet.model.fc = nn.Identity()
        self.ohrc_branch = ohrc_resnet

        # ── Fusion head ──
        # ResNet18 feature dimension = 512 per branch → 1024 after cat
        self.fusion = nn.Sequential(
            nn.Linear(512 + 512, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, dfsar: torch.Tensor, ohrc: torch.Tensor) -> torch.Tensor:
        f_dfsar = self.dfsar_branch(dfsar)   # (B, 512)
        f_ohrc  = self.ohrc_branch(ohrc)     # (B, 512)
        fused   = torch.cat([f_dfsar, f_ohrc], dim=1)  # (B, 1024)
        return self.fusion(fused)


if __name__ == "__main__":
    B, H, W = 4, 256, 256
    dfsar = torch.randn(B, 2, H, W)
    ohrc  = torch.randn(B, 1, H, W)

    early = EarlyFusionCNN()
    print("EarlyFusionCNN out:", early(dfsar, ohrc).shape)

    late = LateFusionResNet(pretrained=False)
    print("LateFusionResNet out:", late(dfsar, ohrc).shape)
