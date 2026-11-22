"""
src/models/simclr_ssl.py
========================
Self-Supervised Contrastive Learning (SimCLR) for Lunar SAR.

Key components:
1. SARSimCLREncoder: ResNet-18 base encoder adapted for 3-channel radar log-intensity
   with an MLP projection head.
2. SARDataAugmentation: SAR-appropriate augmentations (speckle noise, gain jitter,
   flips, 90-degree rotations, random resized crop). Strictly NO RGB color jitter.
3. NTXentLoss: Normalized Temperature-scaled Cross Entropy loss for self-supervised
   contrastive learning.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class SARSimCLR(nn.Module):
    """
    SimCLR framework with a ResNet-18 backbone and 2-layer projection head.
    """
    def __init__(self, in_channels: int = 3, proj_dim: int = 128, hidden_dim: int = 512):
        super(SARSimCLR, self).__init__()
        
        # ResNet-18 backbone
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Feature dimension before fc is 512
        feat_dim = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()  # Replace fc with identity to get embeddings

        # Non-linear projection head (MLP)
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        """Forward pass returns normalized projection vectors for contrastive loss."""
        h = self.backbone(x)          # (B, 512)
        z = self.projector(h)         # (B, 128)
        z = F.normalize(z, dim=1)     # L2 normalization
        return z

    def get_features(self, x):
        """Return backbone representation h before projection."""
        return self.backbone(x)


class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross Entropy Loss (SimCLR loss).
    """
    def __init__(self, temperature: float = 0.2):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        z_i, z_j: (B, D) normalized representations of two augmented views.
        """
        batch_size = z_i.shape[0]
        # Concatenate: [z_i; z_j] has shape (2*B, D)
        z = torch.cat([z_i, z_j], dim=0)

        # Similarity matrix: (2*B, 2*B)
        sim = torch.mm(z, z.t()) / self.temperature

        # Create mask to zero out self-similarity on the diagonal (safe for fp16)
        diag_mask = torch.eye(2 * batch_size, device=z.device, dtype=torch.bool)
        sim.masked_fill_(diag_mask, -1e4)

        # Positive targets: view 1 corresponds to view 2 (idx + B), view 2 corresponds to view 1 (idx - B)
        targets_i = torch.arange(batch_size, 2 * batch_size, device=z.device)
        targets_j = torch.arange(0, batch_size, device=z.device)
        targets = torch.cat([targets_i, targets_j], dim=0)

        loss = F.cross_entropy(sim, targets)
        return loss
