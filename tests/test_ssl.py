"""
tests/test_ssl.py
=================
Unit tests for Self-Supervised Learning (SimCLR) architecture and NT-Xent loss.
"""

import torch
import pytest
from src.models.simclr_ssl import SARSimCLR, NTXentLoss


def test_simclr_forward_shape():
    model = SARSimCLR(in_channels=3, proj_dim=128)
    x = torch.randn(4, 3, 64, 64)
    z = model(x)
    assert z.shape == (4, 128)
    # Check L2 normalization
    norms = torch.norm(z, dim=1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_simclr_feature_extraction():
    model = SARSimCLR(in_channels=3)
    x = torch.randn(2, 3, 64, 64)
    feat = model.get_features(x)
    assert feat.shape == (2, 512)


def test_ntxent_loss():
    criterion = NTXentLoss(temperature=0.2)
    z1 = torch.randn(4, 128)
    z2 = torch.randn(4, 128)
    z1 = torch.nn.functional.normalize(z1, dim=1)
    z2 = torch.nn.functional.normalize(z2, dim=1)
    loss = criterion(z1, z2)
    assert loss.dim() == 0
    assert loss.item() > 0.0
