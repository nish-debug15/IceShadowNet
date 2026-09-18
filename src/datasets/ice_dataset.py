"""
src/datasets/ice_dataset.py
=============================
PyTorch Dataset that wraps pre-extracted ice/non-ice patches.

Changes from Phase 1
---------------------
- __getitem__ now loads from an in-memory patch list (real or synthetic)
  rather than generating random noise on the fly.
- Augmentation uses the real augment() function from preprocessing.data_ops.
- Supports optional per-sample normalisation (zero-mean, unit-variance per channel)
  which is important for SAR data with large dynamic range.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.preprocessing.data_ops import augment as _augment


class IcePatchDataset(Dataset):
    """
    Dataset wrapping a list of (patch_image, label, coord) tuples.

    Args:
        patch_list (list):  Output of generate_patches() — each element is
                            (img_chw float32, label int, (row, col)).
        augment (bool):     Apply random flips/rotations on-the-fly (train only).
        normalize (bool):   Normalise each patch to zero mean / unit std per channel.
    """

    def __init__(self, patch_list: list, augment: bool = False, normalize: bool = True):
        self.patches   = patch_list
        self.do_augment = augment
        self.normalize  = normalize

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        img_chw, label, _ = self.patches[idx]

        # Clone to avoid mutating the shared patch list
        img = img_chw.copy()

        if self.do_augment:
            img = _augment(img)

        if self.normalize:
            # Per-channel zero-mean / unit-std normalisation.
            # Clamp std to avoid division by zero (can happen in synthetic uniform patches).
            for c in range(img.shape[0]):
                mu  = img[c].mean()
                std = img[c].std()
                img[c] = (img[c] - mu) / (std + 1e-8)

        return (
            torch.from_numpy(img),
            torch.tensor([label], dtype=torch.float32),
        )
