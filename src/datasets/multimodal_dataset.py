"""
src/datasets/multimodal_dataset.py
=====================================
PyTorch Dataset for DFSAR + OHRC multimodal patches.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from src.preprocessing.data_ops import augment as _augment


class MultimodalIceDataset(Dataset):
    """
    Wraps a multimodal_*.npz file (images_dfsar, images_ohrc, labels).

    Returns:
        (dfsar_tensor, ohrc_tensor, label_tensor)
    """

    def __init__(self, npz_path: str, augment: bool = False, normalize: bool = True):
        data = np.load(npz_path)
        self.dfsar   = data["images_dfsar"].astype(np.float32)  # (N, 2, H, W)
        self.ohrc    = data["images_ohrc"].astype(np.float32)   # (N, 1, H, W)
        self.labels  = data["labels"].astype(np.float32)
        self.augment    = augment
        self.normalize  = normalize

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        d = self.dfsar[idx].copy()
        o = self.ohrc[idx].copy()
        lbl = self.labels[idx]

        if self.augment:
            # Apply same spatial augmentation to both modalities
            combined = np.concatenate([d, o], axis=0)  # (3, H, W)
            combined = _augment(combined)
            d = combined[:2]
            o = combined[2:]

        if self.normalize:
            for c in range(d.shape[0]):
                d[c] = (d[c] - d[c].mean()) / (d[c].std() + 1e-8)
            o[0] = (o[0] - o[0].mean()) / (o[0].std() + 1e-8)

        return (
            torch.from_numpy(d),
            torch.from_numpy(o),
            torch.tensor([lbl], dtype=torch.float32),
        )
