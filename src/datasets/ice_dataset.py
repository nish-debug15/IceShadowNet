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


def get_ambiguity_mask(
    ice_fracs: np.ndarray,
    min_px: int = 1,
    max_px: int = 4,
    patch_size: int = 64,
) -> np.ndarray:
    """
    Return a boolean mask selecting patches whose positive-pixel count falls
    strictly in the ambiguity band [min_px, max_px] (e.g. 1 to 4 positive pixels).

    Args:
        ice_fracs: Array of float fraction of positive pixels in each patch.
        min_px: Lower bound of ambiguous pixel count (inclusive).
        max_px: Upper bound of ambiguous pixel count (inclusive).
        patch_size: Square patch dimension in pixels.

    Returns:
        Boolean numpy array of shape (N,).
    """
    pos_px = np.round(ice_fracs * (patch_size * patch_size)).astype(int)
    return (pos_px >= min_px) & (pos_px <= max_px)


def get_ambiguous_patches(
    npz_data_or_path,
    min_px: int = 1,
    max_px: int = 4,
    patch_size: int = 64,
) -> dict:
    """
    Extract the subset of patches falling into the ambiguity band.

    Args:
        npz_data_or_path: Path to a processed .npz file or loaded npz dictionary.
        min_px: Minimum positive pixels for ambiguity (default: 1).
        max_px: Maximum positive pixels for ambiguity (default: 4).
        patch_size: Patch size in pixels (default: 64).

    Returns:
        Dictionary containing indices, images, labels, coords, ice_fracs, and count.
    """
    if isinstance(npz_data_or_path, str):
        data = np.load(npz_data_or_path)
    else:
        data = npz_data_or_path

    ice_fracs = data["ice_fracs"]
    mask = get_ambiguity_mask(ice_fracs, min_px=min_px, max_px=max_px, patch_size=patch_size)
    return {
        "indices": np.where(mask)[0],
        "images": data["images"][mask],
        "labels": data["labels"][mask],
        "coords": data["coords"][mask],
        "ice_fracs": data["ice_fracs"][mask],
        "count": int(mask.sum()),
    }

