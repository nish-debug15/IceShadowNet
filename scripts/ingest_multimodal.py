"""
scripts/ingest_multimodal.py
==============================
Extend Block 1 data ingestion to produce a DFSAR+OHRC multimodal patch dataset.

Produces data/processed/{train,val,test}_multimodal.npz
Each sample: images_dfsar (B,2,H,W), images_ohrc (B,1,H,W), labels (B,).

Usage:
    python -m scripts.ingest_multimodal
"""

import os, sys, warnings
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.data_ops import (
    load_dfsar, load_ohrc, compute_cpr_dop,
    label_from_threshold, generate_patches,
)
from src.preprocessing.register_ohrc import register_ohrc_to_dfsar
from src.datasets.spatial_split import spatial_block_split

RAW_DIR       = "data/raw"
PROCESSED_DIR = "data/processed"
REPORTS_DIR   = "reports"
PATCH_SIZE    = 256
STRIDE        = 128
GRID_SIZE     = (4, 4)
TRAIN_RATIO   = 0.70
VAL_RATIO     = 0.15
SEED          = 42

os.makedirs(PROCESSED_DIR, exist_ok=True)


def main():
    np.random.seed(SEED)

    dfsar_path = None
    ohrc_path  = None
    for f in os.listdir(RAW_DIR) if os.path.isdir(RAW_DIR) else []:
        fl = f.lower()
        if "dfsar" in fl or "sar" in fl:
            dfsar_path = os.path.join(RAW_DIR, f)
        elif "ohrc" in fl:
            ohrc_path = os.path.join(RAW_DIR, f)

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        dfsar = load_dfsar(dfsar_path or "data/raw/DFSAR_PLACEHOLDER.img")
        ohrc  = load_ohrc(ohrc_path  or "data/raw/OHRC_PLACEHOLDER.img")

    # Register OHRC to DFSAR grid
    ohrc_reg = register_ohrc_to_dfsar(dfsar, ohrc, dfsar_path, ohrc_path)
    print(f"OHRC registered to DFSAR grid: {ohrc_reg.shape}")

    cpr, dop   = compute_cpr_dop(dfsar)
    label_mask = label_from_threshold(cpr, dop)

    # Build combined (DFSAR || OHRC) patch list
    dfsar_patches = generate_patches(dfsar, label_mask, PATCH_SIZE, STRIDE)

    # Extract matching OHRC patches — use same coords as DFSAR patches
    ohrc_stacked = ohrc_reg[:, :, np.newaxis]  # (H, W, 1)
    ohrc_patches_dict = {}
    for _, lbl, (r, c) in dfsar_patches:
        op = ohrc_stacked[r:r+PATCH_SIZE, c:c+PATCH_SIZE]
        ohrc_patches_dict[(r, c)] = np.transpose(op, (2, 0, 1)).astype(np.float32)

    # Spatial split (same seed → same block assignments as DFSAR-only run)
    coords = [(p[2][0], p[2][1]) for p in dfsar_patches]
    train_c, val_c, test_c = spatial_block_split(coords, GRID_SIZE, TRAIN_RATIO, VAL_RATIO)

    # Leakage assertion
    assert set(train_c).isdisjoint(val_c),  "LEAKAGE: train ∩ val"
    assert set(train_c).isdisjoint(test_c), "LEAKAGE: train ∩ test"
    assert set(val_c).isdisjoint(test_c),   "LEAKAGE: val ∩ test"

    coord_to_dfsar  = {(p[2][0], p[2][1]): p for p in dfsar_patches}

    def save_mm_split(clist, name):
        imgs_d, imgs_o, lbls = [], [], []
        for c in clist:
            if c not in coord_to_dfsar:
                continue
            img_d, lbl, _ = coord_to_dfsar[c]
            img_o = ohrc_patches_dict[c]
            imgs_d.append(img_d)
            imgs_o.append(img_o)
            lbls.append(lbl)
        if not imgs_d:
            print(f"  WARNING: no patches for split {name}")
            return
        path = os.path.join(PROCESSED_DIR, f"{name}_multimodal.npz")
        np.savez_compressed(
            path,
            images_dfsar=np.stack(imgs_d),
            images_ohrc =np.stack(imgs_o),
            labels      =np.array(lbls, dtype=np.int8),
        )
        print(f"  Saved {name}_multimodal.npz  ({len(lbls)} patches)")

    print("Saving multimodal splits...")
    save_mm_split(train_c, "train")
    save_mm_split(val_c,   "val")
    save_mm_split(test_c,  "test")
    print("Multimodal ingestion complete.")


if __name__ == "__main__":
    main()
