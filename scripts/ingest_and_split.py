"""
scripts/ingest_and_split.py
=============================
Block 1 pipeline runner:
  1. Load DFSAR (+ optionally OHRC) tiles from data/raw/ via rasterio.
     Falls back to synthetic scene if no real tiles are present.
  2. Compute CPR/DOP, generate binary label mask.
  3. Extract patches with generate_patches().
  4. Run spatial_block_split() on patch coordinates.
  5. Assert no cross-split leakage (hard assertion, not just print).
  6. Log class balance to reports/class_balance.txt.
  7. Save the patch list as a numpy .npz file to data/processed/
     so train.py can load it without re-running preprocessing.

Usage:
    python -m scripts.ingest_and_split
"""

import os
import sys
import warnings
import numpy as np

# Allow running as `python -m scripts.ingest_and_split` from repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.preprocessing.data_ops import (
    load_dfsar, load_ohrc, compute_cpr_dop,
    label_from_threshold, generate_patches,
)
from src.datasets.spatial_split import spatial_block_split


# ── Config ────────────────────────────────────────────────────────────────────
RAW_DIR        = "data/raw"
PROCESSED_DIR  = "data/processed"
REPORTS_DIR    = "reports"
PATCH_SIZE     = 256
STRIDE         = 128
GRID_SIZE      = (4, 4)   # spatial split grid (rows x cols)
TRAIN_RATIO    = 0.70
VAL_RATIO      = 0.15
SEED           = 42
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR,   exist_ok=True)


def find_raw_files(raw_dir):
    """Scan data/raw/ for .IMG, .img, .tif, .tiff files."""
    exts = {".img", ".tif", ".tiff"}
    found = {}
    if not os.path.isdir(raw_dir):
        return found
    for f in os.listdir(raw_dir):
        lower = f.lower()
        if any(lower.endswith(e) for e in exts):
            key = "dfsar" if "dfsar" in lower or "sar" in lower else "ohrc"
            found.setdefault(key, []).append(os.path.join(raw_dir, f))
    return found


def main():
    np.random.seed(SEED)
    raw_files = find_raw_files(RAW_DIR)

    # ── 1. Load DFSAR ─────────────────────────────────────────────────────────
    dfsar_path = raw_files.get("dfsar", [None])[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        dfsar = load_dfsar(dfsar_path if dfsar_path else "data/raw/DFSAR_PLACEHOLDER.img")
        is_synthetic = any("SYNTHETIC" in str(warning.message) for warning in w)

    if is_synthetic:
        data_source = "SYNTHETIC (no real DFSAR tiles found in data/raw/)"
        print("[SYNTHETIC] No real DFSAR tiles — using physically-motivated synthetic scene.")
        print("            See reports/data_notes.md for acquisition status.")
    else:
        data_source = f"REAL — {dfsar_path}"
        print(f"[REAL] Loaded DFSAR: {dfsar_path}, shape={dfsar.shape}")

    # ── 2. Load OHRC (optional) ───────────────────────────────────────────────
    ohrc_path = raw_files.get("ohrc", [None])[0]
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        ohrc = load_ohrc(ohrc_path if ohrc_path else "data/raw/OHRC_PLACEHOLDER.img")
        ohrc_synthetic = any("SYNTHETIC" in str(warning.message) for warning in w)
    if ohrc_synthetic:
        print("[SYNTHETIC] No real OHRC tile — using synthetic optical scene.")

    # ── 3. CPR / DOP ──────────────────────────────────────────────────────────
    cpr, dop = compute_cpr_dop(dfsar)
    print(f"CPR  stats: min={cpr.min():.3f}  max={cpr.max():.3f}  mean={cpr.mean():.3f}")
    print(f"DOP  stats: min={dop.min():.3f}  max={dop.max():.3f}  mean={dop.mean():.3f}")

    # ── 4. Binary label mask ──────────────────────────────────────────────────
    label_mask = label_from_threshold(cpr, dop)
    ice_px     = int(label_mask.sum())
    total_px   = label_mask.size
    print(f"Ice pixels : {ice_px:,} / {total_px:,}  ({100*ice_px/total_px:.1f}%)")

    # ── 5. Patch extraction ───────────────────────────────────────────────────
    all_patches = generate_patches(dfsar, label_mask, patch_size=PATCH_SIZE, stride=STRIDE)
    coords = [(c[2][0], c[2][1]) for c in all_patches]   # (row, col) per patch

    ice_patches    = [p for p in all_patches if p[1] == 1]
    nonice_patches = [p for p in all_patches if p[1] == 0]
    n_ice   = len(ice_patches)
    n_noice = len(nonice_patches)
    n_total = len(all_patches)
    ice_ratio  = n_ice / max(n_total, 1)
    print(f"\nPatch extraction complete:")
    print(f"  Total patches : {n_total}")
    print(f"  Ice patches   : {n_ice}  ({100*ice_ratio:.1f}%)")
    print(f"  Non-ice       : {n_noice} ({100*(1-ice_ratio):.1f}%)")

    # ── 6. Spatial split ──────────────────────────────────────────────────────
    train_coords, val_coords, test_coords = spatial_block_split(
        coordinates=coords,
        grid_size=GRID_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
    )

    # Hard assertion: no coordinate in more than one split
    train_set = set(train_coords)
    val_set   = set(val_coords)
    test_set  = set(test_coords)
    assert train_set.isdisjoint(val_set),  "LEAKAGE DETECTED: train ∩ val is non-empty!"
    assert train_set.isdisjoint(test_set), "LEAKAGE DETECTED: train ∩ test is non-empty!"
    assert val_set.isdisjoint(test_set),   "LEAKAGE DETECTED: val ∩ test is non-empty!"
    print(f"\nSpatial split verified (no leakage):")
    print(f"  Train: {len(train_coords)}  Val: {len(val_coords)}  Test: {len(test_coords)}")

    # Map coords → patches for saving
    coord_to_patch = {(p[2][0], p[2][1]): p for p in all_patches}
    def coords_to_patches(clist):
        return [coord_to_patch[c] for c in clist if c in coord_to_patch]

    train_patches = coords_to_patches(train_coords)
    val_patches   = coords_to_patches(val_coords)
    test_patches  = coords_to_patches(test_coords)

    # ── 7. Save processed patches ────────────────────────────────────────────
    def save_split(patches, name):
        imgs   = np.stack([p[0] for p in patches])
        labels = np.array([p[1] for p in patches], dtype=np.int8)
        coords_ = np.array([p[2] for p in patches], dtype=np.int32)
        path   = os.path.join(PROCESSED_DIR, f"{name}.npz")
        np.savez_compressed(path, images=imgs, labels=labels, coords=coords_)
        print(f"  Saved {name}.npz -> {len(patches)} patches  shape={imgs.shape}")
        return labels

    print("\nSaving processed splits...")
    lbl_train = save_split(train_patches, "train")
    lbl_val   = save_split(val_patches,   "val")
    lbl_test  = save_split(test_patches,  "test")

    # ── 8. Class balance report ───────────────────────────────────────────────
    pos_weight = n_noice / max(n_ice, 1)   # for BCEWithLogitsLoss pos_weight arg
    balance_txt = (
        f"Data source          : {data_source}\n"
        f"OHRC source          : {'SYNTHETIC' if ohrc_synthetic else ohrc_path}\n"
        f"Patch size / stride  : {PATCH_SIZE} / {STRIDE}\n"
        f"Total patches        : {n_total}\n"
        f"  Ice (positive)     : {n_ice}  ({100*ice_ratio:.1f}%)\n"
        f"  Non-ice (negative) : {n_noice} ({100*(1-ice_ratio):.1f}%)\n"
        f"pos_weight for BCE   : {pos_weight:.2f}  (= n_noice / n_ice)\n"
        f"\nTrain / Val / Test   : {len(train_patches)} / {len(val_patches)} / {len(test_patches)}\n"
        f"  Train ice rate     : {lbl_train.sum()/max(len(lbl_train),1)*100:.1f}%\n"
        f"  Val   ice rate     : {lbl_val.sum()/max(len(lbl_val),1)*100:.1f}%\n"
        f"  Test  ice rate     : {lbl_test.sum()/max(len(lbl_test),1)*100:.1f}%\n"
        f"\nSpatial split grid   : {GRID_SIZE}\n"
        f"Leakage assertion    : PASSED (no coordinate in >1 split)\n"
        f"\nRECOMMENDATION:\n"
        f"  pos_weight = {pos_weight:.2f} -> use BCEWithLogitsLoss(pos_weight=tensor([{pos_weight:.2f}]))\n"
        f"  Focal loss is also viable given strong imbalance if ice_ratio < 0.15\n"
    )

    balance_path = os.path.join(REPORTS_DIR, "class_balance.txt")
    with open(balance_path, "w", encoding="utf-8") as f:
        f.write(balance_txt)
    print(f"\nClass balance report saved to {balance_path}")
    print(balance_txt)
    print("Block 1 pipeline complete.")


if __name__ == "__main__":
    main()
