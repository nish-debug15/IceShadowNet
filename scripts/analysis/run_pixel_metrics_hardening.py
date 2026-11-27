"""
scripts/analysis/run_pixel_metrics_hardening.py
===============================================
Item 4 (Block 5): Hardened Pixel Metrics.
  - Reports Dice & IoU at val-chosen threshold across 3 seeds + 3-seed ensemble.
  - Reports Conditional Dice / Conditional IoU restricted to positive patches (y == 1).
  - Applies >= 5 px connected-component post-processing (8-connectivity) per patch
    and re-evaluates Pooled Dice, Pooled IoU, Conditional Dice, and Patch F1.

Outputs:
  - reports/real/pixel_metrics_hardened.json
  - reports/real/pixel_metrics_hardened.md
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.ndimage import label as cc_label
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.unet import IceUNet
from scripts.train_unet import SegPatchDataset, tune_pixel_threshold

PROCESSED_REAL = "data/processed_real"
MODELS_DIR = "models_real"
REPORTS_DIR = "reports/real"


def dice_iou_from_bin(yt: np.ndarray, yp: np.ndarray) -> tuple:
    tp = int(((yp == 1) & (yt == 1)).sum())
    fp = int(((yp == 1) & (yt == 0)).sum())
    fn = int(((yp == 0) & (yt == 1)).sum())
    dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    return float(dice), float(iou), tp, fp, fn


def filter_small_components(bin_masks: np.ndarray, min_px: int = 5) -> np.ndarray:
    struct = np.ones((3, 3), dtype=int)  # 8-connectivity
    out = np.zeros_like(bin_masks, dtype=np.uint8)
    for i in range(len(bin_masks)):
        m = bin_masks[i]
        if m.sum() < min_px:
            continue
        lbl, n_cc = cc_label(m, structure=struct)
        if n_cc == 0:
            continue
        counts = np.bincount(lbl.ravel())
        keep = counts >= min_px
        keep[0] = False
        out[i] = keep[lbl].astype(np.uint8)
    return out


def eval_masks(M_te: np.ndarray, y_te: np.ndarray, probs: np.ndarray, th: float, label: str) -> dict:
    pos_patch = y_te == 1
    raw_bin = (probs >= th).astype(np.uint8)
    cc_bin = filter_small_components(raw_bin, min_px=5)

    r_dice, r_iou, r_tp, r_fp, r_fn = dice_iou_from_bin(M_te.ravel(), raw_bin.ravel())
    rc_dice, rc_iou, _, _, _ = dice_iou_from_bin(M_te[pos_patch].ravel(), raw_bin[pos_patch].ravel())
    r_patch_f1 = float(f1_score(y_te, (raw_bin.sum(axis=(1, 2)) >= 5).astype(int), zero_division=0))

    c_dice, c_iou, c_tp, c_fp, c_fn = dice_iou_from_bin(M_te.ravel(), cc_bin.ravel())
    cc_dice, cc_iou, _, _, _ = dice_iou_from_bin(M_te[pos_patch].ravel(), cc_bin[pos_patch].ravel())
    c_patch_f1 = float(f1_score(y_te, (cc_bin.sum(axis=(1, 2)) >= 5).astype(int), zero_division=0))

    return {
        "model": label,
        "val_th": float(th),
        "raw": {
            "dice": r_dice, "iou": r_iou, "cond_dice": rc_dice, "cond_iou": rc_iou,
            "tp": r_tp, "fp": r_fp, "fn": r_fn, "patch_ge5_f1": r_patch_f1,
        },
        "cc_ge5": {
            "dice": c_dice, "iou": c_iou, "cond_dice": cc_dice, "cond_iou": cc_iou,
            "tp": c_tp, "fp": c_fp, "fn": c_fn, "patch_ge5_f1": c_patch_f1,
        },
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_va, M_va = d_va["images"], d_va["masks"]
    X_te, M_te, y_te = d_te["images"], d_te["masks"], d_te["labels"].astype(int)

    va_loader = DataLoader(SegPatchDataset(X_va, M_va, augment=False), batch_size=64, shuffle=False)
    te_loader = DataLoader(SegPatchDataset(X_te, M_te, augment=False), batch_size=64, shuffle=False)

    results = []
    va_ens, te_ens = [], []
    for seed in [42, 123, 2024]:
        m = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"IceUNet_seed{seed}.pth"), map_location=device))
        m.eval()
        vp, tp_list = [], []
        with torch.no_grad():
            for xb, _ in va_loader:
                with torch.amp.autocast("cuda"):
                    vp.append(torch.sigmoid(m(xb.to(device))).cpu().numpy()[:, 0])
            for xb, _ in te_loader:
                with torch.amp.autocast("cuda"):
                    tp_list.append(torch.sigmoid(m(xb.to(device))).cpu().numpy()[:, 0])
        va_m = np.concatenate(vp, axis=0)
        te_m = np.concatenate(tp_list, axis=0)
        va_ens.append(va_m)
        te_ens.append(te_m)
        v_th, _ = tune_pixel_threshold(M_va.ravel(), va_m.ravel())
        results.append(eval_masks(M_te, y_te, te_m, v_th, f"IceUNet_seed{seed}"))

    va_ens_m = np.mean(va_ens, axis=0)
    te_ens_m = np.mean(te_ens, axis=0)
    ens_th, _ = tune_pixel_threshold(M_va.ravel(), va_ens_m.ravel())
    results.append(eval_masks(M_te, y_te, te_ens_m, ens_th, "IceUNet_3Seed_Ensemble"))

    with open(os.path.join(REPORTS_DIR, "pixel_metrics_hardened.json"), "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "pixel_metrics_hardened.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 5 Item 4: Hardened Pixel Metrics (Conditional Dice & `>=5 px` CC Post-Processing)\n\n")
        mf.write(f"- **Test Set:** `{len(y_te)}` patches (`{int((y_te==1).sum())}` positive patches), `{M_te.size:,}` pixels (`{int(M_te.sum()):,}` positive pixels)\n\n")
        mf.write("| Model | Val Th | Raw Dice / IoU | Raw Cond. Dice (Pos Patches) | CC>=5 Dice / IoU | CC>=5 Cond. Dice | Raw Patch>=5 F1 | CC>=5 Patch>=5 F1 |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            raw, cc = r["raw"], r["cc_ge5"]
            mf.write(
                f"| **{r['model']}** | {r['val_th']:.2f} | {raw['dice']:.4f} / {raw['iou']:.4f} | {raw['cond_dice']:.4f} | "
                f"{cc['dice']:.4f} / {cc['iou']:.4f} | {cc['cond_dice']:.4f} | {raw['patch_ge5_f1']:.4f} | {cc['patch_ge5_f1']:.4f} |\n"
            )

    print("Saved reports/real/pixel_metrics_hardened.json and reports/real/pixel_metrics_hardened.md")


if __name__ == "__main__":
    main()
