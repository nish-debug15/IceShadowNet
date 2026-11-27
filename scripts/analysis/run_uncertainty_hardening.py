"""
scripts/analysis/run_uncertainty_hardening.py
=============================================
Item 3 (Block 5): Hardened Uncertainty Analysis.
  (a) Selective-prediction curves (patch ROC-AUC & pixel Dice vs coverage 100..50%)
      for MC-entropy, Ensemble-variance, and a |p - 0.5| baseline (0.5 - |p - 0.5|).
  (b) Class-conditional calibration on positives, Brier score, and Brier Skill Score (BSS)
      vs prevalence (0.0527%).
  (c) Exact boundary uncertainty breakdown (interior vs 1-2px ring vs far background).

Outputs:
  - reports/real/uncertainty_hardened.json
  - reports/real/uncertainty_hardened.md
  - reports/real/figures/selective_prediction_hardened.png
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from scipy.ndimage import binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.unet import IceUNet
from scripts.train_unet import SegPatchDataset, tune_pixel_threshold
from scripts.analysis.run_uncertainty_and_stitch import enable_mc_dropout, binary_entropy

PROCESSED_REAL = "data/processed_real"
MODELS_DIR = "models_real"
REPORTS_DIR = "reports/real"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def brier_and_skill(y_true: np.ndarray, probs: np.ndarray) -> dict:
    brier = float(np.mean((probs - y_true) ** 2))
    prev = float(y_true.mean())
    brier_ref = prev * (1.0 - prev)
    bss = 1.0 - (brier / brier_ref) if brier_ref > 0 else 0.0
    pos_mask = y_true == 1
    neg_mask = y_true == 0
    mean_p_pos = float(probs[pos_mask].mean()) if pos_mask.any() else 0.0
    mean_p_neg = float(probs[neg_mask].mean()) if neg_mask.any() else 0.0
    return {
        "brier": brier,
        "brier_ref_prevalence": float(brier_ref),
        "brier_skill_score": float(bss),
        "mean_prob_on_positives": mean_p_pos,
        "mean_prob_on_negatives": mean_p_neg,
        "pos_calibration_gap": float(1.0 - mean_p_pos),
    }


def selective_curves(y_px: np.ndarray, p_px: np.ndarray, y_patch: np.ndarray, p_patch: np.ndarray,
                     unc_px: np.ndarray, unc_patch: np.ndarray, th: float, coverages: list):
    order_px = np.argsort(unc_px)
    order_pt = np.argsort(unc_patch)
    n_px = len(y_px)
    n_pt = len(y_patch)
    rows = []
    for cov in coverages:
        k_px = max(1, int(cov * n_px))
        idx_px = order_px[:k_px]
        yt = y_px[idx_px]
        pred = (p_px[idx_px] >= th).astype(np.uint8)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)

        k_pt = max(2, int(cov * n_pt))
        idx_pt = order_pt[:k_pt]
        yp_t = y_patch[idx_pt]
        pp_t = p_patch[idx_pt]
        pt_auc = float(roc_auc_score(yp_t, pp_t)) if len(np.unique(yp_t)) > 1 else 0.5
        rows.append({
            "coverage": cov,
            "pixel_dice": float(dice),
            "patch_roc_auc": pt_auc,
            "retained_pos_px": int(yt.sum()),
            "retained_pos_patches": int(yp_t.sum()),
        })
    return rows


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_va, M_va = d_va["images"], d_va["masks"]
    X_te, M_te, y_te = d_te["images"], d_te["masks"], d_te["labels"].astype(int)

    va_loader = DataLoader(SegPatchDataset(X_va, M_va, augment=False), batch_size=64, shuffle=False)
    te_loader = DataLoader(SegPatchDataset(X_te, M_te, augment=False), batch_size=64, shuffle=False)

    # 1. MC Dropout (T=20) on seed 42
    model_42 = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
    model_42.load_state_dict(torch.load(os.path.join(MODELS_DIR, "IceUNet_seed42.pth"), map_location=device))

    model_42.eval()
    va_det = []
    with torch.no_grad():
        for xb, _ in va_loader:
            with torch.amp.autocast("cuda"):
                va_det.append(torch.sigmoid(model_42(xb.to(device))).cpu().numpy()[:, 0])
    val_th, _ = tune_pixel_threshold(M_va.ravel(), np.concatenate(va_det, axis=0).ravel())

    enable_mc_dropout(model_42)
    T = 20
    mc_sum = np.zeros_like(M_te, dtype=np.float64)
    with torch.no_grad():
        for _ in range(T):
            preds = []
            for xb, _ in te_loader:
                with torch.amp.autocast("cuda"):
                    preds.append(torch.sigmoid(model_42(xb.to(device))).cpu().numpy()[:, 0])
            mc_sum += np.concatenate(preds, axis=0)
    mc_mean = (mc_sum / T).astype(np.float32)
    mc_ent = binary_entropy(mc_mean).astype(np.float32)

    # 2. 3-Seed Ensemble
    ens_preds = []
    for s in [42, 123, 2024]:
        m = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"IceUNet_seed{s}.pth"), map_location=device))
        m.eval()
        ps = []
        with torch.no_grad():
            for xb, _ in te_loader:
                with torch.amp.autocast("cuda"):
                    ps.append(torch.sigmoid(m(xb.to(device))).cpu().numpy()[:, 0])
        ens_preds.append(np.concatenate(ps, axis=0))
    ens_stack = np.stack(ens_preds, axis=0)
    ens_mean = ens_stack.mean(axis=0).astype(np.float32)
    ens_var = ens_stack.var(axis=0).astype(np.float32)

    # 3. |p - 0.5| uncertainty baseline (0.5 - |p - 0.5| so higher = closer to 0.5 = more uncertain)
    p_dist_mc = (0.5 - np.abs(mc_mean - 0.5)).astype(np.float32)

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    y_px = M_te.ravel()
    mc_px = mc_mean.ravel()
    ens_px = ens_mean.ravel()

    sel_mc = selective_curves(y_px, mc_px, y_te, mc_mean.mean(axis=(1, 2)), mc_ent.ravel(), mc_ent.mean(axis=(1, 2)), val_th, coverages)
    sel_ens = selective_curves(y_px, ens_px, y_te, ens_mean.mean(axis=(1, 2)), ens_var.ravel(), ens_var.mean(axis=(1, 2)), val_th, coverages)
    sel_pdist = selective_curves(y_px, mc_px, y_te, mc_mean.mean(axis=(1, 2)), p_dist_mc.ravel(), p_dist_mc.mean(axis=(1, 2)), val_th, coverages)

    # Plot selective prediction comparison (Patch ROC-AUC & Pixel Dice)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    cov_pct = [c * 100 for c in coverages]
    for label, rows, style in [
        ("MC Entropy", sel_mc, "o-"),
        ("Ensemble Variance", sel_ens, "s--"),
        ("|p - 0.5| Baseline", sel_pdist, "^:"),
    ]:
        ax1.plot(cov_pct, [r["patch_roc_auc"] for r in rows], style, lw=2, label=label)
        ax2.plot(cov_pct, [r["pixel_dice"] for r in rows], style, lw=2, label=label)

    ax1.set_title("Patch ROC-AUC vs. Coverage")
    ax1.set_xlabel("Coverage (% lowest-uncertainty patches)")
    ax1.set_ylabel("Patch Mean-Prob ROC-AUC")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.set_title("Pixel Dice vs. Coverage")
    ax2.set_xlabel("Coverage (% lowest-uncertainty pixels)")
    ax2.set_ylabel("Pixel Dice")
    ax2.invert_xaxis()
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, "selective_prediction_hardened.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()

    # 4. Brier score, BSS, class-conditional calibration
    cal_mc = brier_and_skill(y_px, mc_px)
    cal_ens = brier_and_skill(y_px, ens_px)

    # 5. Boundary breakdown
    gt_int = M_te == 1
    dil = np.stack([binary_dilation(m.astype(bool), iterations=2) for m in M_te], axis=0)
    gt_bnd = dil & (~gt_int)
    far_bg = ~dil
    boundary = {
        rname: {
            "n_px": int(mask.sum()),
            "mc_mean_p": float(mc_mean[mask].mean()),
            "mc_entropy": float(mc_ent[mask].mean()),
            "ens_var": float(ens_var[mask].mean()),
        }
        for rname, mask in [("GT Interior", gt_int), ("GT Boundary Ring (1-2px)", gt_bnd), ("Far Background", far_bg)]
    }

    out = {
        "val_th": val_th,
        "calibration_brier": {"mc_dropout": cal_mc, "deep_ensemble": cal_ens},
        "selective_prediction": {"mc_entropy": sel_mc, "ensemble_variance": sel_ens, "p_minus_05_baseline": sel_pdist},
        "boundary_breakdown": boundary,
    }
    with open(os.path.join(REPORTS_DIR, "uncertainty_hardened.json"), "w", encoding="utf-8") as jf:
        json.dump(out, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "uncertainty_hardened.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 5 Item 3: Hardened Uncertainty & Calibration Analysis\n\n")
        mf.write("## 1. Brier Score, Brier Skill Score (vs. Prevalence 0.0527%), & Class-Conditional Calibration\n\n")
        mf.write("| Method | Brier Score | Ref Brier ($p(1-p)$) | Brier Skill Score | Mean $p$ on Positives | Pos Calibration Gap ($1 - \\bar{p}_{pos}$) |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for label, c in [("MC Dropout (T=20)", cal_mc), ("3-Seed Ensemble", cal_ens)]:
            mf.write(
                f"| **{label}** | {c['brier']:.6f} | {c['brier_ref_prevalence']:.6f} | "
                f"{c['brier_skill_score']:.4f} | {c['mean_prob_on_positives']:.4f} | {c['pos_calibration_gap']:.4f} |\n"
            )

        mf.write("\n## 2. Selective Prediction vs. `|p - 0.5|` Baseline (Coverage 100% → 50%)\n\n")
        mf.write("| Coverage | MC-Ent Patch AUC | Ens-Var Patch AUC | `|p-0.5|` Patch AUC | MC-Ent Px Dice | Ens-Var Px Dice | `|p-0.5|` Px Dice |\n")
        mf.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for i, cov in enumerate(coverages):
            mf.write(
                f"| {100*cov:.0f}% | {sel_mc[i]['patch_roc_auc']:.4f} | {sel_ens[i]['patch_roc_auc']:.4f} | {sel_pdist[i]['patch_roc_auc']:.4f} | "
                f"{sel_mc[i]['pixel_dice']:.4f} | {sel_ens[i]['pixel_dice']:.4f} | {sel_pdist[i]['pixel_dice']:.4f} |\n"
            )

        mf.write("\n## 3. Boundary Uncertainty Breakdown (Exact Measured Behavior)\n\n")
        mf.write("Predictive entropy tracks proximity of $p$ to $0.5$: `GT Interior` pixels have the highest mean predicted probability "
                 f"(`{boundary['GT Interior']['mc_mean_p']:.4f}`) and therefore the highest MC entropy (`{boundary['GT Interior']['mc_entropy']:.5f}`), "
                 f"followed by the `1–2 px` boundary ring (`{boundary['GT Boundary Ring (1-2px)']['mc_entropy']:.5f}`) and far background (`{boundary['Far Background']['mc_entropy']:.5f}`).\n\n")
        mf.write("| Region | Pixels | Mean $p$ (MC) | MC Entropy | Ensemble Variance |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for rname, st in boundary.items():
            mf.write(f"| {rname} | {st['n_px']:,} | {st['mc_mean_p']:.4f} | {st['mc_entropy']:.5f} | {st['ens_var']:.6f} |\n")
        mf.write("\n![Hardened Selective Prediction](figures/selective_prediction_hardened.png)\n")

    print("Saved reports/real/uncertainty_hardened.json and reports/real/uncertainty_hardened.md")


if __name__ == "__main__":
    main()
