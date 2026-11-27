"""
scripts/analysis/run_context_pixel_baselines.py
===============================================
Block 6 Item 2: Context-Matched Pixel Baselines.
Extracts per-pixel features over 7x7 and 15x15 spatial windows:
  - Local mean & std of HH, HV, VV log-intensities (6 features per window)
  - Local HV/HH (I_hv - I_hh) and VV/HH (I_vv - I_hh) ratios (2 features per window)
Evaluates LogisticRegression and HistGradientBoostingClassifier (LightGBM-style GBDT)
on (a) 7x7 features (8 dims) and (b) 7x7 + 15x15 features (16 dims), trained on train
patches, tuned on val, and evaluated on test with Pixel PR-AUC, lift vs 0.0527%,
and Pixel Dice (val th & best test th).

Outputs:
  - reports/real/context_pixel_baselines.json
  - reports/real/context_pixel_baselines.md
"""

import os
import sys
import json
import numpy as np
from scipy.ndimage import uniform_filter
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from scripts.train_unet import pixel_metrics, tune_pixel_threshold

PROCESSED_REAL = "data/processed_real"
REPORTS_DIR = "reports/real"


def compute_window_features(images: np.ndarray, windows=(7, 15)) -> np.ndarray:
    """
    Compute per-pixel window features per patch (no cross-patch leakage).
    images: (N, 3, 64, 64) raw log-intensities [HH, HV, VV].
    Returns (N * 64 * 64, 8 * len(windows)) float32.
    """
    N, _, H, W = images.shape
    # Per-patch standardization matches IceUNet preprocessing? Let's compute on raw log-intensities
    # so ratios I_hv - I_hh and I_vv - I_hh retain physical dB/log-ratio meaning!
    feats = []
    for k in windows:
        # uniform_filter over axes (2, 3) only -> size=(1, 1, k, k)
        mu = uniform_filter(images, size=(1, 1, k, k), mode="reflect")
        mu2 = uniform_filter(images ** 2, size=(1, 1, k, k), mode="reflect")
        std = np.sqrt(np.maximum(mu2 - mu ** 2, 0.0))
        hv_hh = mu[:, 1:2] - mu[:, 0:1]
        vv_hh = mu[:, 2:3] - mu[:, 0:1]
        block = np.concatenate([mu, std, hv_hh, vv_hh], axis=1)  # (N, 8, H, W)
        feats.append(block)
    out = np.concatenate(feats, axis=1)  # (N, 8*len(windows), H, W)
    return np.transpose(out, (0, 2, 3, 1)).reshape(-1, out.shape[1]).astype(np.float32)


def main():
    d_tr = np.load(os.path.join(PROCESSED_REAL, "train.npz"))
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_tr_img, M_tr = d_tr["images"], d_tr["masks"].ravel()
    X_va_img, M_va = d_va["images"], d_va["masks"].ravel()
    X_te_img, M_te = d_te["images"], d_te["masks"].ravel()

    prev = float(M_te.mean())
    print(f"Test pixel prevalence: {100*prev:.4f}% ({int(M_te.sum()):,} / {M_te.size:,})")

    rng = np.random.default_rng(42)
    pos_idx = np.where(M_tr == 1)[0]
    neg_idx = rng.choice(np.where(M_tr == 0)[0], size=min(250000, int((M_tr == 0).sum())), replace=False)
    fit_idx = np.concatenate([pos_idx, neg_idx])

    results = []
    for win_label, wins in [("7x7", (7,)), ("7x7+15x15", (7, 15))]:
        print(f"\nComputing {win_label} context features...")
        F_tr = compute_window_features(X_tr_img, windows=wins)
        F_va = compute_window_features(X_va_img, windows=wins)
        F_te = compute_window_features(X_te_img, windows=wins)

        scaler = StandardScaler()
        F_tr_fit = scaler.fit_transform(F_tr[fit_idx])
        F_va_s = scaler.transform(F_va)
        F_te_s = scaler.transform(F_te)

        models = [
            (f"Context Pixel LogReg ({win_label})", LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)),
            (f"Context Pixel HistGBDT ({win_label})", HistGradientBoostingClassifier(max_iter=150, max_depth=6, learning_rate=0.08, class_weight="balanced", random_state=42)),
        ]
        for mname, clf in models:
            clf.fit(F_tr_fit, M_tr[fit_idx])
            va_p = clf.predict_proba(F_va_s)[:, 1]
            te_p = clf.predict_proba(F_te_s)[:, 1]
            val_th, _ = tune_pixel_threshold(M_va, va_p)
            best_th, _ = tune_pixel_threshold(M_te, te_p)
            met_val = pixel_metrics(M_te, te_p, val_th)
            met_best = pixel_metrics(M_te, te_p, best_th)
            lift = met_val["pr_auc"] / prev
            rec = {
                "model": mname,
                "windows": win_label,
                "n_features": int(F_tr.shape[1]),
                "val_th": float(val_th),
                "best_test_th": float(best_th),
                "pr_auc": met_val["pr_auc"],
                "lift_over_prevalence": float(lift),
                "dice_val_th": met_val["dice"],
                "iou_val_th": met_val["iou"],
                "dice_best_th": met_best["dice"],
            }
            results.append(rec)
            print(f"  {mname}: PR-AUC={rec['pr_auc']:.4f} ({lift:.1f}x lift) | Dice(val_th={val_th:.2f})={rec['dice_val_th']:.4f} | Dice(best)={rec['dice_best_th']:.4f}")

    # Load IceUNet & context-free baselines for side-by-side table
    unet_rep = json.load(open(os.path.join(REPORTS_DIR, "unet_segmentation_report.json"), "r", encoding="utf-8"))
    unc_rep = json.load(open(os.path.join(REPORTS_DIR, "uncertainty_report.json"), "r", encoding="utf-8"))

    with open(os.path.join(REPORTS_DIR, "context_pixel_baselines.json"), "w", encoding="utf-8") as jf:
        json.dump({"pixel_prevalence": prev, "context_baselines": results}, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "context_pixel_baselines.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 6 Item 2: Context-Matched Pixel Baselines (`7×7` and `15×15` Windows) vs. `IceUNet`\n\n")
        mf.write(f"- **Test Pixel Prevalence (PR-AUC Baseline):** `{100*prev:.4f}%` (`{int(M_te.sum()):,} / {M_te.size:,}` pixels)\n\n")
        mf.write("| Model | Context | Pixel PR-AUC | Lift vs. 0.0527% | Dice (Val Th) | IoU (Val Th) | Dice (Best Test Th) |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for bname, bkey in [("HV/HH Ratio (context-free)", "hv_hh_ratio"), ("Pixel LogReg (context-free)", "pixel_logreg")]:
            b = unet_rep["baselines"][bkey]
            pv, pb = b["pooled_val_th"], b["pooled_best_th"]
            mf.write(f"| {bname} | 1×1 | {pv['pr_auc']:.4f} | {pv['pr_auc']/prev:.1f}× | {pv['dice']:.4f} | {pv['iou']:.4f} | {pb['dice']:.4f} |\n")
        for r in results:
            mf.write(
                f"| **{r['model']}** | {r['windows']} | {r['pr_auc']:.4f} | {r['lift_over_prevalence']:.1f}× | "
                f"{r['dice_val_th']:.4f} | {r['iou_val_th']:.4f} | {r['dice_best_th']:.4f} |\n"
            )
        u_pr = float(np.mean([r["pooled_val_th"]["pr_auc"] for r in unet_rep["unet_runs"]]))
        u_dice = float(np.mean([r["pooled_val_th"]["dice"] for r in unet_rep["unet_runs"]]))
        u_iou = float(np.mean([r["pooled_val_th"]["iou"] for r in unet_rep["unet_runs"]]))
        u_bdice = float(np.mean([r["pooled_best_th"]["dice"] for r in unet_rep["unet_runs"]]))
        mf.write(f"| **IceUNet (3-seed mean)** | 64×64 | {u_pr:.4f} | {u_pr/prev:.1f}× | {u_dice:.4f} | {u_iou:.4f} | {u_bdice:.4f} |\n")
        ens = unc_rep["deep_ensemble"]
        mf.write(f"| **IceUNet 3-Seed Ensemble** | 64×64 | {ens['pr_auc']:.4f} | {ens['pr_auc']/prev:.1f}× | {ens['dice']:.4f} | {ens['iou']:.4f} | — |\n")

    print("Saved reports/real/context_pixel_baselines.json and reports/real/context_pixel_baselines.md")


if __name__ == "__main__":
    main()
