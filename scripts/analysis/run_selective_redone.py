"""
scripts/analysis/run_selective_redone.py
========================================
Block 6 Item 3: Selective Prediction Redone (Fair Protocol).
Uses the SAME predictor (3-seed IceUNet ensemble mean) for ALL rejection scores:
  1. Ensemble Variance (across 3 seeds)
  2. MC Dropout Entropy (mean over T=20 MC passes of seed 42, or 3-seed avg MC entropy)
  3. |p_ens - 0.5| baseline (0.5 - |p_ens - 0.5|, higher = closer to 0.5)
  4. Random rejection baseline

Patch level:
  - ROC-AUC & F1 (at val-tuned threshold) vs coverage [100%, 90%, 80%, 70%, 60%, 50%]
  - Sub-block bootstrap 95% bands (within-test-set variability)
  - AURC (Area Under Risk-Coverage curve, using 1 - accuracy as risk)
  - Error-detection AUROC (rejection score vs 1[pred != y_true])

Pixel level:
  - PR-AUC on retained pixels (NO Dice on retained pixels, which degenerates when positives are rejected).
  - Plainly reports whether any score improves over full coverage.

Outputs:
  - reports/real/selective_redone.json
  - reports/real/selective_redone.md
  - reports/real/figures/selective_redone.png
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.unet import IceUNet
from src.evaluate import tune_threshold_on_val
from scripts.train_unet import SegPatchDataset
from scripts.analysis.run_uncertainty_and_stitch import enable_mc_dropout, binary_entropy

PROCESSED_REAL = "data/processed_real"
MODELS_DIR = "models_real"
REPORTS_DIR = "reports/real"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


def subblock_band(y_true, y_score, coords, unc, cov, val_th, n_boot=300, seed=42):
    rows = coords[:, 0]
    blocks = []
    for b0 in range(int(rows.min()), int(rows.max()) + 1, 320):
        idx = np.where((rows >= b0) & (rows + 64 <= b0 + 256))[0]
        if len(idx) > 0:
            blocks.append(idx)
    rng = np.random.default_rng(seed)
    aucs, f1s = [], []
    for _ in range(n_boot):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        idx = np.concatenate([blocks[k] for k in chosen])
        k_keep = max(2, int(cov * len(idx)))
        keep = idx[np.argsort(unc[idx])[:k_keep]]
        yt = y_true[keep]
        if len(np.unique(yt)) < 2:
            continue
        aucs.append(roc_auc_score(yt, y_score[keep]))
        f1s.append(f1_score(yt, (y_score[keep] >= val_th).astype(int), zero_division=0))
    return (
        [float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))] if aucs else [0.5, 0.5],
        [float(np.percentile(f1s, 2.5)), float(np.percentile(f1s, 97.5))] if f1s else [0.0, 0.0],
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_va, M_va, y_va = d_va["images"], d_va["masks"], d_va["labels"].astype(int)
    X_te, M_te, y_te = d_te["images"], d_te["masks"], d_te["labels"].astype(int)
    coords_te = d_te["coords"].astype(int)

    va_loader = DataLoader(SegPatchDataset(X_va, M_va, augment=False), batch_size=64, shuffle=False)
    te_loader = DataLoader(SegPatchDataset(X_te, M_te, augment=False), batch_size=64, shuffle=False)

    # 1. Fixed predictor: 3-seed IceUNet ensemble mean (on val & test)
    va_ens_l, te_ens_l, mc_ent_l = [], [], []
    for s in [42, 123, 2024]:
        m = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"IceUNet_seed{s}.pth"), map_location=device))
        m.eval()
        vp, tp = [], []
        with torch.no_grad():
            for xb, _ in va_loader:
                with torch.amp.autocast("cuda"):
                    vp.append(torch.sigmoid(m(xb.to(device))).cpu().numpy()[:, 0])
            for xb, _ in te_loader:
                with torch.amp.autocast("cuda"):
                    tp.append(torch.sigmoid(m(xb.to(device))).cpu().numpy()[:, 0])
        va_ens_l.append(np.concatenate(vp, axis=0))
        te_ens_l.append(np.concatenate(tp, axis=0))

        # MC Dropout T=20 entropy per seed
        enable_mc_dropout(m)
        mc_sum = np.zeros_like(M_te, dtype=np.float64)
        with torch.no_grad():
            for _ in range(20):
                p_t = []
                for xb, _ in te_loader:
                    with torch.amp.autocast("cuda"):
                        p_t.append(torch.sigmoid(m(xb.to(device))).cpu().numpy()[:, 0])
                mc_sum += np.concatenate(p_t, axis=0)
        mc_ent_l.append(binary_entropy((mc_sum / 20.0).astype(np.float32)))

    p_va_ens = np.mean(va_ens_l, axis=0)
    p_te_ens = np.mean(te_ens_l, axis=0)
    ens_var = np.var(te_ens_l, axis=0)
    mc_ent = np.mean(mc_ent_l, axis=0)
    p_dist = 0.5 - np.abs(p_te_ens - 0.5)

    rng = np.random.default_rng(42)
    rand_px = rng.random(p_te_ens.shape, dtype=np.float32)
    rand_patch = rng.random(len(y_te))

    # Fixed val threshold for patch-level ensemble mean_prob
    patch_va_score = p_va_ens.mean(axis=(1, 2))
    patch_te_score = p_te_ens.mean(axis=(1, 2))
    val_patch_th, _ = tune_threshold_on_val(y_va, patch_va_score)
    patch_pred = (patch_te_score >= val_patch_th).astype(int)
    patch_err = (patch_pred != y_te).astype(int)

    scores = {
        "Ensemble Variance": (ens_var.ravel(), ens_var.mean(axis=(1, 2))),
        "MC Entropy": (mc_ent.ravel(), mc_ent.mean(axis=(1, 2))),
        "|p - 0.5|": (p_dist.ravel(), p_dist.mean(axis=(1, 2))),
        "Random": (rand_px.ravel(), rand_patch),
    }

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
    y_px = M_te.ravel()
    p_px = p_te_ens.ravel()

    all_res = {}
    for sname, (u_px, u_pt) in scores.items():
        ord_px = np.argsort(u_px)
        ord_pt = np.argsort(u_pt)
        err_det_auc = float(roc_auc_score(patch_err, u_pt))

        rows = []
        risks = []
        for cov in coverages:
            k_pt = max(2, int(cov * len(y_te)))
            idx_pt = ord_pt[:k_pt]
            yt = y_te[idx_pt]
            ps = patch_te_score[idx_pt]
            p_auc = float(roc_auc_score(yt, ps)) if len(np.unique(yt)) > 1 else 0.5
            p_f1 = float(f1_score(yt, (ps >= val_patch_th).astype(int), zero_division=0))
            risk = float(( (ps >= val_patch_th).astype(int) != yt ).mean())
            risks.append(risk)
            auc_ci, f1_ci = subblock_band(y_te, patch_te_score, coords_te, u_pt, cov, val_patch_th)

            k_px = max(1, int(cov * len(y_px)))
            idx_px = ord_px[:k_px]
            px_pr = float(average_precision_score(y_px[idx_px], p_px[idx_px])) if y_px[idx_px].sum() > 0 else 0.0
            rows.append({
                "coverage": cov,
                "patch_roc_auc": p_auc, "patch_roc_ci": auc_ci,
                "patch_f1": p_f1, "patch_f1_ci": f1_ci,
                "patch_risk": risk,
                "pixel_pr_auc": px_pr,
                "pixel_pos_retained": int(y_px[idx_px].sum()),
            })

        # AURC via trapezoidal rule over coverage descending -> ascending
        cov_asc = np.array(coverages[::-1])
        risk_asc = np.array(risks[::-1])
        aurc = float(np.trapezoid(risk_asc, cov_asc) / (cov_asc[-1] - cov_asc[0]))
        all_res[sname] = {"aurc": aurc, "error_detection_auroc": err_det_auc, "curve": rows}
        print(f"{sname:18s} | AURC={aurc:.4f} | ErrDet-AUC={err_det_auc:.4f} | 50% Patch AUC={rows[-1]['patch_roc_auc']:.4f} | 50% Px PR-AUC={rows[-1]['pixel_pr_auc']:.4f}")

    # Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    cov_pct = [c * 100 for c in coverages]
    styles = {"Ensemble Variance": "o-", "MC Entropy": "s--", "|p - 0.5|": "^:", "Random": "x-."}
    for sname, d in all_res.items():
        ax1.plot(cov_pct, [r["patch_roc_auc"] for r in d["curve"]], styles[sname], lw=2, label=f"{sname} (ErrAUC={d['error_detection_auroc']:.2f})")
        ax2.plot(cov_pct, [r["pixel_pr_auc"] for r in d["curve"]], styles[sname], lw=2, label=sname)
    ax1.set_title("Patch ROC-AUC vs. Coverage (Same Ensemble Predictor)")
    ax1.set_xlabel("Coverage (%)"); ax1.set_ylabel("Patch ROC-AUC"); ax1.invert_xaxis(); ax1.grid(alpha=0.3); ax1.legend(fontsize=8)
    ax2.set_title("Retained Pixel PR-AUC vs. Coverage (Same Ensemble)")
    ax2.set_xlabel("Coverage (%)"); ax2.set_ylabel("Pixel PR-AUC"); ax2.invert_xaxis(); ax2.grid(alpha=0.3); ax2.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "selective_redone.png"), dpi=150)
    plt.close()

    with open(os.path.join(REPORTS_DIR, "selective_redone.json"), "w", encoding="utf-8") as jf:
        json.dump({"val_patch_th": float(val_patch_th), "results": all_res}, jf, indent=2)

    full_auc = all_res["Ensemble Variance"]["curve"][0]["patch_roc_auc"]
    full_px_pr = all_res["Ensemble Variance"]["curve"][0]["pixel_pr_auc"]
    any_patch_improves = any(r["patch_roc_auc"] > full_auc for s in ["Ensemble Variance", "MC Entropy"] for r in all_res[s]["curve"][1:])
    any_px_improves = any(r["pixel_pr_auc"] > full_px_pr for s in ["Ensemble Variance", "MC Entropy"] for r in all_res[s]["curve"][1:])

    with open(os.path.join(REPORTS_DIR, "selective_redone.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 6 Item 3: Selective Prediction Redone (Unified 3-Seed Ensemble Predictor)\n\n")
        mf.write(f"- **Fixed Predictor:** 3-Seed `IceUNet` Ensemble Mean (Full coverage: Patch ROC-AUC = `{full_auc:.4f}`, Pixel PR-AUC = `{full_px_pr:.4f}`)\n")
        mf.write(f"- **Does any uncertainty score improve Patch ROC-AUC over full coverage?** `{'YES' if any_patch_improves else 'NO'}`\n")
        mf.write(f"- **Does any uncertainty score improve Pixel PR-AUC over full coverage?** `{'YES' if any_px_improves else 'NO'}`\n\n")
        if not any_patch_improves:
            mf.write("> **Verdict:** Neither Ensemble Variance nor MC Entropy improves Patch ROC-AUC over full coverage (both drop as high-scoring uncertain patches are rejected); **uncertainty is not usable for selective rejection on this task.**\n\n")

        mf.write("## 1. Summary: AURC & Error-Detection AUROC\n\n")
        mf.write("| Rejection Score | AURC (Lower Better) | Error-Detection AUROC | 100% Patch AUC | 50% Patch AUC [Sub-block 95% CI] | 100% Px PR-AUC | 50% Px PR-AUC |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for sname, d in all_res.items():
            c0, c5 = d["curve"][0], d["curve"][-1]
            mf.write(
                f"| **{sname}** | {d['aurc']:.4f} | {d['error_detection_auroc']:.4f} | "
                f"{c0['patch_roc_auc']:.4f} | {c5['patch_roc_auc']:.4f} `[{c5['patch_roc_ci'][0]:.3f}, {c5['patch_roc_ci'][1]:.3f}]` | "
                f"{c0['pixel_pr_auc']:.4f} | {c5['pixel_pr_auc']:.4f} |\n"
            )

        mf.write("\n## 2. Full Coverage Curves (100% → 50%)\n\n")
        mf.write("| Cov | Ens-Var AUC / F1 | MC-Ent AUC / F1 | `|p-0.5|` AUC / F1 | Random AUC / F1 | Ens-Var Px PR | MC-Ent Px PR | `|p-0.5|` Px PR |\n")
        mf.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for i, cov in enumerate(coverages):
            ev = all_res["Ensemble Variance"]["curve"][i]
            mc = all_res["MC Entropy"]["curve"][i]
            pd = all_res["|p - 0.5|"]["curve"][i]
            rd = all_res["Random"]["curve"][i]
            mf.write(
                f"| {100*cov:.0f}% | {ev['patch_roc_auc']:.3f} / {ev['patch_f1']:.3f} | "
                f"{mc['patch_roc_auc']:.3f} / {mc['patch_f1']:.3f} | {pd['patch_roc_auc']:.3f} / {pd['patch_f1']:.3f} | "
                f"{rd['patch_roc_auc']:.3f} / {rd['patch_f1']:.3f} | "
                f"{ev['pixel_pr_auc']:.4f} | {mc['pixel_pr_auc']:.4f} | {pd['pixel_pr_auc']:.4f} |\n"
            )
        mf.write("\n![Selective Prediction Redone](figures/selective_redone.png)\n")

    print("Saved reports/real/selective_redone.json and reports/real/selective_redone.md")


if __name__ == "__main__":
    main()
