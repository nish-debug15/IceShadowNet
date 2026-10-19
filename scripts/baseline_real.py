"""
scripts/baseline_real.py
========================
Classical baseline evaluation on real Chandrayaan-2 DFSAR polarimetric data.

Implements classical baselines evaluated on the spatial test split:
1. ORACLE (label rule, not a model)
2. Decision Tree (max_depth=3, class_weight='balanced') on raw [HH, HV, VV]
3. Logistic Regression (class_weight='balanced') on raw [HH, HV, VV]

Protocol enhancements:
- Validation-tuned threshold (maximizing F1 on val, applied to test)
- PR-AUC and test prevalence
- Block-bootstrap 95% Confidence Intervals across test strips (1000 rounds)
- Ambiguity-removed evaluation (patches with 1-4 positive pixels excluded)

Usage:
    python -m scripts.baseline_real
"""

import os
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

from src.evaluate import (
    tune_threshold_on_val,
    compute_comprehensive_metrics,
    block_bootstrap_evaluation,
    get_strip_ids_from_coords,
)
from src.datasets.ice_dataset import get_ambiguity_mask

PROCESSED_REAL = "data/processed_real"
REPORTS_REAL   = "reports/real"
os.makedirs(REPORTS_REAL, exist_ok=True)


def extract_features(data_dict):
    """
    Extract raw backscatter features per patch.
    Model input features: strictly raw log-intensities [HH, HV, VV].
    CPR and DOP are strictly excluded from model inputs.
    """
    images    = data_dict["images"]       # (N, 3, 64, 64): HH, HV, VV log-intensity
    cpr_means = data_dict["cpr_means"]    # (N,)
    dop_means = data_dict["dop_means"]    # (N,)
    ice_fracs = data_dict["ice_fracs"]    # (N,)

    # Compute raw intensity channel statistics
    hh_means = images[:, 0].mean(axis=(1, 2))
    hv_means = images[:, 1].mean(axis=(1, 2))
    vv_means = images[:, 2].mean(axis=(1, 2))

    X = np.stack([hh_means, hv_means, vv_means], axis=-1)
    return X, cpr_means, dop_means, ice_fracs


def main():
    train_path = os.path.join(PROCESSED_REAL, "train.npz")
    val_path   = os.path.join(PROCESSED_REAL, "val.npz")
    test_path  = os.path.join(PROCESSED_REAL, "test.npz")

    for p in (train_path, val_path, test_path):
        if not os.path.isfile(p):
            raise FileNotFoundError(
                f"Processed real split '{p}' not found. "
                "Run `python -m scripts.ingest_and_split --real` first."
            )

    train_data = np.load(train_path)
    val_data   = np.load(val_path)
    test_data  = np.load(test_path)

    y_train = train_data["labels"]
    y_val   = val_data["labels"]
    y_test  = test_data["labels"]

    X_train, _, _, _ = extract_features(train_data)
    X_val,   _, _, _ = extract_features(val_data)
    X_test,  _, _, frac_test = extract_features(test_data)

    test_coords = test_data["coords"]
    test_strips = get_strip_ids_from_coords(test_coords)

    # Ambiguity mask for clean evaluation
    amb_mask = get_ambiguity_mask(test_data["ice_fracs"], min_px=1, max_px=4, patch_size=64)
    clean_mask = ~amb_mask
    y_test_clean = y_test[clean_mask]
    X_test_clean = X_test[clean_mask]
    test_strips_clean = test_strips[clean_mask]

    n_pos_test = int(y_test.sum())
    n_neg_test = int(len(y_test) - n_pos_test)
    test_prev = float(np.mean(y_test))

    print("="*70)
    print("EVALUATING CLASSICAL BASELINES ON REAL DFSAR")
    print(f"Train split : {len(y_train)} patches (Pos: {y_train.sum()}, Neg: {len(y_train)-y_train.sum()})")
    print(f"Val split   : {len(y_val)} patches (Pos: {y_val.sum()}, Neg: {len(y_val)-y_val.sum()})")
    print(f"Test split  : {len(y_test)} patches (Pos: {n_pos_test}, Neg: {n_neg_test}, Prev: {100*test_prev:.2f}%)")
    print(f"Test Clean  : {len(y_test_clean)} patches (Pos: {y_test_clean.sum()}, Neg: {len(y_test_clean)-y_test_clean.sum()}, Dropped Ambiguous: {amb_mask.sum()})")
    print("="*70)

    # ──────────────────────────────────────────────────────────────────────────
    # ORACLE: Label Rule Evaluation (Not an ML model)
    # ──────────────────────────────────────────────────────────────────────────
    oracle_pred = (frac_test >= (5.0 / (64 * 64))).astype(int)
    oracle_metrics = compute_comprehensive_metrics(y_test, frac_test, threshold=(5.0 / (64 * 64)))
    oracle_dict = {
        "name": "ORACLE (label rule, not a model)",
        "accuracy": oracle_metrics["accuracy"],
        "precision": oracle_metrics["precision"],
        "recall": oracle_metrics["recall"],
        "f1": oracle_metrics["f1"],
        "roc_auc": oracle_metrics["roc_auc"],
        "pr_auc": oracle_metrics["pr_auc"],
        "support": oracle_metrics["support"],
    }

    # ──────────────────────────────────────────────────────────────────────────
    # Classical Models
    # ──────────────────────────────────────────────────────────────────────────
    models_spec = [
        ("Decision Tree (max_depth=3, balanced)", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)),
        ("Logistic Regression (balanced)", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)),
    ]

    results_primary = []
    results_clean = []

    for name, clf in models_spec:
        print(f"\n--- Evaluating {name} ---")
        clf.fit(X_train, y_train)

        val_probs  = clf.predict_proba(X_val)[:, 1]
        test_probs = clf.predict_proba(X_test)[:, 1]

        # 1. Tune threshold on validation set
        tuned_th, val_f1 = tune_threshold_on_val(y_val, val_probs)
        print(f"  Validation tuned threshold: {tuned_th:.2f} (Val F1: {val_f1:.4f})")

        # 2. Comprehensive metrics on Test (tuned threshold)
        test_met = compute_comprehensive_metrics(y_test, test_probs, threshold=tuned_th)

        # 3. Block-bootstrap 95% CIs
        boot = block_bootstrap_evaluation(y_test, test_probs, test_strips, threshold=tuned_th, n_bootstraps=1000, seed=42)

        # 4. Clean evaluation (ambiguity removed)
        test_probs_clean = test_probs[clean_mask]
        clean_met = compute_comprehensive_metrics(y_test_clean, test_probs_clean, threshold=tuned_th)
        clean_boot = block_bootstrap_evaluation(y_test_clean, test_probs_clean, test_strips_clean, threshold=tuned_th, n_bootstraps=1000, seed=42)

        row_prim = {
            "model": f"BASELINE: {name}",
            "threshold": tuned_th,
            "accuracy": test_met["accuracy"],
            "precision": test_met["precision"],
            "recall": test_met["recall"],
            "f1": test_met["f1"],
            "f1_ci": f"[{boot['f1_ci'][0]:.4f}, {boot['f1_ci'][1]:.4f}]",
            "roc_auc": test_met["roc_auc"],
            "roc_auc_ci": f"[{boot['roc_auc_ci'][0]:.4f}, {boot['roc_auc_ci'][1]:.4f}]",
            "pr_auc": test_met["pr_auc"],
            "pr_auc_ci": f"[{boot['pr_auc_ci'][0]:.4f}, {boot['pr_auc_ci'][1]:.4f}]",
            "prevalence": f"{100*test_prev:.1f}%",
            "support": test_met["support"],
        }
        results_primary.append(row_prim)

        row_cln = {
            "model": f"BASELINE: {name}",
            "threshold": tuned_th,
            "accuracy": clean_met["accuracy"],
            "precision": clean_met["precision"],
            "recall": clean_met["recall"],
            "f1": clean_met["f1"],
            "f1_ci": f"[{clean_boot['f1_ci'][0]:.4f}, {clean_boot['f1_ci'][1]:.4f}]",
            "roc_auc": clean_met["roc_auc"],
            "roc_auc_ci": f"[{clean_boot['roc_auc_ci'][0]:.4f}, {clean_boot['roc_auc_ci'][1]:.4f}]",
            "pr_auc": clean_met["pr_auc"],
            "pr_auc_ci": f"[{clean_boot['pr_auc_ci'][0]:.4f}, {clean_boot['pr_auc_ci'][1]:.4f}]",
            "prevalence": f"{100*clean_met['prevalence']:.1f}%",
            "support": clean_met["support"],
        }
        results_clean.append(row_cln)

        print(f"  Test Metrics (Tuned Th={tuned_th:.2f}):")
        print(f"    F1       : {row_prim['f1']:.4f} (95% CI: {row_prim['f1_ci']})")
        print(f"    ROC-AUC  : {row_prim['roc_auc']:.4f} (95% CI: {row_prim['roc_auc_ci']})")
        print(f"    PR-AUC   : {row_prim['pr_auc']:.4f} (95% CI: {row_prim['pr_auc_ci']}) [Prev: {row_prim['prevalence']}]")
        print(f"    Support  : {row_prim['support']}")
        print(f"  Clean Test Metrics (Ambiguity Removed):")
        print(f"    F1       : {row_cln['f1']:.4f} (95% CI: {row_cln['f1_ci']})")
        print(f"    ROC-AUC  : {row_cln['roc_auc']:.4f} (95% CI: {row_cln['roc_auc_ci']})")
        print(f"    PR-AUC   : {row_cln['pr_auc']:.4f} (95% CI: {row_cln['pr_auc_ci']}) [Prev: {row_cln['prevalence']}]")

    # Build Markdown Summary Report
    summary_md = (
        "# Classical Baselines on Real Chandrayaan-2 DFSAR L1A\n\n"
        f"> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI, lines 100,000–140,000)\n"
        f"> **Evaluated on:** Spatial Test Split ({len(y_test)} patches: {n_pos_test} positive, {n_neg_test} negative, Prevalence={100*test_prev:.1f}%)\n"
        f"> **Input Features:** Strictly raw radar log-intensities [HH, HV, VV] patch means (no circular CPR/DOP inputs)\n"
        f"> **Threshold Protocol:** Decision threshold tuned on Validation set to maximize Val F1, applied unchanged to Test\n"
        f"> **Confidence Intervals:** 95% block-bootstrap CIs computed by resampling entire spatial strips (1000 rounds)\n\n"
        "## 1. Primary Test Evaluation (Full Spatial Test Set: N=338)\n\n"
        "| Model | Tuned Th | Accuracy | Precision | Recall | F1 (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) | Prevalence | Support |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in results_primary:
        summary_md += (
            f"| {r['model']} | {r['threshold']:.2f} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | **{r['f1']:.4f}**<br>`{r['f1_ci']}` | **{r['roc_auc']:.4f}**<br>`{r['roc_auc_ci']}` | "
            f"**{r['pr_auc']:.4f}**<br>`{r['pr_auc_ci']}` | {r['prevalence']} | {r['support']} |\n"
        )

    summary_md += (
        "\n---\n\n"
        "## 2. Ambiguity-Removed Evaluation (Clean Test Set: N=205)\n\n"
        "> **NOTE ON AMBIGUITY ABLATION:**\n"
        f"> Excludes {amb_mask.sum()} test patches containing between 1 and 4 positive pixels (the boundary transition band). "
        f"Leaves {len(y_test_clean)} clean patches ({y_test_clean.sum()} positive, {len(y_test_clean)-y_test_clean.sum()} negative, Prevalence={100*clean_met['prevalence']:.1f}%).\n\n"
        "| Model | Tuned Th | Accuracy | Precision | Recall | F1 (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) | Prevalence | Support |\n"
        "|---|---|---|---|---|---|---|---|---|---|\n"
    )
    for r in results_clean:
        summary_md += (
            f"| {r['model']} | {r['threshold']:.2f} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | **{r['f1']:.4f}**<br>`{r['f1_ci']}` | **{r['roc_auc']:.4f}**<br>`{r['roc_auc_ci']}` | "
            f"**{r['pr_auc']:.4f}**<br>`{r['pr_auc_ci']}` | {r['prevalence']} | {r['support']} |\n"
        )

    summary_md += (
        "\n---\n\n"
        "## 3. Ground Truth Reference (Separate Note)\n\n"
        "> **ORACLE (label rule, not a model)**\n"
        f"> - **Metrics:** Accuracy={oracle_dict['accuracy']:.4f}, Precision={oracle_dict['precision']:.4f}, "
        f"Recall={oracle_dict['recall']:.4f}, F1={oracle_dict['f1']:.4f}, ROC-AUC={oracle_dict['roc_auc']:.4f}, PR-AUC={oracle_dict['pr_auc']:.4f}\n"
        f"> - **Support:** {oracle_dict['support']}\n"
        "> - **Explanation:** The Oracle directly evaluates the CPR > 1.0 & DOP < 0.13 ground-truth criteria on polarimetric indicators. "
        "It achieves an F1 of 1.0000 by mathematical identity because it defines the labeling rule itself. "
        "It is strictly excluded from machine learning comparison tables to prevent false equivalence.\n"
    )

    out_file = os.path.join(REPORTS_REAL, "baseline_summary.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"\nBaseline summary report saved to {out_file}")

    json_out = os.path.join(REPORTS_REAL, "baseline_results.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump({
            "primary": results_primary,
            "clean": results_clean,
            "oracle": oracle_dict,
        }, f, indent=2)
    print(f"Baseline JSON saved to {json_out}")

    return results_primary, results_clean, oracle_dict


if __name__ == "__main__":
    main()
