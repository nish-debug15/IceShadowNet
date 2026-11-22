"""
scripts/baseline_real.py
========================
Classical baseline evaluation on real Chandrayaan-2 DFSAR polarimetric data.

Implements baselines evaluated on the spatial test split:
1. Trivial Baseline (Majority / All-negative class prediction)
2. Label Rule Baseline (CPR > 1.0 & DOP < 0.13 direct application: the circularity yardstick)
3. Decision Tree (max_depth=3, class_weight='balanced') on raw [HH, HV, VV]
4. Logistic Regression (class_weight='balanced') on raw [HH, HV, VV]

Protocol enhancements:
- Headline metrics: PR-AUC (primary), MCC, Balanced Accuracy, F1, ROC-AUC.
- Validation-tuned threshold (maximizing Val F1, applied unchanged to test).
- Block-bootstrap 95% Confidence Intervals across test strips (1000 rounds).
- Expected Calibration Error (ECE) and reliability diagrams.
- Ambiguity-removed evaluation on clean test set.
- JSON output compatible with downstream reporting.

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
    plot_reliability_diagram,
)
from src.datasets.ice_dataset import get_ambiguity_mask

PROCESSED_REAL = "data/processed_real"
REPORTS_REAL   = "reports/real"
FIGURES_DIR    = os.path.join(REPORTS_REAL, "figures")
os.makedirs(REPORTS_REAL, exist_ok=True)
os.makedirs(FIGURES_DIR,  exist_ok=True)


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
    frac_test_clean = frac_test[clean_mask]

    n_pos_test = int(y_test.sum())
    n_neg_test = int(len(y_test) - n_pos_test)
    test_prev = float(np.mean(y_test))

    print("="*75)
    print("EVALUATING CLASSICAL BASELINES ON REAL DFSAR (OVERHAUL EVALUATION)")
    print(f"Train split : {len(y_train)} patches (Pos: {y_train.sum()}, Neg: {len(y_train)-y_train.sum()})")
    print(f"Val split   : {len(y_val)} patches (Pos: {y_val.sum()}, Neg: {len(y_val)-y_val.sum()})")
    print(f"Test split  : {len(y_test)} patches (Pos: {n_pos_test}, Neg: {n_neg_test}, Prev: {100*test_prev:.2f}%)")
    print(f"Test Clean  : {len(y_test_clean)} patches (Pos: {y_test_clean.sum()}, Neg: {len(y_test_clean)-y_test_clean.sum()}, Dropped Ambiguous: {amb_mask.sum()})")
    print("="*75)

    results_primary = []
    results_clean = []

    # ──────────────────────────────────────────────────────────────────────────
    # 1. TRIVIAL BASELINE (Majority class / constant zero)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Evaluating Trivial Baseline (Majority / All Negative) ---")
    triv_probs = np.zeros_like(y_test, dtype=float)
    triv_met = compute_comprehensive_metrics(y_test, triv_probs, threshold=0.5)
    triv_boot = block_bootstrap_evaluation(y_test, triv_probs, test_strips, threshold=0.5, n_bootstraps=1000, seed=42)

    triv_row = {
        "model": "Trivial Baseline (All Negative)",
        "threshold": 0.5,
        "pr_auc": triv_met["pr_auc"],
        "pr_auc_ci": f"[{triv_boot['pr_auc_ci'][0]:.4f}, {triv_boot['pr_auc_ci'][1]:.4f}]",
        "roc_auc": triv_met["roc_auc"],
        "roc_auc_ci": f"[{triv_boot['roc_auc_ci'][0]:.4f}, {triv_boot['roc_auc_ci'][1]:.4f}]",
        "mcc": triv_met["mcc"],
        "mcc_ci": f"[{triv_boot['mcc_ci'][0]:.4f}, {triv_boot['mcc_ci'][1]:.4f}]",
        "balanced_accuracy": triv_met["balanced_accuracy"],
        "bal_acc_ci": f"[{triv_boot['bal_acc_ci'][0]:.4f}, {triv_boot['bal_acc_ci'][1]:.4f}]",
        "f1": triv_met["f1"],
        "f1_ci": f"[{triv_boot['f1_ci'][0]:.4f}, {triv_boot['f1_ci'][1]:.4f}]",
        "precision": triv_met["precision"],
        "recall": triv_met["recall"],
        "ece": triv_met["ece"],
        "prevalence": f"{100*test_prev:.1f}%",
        "support": triv_met["support"],
    }
    results_primary.append(triv_row)

    # ──────────────────────────────────────────────────────────────────────────
    # 2. RULE BASELINE (Direct CPR>1 & DOP<0.13 rule - Circularity Yardstick)
    # ──────────────────────────────────────────────────────────────────────────
    print("\n--- Evaluating Rule Baseline (Circularity Yardstick) ---")
    rule_probs = frac_test.astype(float)
    rule_th = 5.0 / (64 * 64)
    rule_met = compute_comprehensive_metrics(y_test, rule_probs, threshold=rule_th)
    rule_boot = block_bootstrap_evaluation(y_test, rule_probs, test_strips, threshold=rule_th, n_bootstraps=1000, seed=42)

    rule_row = {
        "model": "Rule Baseline (Circularity Yardstick: CPR>1 & DOP<0.13)",
        "threshold": float(rule_th),
        "pr_auc": rule_met["pr_auc"],
        "pr_auc_ci": f"[{rule_boot['pr_auc_ci'][0]:.4f}, {rule_boot['pr_auc_ci'][1]:.4f}]",
        "roc_auc": rule_met["roc_auc"],
        "roc_auc_ci": f"[{rule_boot['roc_auc_ci'][0]:.4f}, {rule_boot['roc_auc_ci'][1]:.4f}]",
        "mcc": rule_met["mcc"],
        "mcc_ci": f"[{rule_boot['mcc_ci'][0]:.4f}, {rule_boot['mcc_ci'][1]:.4f}]",
        "balanced_accuracy": rule_met["balanced_accuracy"],
        "bal_acc_ci": f"[{rule_boot['bal_acc_ci'][0]:.4f}, {rule_boot['bal_acc_ci'][1]:.4f}]",
        "f1": rule_met["f1"],
        "f1_ci": f"[{rule_boot['f1_ci'][0]:.4f}, {rule_boot['f1_ci'][1]:.4f}]",
        "precision": rule_met["precision"],
        "recall": rule_met["recall"],
        "ece": rule_met["ece"],
        "prevalence": f"{100*test_prev:.1f}%",
        "support": rule_met["support"],
    }
    results_primary.append(rule_row)

    # ──────────────────────────────────────────────────────────────────────────
    # 3. CLASSICAL TRAINABLE BASELINES: Decision Tree & Logistic Regression
    # ──────────────────────────────────────────────────────────────────────────
    models_spec = [
        ("Decision Tree (max_depth=3, balanced)", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42), "dt"),
        ("Logistic Regression (balanced)", LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000), "lr"),
    ]

    for name, clf, short_tag in models_spec:
        print(f"\n--- Evaluating {name} ---")
        clf.fit(X_train, y_train)

        val_probs  = clf.predict_proba(X_val)[:, 1]
        test_probs = clf.predict_proba(X_test)[:, 1]

        # 1. Tune threshold strictly on validation set
        tuned_th, val_f1 = tune_threshold_on_val(y_val, val_probs)
        print(f"  Validation tuned threshold: {tuned_th:.2f} (Val F1: {val_f1:.4f})")

        # 2. Comprehensive metrics on Test (tuned threshold)
        test_met = compute_comprehensive_metrics(y_test, test_probs, threshold=tuned_th)

        # 3. Block-bootstrap 95% CIs
        boot = block_bootstrap_evaluation(y_test, test_probs, test_strips, threshold=tuned_th, n_bootstraps=1000, seed=42)

        # 4. Save Reliability Diagram
        rel_path = os.path.join(FIGURES_DIR, f"calibration_baseline_{short_tag}.png")
        plot_reliability_diagram(y_test, test_probs, title=f"Reliability Diagram: {name}", save_path=rel_path)

        row_prim = {
            "model": f"BASELINE: {name}",
            "threshold": tuned_th,
            "pr_auc": test_met["pr_auc"],
            "pr_auc_ci": f"[{boot['pr_auc_ci'][0]:.4f}, {boot['pr_auc_ci'][1]:.4f}]",
            "roc_auc": test_met["roc_auc"],
            "roc_auc_ci": f"[{boot['roc_auc_ci'][0]:.4f}, {boot['roc_auc_ci'][1]:.4f}]",
            "mcc": test_met["mcc"],
            "mcc_ci": f"[{boot['mcc_ci'][0]:.4f}, {boot['mcc_ci'][1]:.4f}]",
            "balanced_accuracy": test_met["balanced_accuracy"],
            "bal_acc_ci": f"[{boot['bal_acc_ci'][0]:.4f}, {boot['bal_acc_ci'][1]:.4f}]",
            "f1": test_met["f1"],
            "f1_ci": f"[{boot['f1_ci'][0]:.4f}, {boot['f1_ci'][1]:.4f}]",
            "precision": test_met["precision"],
            "recall": test_met["recall"],
            "ece": test_met["ece"],
            "prevalence": f"{100*test_prev:.1f}%",
            "support": test_met["support"],
            "calibration_plot": rel_path,
        }
        results_primary.append(row_prim)

        # 5. Clean evaluation (ambiguity removed)
        test_probs_clean = test_probs[clean_mask]
        clean_met = compute_comprehensive_metrics(y_test_clean, test_probs_clean, threshold=tuned_th)
        clean_boot = block_bootstrap_evaluation(y_test_clean, test_probs_clean, test_strips_clean, threshold=tuned_th, n_bootstraps=1000, seed=42)

        row_cln = {
            "model": f"BASELINE: {name}",
            "threshold": tuned_th,
            "pr_auc": clean_met["pr_auc"],
            "pr_auc_ci": f"[{clean_boot['pr_auc_ci'][0]:.4f}, {clean_boot['pr_auc_ci'][1]:.4f}]",
            "roc_auc": clean_met["roc_auc"],
            "roc_auc_ci": f"[{clean_boot['roc_auc_ci'][0]:.4f}, {clean_boot['roc_auc_ci'][1]:.4f}]",
            "mcc": clean_met["mcc"],
            "mcc_ci": f"[{clean_boot['mcc_ci'][0]:.4f}, {clean_boot['mcc_ci'][1]:.4f}]",
            "balanced_accuracy": clean_met["balanced_accuracy"],
            "bal_acc_ci": f"[{clean_boot['bal_acc_ci'][0]:.4f}, {clean_boot['bal_acc_ci'][1]:.4f}]",
            "f1": clean_met["f1"],
            "f1_ci": f"[{clean_boot['f1_ci'][0]:.4f}, {clean_boot['f1_ci'][1]:.4f}]",
            "precision": clean_met["precision"],
            "recall": clean_met["recall"],
            "ece": clean_met["ece"],
            "prevalence": f"{100*clean_met['prevalence']:.1f}%",
            "support": clean_met["support"],
        }
        results_clean.append(row_cln)

        print(f"  Test Metrics (Tuned Th={tuned_th:.2f}):")
        print(f"    PR-AUC   : {row_prim['pr_auc']:.4f} (95% CI: {row_prim['pr_auc_ci']}) [Headline Metric]")
        print(f"    ROC-AUC  : {row_prim['roc_auc']:.4f} (95% CI: {row_prim['roc_auc_ci']})")
        print(f"    MCC      : {row_prim['mcc']:.4f} (95% CI: {row_prim['mcc_ci']})")
        print(f"    Bal Acc  : {row_prim['balanced_accuracy']:.4f} (95% CI: {row_prim['bal_acc_ci']})")
        print(f"    F1       : {row_prim['f1']:.4f} (95% CI: {row_prim['f1_ci']})")
        print(f"    ECE      : {row_prim['ece']:.4f}")
        print(f"    Support  : {row_prim['support']}")

    # ──────────────────────────────────────────────────────────────────────────
    # Save Baseline Results JSON
    # ──────────────────────────────────────────────────────────────────────────
    json_path = os.path.join(REPORTS_REAL, "baseline_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "primary": results_primary,
            "clean": results_clean,
        }, f, indent=2)
    print(f"\nBaseline results JSON saved to {json_path}")

    # ──────────────────────────────────────────────────────────────────────────
    # Build Markdown Summary Report
    # ──────────────────────────────────────────────────────────────────────────
    md_lines = [
        "# Classical Baselines on Real Chandrayaan-2 DFSAR L1A (Overhaul Evaluation)",
        "",
        "> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI, lines 100,000–140,000)",
        f"> **Spatial Test Split:** {len(y_test)} patches ({n_pos_test} positive, {n_neg_test} negative, Prevalence={100*test_prev:.1f}%)",
        "> **Evaluation Protocol:** PR-AUC is headline metric; MCC and Balanced Accuracy replace unweighted accuracy; thresholds tuned strictly on Val; block-bootstrap 95% CIs across strips; ECE calibration computed.",
        "",
        "## 1. Primary Test Evaluation Table (Full Spatial Test Set: N=338)",
        "",
        "| Model | Tuned Th | PR-AUC (Primary) (95% CI) | ROC-AUC (95% CI) | MCC (95% CI) | Balanced Acc (95% CI) | F1 (95% CI) | Precision | Recall | ECE | Support |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results_primary:
        md_lines.append(
            f"| {r['model']} | {r['threshold']:.2f} | **{r['pr_auc']:.4f}**<br>`{r['pr_auc_ci']}` | "
            f"**{r['roc_auc']:.4f}**<br>`{r['roc_auc_ci']}` | **{r['mcc']:.4f}**<br>`{r['mcc_ci']}` | "
            f"{r['balanced_accuracy']:.4f}<br>`{r['bal_acc_ci']}` | {r['f1']:.4f}<br>`{r['f1_ci']}` | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['ece']:.4f} | {r['support']} |"
        )

    md_lines.extend([
        "",
        "---",
        "",
        "## 2. Ambiguity-Removed Evaluation (Clean Test Set: N=205)",
        "",
        f"> Excludes {amb_mask.sum()} test patches containing between 1 and 4 positive pixels. Leaves {len(y_test_clean)} clean patches ({y_test_clean.sum()} pos, {len(y_test_clean)-y_test_clean.sum()} neg).",
        "",
        "| Model | Tuned Th | PR-AUC (Primary) (95% CI) | ROC-AUC (95% CI) | MCC (95% CI) | Balanced Acc (95% CI) | F1 (95% CI) | Precision | Recall | ECE | Support |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ])
    for r in results_clean:
        md_lines.append(
            f"| {r['model']} | {r['threshold']:.2f} | **{r['pr_auc']:.4f}**<br>`{r['pr_auc_ci']}` | "
            f"**{r['roc_auc']:.4f}**<br>`{r['roc_auc_ci']}` | **{r['mcc']:.4f}**<br>`{r['mcc_ci']}` | "
            f"{r['balanced_accuracy']:.4f}<br>`{r['bal_acc_ci']}` | {r['f1']:.4f}<br>`{r['f1_ci']}` | "
            f"{r['precision']:.4f} | {r['recall']:.4f} | {r['ece']:.4f} | {r['support']} |"
        )

    md_path = os.path.join(REPORTS_REAL, "baseline_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")
    print(f"Baseline markdown summary saved to {md_path}")

    return results_primary, results_clean


if __name__ == "__main__":
    main()
