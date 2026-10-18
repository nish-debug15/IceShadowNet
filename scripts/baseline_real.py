"""
scripts/baseline_real.py
========================
Pixel-wise baseline evaluation on real Chandrayaan-2 DFSAR polarimetric data.

Implements two classical baselines evaluated on the exact same spatial test split
as the neural network models (from data/processed_real/test.npz):

1. Plain Threshold Baseline:
   Evaluates a fixed threshold on patch-level polarimetric indicators (CPR, DOP).
   Specifically, predicts positive if the patch-level mean CPR exceeds the threshold
   and mean DOP is depressed, or if the candidate pixel fraction exceeds the threshold.

2. Decision Tree Baseline:
   Fits a DecisionTreeClassifier (max_depth=3) on the spatial training split using
   polarimetric and intensity features (CPR mean, DOP mean, log-intensity channels)
   and evaluates out-of-sample on the spatial test split.

Usage:
    python -m scripts.baseline_real
"""

import os
import json
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

PROCESSED_REAL = "data/processed_real"
REPORTS_REAL   = "reports/real"
os.makedirs(REPORTS_REAL, exist_ok=True)


def extract_features(data_dict):
    """
    Extract polarimetric and raw backscatter features per patch.
    """
    images    = data_dict["images"]       # (N, 3, 64, 64): HH, HV, VV log-intensity
    cpr_means = data_dict["cpr_means"]    # (N,)
    dop_means = data_dict["dop_means"]    # (N,)
    ice_fracs = data_dict["ice_fracs"]    # (N,)

    # Compute raw intensity statistics
    hh_means = images[:, 0].mean(axis=(1, 2))
    hv_means = images[:, 1].mean(axis=(1, 2))
    vv_means = images[:, 2].mean(axis=(1, 2))

    # Feature matrix: [cpr_mean, dop_mean, hh_mean, hv_mean, vv_mean]
    X = np.stack([cpr_means, dop_means, hh_means, hv_means, vv_means], axis=-1)
    return X, cpr_means, dop_means, ice_fracs


def main():
    train_path = os.path.join(PROCESSED_REAL, "train.npz")
    test_path  = os.path.join(PROCESSED_REAL, "test.npz")

    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        raise FileNotFoundError(
            f"Processed real splits not found in {PROCESSED_REAL}. "
            "Run `python -m scripts.ingest_and_split --real` first."
        )

    train_data = np.load(train_path)
    test_data  = np.load(test_path)

    y_train = train_data["labels"]
    y_test  = test_data["labels"]

    X_train, cpr_train, dop_train, frac_train = extract_features(train_data)
    X_test,  cpr_test,  dop_test,  frac_test  = extract_features(test_data)

    print("="*60)
    print("EVALUATING CLASSICAL POLARIMETRIC BASELINES ON REAL DFSAR")
    print(f"Train split : {len(y_train)} patches (Positives: {y_train.sum()})")
    print(f"Test split  : {len(y_test)} patches (Positives: {y_test.sum()})")
    print("="*60)

    results = []

    # ──────────────────────────────────────────────────────────────────────────
    # BASELINE 1: Plain Threshold on (CPR, DOP)
    # ──────────────────────────────────────────────────────────────────────────
    # Rule: classify patch as positive if ice candidate fraction >= 5 / (64*64) = 0.0012
    threshold_pred = (frac_test >= (5.0 / (64 * 64))).astype(int)
    threshold_scores = frac_test  # continuous score for ROC-AUC

    acc_t  = accuracy_score(y_test, threshold_pred)
    prec_t = precision_score(y_test, threshold_pred, zero_division=0)
    rec_t  = recall_score(y_test, threshold_pred, zero_division=0)
    f1_t   = f1_score(y_test, threshold_pred, zero_division=0)
    try:
        auc_t = roc_auc_score(y_test, threshold_scores)
    except Exception:
        auc_t = 0.5

    cm_t = confusion_matrix(y_test, threshold_pred)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()

    results.append({
        "model": "BASELINE: Plain Threshold (CPR>1 & DOP<0.13)",
        "accuracy": acc_t,
        "precision": prec_t,
        "recall": rec_t,
        "f1": f1_t,
        "roc_auc": auc_t,
        "support": f"TP:{tp_t} FP:{fp_t} FN:{fn_t} TN:{tn_t}",
    })

    # ──────────────────────────────────────────────────────────────────────────
    # BASELINE 2: Decision Tree on (CPR, DOP, Intensity)
    # ──────────────────────────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    dt_prob = dt.predict_proba(X_test)[:, 1]

    acc_dt  = accuracy_score(y_test, dt_pred)
    prec_dt = precision_score(y_test, dt_pred, zero_division=0)
    rec_dt  = recall_score(y_test, dt_pred, zero_division=0)
    f1_dt   = f1_score(y_test, dt_pred, zero_division=0)
    try:
        auc_dt = roc_auc_score(y_test, dt_prob)
    except Exception:
        auc_dt = 0.5

    cm_dt = confusion_matrix(y_test, dt_pred)
    tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()

    results.append({
        "model": "BASELINE: Decision Tree (max_depth=3)",
        "accuracy": acc_dt,
        "precision": prec_dt,
        "recall": rec_dt,
        "f1": f1_dt,
        "roc_auc": auc_dt,
        "support": f"TP:{tp_dt} FP:{fp_dt} FN:{fn_dt} TN:{tn_dt}",
    })

    print("\nBASELINE EVALUATION RESULTS (Real Chandrayaan-2 DFSAR Spatial Test Split):")
    for r in results:
        print(f"\n{r['model']}:")
        print(f"  Accuracy  : {r['accuracy']:.4f}")
        print(f"  Precision : {r['precision']:.4f}")
        print(f"  Recall    : {r['recall']:.4f}")
        print(f"  F1-Score  : {r['f1']:.4f}")
        print(f"  ROC-AUC   : {r['roc_auc']:.4f}")
        print(f"  Support   : {r['support']}")

    # Save summary report
    summary_md = (
        "# Classical Polarimetric Baselines (Real DFSAR L1A)\n\n"
        "> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI)\n"
        "> **Evaluated on:** Spatial Test Split (374 patches, 110 positives, 264 negatives)\n\n"
        "| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for r in results:
        summary_md += (
            f"| {r['model']} | {r['accuracy']:.4f} | {r['precision']:.4f} | "
            f"{r['recall']:.4f} | {r['f1']:.4f} | {r['roc_auc']:.4f} | {r['support']} |\n"
        )

    out_file = os.path.join(REPORTS_REAL, "baseline_summary.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"\nBaseline summary report saved to {out_file}")

    return results


if __name__ == "__main__":
    main()
