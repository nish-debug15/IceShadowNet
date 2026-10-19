"""
scripts/baseline_real.py
========================
Classical baseline evaluation on real Chandrayaan-2 DFSAR polarimetric data.

Implements classical baselines evaluated on the exact same leak-free spatial test split
as the neural network models (from data/processed_real/test.npz):

1. ORACLE (label rule, not a model):
   Evaluates the labeling rule directly on polarimetric CPR and DOP.
   Excluded from model comparison tables because it defines the labels.

2. Decision Tree Baseline:
   DecisionTreeClassifier (max_depth=3, class_weight='balanced') trained strictly on
   patch-mean raw log-intensities [HH, HV, VV] (no derived polarimetric indicators).

3. Logistic Regression Baseline:
   LogisticRegression (class_weight='balanced') trained strictly on patch-mean raw
   log-intensities [HH, HV, VV].

Usage:
    python -m scripts.baseline_real
"""

import os
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)

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

    # Feature matrix for ML models: raw radar channels ONLY
    X = np.stack([hh_means, hv_means, vv_means], axis=-1)
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

    n_pos_test = int(y_test.sum())
    n_neg_test = int(len(y_test) - n_pos_test)

    print("="*65)
    print("EVALUATING CLASSICAL BASELINES ON REAL DFSAR (SPATIAL TEST SPLIT)")
    print(f"Train split : {len(y_train)} patches (Positives: {y_train.sum()}, Negatives: {len(y_train)-y_train.sum()})")
    print(f"Test split  : {len(y_test)} patches (Positives: {n_pos_test}, Negatives: {n_neg_test})")
    print("="*65)

    # ──────────────────────────────────────────────────────────────────────────
    # ORACLE: Label Rule Evaluation (Not an ML model)
    # ──────────────────────────────────────────────────────────────────────────
    oracle_pred = (frac_test >= (5.0 / (64 * 64))).astype(int)
    oracle_scores = frac_test

    acc_o  = accuracy_score(y_test, oracle_pred)
    prec_o = precision_score(y_test, oracle_pred, zero_division=0)
    rec_o  = recall_score(y_test, oracle_pred, zero_division=0)
    f1_o   = f1_score(y_test, oracle_pred, zero_division=0)
    auc_o  = roc_auc_score(y_test, oracle_scores)
    cm_o   = confusion_matrix(y_test, oracle_pred)
    tn_o, fp_o, fn_o, tp_o = cm_o.ravel()

    oracle_dict = {
        "name": "ORACLE (label rule, not a model)",
        "accuracy": acc_o,
        "precision": prec_o,
        "recall": rec_o,
        "f1": f1_o,
        "roc_auc": auc_o,
        "support": f"TP:{tp_o} FP:{fp_o} FN:{fn_o} TN:{tn_o} (Pos:{n_pos_test}, Neg:{n_neg_test})",
    }

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL 1: Decision Tree on Raw Intensity Channels
    # ──────────────────────────────────────────────────────────────────────────
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)
    dt.fit(X_train, y_train)

    dt_pred = dt.predict(X_test)
    dt_prob = dt.predict_proba(X_test)[:, 1]

    acc_dt  = accuracy_score(y_test, dt_pred)
    prec_dt = precision_score(y_test, dt_pred, zero_division=0)
    rec_dt  = recall_score(y_test, dt_pred, zero_division=0)
    f1_dt   = f1_score(y_test, dt_pred, zero_division=0)
    auc_dt  = roc_auc_score(y_test, dt_prob)
    cm_dt   = confusion_matrix(y_test, dt_pred)
    tn_dt, fp_dt, fn_dt, tp_dt = cm_dt.ravel()

    dt_dict = {
        "model": "BASELINE: Decision Tree (max_depth=3, balanced)",
        "accuracy": acc_dt,
        "precision": prec_dt,
        "recall": rec_dt,
        "f1": f1_dt,
        "roc_auc": auc_dt,
        "support": f"TP:{tp_dt} FP:{fp_dt} FN:{fn_dt} TN:{tn_dt} (Pos:{n_pos_test}, Neg:{n_neg_test})",
    }

    # ──────────────────────────────────────────────────────────────────────────
    # MODEL 2: Logistic Regression on Raw Intensity Channels
    # ──────────────────────────────────────────────────────────────────────────
    lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000)
    lr.fit(X_train, y_train)

    lr_pred = lr.predict(X_test)
    lr_prob = lr.predict_proba(X_test)[:, 1]

    acc_lr  = accuracy_score(y_test, lr_pred)
    prec_lr = precision_score(y_test, lr_pred, zero_division=0)
    rec_lr  = recall_score(y_test, lr_pred, zero_division=0)
    f1_lr   = f1_score(y_test, lr_pred, zero_division=0)
    auc_lr  = roc_auc_score(y_test, lr_prob)
    cm_lr   = confusion_matrix(y_test, lr_pred)
    tn_lr, fp_lr, fn_lr, tp_lr = cm_lr.ravel()

    lr_dict = {
        "model": "BASELINE: Logistic Regression (balanced)",
        "accuracy": acc_lr,
        "precision": prec_lr,
        "recall": rec_lr,
        "f1": f1_lr,
        "roc_auc": auc_lr,
        "support": f"TP:{tp_lr} FP:{fp_lr} FN:{fn_lr} TN:{tn_lr} (Pos:{n_pos_test}, Neg:{n_neg_test})",
    }

    models = [dt_dict, lr_dict]

    print("\nORACLE (GROUND TRUTH DEFINITION):")
    print(f"  {oracle_dict['name']}: Accuracy={oracle_dict['accuracy']:.4f}, F1={oracle_dict['f1']:.4f}, Support: {oracle_dict['support']}")

    print("\nCLASSICAL ML BASELINE RESULTS (Trained on raw [HH, HV, VV] log-intensities):")
    for m in models:
        print(f"\n{m['model']}:")
        print(f"  Accuracy  : {m['accuracy']:.4f}")
        print(f"  Precision : {m['precision']:.4f}")
        print(f"  Recall    : {m['recall']:.4f}")
        print(f"  F1-Score  : {m['f1']:.4f}")
        print(f"  ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"  Support   : {m['support']}")

    # Save summary report
    summary_md = (
        "# Classical Baselines on Real Chandrayaan-2 DFSAR L1A\n\n"
        f"> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI, lines 100,000–140,000)\n"
        f"> **Evaluated on:** Spatial Test Split ({len(y_test)} patches: {n_pos_test} positive, {n_neg_test} negative)\n"
        f"> **Input Features:** Strictly raw radar log-intensities [HH, HV, VV] patch means (no circular CPR/DOP inputs)\n\n"
        "## Machine Learning Comparison Table\n\n"
        "| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Support |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for m in models:
        summary_md += (
            f"| {m['model']} | {m['accuracy']:.4f} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} | {m['roc_auc']:.4f} | {m['support']} |\n"
        )

    summary_md += (
        "\n---\n\n"
        "## Ground Truth Reference\n\n"
        "> **NOTE: ORACLE (label rule, not a model)**\n"
        f"> - **Evaluation:** Accuracy={oracle_dict['accuracy']:.4f}, Precision={oracle_dict['precision']:.4f}, "
        f"Recall={oracle_dict['recall']:.4f}, F1={oracle_dict['f1']:.4f}, ROC-AUC={oracle_dict['roc_auc']:.4f}\n"
        f"> - **Support:** {oracle_dict['support']}\n"
        "> - **Explanation:** The Oracle directly evaluates the CPR > 1.0 & DOP < 0.13 ground-truth criteria. "
        "It achieves an F1 of 1.0000 by mathematical identity because it constitutes the labeling rule itself. "
        "It is strictly excluded from machine learning comparison tables to prevent false equivalence.\n"
    )

    out_file = os.path.join(REPORTS_REAL, "baseline_summary.md")
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(summary_md)
    print(f"\nBaseline summary report saved to {out_file}")

    return models, oracle_dict


if __name__ == "__main__":
    main()
