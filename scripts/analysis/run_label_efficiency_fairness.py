"""
scripts/analysis/run_label_efficiency_fairness.py
=================================================
Item 2 (Block 5): Label-Efficiency Fairness Check.
Evaluates LogisticRegression (10/25/50/100% stratified train subsamples, 3 seeds)
on the identical frozen split and compares against CustomCNN, ResNet_SSL,
ResNet_scratch, and ResNet_pretrained from reports/real/ssl_label_efficiency.json.

Outputs:
  - reports/real/label_efficiency_fairness.json
  - reports/real/label_efficiency_fairness.md
"""

import os
import sys
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.evaluate import tune_threshold_on_val

PROCESSED_REAL = "data/processed_real"
REPORTS_REAL = "reports/real"


def extract_tabular_features(imgs):
    feats = []
    for c in range(3):
        ch = imgs[:, c]
        feats.append(ch.mean(axis=(1, 2)))
        feats.append(ch.std(axis=(1, 2)))
        feats.append(np.percentile(ch, 90, axis=(1, 2)))
        feats.append(np.percentile(ch, 10, axis=(1, 2)))
    return np.column_stack(feats)


def strip_bootstrap_ci(y_true, y_probs, strip_ids, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_strips = np.unique(strip_ids)
    strip_idx = {s: np.where(strip_ids == s)[0] for s in unique_strips}
    cache = {}
    rocs, prs = [], []
    for _ in range(n_boot):
        sampled = tuple(sorted(rng.choice(unique_strips, size=len(unique_strips), replace=True).tolist()))
        if sampled not in cache:
            idx = np.concatenate([strip_idx[s] for s in sampled])
            yt, yp = y_true[idx], y_probs[idx]
            cache[sampled] = None if len(np.unique(yt)) < 2 else (roc_auc_score(yt, yp), average_precision_score(yt, yp))
        if cache[sampled] is not None:
            rocs.append(cache[sampled][0])
            prs.append(cache[sampled][1])
    return (
        [float(np.percentile(rocs, 2.5)), float(np.percentile(rocs, 97.5))],
        [float(np.percentile(prs, 2.5)), float(np.percentile(prs, 97.5))],
    )


def main():
    d_tr = np.load(os.path.join(PROCESSED_REAL, "train.npz"))
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_tr = extract_tabular_features(d_tr["images"])
    y_tr = d_tr["labels"].astype(int)
    X_va = extract_tabular_features(d_va["images"])
    y_va = d_va["labels"].astype(int)
    X_te = extract_tabular_features(d_te["images"])
    y_te = d_te["labels"].astype(int)
    te_strips = d_te["strip_ids"].astype(int)

    fractions = [0.10, 0.25, 0.50, 1.00]
    seeds = [42, 123, 2024]

    lr_runs = []
    lr_summary = []

    for frac in fractions:
        seed_metrics = []
        for seed in seeds:
            if frac < 1.0:
                sss = StratifiedShuffleSplit(n_splits=1, train_size=frac, random_state=seed)
                sub_idx, _ = next(sss.split(X_tr, y_tr))
            else:
                sub_idx = np.arange(len(y_tr))

            clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
            clf.fit(X_tr[sub_idx], y_tr[sub_idx])
            va_p = clf.predict_proba(X_va)[:, 1]
            te_p = clf.predict_proba(X_te)[:, 1]
            val_th, _ = tune_threshold_on_val(y_va, va_p)
            pred = (te_p >= val_th).astype(int)

            roc_ci, pr_ci = strip_bootstrap_ci(y_te, te_p, te_strips, seed=seed)
            rec = {
                "model": "LogisticRegression",
                "fraction": frac,
                "percent": int(round(frac * 100)),
                "n_samples": int(len(sub_idx)),
                "seed": seed,
                "pr_auc": float(average_precision_score(y_te, te_p)),
                "roc_auc": float(roc_auc_score(y_te, te_p)),
                "f1": float(f1_score(y_te, pred, zero_division=0)),
                "bal_acc": float(balanced_accuracy_score(y_te, pred)),
                "roc_ci": roc_ci,
                "pr_ci": pr_ci,
            }
            lr_runs.append(rec)
            seed_metrics.append(rec)

        lr_summary.append({
            "model": "LogisticRegression",
            "fraction": frac,
            "percent": int(round(frac * 100)),
            "n_samples": seed_metrics[0]["n_samples"],
            "pr_auc_mean": float(np.mean([m["pr_auc"] for m in seed_metrics])),
            "pr_auc_std": float(np.std([m["pr_auc"] for m in seed_metrics])),
            "roc_auc_mean": float(np.mean([m["roc_auc"] for m in seed_metrics])),
            "roc_auc_std": float(np.std([m["roc_auc"] for m in seed_metrics])),
            "f1_mean": float(np.mean([m["f1"] for m in seed_metrics])),
            "f1_std": float(np.std([m["f1"] for m in seed_metrics])),
            "bal_acc_mean": float(np.mean([m["bal_acc"] for m in seed_metrics])),
            "bal_acc_std": float(np.std([m["bal_acc"] for m in seed_metrics])),
            "strip_roc_ci_seed42": seed_metrics[0]["roc_ci"],
            "strip_pr_ci_seed42": seed_metrics[0]["pr_ci"],
        })

    ssl_data = json.load(open(os.path.join(REPORTS_REAL, "ssl_label_efficiency.json"), "r", encoding="utf-8"))
    ssl_summary = ssl_data["summary"]

    combined = ssl_summary + lr_summary
    with open(os.path.join(REPORTS_REAL, "label_efficiency_fairness.json"), "w", encoding="utf-8") as jf:
        json.dump({"summary": combined, "lr_runs": lr_runs}, jf, indent=2)

    with open(os.path.join(REPORTS_REAL, "label_efficiency_fairness.md"), "w", encoding="utf-8") as mf:
        mf.write("# Label-Efficiency Fairness Benchmark (10% / 25% / 50% / 100%, 3 Seeds)\n\n")
        mf.write("Includes `LogisticRegression` (tabular raw-intensity stats) and `CustomCNN` alongside `ResNet_SSL`, `ResNet_scratch`, and `ResNet_pretrained` on the identical frozen split (`N_test = 2,018`, prevalence `16.45%`).\n\n")
        mf.write("| Fraction | N Train | Model | ROC-AUC (Mean ± Std) | 95% Seed CI | PR-AUC (Mean ± Std) | F1 (Val Th) |\n")
        mf.write("| :---: | :---: | :--- | :---: | :---: | :---: | :---: |\n")
        order = ["LogisticRegression", "CustomCNN", "ResNet_SSL", "ResNet_scratch", "ResNet_pretrained"]
        for pct in [10, 25, 50, 100]:
            for mname in order:
                rows = [r for r in combined if r["percent"] == pct and r["model"] == mname]
                if not rows:
                    continue
                r = rows[0]
                half_w = 1.96 * r["roc_auc_std"] / np.sqrt(3.0)
                ci_str = f"[{r['roc_auc_mean'] - half_w:.4f}, {r['roc_auc_mean'] + half_w:.4f}]"
                mf.write(
                    f"| {pct}% | {r['n_samples']:,} | **{mname}** | "
                    f"{r['roc_auc_mean']:.4f} ± {r['roc_auc_std']:.4f} | {ci_str} | "
                    f"{r['pr_auc_mean']:.4f} ± {r['pr_auc_std']:.4f} | {r['f1_mean']:.4f} |\n"
                )
        mf.write("\n## Fairness Verdict: Does `ResNet_SSL` Beat `LogisticRegression` & `CustomCNN`?\n\n")
        for pct in [10, 25, 50, 100]:
            lr_r = [r for r in combined if r["percent"] == pct and r["model"] == "LogisticRegression"][0]
            cnn_r = [r for r in combined if r["percent"] == pct and r["model"] == "CustomCNN"][0]
            ssl_r = [r for r in combined if r["percent"] == pct and r["model"] == "ResNet_SSL"][0]
            mf.write(
                f"- **{pct}% labels:** `ResNet_SSL` ({ssl_r['roc_auc_mean']:.4f} ROC-AUC, {ssl_r['pr_auc_mean']:.4f} PR-AUC) vs. "
                f"`LogisticRegression` ({lr_r['roc_auc_mean']:.4f} / {lr_r['pr_auc_mean']:.4f}) vs. "
                f"`CustomCNN` ({cnn_r['roc_auc_mean']:.4f} / {cnn_r['pr_auc_mean']:.4f})\n"
            )

    print("Saved reports/real/label_efficiency_fairness.json and reports/real/label_efficiency_fairness.md")


if __name__ == "__main__":
    main()
