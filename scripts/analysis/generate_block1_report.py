"""
scripts/analysis/generate_block1_report.py
==========================================
Amendment B comprehensive reporter for Block 1 (frozen split [90624, 319812]):
  1. Pooled & per-test-strip metrics (strips 0, 8, 15) with 1000-round strip-block bootstrap CIs.
  2. Headline metrics: ROC-AUC & PR-AUC (with prevalence shown as PR baseline).
  3. Threshold transfer: F1 & Accuracy at val-chosen threshold AND best-possible test threshold.
  4. Unambiguous subset ablation (exclude 1-4 px patches): Balanced Accuracy, Accuracy, ROC-AUC, PR-AUC, F1.
  5. Old split [100000, 140000] vs New frozen split [90624, 319812] before/after comparison table.

Outputs:
  - reports/real/block1_benchmark_report.json
  - reports/real/block1_benchmark_report.md
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, balanced_accuracy_score,
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.datasets.ice_dataset import IcePatchDataset, get_ambiguity_mask
from src.models.custom_cnn import CustomCNN
from src.models.resnet import IceResNet
from src.evaluate import tune_threshold_on_val


def best_test_threshold(y_true, y_probs):
    best_f1, best_th = 0.0, 0.5
    for th in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_probs >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)
    return best_th, best_f1


def strip_bootstrap_ci(y_true, y_probs, strip_ids, val_th, n_boot=1000, seed=42):
    rng = np.random.default_rng(seed)
    unique_strips = np.unique(strip_ids)
    strip_idx = {s: np.where(strip_ids == s)[0] for s in unique_strips}

    rocs, prs, f1s, accs = [], [], [], []
    for _ in range(n_boot):
        sampled = rng.choice(unique_strips, size=len(unique_strips), replace=True)
        idx = np.concatenate([strip_idx[s] for s in sampled])
        yt, yp = y_true[idx], y_probs[idx]
        if len(np.unique(yt)) < 2:
            continue
        rocs.append(roc_auc_score(yt, yp))
        prs.append(average_precision_score(yt, yp))
        pred = (yp >= val_th).astype(int)
        f1s.append(f1_score(yt, pred, zero_division=0))
        accs.append(accuracy_score(yt, pred))

    def ci(arr):
        if not arr:
            return [0.0, 0.0]
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))]

    return {"roc_auc_ci": ci(rocs), "pr_auc_ci": ci(prs), "f1_ci": ci(f1s), "acc_ci": ci(accs)}


def compute_split_metrics(y_true, y_probs, val_th):
    pred_val = (y_probs >= val_th).astype(int)
    best_th, best_f1 = best_test_threshold(y_true, y_probs)
    pred_best = (y_probs >= best_th).astype(int)
    return {
        "n": int(len(y_true)),
        "pos": int(y_true.sum()),
        "prevalence": float(y_true.mean()),
        "roc_auc": float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, y_probs)) if y_true.sum() > 0 else 0.0,
        "val_th": float(val_th),
        "f1_val_th": float(f1_score(y_true, pred_val, zero_division=0)),
        "acc_val_th": float(accuracy_score(y_true, pred_val)),
        "bal_acc_val_th": float(balanced_accuracy_score(y_true, pred_val)) if len(np.unique(y_true)) > 1 else 0.5,
        "best_test_th": float(best_th),
        "f1_best_test_th": float(best_f1),
        "acc_best_test_th": float(accuracy_score(y_true, pred_best)),
    }


def predict_dl(ckpt_path, model_factory, loader, device):
    model = model_factory().to(device)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    probs = []
    with torch.no_grad():
        for xb, _ in loader:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                logits = model(xb.to(device))
            probs.extend(torch.sigmoid(logits).cpu().numpy().flatten())
    return np.array(probs, dtype=np.float64)


def extract_tabular_features(d):
    imgs = d["images"]
    feats = []
    for c in range(3):
        ch = imgs[:, c]
        feats.append(ch.mean(axis=(1, 2)))
        feats.append(ch.std(axis=(1, 2)))
        feats.append(np.percentile(ch, 90, axis=(1, 2)))
        feats.append(np.percentile(ch, 10, axis=(1, 2)))
    return np.column_stack(feats)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_tr = np.load("data/processed_real/train.npz")
    d_va = np.load("data/processed_real/val.npz")
    d_te = np.load("data/processed_real/test.npz")

    y_tr = d_tr["labels"].astype(int)
    y_va = d_va["labels"].astype(int)
    y_te = d_te["labels"].astype(int)
    te_strips = d_te["strip_ids"].astype(int)
    clean_mask = ~get_ambiguity_mask(d_te["ice_fracs"], min_px=1, max_px=4, patch_size=64)

    va_loader = DataLoader(
        IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(d_va["images"], y_va)], augment=False),
        batch_size=64, shuffle=False,
    )
    te_loader = DataLoader(
        IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(d_te["images"], y_te)], augment=False),
        batch_size=64, shuffle=False,
    )

    all_records = []

    # 1. Classical Baselines
    X_tr_tab = extract_tabular_features(d_tr)
    X_va_tab = extract_tabular_features(d_va)
    X_te_tab = extract_tabular_features(d_te)

    for bname, clf in [
        ("DecisionTree", DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42)),
        ("LogisticRegression", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)),
    ]:
        clf.fit(X_tr_tab, y_tr)
        va_p = clf.predict_proba(X_va_tab)[:, 1]
        te_p = clf.predict_proba(X_te_tab)[:, 1]
        val_th, _ = tune_threshold_on_val(y_va, va_p)
        pooled = compute_split_metrics(y_te, te_p, val_th)
        cis = strip_bootstrap_ci(y_te, te_p, te_strips, val_th)
        clean = compute_split_metrics(y_te[clean_mask], te_p[clean_mask], val_th)
        per_strip = {
            int(s): compute_split_metrics(y_te[te_strips == s], te_p[te_strips == s], val_th)
            for s in sorted(np.unique(te_strips))
        }
        all_records.append({
            "model": bname, "seed": 42,
            "pooled": pooled, "cis": cis, "clean_ablation": clean, "per_strip": per_strip,
        })

    # 2. Deep Models (3 seeds each)
    dl_configs = [
        ("CustomCNN", lambda: CustomCNN(in_channels=3, num_classes=1)),
        ("ResNet_scratch", lambda: IceResNet(in_channels=3, num_classes=1, pretrained=False)),
        ("ResNet_pretrained", lambda: IceResNet(in_channels=3, num_classes=1, pretrained=True)),
    ]
    for mname, factory in dl_configs:
        for seed in [42, 123, 2024]:
            ckpt = f"models_real/{mname}_seed{seed}.pth"
            if not os.path.isfile(ckpt):
                print(f"Warning: missing {ckpt}, skipping")
                continue
            va_p = predict_dl(ckpt, factory, va_loader, device)
            te_p = predict_dl(ckpt, factory, te_loader, device)
            val_th, _ = tune_threshold_on_val(y_va, va_p)
            pooled = compute_split_metrics(y_te, te_p, val_th)
            cis = strip_bootstrap_ci(y_te, te_p, te_strips, val_th)
            clean = compute_split_metrics(y_te[clean_mask], te_p[clean_mask], val_th)
            per_strip = {
                int(s): compute_split_metrics(y_te[te_strips == s], te_p[te_strips == s], val_th)
                for s in sorted(np.unique(te_strips))
            }
            all_records.append({
                "model": mname, "seed": seed,
                "pooled": pooled, "cis": cis, "clean_ablation": clean, "per_strip": per_strip,
            })

    os.makedirs("reports/real", exist_ok=True)
    with open("reports/real/block1_benchmark_report.json", "w", encoding="utf-8") as jf:
        json.dump(all_records, jf, indent=2)

    # Aggregate mean ± std by model
    models_order = ["CustomCNN", "ResNet_scratch", "ResNet_pretrained", "LogisticRegression", "DecisionTree"]
    with open("reports/real/block1_benchmark_report.md", "w", encoding="utf-8") as mf:
        mf.write("# Block 1 Benchmark Report (Frozen Split `[90624, 319812]`, 256-Row Buffer)\n\n")
        mf.write(f"- **Test Set:** `N = {len(y_te)}`, `Positives = {int(y_te.sum())}`, **Prevalence (PR-AUC baseline) = `{100*y_te.mean():.2f}%`**\n")
        mf.write(f"- **Test Strips:** `0` (`{int((y_te[te_strips==0]).sum())}/{int((te_strips==0).sum())}`), "
                 f"`8` (`{int((y_te[te_strips==8]).sum())}/{int((te_strips==8).sum())}`), "
                 f"`15` (`{int((y_te[te_strips==15]).sum())}/{int((te_strips==15).sum())}`)\n")
        mf.write(f"- **Unambiguous Clean Subset (Ablation, 1–4 px excluded):** `N = {int(clean_mask.sum())}`, `Positives = {int(y_te[clean_mask].sum())}` (`{100*y_te[clean_mask].mean():.2f}%`)\n\n")

        mf.write("## 1. Pooled Test Metrics (Mean ± Std Across Seeds)\n\n")
        mf.write("| Model | ROC-AUC | PR-AUC (Baseline 16.45%) | F1 (Val Th) | Acc (Val Th) | F1 (Best Test Th) | Acc (Best Test Th) | Bal Acc (Val Th) |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        summary_by_model = {}
        for mname in models_order:
            sub = [r for r in all_records if r["model"] == mname]
            if not sub:
                continue
            def ms(key, bucket="pooled"):
                vals = [r[bucket][key] for r in sub]
                return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"
            summary_by_model[mname] = {
                "roc": float(np.mean([r["pooled"]["roc_auc"] for r in sub])),
                "pr": float(np.mean([r["pooled"]["pr_auc"] for r in sub])),
                "f1_val": float(np.mean([r["pooled"]["f1_val_th"] for r in sub])),
                "f1_best": float(np.mean([r["pooled"]["f1_best_test_th"] for r in sub])),
            }
            mf.write(
                f"| **{mname}** | {ms('roc_auc')} | {ms('pr_auc')} | {ms('f1_val_th')} | {ms('acc_val_th')} | "
                f"{ms('f1_best_test_th')} | {ms('acc_best_test_th')} | {ms('bal_acc_val_th')} |\n"
            )

        mf.write("\n## 2. Per-Test-Strip Breakdown & Strip-Block Bootstrap 95% CIs (Seed 42 Representative)\n\n")
        mf.write("| Model | Pooled ROC-AUC [95% CI] | Pooled PR-AUC [95% CI] | Strip 0 AUC / PR | Strip 8 AUC / PR | Strip 15 AUC / PR |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for mname in models_order:
            sub = [r for r in all_records if r["model"] == mname and r["seed"] == 42]
            if not sub:
                continue
            r = sub[0]
            ps = r["per_strip"]
            mf.write(
                f"| **{mname}** | {r['pooled']['roc_auc']:.4f} `[{r['cis']['roc_auc_ci'][0]:.4f}, {r['cis']['roc_auc_ci'][1]:.4f}]` | "
                f"{r['pooled']['pr_auc']:.4f} `[{r['cis']['pr_auc_ci'][0]:.4f}, {r['cis']['pr_auc_ci'][1]:.4f}]` | "
                f"{ps[0]['roc_auc']:.3f} / {ps[0]['pr_auc']:.3f} | "
                f"{ps[8]['roc_auc']:.3f} / {ps[8]['pr_auc']:.3f} | "
                f"{ps[15]['roc_auc']:.3f} / {ps[15]['pr_auc']:.3f} |\n"
            )

        mf.write("\n## 3. Unambiguous Subset Ablation (1–4 px Ambiguity Band Excluded)\n\n")
        mf.write("| Model | Clean ROC-AUC | Clean PR-AUC (Baseline 27.39%) | Clean F1 (Val Th) | Clean Acc (Val Th) | Clean Bal Acc (Val Th) |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for mname in models_order:
            sub = [r for r in all_records if r["model"] == mname]
            if not sub:
                continue
            def msc(key):
                vals = [r["clean_ablation"][key] for r in sub]
                return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}" if len(vals) > 1 else f"{vals[0]:.4f}"
            mf.write(
                f"| **{mname}** | {msc('roc_auc')} | {msc('pr_auc')} | {msc('f1_val_th')} | {msc('acc_val_th')} | {msc('bal_acc_val_th')} |\n"
            )

        mf.write("\n## 4. Before vs. After Split Comparison (Old `[100000, 140000]` vs. New Frozen `[90624, 319812]`)\n\n")
        mf.write("| Metric / Model | Old Split (`[100k, 140k]`, 64px buf) | New Frozen Split (`[90.6k, 319.8k]`, 256px buf) |\n")
        mf.write("| :--- | :---: | :---: |\n")
        mf.write("| **Total Post-Buffer Patches** | `2,366` (`1686 / 342 / 338`) | `13,886` (`9828 / 2040 / 2018`) |\n")
        mf.write("| **Test Positives (Prevalence)** | `101` (`29.9%`, min strip `5` pos) | `332` (`16.45%`, min strip `70` pos) |\n")
        mf.write("| **CustomCNN (ROC-AUC / PR-AUC / F1)** | `0.7564 / 0.5096 / 0.6113` | "
                 f"`{summary_by_model['CustomCNN']['roc']:.4f} / {summary_by_model['CustomCNN']['pr']:.4f} / {summary_by_model['CustomCNN']['f1_val']:.4f}` |\n")
        mf.write("| **ResNet_scratch (ROC-AUC / PR-AUC / F1)** | `0.7531 / 0.4988 / 0.5814` | "
                 f"`{summary_by_model['ResNet_scratch']['roc']:.4f} / {summary_by_model['ResNet_scratch']['pr']:.4f} / {summary_by_model['ResNet_scratch']['f1_val']:.4f}` |\n")
        mf.write("| **ResNet_pretrained (ROC-AUC / PR-AUC / F1)** | `0.6876 / 0.4561 / 0.4871` *(v2a)* | "
                 f"`{summary_by_model['ResNet_pretrained']['roc']:.4f} / {summary_by_model['ResNet_pretrained']['pr']:.4f} / {summary_by_model['ResNet_pretrained']['f1_val']:.4f}` |\n")
        mf.write("| **LogisticRegression (ROC-AUC / PR-AUC / F1)** | `0.7314 / 0.4953 / 0.5145` | "
                 f"`{summary_by_model['LogisticRegression']['roc']:.4f} / {summary_by_model['LogisticRegression']['pr']:.4f} / {summary_by_model['LogisticRegression']['f1_val']:.4f}` |\n")

    print("Saved reports/real/block1_benchmark_report.json and reports/real/block1_benchmark_report.md")


if __name__ == "__main__":
    main()
