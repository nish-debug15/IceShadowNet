import os, sys, json, torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
from src.models.custom_cnn import CustomCNN
from src.models.resnet import IceResNet
from src.datasets.ice_dataset import IcePatchDataset, get_ambiguity_mask
from src.evaluate import (
    get_strip_ids_from_coords,
    evaluate_model,
    block_bootstrap_evaluation,
    compute_comprehensive_metrics,
)
from scripts.baseline_real import extract_features
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_REAL = "data/processed_real"

# Load datasets
train_data = np.load(os.path.join(PROCESSED_REAL, "train.npz"))
val_data   = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
test_data  = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

X_test_img = test_data["images"]
y_test = test_data["labels"]
test_coords = test_data["coords"]
test_strips = get_strip_ids_from_coords(test_coords)

patches = list(zip(X_test_img, y_test.tolist(), [tuple(c) for c in test_coords]))
test_ds = IcePatchDataset(patches, augment=False, normalize=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

# Clean mask
test_ice_fracs = test_data["ice_fracs"]
amb_mask = get_ambiguity_mask(test_ice_fracs, min_px=1, max_px=4, patch_size=64)
clean_mask = ~amb_mask
test_strips_clean = test_strips[clean_mask]

# Baselines
X_train_feat, _, _, _ = extract_features(train_data)
y_train = train_data["labels"]
X_test_feat, _, _, _ = extract_features(test_data)

dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(X_train_feat, y_train)
lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000).fit(X_train_feat, y_train)

dt_probs = dt.predict_proba(X_test_feat)[:, 1]
lr_probs = lr.predict_proba(X_test_feat)[:, 1]

dt_th = 0.42
lr_th = 0.50

# Load stored records
with open('reports/real/training_summary.json') as f: orig = json.load(f)
with open('reports/real/training_summary_v2a.json') as f: v2a = json.load(f)
with open('reports/real/training_summary_v2b.json') as f: v2b = json.load(f)
with open('reports/real/baseline_results.json') as f: base = json.load(f)

all_primary = orig['primary'] + v2a['primary'] + v2b['primary']
all_clean   = orig['clean'] + v2a['clean'] + v2b['clean']
primary_clean_map = {(r['model'], r['seed']): r for r in all_clean}

# Add baselines to all_primary and all_clean
base_prim = base['primary']
base_clean = base['clean']

eval_runs = []
for r in all_primary:
    eval_runs.append({
        'type': 'dl',
        'model': r['model'],
        'seed': r['seed'],
        'best_epoch': r['best_epoch'],
        'val_auc': r['val_auc'],
        'th': r['tuned_th'],
        'orig_record': r,
    })

eval_runs.append({
    'type': 'base',
    'model': 'Decision Tree (max_depth=3, balanced)',
    'seed': None,
    'best_epoch': '-',
    'val_auc': None,
    'th': dt_th,
    'probs': dt_probs,
    'orig_record': base_prim[0],
})

eval_runs.append({
    'type': 'base',
    'model': 'Logistic Regression (balanced)',
    'seed': None,
    'best_epoch': '-',
    'val_auc': None,
    'th': lr_th,
    'probs': lr_probs,
    'orig_record': base_prim[1],
})

final_primary_rows = []
final_clean_rows = []
per_strip_rows = []

print("Running full re-evaluation & assertion...")

for run in eval_runs:
    mname = run['model']
    seed = run['seed']
    th = run['th']

    if run['type'] == 'dl':
        ckpt_path = os.path.join("models_real", f"{mname}_seed{seed}.pth")
        if 'CustomCNN' in mname: m = CustomCNN(3, 1).to(DEVICE)
        elif 'scratch' in mname: m = IceResNet(3, 1, pretrained=False).to(DEVICE)
        else: m = IceResNet(3, 1, pretrained=True).to(DEVICE)
        m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
        res = evaluate_model(m, test_loader, DEVICE, threshold=th)
        probs = res['probs']
        labels = res['labels']
    else:
        probs = run['probs']
        labels = y_test
        res = compute_comprehensive_metrics(labels, probs, threshold=th)

    preds = (probs >= th).astype(int)

    # Assert point estimates
    orig_rec = run['orig_record']
    np.testing.assert_allclose(res['accuracy'], orig_rec['accuracy'], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res['precision'], orig_rec['precision'], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res['recall'], orig_rec['recall'], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res['f1'], orig_rec['f1'], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res['roc_auc'], orig_rec['roc_auc'], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(res['pr_auc'], orig_rec['pr_auc'], rtol=1e-4, atol=1e-4)

    # Primary bootstrap (default + conditional)
    boot_seed = seed if seed is not None else 42
    boot = block_bootstrap_evaluation(labels, probs, test_strips, threshold=th, n_bootstraps=1000, seed=boot_seed)

    row_prim = {
        'model': mname,
        'seed': seed if seed is not None else '-',
        'best_epoch': run['best_epoch'],
        'val_auc': f"{run['val_auc']:.4f}" if run['val_auc'] is not None else "-",
        'roc_auc': float(res['roc_auc']),
        'roc_auc_ci': f"[{boot['roc_auc_ci'][0]:.4f}, {boot['roc_auc_ci'][1]:.4f}]",
        'pr_auc': float(res['pr_auc']),
        'pr_auc_ci': f"[{boot['pr_auc_ci'][0]:.4f}, {boot['pr_auc_ci'][1]:.4f}]",
        'f1': float(res['f1']),
        'f1_ci': f"[{boot['f1_ci'][0]:.4f}, {boot['f1_ci'][1]:.4f}]",
        'f1_cond_ci': f"[{boot['f1_conditional_ci'][0]:.4f}, {boot['f1_conditional_ci'][1]:.4f}]",
        'tuned_th': th,
        'prevalence': f"{100*np.mean(labels):.1f}%",
        'support': res['support'],
    }
    final_primary_rows.append(row_prim)

    # Clean test
    clean_eval = compute_comprehensive_metrics(labels[clean_mask], probs[clean_mask], threshold=th)
    clean_boot = block_bootstrap_evaluation(labels[clean_mask], probs[clean_mask], test_strips_clean, threshold=th, n_bootstraps=1000, seed=boot_seed)

    row_clean = {
        'model': mname,
        'seed': seed if seed is not None else '-',
        'best_epoch': run['best_epoch'],
        'val_auc': f"{run['val_auc']:.4f}" if run['val_auc'] is not None else "-",
        'roc_auc': float(clean_eval['roc_auc']),
        'roc_auc_ci': f"[{clean_boot['roc_auc_ci'][0]:.4f}, {clean_boot['roc_auc_ci'][1]:.4f}]",
        'pr_auc': float(clean_eval['pr_auc']),
        'pr_auc_ci': f"[{clean_boot['pr_auc_ci'][0]:.4f}, {clean_boot['pr_auc_ci'][1]:.4f}]",
        'f1': float(clean_eval['f1']),
        'f1_ci': f"[{clean_boot['f1_ci'][0]:.4f}, {clean_boot['f1_ci'][1]:.4f}]",
        'f1_cond_ci': f"[{clean_boot['f1_conditional_ci'][0]:.4f}, {clean_boot['f1_conditional_ci'][1]:.4f}]",
        'tuned_th': th,
        'prevalence': f"{100*np.mean(labels[clean_mask]):.1f}%",
        'support': clean_eval['support'],
    }
    final_clean_rows.append(row_clean)

    # Per-strip metrics (6, 10, 14)
    for s in [6, 10, 14]:
        idx = np.where(test_strips == s)[0]
        y_s = labels[idx]
        p_s = probs[idx]
        pred_s = preds[idx]

        n_p = int(np.sum(y_s == 1))
        n_n = int(np.sum(y_s == 0))
        cm_s = confusion_matrix(y_s, pred_s, labels=[0, 1])
        tn_s, fp_s, fn_s, tp_s = cm_s.ravel()

        rec_s = float(recall_score(y_s, pred_s, zero_division=0))
        prec_s = float(precision_score(y_s, pred_s, zero_division=0))
        f1_s = float(f1_score(y_s, pred_s, zero_division=0))

        if len(np.unique(y_s)) > 1:
            try: roc_s = float(roc_auc_score(y_s, p_s))
            except Exception: roc_s = None
        else:
            roc_s = None

        per_strip_rows.append({
            'model': mname,
            'seed': seed,
            'strip': s,
            'n_pos': n_p,
            'n_neg': n_n,
            'tp': int(tp_s),
            'fp': int(fp_s),
            'fn': int(fn_s),
            'tn': int(tn_s),
            'recall': rec_s,
            'precision': prec_s,
            'f1': f1_s,
            'roc_auc': roc_s,
        })

print("All assertions passed and metrics gathered.")

# ──────────────────────────────────────────────────────────────────────────────
# 1. Write reports/real/results_real_ci_all.md
# ──────────────────────────────────────────────────────────────────────────────
md_ci = """# Real-Data Benchmark Results (Default Spatial Block-Bootstrap CI)

> **DATA SOURCE:** Real Chandrayaan-2 DFSAR L1A quad-pol SLI (lines 100,000–140,000).  
> **Bootstrap Configuration:** **3 blocks** (along-track strips 6, 10, 14), **10 distinct resamples**, 1000 bootstrap iterations.  
> **Default CI:** Includes all resamples with $\ge 1$ ground-truth positive (all 1000 rounds). Accurately captures the lower tail from spatial heterogeneity where strip 14 has zero true positive detections.  
> **Conditional CI (F1 only):** Sensitivity analysis conditioning on $\ge 1$ positive detection (excluding degenerate $\\text{TP}=0$ rounds).

## 1. Primary Benchmark (Full Spatial Test Set: N=338, Prevalence=29.9%)

| Model | Seed | Best Ep | Val AUC | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test F1 (Default 95% CI) | F1 (Conditional 95% CI) | Tuned Th | Prev | Support |
|---|---|---|---|---|---|---|---|---|---|---|
"""

for r in final_primary_rows:
    md_ci += f"| {r['model']} | {r['seed']} | {r['best_epoch']} | {r['val_auc']} | {r['roc_auc']:.4f} `{r['roc_auc_ci']}` | {r['pr_auc']:.4f} `{r['pr_auc_ci']}` | **{r['f1']:.4f}** `{r['f1_ci']}` | `{r['f1_cond_ci']}` | {r['tuned_th']:.2f} | {r['prevalence']} | {r['support']} |\n"

md_ci += """
---

## 2. Ambiguity-Removed Benchmark (Clean Test Set: N=205, Prevalence=49.3%)

| Model | Seed | Best Ep | Val AUC | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test F1 (Default 95% CI) | F1 (Conditional 95% CI) | Tuned Th | Prev | Support |
|---|---|---|---|---|---|---|---|---|---|---|
"""

for r in final_clean_rows:
    md_ci += f"| {r['model']} | {r['seed']} | {r['best_epoch']} | {r['val_auc']} | {r['roc_auc']:.4f} `{r['roc_auc_ci']}` | {r['pr_auc']:.4f} `{r['pr_auc_ci']}` | **{r['f1']:.4f}** `{r['f1_ci']}` | `{r['f1_cond_ci']}` | {r['tuned_th']:.2f} | {r['prevalence']} | {r['support']} |\n"

with open("reports/real/results_real_ci_all.md", "w", encoding="utf-8") as f:
    f.write(md_ci)
print("Saved reports/real/results_real_ci_all.md")

# ──────────────────────────────────────────────────────────────────────────────
# 2. Write reports/real/per_strip_metrics.md
# ──────────────────────────────────────────────────────────────────────────────
md_strip = """# Real DFSAR Along-Track Per-Strip Metrics Breakdown

> **Spatial Test Set Geometry:** Three along-track strips spanning radar lines 100,000–140,000:
> - **Strip 6**: Total = 112 ($N_{pos} = 43$, $N_{neg} = 69$, Prevalence = 38.4%)
> - **Strip 10**: Total = 112 ($N_{pos} = 53$, $N_{neg} = 59$, Prevalence = 47.3%)
> - **Strip 14**: Total = 114 ($N_{pos} = 5$, $N_{neg} = 109$, Prevalence = 4.4%)

---

## 1. Per-Strip Metrics Table (All 15 DL Runs + 2 Baselines)

| Model Architecture | Seed | Strip | N Pos | N Neg | TP | FP | FN | TN | Recall | Precision | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""

for r in per_strip_rows:
    s_label = r['seed'] if r['seed'] is not None else "-"
    auc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "-"
    md_strip += f"| {r['model']} | {s_label} | {r['strip']} | {r['n_pos']} | {r['n_neg']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['recall']:.4f} | {r['precision']:.4f} | {r['f1']:.4f} | {auc_str} |\n"

md_strip += """
---

## 2. Mean Across Seeds per Model Architecture per Strip

| Model Architecture | Strip | Mean TP | Mean FP | Mean FN | Mean TN | Mean Recall | Mean Precision | Mean F1 Score | Mean ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
"""

models_order = ['CustomCNN', 'ResNet_scratch', 'ResNet_pretrained', 'ResNet_pretrained_v2a', 'ResNet_pretrained_v2b', 'Decision Tree', 'Logistic Regression']

strip_means = []
for m in models_order:
    for s in [6, 10, 14]:
        subset = [r for r in per_strip_rows if r['model'] == m and r['strip'] == s]
        if not subset: continue
        tp_m = np.mean([r['tp'] for r in subset])
        fp_m = np.mean([r['fp'] for r in subset])
        fn_m = np.mean([r['fn'] for r in subset])
        tn_m = np.mean([r['tn'] for r in subset])
        rec_m = np.mean([r['recall'] for r in subset])
        prec_m = np.mean([r['precision'] for r in subset])
        f1_m = np.mean([r['f1'] for r in subset])
        aucs = [r['roc_auc'] for r in subset if r['roc_auc'] is not None]
        auc_m_str = f"{np.mean(aucs):.4f}" if aucs else "-"
        md_strip += f"| **{m}** | Strip {s} | {tp_m:.1f} | {fp_m:.1f} | {fn_m:.1f} | {tn_m:.1f} | {rec_m:.4f} | {prec_m:.4f} | **{f1_m:.4f}** | {auc_m_str} |\n"

md_strip += """
---

## 3. Strip 14 Detection Analysis (Zero-TP Prevalence)

- **Ground Truth on Strip 14:** Only 5 positive patches out of 114 total patches (prevalence 4.4%).
- **Detection Results on Strip 14:**
  - **CustomCNN** (all 3 seeds): $\\text{TP} = 0, \\text{FP} = 0, \\text{FN} = 5, \\text{TN} = 109 \\implies \\text{F1} = 0.0000$ (0/3 seeds detected positives).
  - **ResNet_pretrained_v2a** (all 3 seeds): $\\text{TP} = 0, \\text{FP} \\in [5, 18], \\text{FN} = 5 \\implies \\text{F1} = 0.0000$ (0/3 seeds detected positives).
  - **Logistic Regression**: $\\text{TP} = 0, \\text{FP} = 0, \\text{FN} = 5, \\text{TN} = 109 \\implies \\text{F1} = 0.0000$.
  - **ResNet_scratch**: Seed 42 and Seed 123 have $\\text{TP} = 0$; Seed 2024 has $\\text{TP} = 1, \\text{FP} = 9 \\implies \\text{F1} = 0.1333$.
  - **ResNet_pretrained_v2b**: Seed 42 and Seed 123 have $\\text{TP} = 0$; Seed 2024 has $\\text{TP} = 1, \\text{FP} = 17 \\implies \\text{F1} = 0.0870$.
  - **ResNet_pretrained (Orig)**: Low thresholds produce high false positives ($\text{FP} \in [23, 41]$) picking up 1 to 3 positives ($\text{TP} \in [1, 3]$).
  - **Decision Tree**: High false positives ($\\text{FP} = 50$) with $\\text{TP} = 1 \\implies \\text{F1} = 0.0357$.
- **Conclusion:** Strip 14 does **not** have $\\text{TP}=0$ for every single model, but has $\\text{TP}=0$ across **11 of the 17 total evaluated runs (64.7%)**, including 100% of CustomCNN, 100% of ResNet_pretrained_v2a, and Logistic Regression. In a 3-block bootstrap, $(1/3)^3 = 1/27 \\approx 3.7\\%$ of resamples select only Strip 14, directly explaining why the 2.5th percentile lower bound of F1 drops to `0.0000` under the default bootstrap procedure.
"""

with open("reports/real/per_strip_metrics.md", "w", encoding="utf-8") as f:
    f.write(md_strip)
print("Saved reports/real/per_strip_metrics.md")
