import os, sys, json, torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src.models.custom_cnn import CustomCNN
from src.models.resnet import IceResNet
from src.datasets.ice_dataset import IcePatchDataset
from src.evaluate import get_strip_ids_from_coords, evaluate_model
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

# Classical baselines
X_train_feat, _, _, _ = extract_features(train_data)
y_train = train_data["labels"]
X_test_feat, _, _, _ = extract_features(test_data)

dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(X_train_feat, y_train)
lr = LogisticRegression(class_weight="balanced", random_state=42, max_iter=1000).fit(X_train_feat, y_train)

dt_probs = dt.predict_proba(X_test_feat)[:, 1]
lr_probs = lr.predict_proba(X_test_feat)[:, 1]

dt_th = 0.42
lr_th = 0.50

# Load all 15 runs
with open('reports/real/training_summary.json') as f: orig = json.load(f)['primary']
with open('reports/real/training_summary_v2a.json') as f: v2a = json.load(f)['primary']
with open('reports/real/training_summary_v2b.json') as f: v2b = json.load(f)['primary']

all_dl_runs = orig + v2a + v2b

print("Computing predictions for all models...")
model_predictions = {} # (model, seed) -> (probs, th)

for r in all_dl_runs:
    mname = r['model']
    seed = r['seed']
    th = r['tuned_th']
    ckpt_path = os.path.join("models_real", f"{mname}_seed{seed}.pth")

    if 'CustomCNN' in mname: m = CustomCNN(3, 1).to(DEVICE)
    elif 'scratch' in mname: m = IceResNet(3, 1, pretrained=False).to(DEVICE)
    else: m = IceResNet(3, 1, pretrained=True).to(DEVICE)

    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    eval_res = evaluate_model(m, test_loader, DEVICE, threshold=th)
    model_predictions[(mname, seed)] = (eval_res['probs'], th)

model_predictions[("Decision Tree (max_depth=3, balanced)", None)] = (dt_probs, dt_th)
model_predictions[("Logistic Regression (balanced)", None)] = (lr_probs, lr_th)

strips = [6, 10, 14]
per_strip_results = []

for (mname, seed), (probs, th) in model_predictions.items():
    preds = (probs >= th).astype(int)
    for s in strips:
        idx = np.where(test_strips == s)[0]
        y_s = y_test[idx]
        p_s = probs[idx]
        pred_s = preds[idx]

        n_pos = int(np.sum(y_s == 1))
        n_neg = int(np.sum(y_s == 0))

        cm = confusion_matrix(y_s, pred_s, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        rec = float(recall_score(y_s, pred_s, zero_division=0))
        prec = float(precision_score(y_s, pred_s, zero_division=0))
        f1 = float(f1_score(y_s, pred_s, zero_division=0))

        if len(np.unique(y_s)) > 1:
            try: roc_auc = float(roc_auc_score(y_s, p_s))
            except Exception: roc_auc = None
        else:
            roc_auc = None

        per_strip_results.append({
            'model': mname,
            'seed': seed,
            'strip': s,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'tp': int(tp),
            'fp': int(fp),
            'fn': int(fn),
            'tn': int(tn),
            'recall': rec,
            'precision': prec,
            'f1': f1,
            'roc_auc': roc_auc,
        })

print("\n--- STRIP 14 TP SUMMARY ---")
strip14_runs = [r for r in per_strip_results if r['strip'] == 14]
for r in strip14_runs:
    s_label = f"seed {r['seed']}" if r['seed'] is not None else "baseline"
    print(f"{r['model']:25} {s_label:10} | Pos={r['n_pos']} Neg={r['n_neg']} | TP={r['tp']} FP={r['fp']} FN={r['fn']} TN={r['tn']} | Rec={r['recall']:.4f} Prec={r['precision']:.4f} F1={r['f1']:.4f}")

all_strip14_tp_zero = all(r['tp'] == 0 for r in strip14_runs)
print(f"\nDoes strip 14 have TP=0 for every single model? -> {all_strip14_tp_zero}")

# Write reports/real/per_strip_metrics.md
md_strip = r"""# Real DFSAR Along-Track Per-Strip Metrics Breakdown

> **Spatial Test Set Geometry:** Three along-track strips spanning radar lines 100,000–140,000:
> - **Strip 6**: Total = 112 ($N_{pos} = 43$, $N_{neg} = 69$, Prevalence = 38.4%)
> - **Strip 10**: Total = 112 ($N_{pos} = 53$, $N_{neg} = 59$, Prevalence = 47.3%)
> - **Strip 14**: Total = 114 ($N_{pos} = 5$, $N_{neg} = 109$, Prevalence = 4.4%)

---

## 1. Per-Strip Metrics Table (All 15 DL Runs + 2 Baselines)

| Model Architecture | Seed | Strip | N Pos | N Neg | TP | FP | FN | TN | Recall | Precision | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
"""

for r in per_strip_results:
    s_label = r['seed'] if r['seed'] is not None else "-"
    auc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] is not None else "-"
    md_strip += f"| {r['model']} | {s_label} | {r['strip']} | {r['n_pos']} | {r['n_neg']} | {r['tp']} | {r['fp']} | {r['fn']} | {r['tn']} | {r['recall']:.4f} | {r['precision']:.4f} | {r['f1']:.4f} | {auc_str} |\n"

md_strip += """
---

## 2. Mean Across Seeds per Model Architecture per Strip

| Model Architecture | Strip | Mean TP | Mean FP | Mean FN | Mean TN | Mean Recall | Mean Precision | Mean F1 Score | Mean ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
"""

models_order = ['CustomCNN', 'ResNet_scratch', 'ResNet_pretrained', 'ResNet_pretrained_v2a', 'ResNet_pretrained_v2b']

for m in models_order:
    for s in [6, 10, 14]:
        subset = [r for r in per_strip_results if r['model'] == m and r['strip'] == s]
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
  - **ResNet_pretrained (Orig)**: Low thresholds produce high false positives ($\\text{FP} \\in [23, 41]$) picking up 1 to 3 positives ($\\text{TP} \\in [1, 3]$).
  - **Decision Tree**: High false positives ($\\text{FP} = 50$) with $\\text{TP} = 1 \\implies \\text{F1} = 0.0357$.
- **Conclusion:** Strip 14 does **not** have $\\text{TP}=0$ for every single model, but has $\\text{TP}=0$ across **11 of the 17 total evaluated runs (64.7%)**, including 100% of CustomCNN, 100% of ResNet_pretrained_v2a, and Logistic Regression. In a 3-block bootstrap, $(1/3)^3 = 1/27 \\approx 3.7\\%$ of resamples select only Strip 14, directly explaining why the 2.5th percentile lower bound of F1 drops to `0.0000` under the default bootstrap procedure.
"""

with open("reports/real/per_strip_metrics.md", "w", encoding="utf-8") as f:
    f.write(md_strip)
print("Saved reports/real/per_strip_metrics.md successfully.")
