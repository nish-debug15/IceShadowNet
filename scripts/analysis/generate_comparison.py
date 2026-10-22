import os, sys, json
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

with open('reports/real/training_summary.json') as f:
    orig = json.load(f)['primary']
with open('reports/real/training_summary_v2a.json') as f:
    v2a = json.load(f)['primary']
with open('reports/real/training_summary_v2b.json') as f:
    v2b = json.load(f)['primary']

all_runs = orig + v2a + v2b

md = """# Real-Data Benchmark Comparison: Pretrained ResNet Variants vs. Baselines

> **Protocol:** Evaluated on real Chandrayaan-2 DFSAR L1A quad-pol SLI dataset (lines 100,000–140,000). Max 30 epochs, early stopping on Val ROC-AUC (patience 5), validation-tuned decision threshold, 95% block-bootstrap CIs across strips (1000 rounds).
> **Methodological Note on Ranking:** As established in the evaluation protocol, model architectures are ranked strictly by **Validation ROC-AUC** to prevent data leakage and cherry-picking based on test set metrics.

---

## 1. Per-Run Metrics (Seeds 42, 123, 2024)

### Full Spatial Test Set (N=338, Prevalence=29.9%)

| Model Architecture | Seed | Best Ep | Val AUC | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test F1 (95% CI) | Tuned Th | Prev | Support |
|---|---|---|---|---|---|---|---|---|---|
"""

name_map = {
    'CustomCNN': '**CustomCNN**',
    'ResNet_scratch': '**ResNet_scratch**',
    'ResNet_pretrained': '**ResNet_pretrained (Original)**',
    'ResNet_pretrained_v2a': '**ResNet_pretrained_v2a (Fixed Freeze + Native Conv1)**',
    'ResNet_pretrained_v2b': '**ResNet_pretrained_v2b (Differential LR: 1e-4 / 1e-3)**',
}

for r in all_runs:
    display_name = name_map.get(r['model'], f"**{r['model']}**")
    md += f"| {display_name} | {r['seed']} | {r['best_epoch']} | {r['val_auc']:.4f} | {r['roc_auc']:.4f} `{r['roc_auc_ci']}` | {r['pr_auc']:.4f} `{r['pr_auc_ci']}` | {r['f1']:.4f} `{r['f1_ci']}` | {r['tuned_th']:.2f} | {r['prevalence']} | {r['support']} |\n"

md += """
---

## 2. Summary Benchmark: Mean ± Std Across Seeds

| Model Architecture | Trainable Params | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|---|---|
"""

models_order = ['CustomCNN', 'ResNet_pretrained_v2a', 'ResNet_scratch', 'ResNet_pretrained_v2b', 'ResNet_pretrained']
stats = {}
for m in set(r['model'] for r in all_runs):
    subset = [r for r in all_runs if r['model'] == m]
    params = subset[0]['params']
    val_aucs = [r['val_auc'] for r in subset]
    roc_aucs = [r['roc_auc'] for r in subset]
    pr_aucs = [r['pr_auc'] for r in subset]
    f1s = [r['f1'] for r in subset]
    stats[m] = {
        'params': params,
        'val_m': np.mean(val_aucs), 'val_s': np.std(val_aucs, ddof=1),
        'roc_m': np.mean(roc_aucs), 'roc_s': np.std(roc_aucs, ddof=1),
        'pr_m': np.mean(pr_aucs), 'pr_s': np.std(pr_aucs, ddof=1),
        'f1_m': np.mean(f1s), 'f1_s': np.std(f1s, ddof=1),
    }

# Rank models by validation ROC-AUC mean
ranked_models = sorted(stats.keys(), key=lambda m: stats[m]['val_m'], reverse=True)

for m in ranked_models:
    s = stats[m]
    dname = name_map.get(m, f"**{m}**")
    md += f"| {dname} | {s['params']:,} | **{s['val_m']:.4f} ± {s['val_s']:.4f}** | {s['roc_m']:.4f} ± {s['roc_s']:.4f} | {s['pr_m']:.4f} ± {s['pr_s']:.4f} | {s['f1_m']:.4f} ± {s['f1_s']:.4f} |\n"

md += """
---

## 3. Architecture Ranking (Strictly by Mean Validation ROC-AUC)

1. **CustomCNN** (`0.6784 ± 0.0131`): Top performer on validation criteria; compact 3-layer representation avoids overfitting to radar speckle.
2. **ResNet_pretrained_v2a** (`0.6657 ± 0.0063`): Substantial improvement over original pretrained (+0.0992 Val AUC). Fixing the substring-freezing bug in `layer3`/`layer4` and maintaining native 3-channel weights on `conv1` elevated validation discriminability above `ResNet_scratch`.
3. **ResNet_scratch** (`0.6504 ± 0.0134`): Solid baseline; learns SAR-specific representations from scratch but exhibits wider variance across seeds on validation splits.
4. **ResNet_pretrained_v2b** (`0.6217 ± 0.0240`): Differential learning rate (backbone `1e-4`, head `1e-3`) underperformed v2a (`1e-3`), indicating that `1e-4` is overly conservative to bridge the cross-domain optical-to-radar distribution shift within 30 epochs.
5. **ResNet_pretrained (Original)** (`0.5665 ± 0.0126`): Severely impaired by the dual issues of channel collapse in `conv1` (averaged RGB weights) and accidental intra-block freezing of `conv1`/`bn1` across `layer3` and `layer4`.
"""

with open('reports/real/results_real_variants_comparison.md', 'w', encoding='utf-8') as f:
    f.write(md)
print("Regenerated reports/real/results_real_variants_comparison.md successfully.")
