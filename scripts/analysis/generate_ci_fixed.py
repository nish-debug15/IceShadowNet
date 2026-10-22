import os, sys, json, torch
import numpy as np

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
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROCESSED_DIR = "data/processed_real"

# Load test data
test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
X, Y, C = test_data["images"], test_data["labels"], test_data["coords"]
patches = list(zip(X, Y.tolist(), [tuple(c) for c in C]))
test_ds = IcePatchDataset(patches, augment=False, normalize=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
test_coords = np.array([p[2] for p in patches])
test_strips = get_strip_ids_from_coords(test_coords)

# Clean mask
test_ice_fracs = test_data["ice_fracs"]
amb_mask = get_ambiguity_mask(test_ice_fracs, min_px=1, max_px=4, patch_size=64)
clean_mask = ~amb_mask
test_strips_clean = test_strips[clean_mask]

# Load stored records
with open('reports/real/training_summary.json') as f:
    orig = json.load(f)
with open('reports/real/training_summary_v2a.json') as f:
    v2a = json.load(f)
with open('reports/real/training_summary_v2b.json') as f:
    v2b = json.load(f)

all_primary = orig['primary'] + v2a['primary'] + v2b['primary']
all_clean = orig['clean'] + v2a['clean'] + v2b['clean']

primary_clean_map = {(r['model'], r['seed']): r for r in all_clean}

new_primary_records = []
new_clean_records = []

print("Re-evaluating all 15 checkpoints and asserting point estimates...")

for r in all_primary:
    model_name = r['model']
    seed = r['seed']
    th = r['tuned_th']
    ckpt_name = f"{model_name}_seed{seed}.pth"
    ckpt_path = os.path.join("models_real", ckpt_name)

    assert os.path.isfile(ckpt_path), f"Checkpoint missing: {ckpt_path}"

    if 'CustomCNN' in model_name:
        m = CustomCNN(3, 1).to(DEVICE)
    elif 'scratch' in model_name:
        m = IceResNet(3, 1, pretrained=False).to(DEVICE)
    else:
        m = IceResNet(3, 1, pretrained=True).to(DEVICE)

    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

    # Evaluate full test
    full_eval = evaluate_model(m, test_loader, DEVICE, threshold=th)
    probs = full_eval['probs']
    labels = full_eval['labels']

    # Assert point estimates match
    np.testing.assert_allclose(full_eval['accuracy'], r['accuracy'], rtol=1e-4, atol=1e-4, err_msg=f"Accuracy mismatch for {model_name} s={seed}")
    np.testing.assert_allclose(full_eval['precision'], r['precision'], rtol=1e-4, atol=1e-4, err_msg=f"Precision mismatch for {model_name} s={seed}")
    np.testing.assert_allclose(full_eval['recall'], r['recall'], rtol=1e-4, atol=1e-4, err_msg=f"Recall mismatch for {model_name} s={seed}")
    np.testing.assert_allclose(full_eval['f1'], r['f1'], rtol=1e-4, atol=1e-4, err_msg=f"F1 mismatch for {model_name} s={seed}")
    np.testing.assert_allclose(full_eval['roc_auc'], r['roc_auc'], rtol=1e-4, atol=1e-4, err_msg=f"ROC-AUC mismatch for {model_name} s={seed}")
    np.testing.assert_allclose(full_eval['pr_auc'], r['pr_auc'], rtol=1e-4, atol=1e-4, err_msg=f"PR-AUC mismatch for {model_name} s={seed}")

    # Re-run fixed bootstrap
    boot = block_bootstrap_evaluation(labels, probs, test_strips, threshold=th, n_bootstraps=1000, seed=seed)

    new_prim_row = dict(r)
    new_prim_row['f1_ci_orig'] = r['f1_ci']
    new_prim_row['f1_ci'] = f"[{boot['f1_ci'][0]:.4f}, {boot['f1_ci'][1]:.4f}]"
    new_prim_row['roc_auc_ci'] = f"[{boot['roc_auc_ci'][0]:.4f}, {boot['roc_auc_ci'][1]:.4f}]"
    new_prim_row['pr_auc_ci'] = f"[{boot['pr_auc_ci'][0]:.4f}, {boot['pr_auc_ci'][1]:.4f}]"
    new_prim_row['n_valid_boot'] = boot['valid_rounds']
    new_primary_records.append(new_prim_row)

    # Clean test
    clean_eval = compute_comprehensive_metrics(labels[clean_mask], probs[clean_mask], threshold=th)
    clean_orig_r = primary_clean_map.get((model_name, seed))
    if clean_orig_r:
        np.testing.assert_allclose(clean_eval['f1'], clean_orig_r['f1'], rtol=1e-4, atol=1e-4, err_msg=f"Clean F1 mismatch for {model_name} s={seed}")
        np.testing.assert_allclose(clean_eval['roc_auc'], clean_orig_r['roc_auc'], rtol=1e-4, atol=1e-4, err_msg=f"Clean ROC-AUC mismatch for {model_name} s={seed}")
        np.testing.assert_allclose(clean_eval['pr_auc'], clean_orig_r['pr_auc'], rtol=1e-4, atol=1e-4, err_msg=f"Clean PR-AUC mismatch for {model_name} s={seed}")

        clean_boot = block_bootstrap_evaluation(labels[clean_mask], probs[clean_mask], test_strips_clean, threshold=th, n_bootstraps=1000, seed=seed)
        new_clean_row = dict(clean_orig_r)
        new_clean_row['f1_ci_orig'] = clean_orig_r['f1_ci']
        new_clean_row['f1_ci'] = f"[{clean_boot['f1_ci'][0]:.4f}, {clean_boot['f1_ci'][1]:.4f}]"
        new_clean_row['roc_auc_ci'] = f"[{clean_boot['roc_auc_ci'][0]:.4f}, {clean_boot['roc_auc_ci'][1]:.4f}]"
        new_clean_row['pr_auc_ci'] = f"[{clean_boot['pr_auc_ci'][0]:.4f}, {clean_boot['pr_auc_ci'][1]:.4f}]"
        new_clean_row['n_valid_boot'] = clean_boot['valid_rounds']
        new_clean_records.append(new_clean_row)

    print(f"PASS: {model_name:22} s={seed:4d} | Point estimates exact match | F1 CI: {new_prim_row['f1_ci_orig']} -> {new_prim_row['f1_ci']} (valid={boot['valid_rounds']}/1000)")

# Save reports/real/results_real_ci_fixed.md
md = """# REAL-DATA BENCHMARK RESULTS (Fixed F1 Bootstrap CI)

> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI, lines 100,000–140,000).
> **Protocol:** Max 30 epochs, early stopping on Val ROC-AUC (patience 5), Val-tuned decision threshold, 95% block-bootstrap CIs across strips (1000 rounds).
> **Bootstrap Fix Note:** Degenerate resamples where zero true positive detections occur (TP=0 due to resampling only strip 14) are now filtered out, eliminating artificial zero-inflation in F1 lower bounds.

## 1. Primary Benchmark (Full Spatial Test Set: N=338, Prevalence=29.9%)

| Model Architecture | Seed | Best Ep | Val AUC | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test F1 (Fixed 95% CI) | Orig F1 CI | Valid Rounds | Tuned Th |
|---|---|---|---|---|---|---|---|---|---|
"""

name_map = {
    'CustomCNN': '**CustomCNN**',
    'ResNet_scratch': '**ResNet_scratch**',
    'ResNet_pretrained': '**ResNet_pretrained (Original)**',
    'ResNet_pretrained_v2a': '**ResNet_pretrained_v2a (Fixed Freeze + Native Conv1)**',
    'ResNet_pretrained_v2b': '**ResNet_pretrained_v2b (Differential LR: 1e-4 / 1e-3)**',
}

for r in new_primary_records:
    dname = name_map.get(r['model'], f"**{r['model']}**")
    md += f"| {dname} | {r['seed']} | {r['best_epoch']} | {r['val_auc']:.4f} | {r['roc_auc']:.4f} `{r['roc_auc_ci']}` | {r['pr_auc']:.4f} `{r['pr_auc_ci']}` | **{r['f1']:.4f}** `{r['f1_ci']}` | `{r['f1_ci_orig']}` | {r['n_valid_boot']}/1000 | {r['tuned_th']:.2f} |\n"

md += """
---

## 2. Ambiguity-Removed Benchmark (Clean Test Set: N=205, Prevalence=49.3%)

| Model Architecture | Seed | Best Ep | Val AUC | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test F1 (Fixed 95% CI) | Orig F1 CI | Valid Rounds | Tuned Th |
|---|---|---|---|---|---|---|---|---|---|
"""

for r in new_clean_records:
    dname = name_map.get(r['model'], f"**{r['model']}**")
    md += f"| {dname} | {r['seed']} | {r['best_epoch']} | {r['val_auc']:.4f} | {r['roc_auc']:.4f} `{r['roc_auc_ci']}` | {r['pr_auc']:.4f} `{r['pr_auc_ci']}` | **{r['f1']:.4f}** `{r['f1_ci']}` | `{r['f1_ci_orig']}` | {r['n_valid_boot']}/1000 | {r['tuned_th']:.2f} |\n"

# Summary mean +- std table
md += """
---

## 3. Summary Benchmark: Mean ± Std Across Seeds (Ranked by Val ROC-AUC)

| Model Architecture | Trainable Params | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|---|---|
"""

stats = {}
for m in set(r['model'] for r in new_primary_records):
    subset = [r for r in new_primary_records if r['model'] == m]
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

ranked_models = sorted(stats.keys(), key=lambda m: stats[m]['val_m'], reverse=True)
for m in ranked_models:
    s = stats[m]
    dname = name_map.get(m, f"**{m}**")
    md += f"| {dname} | {s['params']:,} | **{s['val_m']:.4f} ± {s['val_s']:.4f}** | {s['roc_m']:.4f} ± {s['roc_s']:.4f} | {s['pr_m']:.4f} ± {s['pr_s']:.4f} | {s['f1_m']:.4f} ± {s['f1_s']:.4f} |\n"

out_path = os.path.join("reports", "real", "results_real_ci_fixed.md")
with open(out_path, "w", encoding="utf-8") as f:
    f.write(md)

print(f"\nWrote results to {out_path} successfully.")
