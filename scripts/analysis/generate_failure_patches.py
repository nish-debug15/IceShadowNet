import os, sys, json, torch
import numpy as np
import matplotlib.pyplot as plt

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)

from src.models.custom_cnn import CustomCNN
from src.datasets.ice_dataset import IcePatchDataset
from src.evaluate import get_strip_ids_from_coords, evaluate_model
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Identify CustomCNN seed with highest Val ROC-AUC
with open('reports/real/training_summary.json') as f:
    orig = json.load(f)['primary']
cnn_runs = [r for r in orig if r['model'] == 'CustomCNN']
for r in cnn_runs:
    print(f"Seed {r['seed']}: Val AUC={r['val_auc']:.4f}, Test AUC={r['roc_auc']:.4f}, Th={r['tuned_th']}")

best_run = max(cnn_runs, key=lambda r: r['val_auc'])
best_seed = best_run['seed']
best_th = best_run['tuned_th']
print(f"\nSelected Model: CustomCNN seed {best_seed} (Val AUC={best_run['val_auc']:.4f}, Threshold={best_th})")

# 2. Load test data
PROCESSED_DIR = "data/processed_real"
test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
X_test = test_data["images"]
y_test = test_data["labels"]
coords_test = test_data["coords"]
strips_test = get_strip_ids_from_coords(coords_test)
cpr_means = test_data.get("cpr_means", None)
dop_means = test_data.get("dop_means", None)

patches = list(zip(X_test, y_test.tolist(), [tuple(c) for c in coords_test]))
test_ds = IcePatchDataset(patches, augment=False, normalize=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)

# 3. Load model and predict
ckpt_path = os.path.join("models_real", f"CustomCNN_seed{best_seed}.pth")
model = CustomCNN(3, 1).to(DEVICE)
model.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))

eval_res = evaluate_model(model, test_loader, DEVICE, threshold=best_th)
probs = eval_res['probs']
labels = eval_res['labels']
preds = (probs >= best_th).astype(int)

# Verify assertions
np.testing.assert_allclose(labels, y_test)
print(f"Test Accuracy: {eval_res['accuracy']:.4f}, F1: {eval_res['f1']:.4f}, ROC-AUC: {eval_res['roc_auc']:.4f}")

# 4. Fixed Selection Rule
# FP: y=0 and pred=1 (i.e. probs >= best_th). Top-2 highest confidence (highest probs), tie-break by index
fp_indices = np.where((labels == 0) & (preds == 1))[0]
# Sort descending by probs, then ascending by index
sorted_fps = sorted(fp_indices, key=lambda i: (-probs[i], i))
top2_fps = sorted_fps[:2]

# FN: y=1 and pred=0 (i.e. probs < best_th). Top-2 highest confidence (lowest probs), tie-break by index
fn_indices = np.where((labels == 1) & (preds == 0))[0]
# Sort ascending by probs, then ascending by index
sorted_fns = sorted(fn_indices, key=lambda i: (probs[i], i))
top2_fns = sorted_fns[:2]

print("\n--- SELECTED FALSE POSITIVES ---")
for rank, idx in enumerate(top2_fps, 1):
    print(f"FP #{rank}: Patch {idx} | Prob={probs[idx]:.4f} (Th={best_th}) | Label={labels[idx]} | Strip {strips_test[idx]} | Coords={coords_test[idx]}")

print("\n--- SELECTED FALSE NEGATIVES ---")
for rank, idx in enumerate(top2_fns, 1):
    print(f"FN #{rank}: Patch {idx} | Prob={probs[idx]:.4f} (Th={best_th}) | Label={labels[idx]} | Strip {strips_test[idx]} | Coords={coords_test[idx]}")

# 5. Create figures directory and plot
figures_dir = os.path.join("reports", "real", "figures")
os.makedirs(figures_dir, exist_ok=True)

ch_names = ["HH Log-Intensity", "HV Log-Intensity", "VV Log-Intensity"]

# Plot False Positives
for rank, idx in enumerate(top2_fps, 1):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    img = X_test[idx]
    for c in range(3):
        im = axes[c].imshow(img[c], cmap="viridis")
        axes[c].set_title(ch_names[c], fontsize=10)
        axes[c].axis("off")
        plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)

    cpr_s = f", CPR={cpr_means[idx]:.2f}" if cpr_means is not None else ""
    dop_s = f", DOP={dop_means[idx]:.2f}" if dop_means is not None else ""
    fig.suptitle(
        f"False Positive #{rank} [Patch {idx} | Strip {strips_test[idx]} | True Label: 0 | Score: {probs[idx]:.4f} | Th: {best_th}{cpr_s}{dop_s}]",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    out_path = os.path.join(figures_dir, f"failure_fp{rank}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path} ({size_kb:.1f} KB)")

# Plot False Negatives
for rank, idx in enumerate(top2_fns, 1):
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    img = X_test[idx]
    for c in range(3):
        im = axes[c].imshow(img[c], cmap="viridis")
        axes[c].set_title(ch_names[c], fontsize=10)
        axes[c].axis("off")
        plt.colorbar(im, ax=axes[c], fraction=0.046, pad=0.04)

    cpr_s = f", CPR={cpr_means[idx]:.2f}" if cpr_means is not None else ""
    dop_s = f", DOP={dop_means[idx]:.2f}" if dop_means is not None else ""
    fig.suptitle(
        f"False Negative #{rank} [Patch {idx} | Strip {strips_test[idx]} | True Label: 1 | Score: {probs[idx]:.4f} | Th: {best_th}{cpr_s}{dop_s}]",
        fontsize=10, fontweight="bold"
    )
    plt.tight_layout()
    out_path = os.path.join(figures_dir, f"failure_fn{rank}.png")
    plt.savefig(out_path, dpi=120)
    plt.close()
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Saved {out_path} ({size_kb:.1f} KB)")
