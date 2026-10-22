import os, sys, json, torch
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
os.chdir(REPO_ROOT)
from src.models.custom_cnn import CustomCNN
from src.models.resnet import IceResNet
from src.datasets.ice_dataset import IcePatchDataset
from src.evaluate import get_strip_ids_from_coords, evaluate_model
from torch.utils.data import DataLoader

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load test data
PROCESSED_DIR = "data/processed_real"
test_data = np.load(os.path.join(PROCESSED_DIR, "test.npz"))
X, Y, C = test_data["images"], test_data["labels"], test_data["coords"]
patches = list(zip(X, Y.tolist(), [tuple(c) for c in C]))
test_ds = IcePatchDataset(patches, augment=False, normalize=True)
test_loader = DataLoader(test_ds, batch_size=16, shuffle=False, num_workers=0)
test_coords = np.array([p[2] for p in patches])
test_strips = get_strip_ids_from_coords(test_coords)

# Collect all 15 runs from training_summary json files
with open('reports/real/training_summary.json') as f:
    orig = json.load(f)['primary']
with open('reports/real/training_summary_v2a.json') as f:
    v2a = json.load(f)['primary']
with open('reports/real/training_summary_v2b.json') as f:
    v2b = json.load(f)['primary']

all_runs = orig + v2a + v2b

print(f"Total runs: {len(all_runs)}")

for r in all_runs:
    model_name = r['model']
    seed = r['seed']
    th = r['tuned_th']
    ckpt_name = f"{model_name}_seed{seed}.pth"
    ckpt_path = os.path.join("models_real", ckpt_name)

    # instantiate model
    if 'CustomCNN' in model_name:
        m = CustomCNN(3, 1).to(DEVICE)
    elif 'scratch' in model_name:
        m = IceResNet(3, 1, pretrained=False).to(DEVICE)
    else:
        m = IceResNet(3, 1, pretrained=True).to(DEVICE)

    m.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    eval_res = evaluate_model(m, test_loader, DEVICE, threshold=th)
    probs = eval_res['probs']
    labels = eval_res['labels']

    # Test assert point estimates
    point_f1 = eval_res['f1']
    point_roc = eval_res['roc_auc']
    point_pr = eval_res['pr_auc']

    diff_f1 = abs(point_f1 - r['f1'])
    diff_roc = abs(point_roc - r['roc_auc'])
    diff_pr = abs(point_pr - r['pr_auc'])

    # Bootstrap investigation
    unique_strips = np.unique(test_strips)
    strip_map = {s: np.where(test_strips == s)[0] for s in unique_strips}
    rng = np.random.default_rng(seed)

    zero_pos_count = 0
    zero_pred_pos_count = 0
    zero_f1_count = 0
    resamples_evaluated = 0

    for _ in range(1000):
        sampled_strips = rng.choice(unique_strips, size=len(unique_strips), replace=True)
        sampled_indices = np.concatenate([strip_map[s] for s in sampled_strips])
        y_boot = labels[sampled_indices]
        p_boot = probs[sampled_indices]

        n_pos = np.sum(y_boot == 1)
        pred_boot = (p_boot >= th).astype(int)
        n_pred_pos = np.sum(pred_boot == 1)

        if n_pos == 0:
            zero_pos_count += 1
        if n_pred_pos == 0:
            zero_pred_pos_count += 1

        if len(np.unique(y_boot)) < 2:
            continue

        resamples_evaluated += 1
        f1_val = f1_score(y_boot, pred_boot, zero_division=0)
        if f1_val == 0.0:
            zero_f1_count += 1

    print(f"{model_name:22} s={seed:4d} | th={th:.2f} | point diffs: f1={diff_f1:.6f}, roc={diff_roc:.6f}, pr={diff_pr:.6f} | zero_gt_pos={zero_pos_count:3d}, zero_pred_pos={zero_pred_pos_count:3d}, eval_rounds={resamples_evaluated:4d}, zero_f1={zero_f1_count:3d}")
