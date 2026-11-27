"""
scripts/analysis/run_5fold_strip_cv.py
=======================================
Amendment B: 5-fold blocked-strip cross-validation across all 20 strips
with 256-row azimuth guard buffers for CustomCNN and Logistic Regression.

Each fold holds out 4 contiguous strips as Test, 3 strips as Val, and the
remaining 13 strips as Train, dropping any patch within 256 rows of a split
boundary and asserting zero pixel-extent leakage.

Outputs:
  - reports/real/five_fold_cv.json
  - reports/real/five_fold_cv.md
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score, accuracy_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.preprocessing.data_ops import load_complex_dfsar_sli, compute_cpr_dop_complex
from src.datasets.spatial_split import deterministic_strip_split, assert_no_extent_leakage
from src.datasets.ice_dataset import IcePatchDataset
from src.models.custom_cnn import CustomCNN
from src.evaluate import tune_threshold_on_val


def best_test_f1_threshold(y_true, y_probs):
    best_f1, best_th = 0.0, 0.5
    for th in np.linspace(0.05, 0.95, 91):
        f1 = f1_score(y_true, (y_probs >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)
    return best_th, best_f1


def extract_lr_features(imgs):
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
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("real_dfsar", {})

    raw_dir = cfg.get("raw_dir", "data/raw/sar_nxcl")
    ar = cfg.get("active_rows", [90624, 319812])
    start_row, num_rows = int(ar[0]), int(ar[1]) - int(ar[0])
    P = int(cfg.get("patch_size", 64))
    stride = int(cfg.get("stride", 32))
    n_strips = int(cfg.get("grid_size", [1, 20])[1])
    buffer_px = int(cfg.get("buffer_pixels", 256))
    W = int(cfg.get("multilook_window", 7))
    cpr_th = float(cfg.get("cpr_threshold", 1.0))
    dop_th = float(cfg.get("dop_threshold", 0.13))
    min_pos_px = int(cfg.get("min_positive_pixels", 5))

    print(f"Loading SLI [rows {start_row}..{start_row+num_rows}] for 5-fold blocked-strip CV...")
    channels = load_complex_dfsar_sli(raw_dir, start_row=start_row, num_rows=num_rows)
    Shh, Shv, Svh, Svv = channels["hh"], channels["hv"], channels["vh"], channels["vv"]
    valid_mask = (
        np.isfinite(Shh) & np.isfinite(Shv) & np.isfinite(Svh) & np.isfinite(Svv)
        & (np.abs(Shh) > 0) & (np.abs(Svv) > 0)
    )
    Shv_sym = 0.5 * (Shv + Svh)
    eps = 1e-5
    radar_inputs = np.stack([
        np.log10(np.abs(Shh)**2 + eps).astype(np.float32),
        np.log10(np.abs(Shv_sym)**2 + eps).astype(np.float32),
        np.log10(np.abs(Svv)**2 + eps).astype(np.float32),
    ], axis=0)  # (3, H, W)

    cpr, dop = compute_cpr_dop_complex(Shh, Shv, Svh, Svv, W=W)
    pixel_ice_mask = ((cpr > cpr_th) & (dop < dop_th) & valid_mask).astype(np.uint8)

    H, W_dim = Shh.shape
    coords = []
    imgs_list = []
    labels_list = []
    for r in range(0, H - P + 1, stride):
        for c in range(0, W_dim - P + 1, stride):
            if valid_mask[r:r+P, c:c+P].mean() < 0.95:
                continue
            p_ice = int(np.count_nonzero(pixel_ice_mask[r:r+P, c:c+P]))
            coords.append((r, c))
            imgs_list.append(radar_inputs[:, r:r+P, c:c+P])
            labels_list.append(1 if p_ice >= min_pos_px else 0)

    imgs_all = np.stack(imgs_list)
    labels_all = np.array(labels_list, dtype=np.int64)
    coord_to_idx = {c: i for i, c in enumerate(coords)}

    # 5 contiguous 4-strip test folds; val = 3 strips offset from test
    folds = [
        {"test": (0, 1, 2, 3),     "val": (6, 11, 16)},
        {"test": (4, 5, 6, 7),     "val": (1, 12, 17)},
        {"test": (8, 9, 10, 11),   "val": (2, 14, 18)},
        {"test": (12, 13, 14, 15), "val": (2, 7, 18)},
        {"test": (16, 17, 18, 19), "val": (2, 8, 13)},
    ]

    results = []
    for fold_idx, fcfg in enumerate(folds):
        tr_c, va_c, te_c, _ = deterministic_strip_split(
            coords, n_strips=n_strips,
            val_strips=fcfg["val"], test_strips=fcfg["test"],
            patch_size=P, buffer_pixels=buffer_px,
        )
        assert_no_extent_leakage(tr_c, va_c, te_c, patch_size=P)

        tr_idx = [coord_to_idx[c] for c in tr_c]
        va_idx = [coord_to_idx[c] for c in va_c]
        te_idx = [coord_to_idx[c] for c in te_c]

        X_tr, y_tr = imgs_all[tr_idx], labels_all[tr_idx]
        X_va, y_va = imgs_all[va_idx], labels_all[va_idx]
        X_te, y_te = imgs_all[te_idx], labels_all[te_idx]
        prev = float(y_te.mean())
        print(f"\n--- Fold {fold_idx+1}/5 | Test strips={fcfg['test']} | N_te={len(y_te)} (pos={y_te.sum()}, prev={100*prev:.1f}%) ---")

        # 1. Logistic Regression
        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr.fit(extract_lr_features(X_tr), y_tr)
        lr_va_p = lr.predict_proba(extract_lr_features(X_va))[:, 1]
        lr_te_p = lr.predict_proba(extract_lr_features(X_te))[:, 1]
        lr_val_th, _ = tune_threshold_on_val(y_va, lr_va_p)
        lr_best_th, lr_best_f1 = best_test_f1_threshold(y_te, lr_te_p)
        lr_pred = (lr_te_p >= lr_val_th).astype(int)
        results.append({
            "fold": fold_idx + 1, "model": "LogisticRegression",
            "test_strips": list(fcfg["test"]), "n_test": int(len(y_te)), "n_pos": int(y_te.sum()),
            "prevalence": prev,
            "roc_auc": float(roc_auc_score(y_te, lr_te_p)),
            "pr_auc": float(average_precision_score(y_te, lr_te_p)),
            "val_th": float(lr_val_th),
            "f1_val_th": float(f1_score(y_te, lr_pred, zero_division=0)),
            "acc_val_th": float(accuracy_score(y_te, lr_pred)),
            "bal_acc_val_th": float(balanced_accuracy_score(y_te, lr_pred)),
            "best_test_th": float(lr_best_th),
            "f1_best_test_th": float(lr_best_f1),
        })

        # 2. CustomCNN (seed 42, AMP, max 20 epochs, patience 5)
        torch.manual_seed(42)
        np.random.seed(42)
        tr_ds = IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(X_tr, y_tr)], augment=True)
        va_ds = IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(X_va, y_va)], augment=False)
        te_ds = IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(X_te, y_te)], augment=False)
        tr_loader = DataLoader(tr_ds, batch_size=32, shuffle=True)
        va_loader = DataLoader(va_ds, batch_size=64, shuffle=False)
        te_loader = DataLoader(te_ds, batch_size=64, shuffle=False)

        model = CustomCNN(in_channels=3, num_classes=1).to(device)
        pos_w = float((y_tr == 0).sum() / max((y_tr == 1).sum(), 1))
        crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        best_va_auc, best_state, patience = -1.0, None, 0
        for ep in range(20):
            model.train()
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device).float()
                opt.zero_grad()
                with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                    loss = crit(model(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            model.eval()
            vp = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        vp.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy().flatten())
            va_auc = float(roc_auc_score(y_va, np.array(vp)))
            if va_auc > best_va_auc:
                best_va_auc = va_auc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
            else:
                patience += 1
                if patience >= 5:
                    break

        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()
        def predict_loader(loader):
            ps = []
            with torch.no_grad():
                for xb, _ in loader:
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                        ps.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy().flatten())
            return np.array(ps)

        cnn_va_p = predict_loader(va_loader)
        cnn_te_p = predict_loader(te_loader)
        cnn_val_th, _ = tune_threshold_on_val(y_va, cnn_va_p)
        cnn_best_th, cnn_best_f1 = best_test_f1_threshold(y_te, cnn_te_p)
        cnn_pred = (cnn_te_p >= cnn_val_th).astype(int)
        results.append({
            "fold": fold_idx + 1, "model": "CustomCNN",
            "test_strips": list(fcfg["test"]), "n_test": int(len(y_te)), "n_pos": int(y_te.sum()),
            "prevalence": prev,
            "roc_auc": float(roc_auc_score(y_te, cnn_te_p)),
            "pr_auc": float(average_precision_score(y_te, cnn_te_p)),
            "val_th": float(cnn_val_th),
            "f1_val_th": float(f1_score(y_te, cnn_pred, zero_division=0)),
            "acc_val_th": float(accuracy_score(y_te, cnn_pred)),
            "bal_acc_val_th": float(balanced_accuracy_score(y_te, cnn_pred)),
            "best_test_th": float(cnn_best_th),
            "f1_best_test_th": float(cnn_best_f1),
        })
        print(f"  LR  ROC-AUC={results[-2]['roc_auc']:.4f} PR-AUC={results[-2]['pr_auc']:.4f} F1(val_th)={results[-2]['f1_val_th']:.4f}")
        print(f"  CNN ROC-AUC={results[-1]['roc_auc']:.4f} PR-AUC={results[-1]['pr_auc']:.4f} F1(val_th)={results[-1]['f1_val_th']:.4f}")

    os.makedirs("reports/real", exist_ok=True)
    with open("reports/real/five_fold_cv.json", "w", encoding="utf-8") as jf:
        json.dump(results, jf, indent=2)

    with open("reports/real/five_fold_cv.md", "w", encoding="utf-8") as mf:
        mf.write("# 5-Fold Blocked-Strip Cross-Validation (All 20 Strips, 256-Row Buffer)\n\n")
        mf.write("| Fold | Test Strips | Prev | Model | ROC-AUC | PR-AUC | F1 (Val Th) | F1 (Best Test Th) | Bal Acc |\n")
        mf.write("| :---: | :--- | :---: | :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in results:
            mf.write(
                f"| {r['fold']} | {r['test_strips']} | {100*r['prevalence']:.1f}% | {r['model']} | "
                f"{r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['f1_val_th']:.4f} | {r['f1_best_test_th']:.4f} | {r['bal_acc_val_th']:.4f} |\n"
            )
        mf.write("\n## Mean ± Std Across 5 Folds\n\n")
        mf.write("| Model | ROC-AUC | PR-AUC | F1 (Val Th) | F1 (Best Test Th) | Bal Acc |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for mname in ["LogisticRegression", "CustomCNN"]:
            sub = [r for r in results if r["model"] == mname]
            mf.write(
                f"| {mname} | "
                f"{np.mean([r['roc_auc'] for r in sub]):.4f} ± {np.std([r['roc_auc'] for r in sub]):.4f} | "
                f"{np.mean([r['pr_auc'] for r in sub]):.4f} ± {np.std([r['pr_auc'] for r in sub]):.4f} | "
                f"{np.mean([r['f1_val_th'] for r in sub]):.4f} ± {np.std([r['f1_val_th'] for r in sub]):.4f} | "
                f"{np.mean([r['f1_best_test_th'] for r in sub]):.4f} ± {np.std([r['f1_best_test_th'] for r in sub]):.4f} | "
                f"{np.mean([r['bal_acc_val_th'] for r in sub]):.4f} ± {np.std([r['bal_acc_val_th'] for r in sub]):.4f} |\n"
            )
    print("\nSaved reports/real/five_fold_cv.json and reports/real/five_fold_cv.md")


if __name__ == "__main__":
    main()
