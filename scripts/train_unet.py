"""
scripts/train_unet.py
=====================
Block 2: Pixel-level lunar SAR anomaly segmentation using lightweight MC-Dropout
IceUNet on the frozen spatial split [90624, 319812].

Protocol:
  - Inputs: strictly raw log-intensities [HH, HV, VV] (anti-circularity).
  - Targets: pixel-level (CPR > 1.0 & DOP < 0.13) masks on 7x7 multilooked grid.
  - Loss: 0.5 * BCEWithLogitsLoss(pos_weight) + 0.5 * SoftDiceLoss.
  - 3 seeds (42, 123, 2024), AMP, early stopping on Val Pixel PR-AUC.
  - Mandatory pixel baselines: HV/HH ratio threshold + per-pixel Logistic Regression.
  - Evaluates pooled & per-strip Dice, IoU, Pixel PR-AUC, Pixel ROC-AUC with
    strip-block bootstrap 95% CIs, plus val-tuned vs best-possible test thresholds.
  - Patch-level aggregation (max_prob, mean_prob, pred_px_count>=5) compared against
    CustomCNN, ResNet_scratch, ResNet_SSL, and LogisticRegression on identical split
    (full test + unambiguous subset ablation).
  - Saves visual overlays to reports/real/figures/unet_overlays.png.
"""

import os
import sys
import json
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, balanced_accuracy_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from src.models.unet import IceUNet
from src.datasets.ice_dataset import get_ambiguity_mask
from src.evaluate import tune_threshold_on_val

PROCESSED_REAL = "data/processed_real"
MODELS_DIR = "models_real"
REPORTS_DIR = "reports/real"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)


class SegPatchDataset(Dataset):
    def __init__(self, images: np.ndarray, masks: np.ndarray, augment: bool = False):
        self.images = images
        self.masks = masks
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx].copy()   # (3, 64, 64)
        msk = self.masks[idx].copy()    # (64, 64)

        if self.augment:
            k = np.random.randint(0, 4)
            if k > 0:
                img = np.rot90(img, k, axes=(1, 2)).copy()
                msk = np.rot90(msk, k, axes=(0, 1)).copy()
            if np.random.rand() < 0.5:
                img = img[:, :, ::-1].copy()
                msk = msk[:, ::-1].copy()
            if np.random.rand() < 0.5:
                img = img[:, ::-1, :].copy()
                msk = msk[::-1, :].copy()

        for c in range(img.shape[0]):
            mu = img[c].mean()
            std = img[c].std()
            img[c] = (img[c] - mu) / (std + 1e-8)

        return torch.from_numpy(img).float(), torch.from_numpy(msk[None, ...]).float()


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        num = 2.0 * (probs * targets).sum(dim=(1, 2, 3)) + self.smooth
        den = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + self.smooth
        return (1.0 - num / den).mean()


def pixel_metrics(y_true_flat: np.ndarray, probs_flat: np.ndarray, th: float) -> dict:
    pred = (probs_flat >= th).astype(np.uint8)
    tp = int(((pred == 1) & (y_true_flat == 1)).sum())
    fp = int(((pred == 1) & (y_true_flat == 0)).sum())
    fn = int(((pred == 0) & (y_true_flat == 1)).sum())
    dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)
    iou = tp / max(tp + fp + fn, 1)
    pr_auc = float(average_precision_score(y_true_flat, probs_flat)) if y_true_flat.sum() > 0 else 0.0
    roc_auc = float(roc_auc_score(y_true_flat, probs_flat)) if len(np.unique(y_true_flat)) > 1 else 0.5
    return {
        "dice": float(dice),
        "iou": float(iou),
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "tp": tp, "fp": fp, "fn": fn,
        "pos_px": int(y_true_flat.sum()),
        "total_px": int(y_true_flat.size),
        "prevalence": float(y_true_flat.mean()),
    }


def tune_pixel_threshold(y_true_flat: np.ndarray, probs_flat: np.ndarray, grid=None) -> tuple:
    if grid is None:
        grid = np.linspace(0.05, 0.95, 19)
    best_th, best_dice = 0.5, -1.0
    for th in grid:
        pred = probs_flat >= th
        tp = int((pred & (y_true_flat == 1)).sum())
        fp = int((pred & (y_true_flat == 0)).sum())
        fn = int((~pred & (y_true_flat == 1)).sum())
        dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)
        if dice > best_dice:
            best_dice, best_th = float(dice), float(th)
    return best_th, best_dice


def strip_bootstrap_pixel_ci(masks: np.ndarray, probs: np.ndarray, strip_ids: np.ndarray, th: float, n_boot: int = 300, seed: int = 42):
    rng = np.random.default_rng(seed)
    unique_strips = np.unique(strip_ids)
    strip_idx = {s: np.where(strip_ids == s)[0] for s in unique_strips}
    cache = {}
    dices, ious, prs, rocs = [], [], [], []
    for _ in range(n_boot):
        sampled = tuple(sorted(rng.choice(unique_strips, size=len(unique_strips), replace=True).tolist()))
        if sampled not in cache:
            idx = np.concatenate([strip_idx[s] for s in sampled])
            yt = masks[idx].ravel()
            yp = probs[idx].ravel()
            if yt.sum() == 0:
                cache[sampled] = None
            else:
                cache[sampled] = pixel_metrics(yt, yp, th)
        m = cache[sampled]
        if m is None:
            continue
        dices.append(m["dice"])
        ious.append(m["iou"])
        prs.append(m["pr_auc"])
        rocs.append(m["roc_auc"])

    def ci(arr):
        return [float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))] if arr else [0.0, 0.0]

    return {"dice_ci": ci(dices), "iou_ci": ci(ious), "pr_auc_ci": ci(prs), "roc_auc_ci": ci(rocs)}


def patch_eval(y_true: np.ndarray, scores: np.ndarray, val_th: float) -> dict:
    pred_v = (scores >= val_th).astype(int)
    best_th, best_f1 = 0.5, 0.0
    for th in np.linspace(float(scores.min()), float(scores.max()), 50):
        f1 = f1_score(y_true, (scores >= th).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_th = float(f1), float(th)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.5,
        "pr_auc": float(average_precision_score(y_true, scores)) if y_true.sum() > 0 else 0.0,
        "val_th": float(val_th),
        "f1_val_th": float(f1_score(y_true, pred_v, zero_division=0)),
        "acc_val_th": float(accuracy_score(y_true, pred_v)),
        "bal_acc_val_th": float(balanced_accuracy_score(y_true, pred_v)) if len(np.unique(y_true)) > 1 else 0.5,
        "best_test_th": float(best_th),
        "f1_best_test_th": float(best_f1),
    }


def save_overlays(images: np.ndarray, masks: np.ndarray, probs: np.ndarray, labels: np.ndarray, pos_counts: np.ndarray, th: float):
    # Pick 2 TP, 2 sub-threshold (1-4 px) with high prob, 2 FN
    tp_idx = np.where((labels == 1) & (probs.max(axis=(1, 2)) >= th))[0]
    sub_idx = np.where((pos_counts >= 1) & (pos_counts <= 4))[0]
    fn_idx = np.where((labels == 1) & (probs.max(axis=(1, 2)) < th))[0]

    chosen = []
    if len(tp_idx) > 0:
        tp_sorted = tp_idx[np.argsort(-pos_counts[tp_idx])]
        chosen.extend([(int(tp_sorted[0]), "TP (High Ice)"), (int(tp_sorted[min(5, len(tp_sorted)-1)]), "TP")])
    if len(sub_idx) > 0:
        sub_sorted = sub_idx[np.argsort(-probs[sub_idx].max(axis=(1, 2)))]
        chosen.extend([(int(sub_sorted[0]), "Sub-Threshold (1-4px)")])
    if len(fn_idx) > 0:
        chosen.extend([(int(fn_idx[0]), "FN (Missed)")])

    if not chosen:
        return

    fig, axes = plt.subplots(len(chosen), 4, figsize=(12, 3.0 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[None, :]

    for row_i, (idx, tag) in enumerate(chosen):
        img = images[idx]  # (3, 64, 64)
        # Normalize each channel to [0, 1] for RGB composite [HH, HV, VV]
        rgb = np.zeros((64, 64, 3), dtype=np.float32)
        for c in range(3):
            lo, hi = np.percentile(img[c], 2), np.percentile(img[c], 98)
            rgb[:, :, c] = np.clip((img[c] - lo) / max(hi - lo, 1e-6), 0, 1)

        axes[row_i, 0].imshow(rgb)
        axes[row_i, 0].set_title(f"Patch #{idx} [{tag}]\nInput [HH, HV, VV]", fontsize=9)
        axes[row_i, 1].imshow(masks[idx], cmap="gray", vmin=0, vmax=1)
        axes[row_i, 1].set_title(f"GT Mask (pos_px={pos_counts[idx]})", fontsize=9)
        axes[row_i, 2].imshow(probs[idx], cmap="magma", vmin=0, vmax=1)
        axes[row_i, 2].set_title(f"U-Net Prob (max={probs[idx].max():.2f})", fontsize=9)
        axes[row_i, 3].imshow((probs[idx] >= th).astype(float), cmap="cyan" if False else "viridis", vmin=0, vmax=1)
        axes[row_i, 3].set_title(f"Binary Pred (th={th:.2f})", fontsize=9)
        for c_i in range(4):
            axes[row_i, c_i].axis("off")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "unet_overlays.png")
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved visual overlays -> {out_path}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", f"CUDA required, got {device}"

    d_tr = np.load(os.path.join(PROCESSED_REAL, "train.npz"))
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_tr, M_tr, y_tr = d_tr["images"], d_tr["masks"], d_tr["labels"].astype(int)
    X_va, M_va, y_va = d_va["images"], d_va["masks"], d_va["labels"].astype(int)
    X_te, M_te, y_te = d_te["images"], d_te["masks"], d_te["labels"].astype(int)
    te_strips = d_te["strip_ids"].astype(int)
    te_pos_counts = d_te["pos_px_counts"].astype(int)
    clean_mask = ~get_ambiguity_mask(d_te["ice_fracs"], min_px=1, max_px=4, patch_size=64)

    print(f"Loaded frozen split: Train={len(X_tr)}, Val={len(X_va)}, Test={len(X_te)}")
    print(f"Pixel prevalence: Train={100*M_tr.mean():.4f}%, Val={100*M_va.mean():.4f}%, Test={100*M_te.mean():.4f}%")

    # ── 1. Mandatory Pixel Baselines ───────────────────────────────────────────
    print("\n--- Running Mandatory Pixel Baselines ---")
    # Baseline A: HV/HH log-ratio (I_hv - I_hh)
    hv_hh_va = (X_va[:, 1] - X_va[:, 0]).ravel()
    hv_hh_te = (X_te[:, 1] - X_te[:, 0]).ravel()
    # Map to [0, 1] via sigmoid-like min-max from val for AUC/thresholding
    lo, hi = np.percentile(hv_hh_va, 1), np.percentile(hv_hh_va, 99.9)
    hv_hh_va_s = np.clip((hv_hh_va - lo) / max(hi - lo, 1e-6), 0, 1)
    hv_hh_te_s = np.clip((hv_hh_te - lo) / max(hi - lo, 1e-6), 0, 1)
    hvhh_val_th, _ = tune_pixel_threshold(M_va.ravel(), hv_hh_va_s)
    hvhh_best_th, _ = tune_pixel_threshold(M_te.ravel(), hv_hh_te_s)
    hvhh_res = {
        "val_th": hvhh_val_th,
        "pooled_val_th": pixel_metrics(M_te.ravel(), hv_hh_te_s, hvhh_val_th),
        "pooled_best_th": pixel_metrics(M_te.ravel(), hv_hh_te_s, hvhh_best_th),
        "best_test_th": hvhh_best_th,
        "cis": strip_bootstrap_pixel_ci(M_te, hv_hh_te_s.reshape(M_te.shape), te_strips, hvhh_val_th, n_boot=200),
    }
    print(f"  HV/HH Ratio: PR-AUC={hvhh_res['pooled_val_th']['pr_auc']:.4f} | ROC-AUC={hvhh_res['pooled_val_th']['roc_auc']:.4f} | Dice(val_th)={hvhh_res['pooled_val_th']['dice']:.4f}")

    # Baseline B: Per-pixel Logistic Regression on [HH, HV, VV]
    rng = np.random.default_rng(42)
    X_tr_px = np.transpose(X_tr, (0, 2, 3, 1)).reshape(-1, 3)
    y_tr_px = M_tr.ravel()
    pos_px_idx = np.where(y_tr_px == 1)[0]
    neg_px_idx = rng.choice(np.where(y_tr_px == 0)[0], size=min(150000, int((y_tr_px == 0).sum())), replace=False)
    fit_idx = np.concatenate([pos_px_idx, neg_px_idx])
    px_lr = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42)
    px_lr.fit(X_tr_px[fit_idx], y_tr_px[fit_idx])

    px_lr_va = px_lr.predict_proba(np.transpose(X_va, (0, 2, 3, 1)).reshape(-1, 3))[:, 1]
    px_lr_te = px_lr.predict_proba(np.transpose(X_te, (0, 2, 3, 1)).reshape(-1, 3))[:, 1]
    pxlr_val_th, _ = tune_pixel_threshold(M_va.ravel(), px_lr_va)
    pxlr_best_th, _ = tune_pixel_threshold(M_te.ravel(), px_lr_te)
    pxlr_res = {
        "val_th": pxlr_val_th,
        "pooled_val_th": pixel_metrics(M_te.ravel(), px_lr_te, pxlr_val_th),
        "pooled_best_th": pixel_metrics(M_te.ravel(), px_lr_te, pxlr_best_th),
        "best_test_th": pxlr_best_th,
        "cis": strip_bootstrap_pixel_ci(M_te, px_lr_te.reshape(M_te.shape), te_strips, pxlr_val_th, n_boot=200),
    }
    print(f"  Pixel LR:    PR-AUC={pxlr_res['pooled_val_th']['pr_auc']:.4f} | ROC-AUC={pxlr_res['pooled_val_th']['roc_auc']:.4f} | Dice(val_th)={pxlr_res['pooled_val_th']['dice']:.4f}")

    # ── 2. Train IceUNet (3 seeds) ─────────────────────────────────────────────
    seeds = [42, 123, 2024]
    unet_runs = []
    best_seed42_probs_te = None
    best_seed42_th = 0.5

    for seed in seeds:
        print(f"\n--- Training IceUNet | Seed {seed} ---")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        tr_loader = DataLoader(SegPatchDataset(X_tr, M_tr, augment=True), batch_size=32, shuffle=True, pin_memory=True)
        va_loader = DataLoader(SegPatchDataset(X_va, M_va, augment=False), batch_size=64, shuffle=False, pin_memory=True)
        te_loader = DataLoader(SegPatchDataset(X_te, M_te, augment=False), batch_size=64, shuffle=False, pin_memory=True)

        model = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        pos_weight_val = min(float((M_tr == 0).sum() / max((M_tr == 1).sum(), 1)), 50.0)
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
        dice_loss = SoftDiceLoss()
        opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
        scaler = torch.amp.GradScaler("cuda")

        best_val_pr, best_state, patience = -1.0, None, 0
        t0 = time.time()
        for ep in range(1, 26):
            model.train()
            tr_loss = 0.0
            for xb, mb in tr_loader:
                xb, mb = xb.to(device, non_blocking=True), mb.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = 0.5 * bce(logits, mb) + 0.5 * dice_loss(logits, mb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                tr_loss += float(loss.item()) * len(xb)
            sched.step()

            model.eval()
            va_probs = []
            with torch.no_grad():
                for xb, _ in va_loader:
                    with torch.amp.autocast("cuda"):
                        va_probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0])
            va_probs = np.concatenate(va_probs, axis=0)
            va_pr = float(average_precision_score(M_va.ravel(), va_probs.ravel()))
            if va_pr > best_val_pr:
                best_val_pr = va_pr
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience = 0
                best_ep = ep
            else:
                patience += 1
                if ep % 5 == 0:
                    print(f"  Ep {ep:2d}/25 | Train Loss={tr_loss/len(X_tr):.4f} | Val Px PR-AUC={va_pr:.4f} (Best={best_val_pr:.4f} @ Ep {best_ep})")
                if patience >= 5:
                    print(f"  Early stopping at Ep {ep} (Best Val Px PR-AUC={best_val_pr:.4f} @ Ep {best_ep})")
                    break
            if ep % 5 == 0 and patience == 0:
                print(f"  Ep {ep:2d}/25 | Train Loss={tr_loss/len(X_tr):.4f} | Val Px PR-AUC={va_pr:.4f} (Best={best_val_pr:.4f} @ Ep {best_ep})")

        train_time = time.time() - t0
        ckpt_path = os.path.join(MODELS_DIR, f"IceUNet_seed{seed}.pth")
        torch.save(best_state, ckpt_path)
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        model.eval()

        def predict_maps(loader):
            ps = []
            with torch.no_grad():
                for xb, _ in loader:
                    with torch.amp.autocast("cuda"):
                        ps.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0])
            return np.concatenate(ps, axis=0)

        va_maps = predict_maps(va_loader)
        te_maps = predict_maps(te_loader)

        val_px_th, _ = tune_pixel_threshold(M_va.ravel(), va_maps.ravel())
        best_px_th, _ = tune_pixel_threshold(M_te.ravel(), te_maps.ravel())

        pooled_val = pixel_metrics(M_te.ravel(), te_maps.ravel(), val_px_th)
        pooled_best = pixel_metrics(M_te.ravel(), te_maps.ravel(), best_px_th)
        cis = strip_bootstrap_pixel_ci(M_te, te_maps, te_strips, val_px_th, n_boot=300, seed=seed)
        per_strip = {
            int(s): pixel_metrics(M_te[te_strips == s].ravel(), te_maps[te_strips == s].ravel(), val_px_th)
            for s in sorted(np.unique(te_strips))
        }

        # Patch-level aggregations: max_prob, mean_prob, pred_px_count (>=5 at val_px_th)
        agg_results = {}
        for agg_name, va_sc, te_sc, fixed_th in [
            ("max_prob", va_maps.max(axis=(1, 2)), te_maps.max(axis=(1, 2)), None),
            ("mean_prob", va_maps.mean(axis=(1, 2)), te_maps.mean(axis=(1, 2)), None),
            ("pred_px_count_ge5", (va_maps >= val_px_th).sum(axis=(1, 2)).astype(float), (te_maps >= val_px_th).sum(axis=(1, 2)).astype(float), 5.0),
        ]:
            if fixed_th is None:
                v_th, _ = tune_threshold_on_val(y_va, va_sc)
            else:
                v_th = fixed_th
            agg_results[agg_name] = {
                "full": patch_eval(y_te, te_sc, v_th),
                "clean_ablation": patch_eval(y_te[clean_mask], te_sc[clean_mask], v_th),
            }

        print(f"  Seed {seed} ({train_time:.1f}s, {n_params:,} params): Px PR-AUC={pooled_val['pr_auc']:.4f}, "
              f"Dice(val_th={val_px_th:.2f})={pooled_val['dice']:.4f}, Dice(best_th={best_px_th:.2f})={pooled_best['dice']:.4f}, "
              f"Patch MaxProb ROC-AUC={agg_results['max_prob']['full']['roc_auc']:.4f}")

        if seed == 42:
            best_seed42_probs_te = te_maps
            best_seed42_th = val_px_th

        unet_runs.append({
            "seed": seed, "params": n_params, "train_time_s": train_time, "best_val_px_pr_auc": best_val_pr,
            "val_px_th": val_px_th, "best_test_px_th": best_px_th,
            "pooled_val_th": pooled_val, "pooled_best_th": pooled_best,
            "cis": cis, "per_strip": per_strip, "patch_agg": agg_results,
        })

    if best_seed42_probs_te is not None:
        save_overlays(X_te, M_te, best_seed42_probs_te, y_te, te_pos_counts, best_seed42_th)

    # Save JSON + Markdown report
    report = {
        "pixel_prevalence_test": float(M_te.mean()),
        "patch_prevalence_test": float(y_te.mean()),
        "baselines": {"hv_hh_ratio": hvhh_res, "pixel_logreg": pxlr_res},
        "unet_runs": unet_runs,
    }
    with open(os.path.join(REPORTS_DIR, "unet_segmentation_report.json"), "w", encoding="utf-8") as jf:
        json.dump(report, jf, indent=2)

    # Load Block 1 patch comparison for identical-split table
    b1_path = os.path.join(REPORTS_DIR, "block1_benchmark_report.json")
    b1_records = json.load(open(b1_path)) if os.path.isfile(b1_path) else []

    with open(os.path.join(REPORTS_DIR, "unet_segmentation_report.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 2: U-Net Pixel-Level Segmentation Report (Frozen Split `[90624, 319812]`)\n\n")
        mf.write(f"- **Inputs:** Raw `[HH, HV, VV]` log-intensities only (anti-circularity)\n")
        mf.write(f"- **Test Pixel Prevalence (PR-AUC baseline):** `{100*M_te.mean():.4f}%` (`{int(M_te.sum()):,} / {M_te.size:,}` pixels)\n\n")

        mf.write("## 1. Pixel-Level Segmentation Benchmark (Test Set)\n\n")
        mf.write("| Model | Pixel PR-AUC | Pixel ROC-AUC | Dice (Val Th) | IoU (Val Th) | Dice (Best Test Th) | IoU (Best Test Th) |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for bname, bres in [("Baseline: HV/HH Ratio", hvhh_res), ("Baseline: Pixel LogReg", pxlr_res)]:
            pv, pb = bres["pooled_val_th"], bres["pooled_best_th"]
            mf.write(f"| {bname} | {pv['pr_auc']:.4f} | {pv['roc_auc']:.4f} | {pv['dice']:.4f} | {pv['iou']:.4f} | {pb['dice']:.4f} | {pb['iou']:.4f} |\n")

        def umean(path_fn):
            vals = [path_fn(r) for r in unet_runs]
            return f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"

        mf.write(
            f"| **IceUNet (3 seeds)** | {umean(lambda r: r['pooled_val_th']['pr_auc'])} | "
            f"{umean(lambda r: r['pooled_val_th']['roc_auc'])} | "
            f"{umean(lambda r: r['pooled_val_th']['dice'])} | "
            f"{umean(lambda r: r['pooled_val_th']['iou'])} | "
            f"{umean(lambda r: r['pooled_best_th']['dice'])} | "
            f"{umean(lambda r: r['pooled_best_th']['iou'])} |\n"
        )

        r42 = unet_runs[0]
        mf.write("\n### Seed 42 Strip-Block Bootstrap 95% CIs & Per-Strip Breakdown\n\n")
        mf.write(f"- **Pooled Pixel PR-AUC:** `{r42['pooled_val_th']['pr_auc']:.4f}` `[{r42['cis']['pr_auc_ci'][0]:.4f}, {r42['cis']['pr_auc_ci'][1]:.4f}]`\n")
        mf.write(f"- **Pooled Pixel Dice (Val Th={r42['val_px_th']:.2f}):** `{r42['pooled_val_th']['dice']:.4f}` `[{r42['cis']['dice_ci'][0]:.4f}, {r42['cis']['dice_ci'][1]:.4f}]`\n\n")
        mf.write("| Test Strip | Pos Px / Total Px | Pixel PR-AUC | Pixel ROC-AUC | Dice (Val Th) | IoU (Val Th) |\n")
        mf.write("| :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for s in sorted(r42["per_strip"].keys()):
            ps = r42["per_strip"][s]
            mf.write(f"| Strip {s} | {ps['pos_px']} / {ps['total_px']:,} | {ps['pr_auc']:.4f} | {ps['roc_auc']:.4f} | {ps['dice']:.4f} | {ps['iou']:.4f} |\n")

        mf.write("\n## 2. Patch-Level Aggregation vs. Classifiers (Identical Frozen Split)\n\n")
        mf.write("| Model | Full ROC-AUC | Full PR-AUC (16.45%) | Full F1 (Val Th) | Full F1 (Best Test Th) | Clean ROC-AUC (Ablation) | Clean F1 (Ablation) |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for agg_key, label in [("max_prob", "IceUNet (agg: max_prob)"), ("mean_prob", "IceUNet (agg: mean_prob)"), ("pred_px_count_ge5", "IceUNet (agg: px_count>=5)")]:
            f_roc = np.mean([r["patch_agg"][agg_key]["full"]["roc_auc"] for r in unet_runs])
            f_pr = np.mean([r["patch_agg"][agg_key]["full"]["pr_auc"] for r in unet_runs])
            f_f1v = np.mean([r["patch_agg"][agg_key]["full"]["f1_val_th"] for r in unet_runs])
            f_f1b = np.mean([r["patch_agg"][agg_key]["full"]["f1_best_test_th"] for r in unet_runs])
            c_roc = np.mean([r["patch_agg"][agg_key]["clean_ablation"]["roc_auc"] for r in unet_runs])
            c_f1 = np.mean([r["patch_agg"][agg_key]["clean_ablation"]["f1_val_th"] for r in unet_runs])
            mf.write(f"| **{label}** | {f_roc:.4f} | {f_pr:.4f} | {f_f1v:.4f} | {f_f1b:.4f} | {c_roc:.4f} | {c_f1:.4f} |\n")

        for mname in ["CustomCNN", "ResNet_scratch", "ResNet_pretrained", "LogisticRegression"]:
            sub = [r for r in b1_records if r["model"] == mname]
            if not sub:
                continue
            mf.write(
                f"| {mname} | {np.mean([r['pooled']['roc_auc'] for r in sub]):.4f} | "
                f"{np.mean([r['pooled']['pr_auc'] for r in sub]):.4f} | "
                f"{np.mean([r['pooled']['f1_val_th'] for r in sub]):.4f} | "
                f"{np.mean([r['pooled']['f1_best_test_th'] for r in sub]):.4f} | "
                f"{np.mean([r['clean_ablation']['roc_auc'] for r in sub]):.4f} | "
                f"{np.mean([r['clean_ablation']['f1_val_th'] for r in sub]):.4f} |\n"
            )

        mf.write("\n## 3. Visual Overlays\n\n")
        mf.write("![U-Net Segmentation Overlays](figures/unet_overlays.png)\n")

    print("\nSaved reports/real/unet_segmentation_report.json and reports/real/unet_segmentation_report.md")


if __name__ == "__main__":
    main()
