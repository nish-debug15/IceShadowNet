"""
scripts/analysis/run_pruning_latency_table.py
=============================================
Block 6 Item 5: Unified Pruning & Latency Table.
Evaluates Base (0%), Structured Channel Pruning (30%, 50%), and Unstructured
L1 Pruning (50%, 90%) in ONE table with:
  - Patch ROC-AUC (mean_prob)
  - Pixel PR-AUC
  - Non-Zero Params
  - CUDA Latency: warmup 50 iters, torch.cuda.synchronize(), median of 200 runs,
    for BOTH batch=1 and batch=64.

Outputs:
  - reports/real/pruning_unified.json
  - reports/real/pruning_unified.md
"""

import os
import sys
import copy
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.unet import IceUNet
from scripts.train_unet import SegPatchDataset, SoftDiceLoss
from scripts.analysis.run_structured_pruning import build_structurally_pruned_unet

PROCESSED_REAL = "data/processed_real"
MODELS_DIR = "models_real"
REPORTS_DIR = "reports/real"


def eval_maps(model, loader, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _ in loader:
            with torch.amp.autocast("cuda"):
                ps.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0])
    return np.concatenate(ps, axis=0)


def median_latency_ms(model, device, batch_size: int, warmup: int = 50, n_runs: int = 200) -> float:
    model.eval()
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            with torch.amp.autocast("cuda"):
                _ = model(x)
        torch.cuda.synchronize()
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.amp.autocast("cuda"):
                _ = model(x)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000.0)
    return float(np.median(times))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"

    d_tr = np.load(os.path.join(PROCESSED_REAL, "train.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))
    X_tr, M_tr = d_tr["images"], d_tr["masks"]
    X_te, M_te, y_te = d_te["images"], d_te["masks"], d_te["labels"].astype(int)

    tr_loader = DataLoader(SegPatchDataset(X_tr, M_tr, augment=True), batch_size=32, shuffle=True, pin_memory=True)
    te_loader = DataLoader(SegPatchDataset(X_te, M_te, augment=False), batch_size=64, shuffle=False, pin_memory=True)

    pw = min(float((M_tr == 0).sum() / max((M_tr == 1).sum(), 1)), 50.0)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    dice_fn = SoftDiceLoss()
    base_state = torch.load(os.path.join(MODELS_DIR, "IceUNet_seed42.pth"), map_location=device)

    def finetune(m, epochs=3, lr=2e-4):
        opt = optim.AdamW(m.parameters(), lr=lr, weight_decay=1e-4)
        sc = torch.amp.GradScaler("cuda")
        for _ in range(epochs):
            m.train()
            for xb, mb in tr_loader:
                xb, mb = xb.to(device, non_blocking=True), mb.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    lg = m(xb)
                    loss = 0.5 * bce_fn(lg, mb) + 0.5 * dice_fn(lg, mb)
                sc.scale(loss).backward()
                sc.step(opt)
                sc.update()

    specs = [
        ("Base (0%)", "none", 0.0),
        ("Structured Channel 30% (base_ch=17)", "structured", 0.70),
        ("Structured Channel 50% (base_ch=12)", "structured", 0.50),
        ("Unstructured L1 50% (param reduction)", "unstructured", 0.50),
        ("Unstructured L1 90% (param reduction)", "unstructured", 0.90),
    ]

    rows = []
    for name, kind, val in specs:
        base_m = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        base_m.load_state_dict(copy.deepcopy(base_state))
        if kind == "none":
            m = base_m
        elif kind == "structured":
            m = build_structurally_pruned_unet(base_m, val, device)
            finetune(m, epochs=3, lr=5e-4)
        else:
            m = base_m
            params_to_prune = [(mod, "weight") for mod in m.modules() if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d))]
            prune.global_unstructured(params_to_prune, pruning_method=prune.L1Unstructured, amount=val)
            finetune(m, epochs=3, lr=2e-4)
            for mod, pname in params_to_prune:
                prune.remove(mod, pname)

        nz_params = sum(int((p != 0).sum().item()) for p in m.parameters())
        te_maps = eval_maps(m, te_loader, device)
        px_pr = float(average_precision_score(M_te.ravel(), te_maps.ravel()))
        patch_auc = float(roc_auc_score(y_te, te_maps.mean(axis=(1, 2))))
        lat_b1 = median_latency_ms(m, device, batch_size=1)
        lat_b64 = median_latency_ms(m, device, batch_size=64)

        row = {
            "config": name, "nonzero_params": nz_params,
            "patch_roc_auc": patch_auc, "pixel_pr_auc": px_pr,
            "lat_b1_ms": lat_b1, "lat_b64_ms": lat_b64,
        }
        rows.append(row)
        print(f"{name:40s} | NZ={nz_params:,} | Patch AUC={patch_auc:.4f} | Px PR={px_pr:.4f} | B=1: {lat_b1:.3f} ms | B=64: {lat_b64:.3f} ms")

    with open(os.path.join(REPORTS_DIR, "pruning_unified.json"), "w", encoding="utf-8") as jf:
        json.dump(rows, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "pruning_unified.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 6 Item 5: Unified Pruning & Latency Table (Median of 200 Runs, 50 Warmup, CUDA Sync)\n\n")
        mf.write("| Configuration | Non-Zero Params | Patch ROC-AUC (`mean_prob`) | Pixel PR-AUC | CUDA Latency B=1 (ms) | CUDA Latency B=64 (ms) |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for r in rows:
            mf.write(
                f"| {r['config']} | {r['nonzero_params']:,} | {r['patch_roc_auc']:.4f} | "
                f"{r['pixel_pr_auc']:.4f} | {r['lat_b1_ms']:.3f} | {r['lat_b64_ms']:.3f} |\n"
            )

    print("Saved reports/real/pruning_unified.json and reports/real/pruning_unified.md")


if __name__ == "__main__":
    main()
