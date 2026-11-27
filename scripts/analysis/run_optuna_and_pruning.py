"""
scripts/analysis/run_optuna_and_pruning.py
==========================================
Block 3b:
  1. Val-only Optuna hyperparameter sweep (12 trials, max 10 epochs/trial) over
     lr, base_ch, dropout_p, and dice_weight. Zero test-set peeking; winning
     config evaluated on test once.
  2. Magnitude pruning (torch.nn.utils.prune.l1_unstructured) on IceUNet_seed42
     at 0%, 30%, 50%, 70%, 90% sparsity with 3-epoch fine-tune:
     reports Pixel Dice, Pixel PR-AUC, Patch Mean-Prob ROC-AUC, non-zero params,
     and CUDA inference latency (ms / batch of 64).

Outputs:
  - reports/real/efficiency_report.json
  - reports/real/efficiency_report.md
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
from sklearn.metrics import average_precision_score, roc_auc_score
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.unet import IceUNet
from scripts.train_unet import SegPatchDataset, SoftDiceLoss, pixel_metrics, tune_pixel_threshold

PROCESSED_REAL = "data/processed_real"
MODELS_DIR = "models_real"
REPORTS_DIR = "reports/real"


def eval_unet_on_loader(model, loader, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _ in loader:
            with torch.amp.autocast("cuda"):
                ps.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0])
    return np.concatenate(ps, axis=0)


def measure_latency_ms(model, device, batch_size=64, n_iters=40):
    model.eval()
    x = torch.randn(batch_size, 3, 64, 64, device=device)
    with torch.no_grad():
        for _ in range(10):
            with torch.amp.autocast("cuda"):
                _ = model(x)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters):
            with torch.amp.autocast("cuda"):
                _ = model(x)
        torch.cuda.synchronize()
        dt = (time.perf_counter() - t0) / n_iters
    return float(dt * 1000.0)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"

    d_tr = np.load(os.path.join(PROCESSED_REAL, "train.npz"))
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_tr, M_tr = d_tr["images"], d_tr["masks"]
    X_va, M_va = d_va["images"], d_va["masks"]
    X_te, M_te, y_te = d_te["images"], d_te["masks"], d_te["labels"].astype(int)

    tr_loader = DataLoader(SegPatchDataset(X_tr, M_tr, augment=True), batch_size=32, shuffle=True, pin_memory=True)
    va_loader = DataLoader(SegPatchDataset(X_va, M_va, augment=False), batch_size=64, shuffle=False, pin_memory=True)
    te_loader = DataLoader(SegPatchDataset(X_te, M_te, augment=False), batch_size=64, shuffle=False, pin_memory=True)

    pos_weight_val = min(float((M_tr == 0).sum() / max((M_tr == 1).sum(), 1)), 50.0)
    bce_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
    dice_fn = SoftDiceLoss()

    # ── 1. Val-Only Optuna Sweep (12 trials, max 8 epochs/trial) ───────────────
    print("--- Part 1: Val-Only Optuna Sweep (12 trials, zero test peeking) ---")
    trial_logs = []

    def objective(trial):
        lr = trial.suggest_float("lr", 3e-4, 3e-3, log=True)
        base_ch = trial.suggest_categorical("base_ch", [16, 24, 32])
        dropout_p = trial.suggest_float("dropout_p", 0.05, 0.30)
        dice_w = trial.suggest_float("dice_weight", 0.2, 0.8)

        torch.manual_seed(42)
        np.random.seed(42)
        model = IceUNet(in_channels=3, base_ch=base_ch, dropout_p=dropout_p).to(device)
        opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.amp.GradScaler("cuda")

        best_va_pr = -1.0
        for ep in range(8):
            model.train()
            for xb, mb in tr_loader:
                xb, mb = xb.to(device, non_blocking=True), mb.to(device, non_blocking=True)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    logits = model(xb)
                    loss = (1.0 - dice_w) * bce_fn(logits, mb) + dice_w * dice_fn(logits, mb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

            va_maps = eval_unet_on_loader(model, va_loader, device)
            va_pr = float(average_precision_score(M_va.ravel(), va_maps.ravel()))
            if va_pr > best_va_pr:
                best_va_pr = va_pr

        trial_logs.append({
            "trial": trial.number, "lr": lr, "base_ch": base_ch,
            "dropout_p": dropout_p, "dice_weight": dice_w, "val_px_pr_auc": best_va_pr,
        })
        print(f"  Trial {trial.number:2d} | base_ch={base_ch} lr={lr:.4e} drop={dropout_p:.2f} dice_w={dice_w:.2f} -> Val Px PR-AUC={best_va_pr:.4f}")
        return best_va_pr

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=12)
    best_p = study.best_params
    print(f"Best Optuna params: {best_p} (Val Px PR-AUC={study.best_value:.4f})")

    # Train winning Optuna config for full 15 epochs (patience 5) and evaluate ONCE on test
    torch.manual_seed(42)
    opt_model = IceUNet(in_channels=3, base_ch=best_p["base_ch"], dropout_p=best_p["dropout_p"]).to(device)
    opt_opt = optim.AdamW(opt_model.parameters(), lr=best_p["lr"], weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda")
    best_va_pr, best_state, patience = -1.0, None, 0
    for ep in range(15):
        opt_model.train()
        for xb, mb in tr_loader:
            xb, mb = xb.to(device, non_blocking=True), mb.to(device, non_blocking=True)
            opt_opt.zero_grad()
            with torch.amp.autocast("cuda"):
                logits = opt_model(xb)
                loss = (1.0 - best_p["dice_weight"]) * bce_fn(logits, mb) + best_p["dice_weight"] * dice_fn(logits, mb)
            scaler.scale(loss).backward()
            scaler.step(opt_opt)
            scaler.update()
        va_maps = eval_unet_on_loader(opt_model, va_loader, device)
        va_pr = float(average_precision_score(M_va.ravel(), va_maps.ravel()))
        if va_pr > best_va_pr:
            best_va_pr = va_pr
            best_state = {k: v.cpu().clone() for k, v in opt_model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 5:
                break

    opt_model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    va_maps = eval_unet_on_loader(opt_model, va_loader, device)
    te_maps = eval_unet_on_loader(opt_model, te_loader, device)
    opt_th, _ = tune_pixel_threshold(M_va.ravel(), va_maps.ravel())
    opt_te_met = pixel_metrics(M_te.ravel(), te_maps.ravel(), opt_th)
    opt_patch_auc = float(roc_auc_score(y_te, te_maps.mean(axis=(1, 2))))

    # ── 2. Magnitude Pruning (0, 30, 50, 70, 90%) on IceUNet_seed42 ────────────
    print("\n--- Part 2: Magnitude Pruning (0%, 30%, 50%, 70%, 90%) + Fine-Tune ---")
    base_state = torch.load(os.path.join(MODELS_DIR, "IceUNet_seed42.pth"), map_location=device)
    sparsities = [0.0, 0.30, 0.50, 0.70, 0.90]
    pruning_rows = []

    for sp in sparsities:
        m = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        m.load_state_dict(copy.deepcopy(base_state))

        if sp > 0.0:
            params_to_prune = [
                (mod, "weight")
                for mod in m.modules()
                if isinstance(mod, (nn.Conv2d, nn.ConvTranspose2d))
            ]
            prune.global_unstructured(
                params_to_prune,
                pruning_method=prune.L1Unstructured,
                amount=sp,
            )
            # Fine-tune 3 epochs with low lr
            f_opt = optim.AdamW(m.parameters(), lr=2e-4, weight_decay=1e-4)
            f_scaler = torch.amp.GradScaler("cuda")
            for ep in range(3):
                m.train()
                for xb, mb in tr_loader:
                    xb, mb = xb.to(device, non_blocking=True), mb.to(device, non_blocking=True)
                    f_opt.zero_grad()
                    with torch.amp.autocast("cuda"):
                        logits = m(xb)
                        loss = 0.5 * bce_fn(logits, mb) + 0.5 * dice_fn(logits, mb)
                    f_scaler.scale(loss).backward()
                    f_scaler.step(f_opt)
                    f_scaler.update()

            for mod, name in params_to_prune:
                prune.remove(mod, name)

        # Count exact zero / non-zero parameters
        total_p = sum(p.numel() for p in m.parameters())
        nonzero_p = sum(int((p != 0).sum().item()) for p in m.parameters())
        actual_sp = 1.0 - (nonzero_p / total_p)

        va_p = eval_unet_on_loader(m, va_loader, device)
        te_p = eval_unet_on_loader(m, te_loader, device)
        th, _ = tune_pixel_threshold(M_va.ravel(), va_p.ravel())
        met = pixel_metrics(M_te.ravel(), te_p.ravel(), th)
        patch_auc = float(roc_auc_score(y_te, te_p.mean(axis=(1, 2))))
        lat_ms = measure_latency_ms(m, device, batch_size=64)

        row = {
            "target_sparsity": sp,
            "actual_sparsity": float(actual_sp),
            "total_params": int(total_p),
            "nonzero_params": int(nonzero_p),
            "px_pr_auc": met["pr_auc"],
            "px_roc_auc": met["roc_auc"],
            "dice": met["dice"],
            "patch_mean_roc_auc": patch_auc,
            "latency_ms_b64": lat_ms,
        }
        pruning_rows.append(row)
        print(f"  Sparsity {100*sp:2.0f}% (actual {100*actual_sp:4.1f}%) | NZ={nonzero_p:,} | "
              f"Px PR-AUC={met['pr_auc']:.4f} | Dice={met['dice']:.4f} | Patch AUC={patch_auc:.4f} | Lat={lat_ms:.2f} ms")

    out_json = {
        "optuna": {
            "n_trials": 12, "best_params": best_p, "best_val_px_pr_auc": float(study.best_value),
            "test_metrics": opt_te_met, "test_patch_mean_roc_auc": opt_patch_auc, "trials": trial_logs,
        },
        "pruning": pruning_rows,
    }
    with open(os.path.join(REPORTS_DIR, "efficiency_report.json"), "w", encoding="utf-8") as jf:
        json.dump(out_json, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "efficiency_report.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 3b: Val-Only Optuna Sweep & Magnitude Pruning\n\n")
        mf.write("## 1. Val-Only Optuna Hyperparameter Sweep (12 Trials, Zero Test Peeking)\n\n")
        mf.write(f"- **Winning Config:** `base_ch={best_p['base_ch']}`, `lr={best_p['lr']:.4e}`, `dropout_p={best_p['dropout_p']:.3f}`, `dice_weight={best_p['dice_weight']:.3f}`\n")
        mf.write(f"- **Best Val Pixel PR-AUC:** `{study.best_value:.4f}`\n")
        mf.write(f"- **Evaluated Once on Test:** Pixel PR-AUC = `{opt_te_met['pr_auc']:.4f}`, Pixel ROC-AUC = `{opt_te_met['roc_auc']:.4f}`, Dice (val th={opt_th:.2f}) = `{opt_te_met['dice']:.4f}`, Patch Mean-Prob ROC-AUC = `{opt_patch_auc:.4f}`\n\n")

        mf.write("## 2. Global Magnitude Pruning + 3-Epoch Fine-Tune (`IceUNet_seed42`)\n\n")
        mf.write("| Target Sparsity | Actual Sparsity | Non-Zero Params | Pixel PR-AUC | Dice (Val Th) | Patch Mean-Prob ROC-AUC | CUDA Latency (ms / B=64) |\n")
        mf.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in pruning_rows:
            mf.write(
                f"| {100*r['target_sparsity']:.0f}% | {100*r['actual_sparsity']:.1f}% | {r['nonzero_params']:,} | "
                f"{r['px_pr_auc']:.4f} | {r['dice']:.4f} | {r['patch_mean_roc_auc']:.4f} | {r['latency_ms_b64']:.2f} |\n"
            )
    print("Saved reports/real/efficiency_report.json and reports/real/efficiency_report.md")


if __name__ == "__main__":
    main()
