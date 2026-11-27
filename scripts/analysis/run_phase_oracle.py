"""
scripts/analysis/run_phase_oracle.py
====================================
Block 6 Item 4: Phase Oracle Diagnostic (1 seed, explicitly CIRCULAR — NOT a headline model).
Adds a 4th input channel: 7x7 multilooked inter-channel phase coherence
  gamma_HH_VV = |<Shh * conj(Svv)>_7x7| / sqrt(<|Shh|^2>_7x7 * <|Svv|^2>_7x7 + eps)
to test the "phase-information ceiling" hypothesis against the 3-channel intensity-only
IceUNet_seed42.

Outputs:
  - reports/real/phase_oracle_diagnostic.json
  - reports/real/phase_oracle_diagnostic.md
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
from scipy.ndimage import uniform_filter
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.preprocessing.data_ops import load_complex_dfsar_sli
from src.models.unet import IceUNet
from scripts.train_unet import SegPatchDataset, SoftDiceLoss, pixel_metrics, tune_pixel_threshold

PROCESSED_REAL = "data/processed_real"
REPORTS_DIR = "reports/real"


def compute_hh_vv_coherence(Shh: np.ndarray, Svv: np.ndarray, W: int = 7) -> np.ndarray:
    cross = Shh * np.conj(Svv)
    c_re = uniform_filter(cross.real.astype(np.float64), size=W, mode="reflect")
    c_im = uniform_filter(cross.imag.astype(np.float64), size=W, mode="reflect")
    p_hh = uniform_filter((np.abs(Shh) ** 2).astype(np.float64), size=W, mode="reflect")
    p_vv = uniform_filter((np.abs(Svv) ** 2).astype(np.float64), size=W, mode="reflect")
    coh = np.sqrt(c_re ** 2 + c_im ** 2) / np.sqrt(np.maximum(p_hh * p_vv, 1e-12))
    return np.clip(coh, 0.0, 1.0).astype(np.float32)


def add_4th_channel(images: np.ndarray, coords: np.ndarray, coh_map: np.ndarray, P: int = 64) -> np.ndarray:
    ch4 = np.stack([coh_map[r:r+P, c:c+P] for r, c in coords], axis=0)[:, None, :, :]
    return np.concatenate([images, ch4], axis=1).astype(np.float32)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"

    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("real_dfsar", {})
    raw_dir = cfg.get("raw_dir", "data/raw/sar_nxcl")
    ar = cfg.get("active_rows", [90624, 319812])
    start_row, num_rows = int(ar[0]), int(ar[1]) - int(ar[0])

    print("Loading SLI complex channels to compute 7x7 |HH*conj(VV)| coherence...")
    channels = load_complex_dfsar_sli(raw_dir, start_row=start_row, num_rows=num_rows)
    coh_map = compute_hh_vv_coherence(channels["hh"], channels["vv"], W=7)

    d_tr = np.load(os.path.join(PROCESSED_REAL, "train.npz"))
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_tr4 = add_4th_channel(d_tr["images"], d_tr["coords"], coh_map)
    X_va4 = add_4th_channel(d_va["images"], d_va["coords"], coh_map)
    X_te4 = add_4th_channel(d_te["images"], d_te["coords"], coh_map)
    M_tr, M_va, M_te = d_tr["masks"], d_va["masks"], d_te["masks"]

    tr_loader = DataLoader(SegPatchDataset(X_tr4, M_tr, augment=True), batch_size=32, shuffle=True, pin_memory=True)
    va_loader = DataLoader(SegPatchDataset(X_va4, M_va, augment=False), batch_size=64, shuffle=False, pin_memory=True)
    te_loader = DataLoader(SegPatchDataset(X_te4, M_te, augment=False), batch_size=64, shuffle=False, pin_memory=True)

    torch.manual_seed(42)
    np.random.seed(42)
    model = IceUNet(in_channels=4, base_ch=24, dropout_p=0.15).to(device)
    pw = min(float((M_tr == 0).sum() / max((M_tr == 1).sum(), 1)), 50.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    dice = SoftDiceLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    scaler = torch.amp.GradScaler("cuda")

    best_pr, best_st, pat = -1.0, None, 0
    for ep in range(1, 26):
        model.train()
        for xb, mb in tr_loader:
            xb, mb = xb.to(device, non_blocking=True), mb.to(device, non_blocking=True)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                lg = model(xb)
                loss = 0.5 * bce(lg, mb) + 0.5 * dice(lg, mb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()

        model.eval()
        vp = []
        with torch.no_grad():
            for xb, _ in va_loader:
                with torch.amp.autocast("cuda"):
                    vp.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0])
        vpr = float(average_precision_score(M_va.ravel(), np.concatenate(vp, axis=0).ravel()))
        if vpr > best_pr:
            best_pr, best_st, pat = vpr, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            pat += 1
            if pat >= 5:
                print(f"Early stop at Ep {ep} (Best Val Px PR-AUC={best_pr:.4f})")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_st.items()})
    model.eval()
    vp, tp = [], []
    with torch.no_grad():
        for xb, _ in va_loader:
            with torch.amp.autocast("cuda"):
                vp.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0])
        for xb, _ in te_loader:
            with torch.amp.autocast("cuda"):
                tp.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0])
    va_maps = np.concatenate(vp, axis=0)
    te_maps = np.concatenate(tp, axis=0)

    val_th, _ = tune_pixel_threshold(M_va.ravel(), va_maps.ravel())
    best_th, _ = tune_pixel_threshold(M_te.ravel(), te_maps.ravel())
    met_val = pixel_metrics(M_te.ravel(), te_maps.ravel(), val_th)
    met_best = pixel_metrics(M_te.ravel(), te_maps.ravel(), best_th)
    prev = float(M_te.mean())

    unet_rep = json.load(open(os.path.join(REPORTS_DIR, "unet_segmentation_report.json"), "r", encoding="utf-8"))
    s42 = [r for r in unet_rep["unet_runs"] if r["seed"] == 42][0]

    out = {
        "diagnostic_circular": True,
        "seed": 42,
        "prevalence": prev,
        "intensity_3ch_seed42": {
            "pr_auc": s42["pooled_val_th"]["pr_auc"],
            "lift": s42["pooled_val_th"]["pr_auc"] / prev,
            "dice_val_th": s42["pooled_val_th"]["dice"],
            "dice_best_th": s42["pooled_best_th"]["dice"],
        },
        "phase_oracle_4ch_seed42": {
            "val_th": val_th,
            "best_th": best_th,
            "pr_auc": met_val["pr_auc"],
            "lift": met_val["pr_auc"] / prev,
            "dice_val_th": met_val["dice"],
            "iou_val_th": met_val["iou"],
            "dice_best_th": met_best["dice"],
        },
    }
    with open(os.path.join(REPORTS_DIR, "phase_oracle_diagnostic.json"), "w", encoding="utf-8") as jf:
        json.dump(out, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "phase_oracle_diagnostic.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 6 Item 4: Phase Oracle Diagnostic (Circular — NOT a Headline Model)\n\n")
        mf.write("> **Warning (Circular Diagnostic):** Adding `7×7` multilooked `|HH*conj(VV)|` coherence exposes inter-channel phase stability directly related to `DOP`/`CPR` labeling. This experiment tests the **phase-information ceiling** hypothesis and is **not** a headline model.\n\n")
        mf.write("| Model (Seed 42) | Inputs | Pixel PR-AUC | Lift vs. 0.0527% | Dice (Val Th) | Dice (Best Test Th) |\n")
        mf.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        i3 = out["intensity_3ch_seed42"]
        o4 = out["phase_oracle_4ch_seed42"]
        mf.write(f"| **IceUNet (3-ch Intensity Only)** | `[HH, HV, VV]` | {i3['pr_auc']:.4f} | {i3['lift']:.1f}× | {i3['dice_val_th']:.4f} | {i3['dice_best_th']:.4f} |\n")
        mf.write(f"| **Phase Oracle IceUNet (4-ch, CIRCULAR)** | `[HH, HV, VV, γ_HHVV]` | {o4['pr_auc']:.4f} | {o4['lift']:.1f}× | {o4['dice_val_th']:.4f} | {o4['dice_best_th']:.4f} |\n")

    print(f"3-ch Seed 42: PR-AUC={i3['pr_auc']:.4f}, Dice={i3['dice_val_th']:.4f} | 4-ch Phase Oracle: PR-AUC={o4['pr_auc']:.4f}, Dice={o4['dice_val_th']:.4f}")


if __name__ == "__main__":
    main()
