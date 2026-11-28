"""
scripts/analysis/run_phase_oracle.py
====================================
Block 7 Item 2: 3-Seed Phase Oracle Diagnostic (Seeds 42, 123, 2024).
Explicitly CIRCULAR — diagnostic of the phase-information ceiling, NOT a model contribution.

Trains both:
  - 3-ch intensity-only IceUNet: [HH, HV, VV]
  - 4-ch Phase Oracle IceUNet:   [HH, HV, VV, gamma_HHVV] where
    gamma_HHVV = |<Shh * conj(Svv)>_7x7| / sqrt(<|Shh|^2>_7x7 * <|Svv|^2>_7x7 + 1e-12)
on the identical frozen split, identical training loop (25 epochs, patience 8 on Val Px PR-AUC,
val-chosen threshold), and reports Test PR-AUC, lift over prevalence, Dice, and IoU as mean ± std.

Outputs:
  - reports/real/phase_oracle_diagnostic.json
  - reports/real/phase_oracle_diagnostic.md
"""

import os
import sys
import json
import yaml
import random
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


def train_and_eval_one(X_tr, M_tr, X_va, M_va, X_te, M_te, in_channels: int, seed: int, device) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    g = torch.Generator()
    g.manual_seed(seed)
    tr_loader = DataLoader(SegPatchDataset(X_tr, M_tr, augment=True), batch_size=32, shuffle=True, generator=g, pin_memory=True)
    va_loader = DataLoader(SegPatchDataset(X_va, M_va, augment=False), batch_size=64, shuffle=False, pin_memory=True)
    te_loader = DataLoader(SegPatchDataset(X_te, M_te, augment=False), batch_size=64, shuffle=False, pin_memory=True)

    model = IceUNet(in_channels=in_channels, base_ch=24, dropout_p=0.15).to(device)
    pw = min(float((M_tr == 0).sum() / max((M_tr == 1).sum(), 1)), 50.0)
    bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw], device=device))
    dice = SoftDiceLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=25)
    scaler = torch.amp.GradScaler("cuda")

    best_pr, best_st, pat, best_ep = -1.0, None, 0, 0
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
            best_pr, best_st, pat, best_ep = vpr, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0, ep
        else:
            pat += 1
            if pat >= 8:
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
    met_val = pixel_metrics(M_te.ravel(), te_maps.ravel(), val_th)
    prev = float(M_te.mean())
    return {
        "seed": seed,
        "best_ep": best_ep,
        "best_val_px_pr_auc": float(best_pr),
        "val_th": float(val_th),
        "pr_auc": float(met_val["pr_auc"]),
        "lift": float(met_val["pr_auc"] / prev),
        "dice": float(met_val["dice"]),
        "iou": float(met_val["iou"]),
    }


def summarize(runs: list) -> dict:
    return {
        k: {
            "mean": float(np.mean([r[k] for r in runs])),
            "std": float(np.std([r[k] for r in runs])),
        }
        for k in ["pr_auc", "lift", "dice", "iou"]
    }


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

    X_tr3, X_va3, X_te3 = d_tr["images"], d_va["images"], d_te["images"]
    X_tr4 = add_4th_channel(X_tr3, d_tr["coords"], coh_map)
    X_va4 = add_4th_channel(X_va3, d_va["coords"], coh_map)
    X_te4 = add_4th_channel(X_te3, d_te["coords"], coh_map)
    M_tr, M_va, M_te = d_tr["masks"], d_va["masks"], d_te["masks"]
    prev = float(M_te.mean())

    seeds = [42, 123, 2024]
    runs_3ch, runs_4ch = [], []
    for s in seeds:
        r3 = train_and_eval_one(X_tr3, M_tr, X_va3, M_va, X_te3, M_te, in_channels=3, seed=s, device=device)
        print(f"  3-ch Seed {s:4d} (ep={r3['best_ep']:2d}): PR-AUC={r3['pr_auc']:.4f} ({r3['lift']:.1f}x) | Dice={r3['dice']:.4f} | IoU={r3['iou']:.4f}")
        runs_3ch.append(r3)

        r4 = train_and_eval_one(X_tr4, M_tr, X_va4, M_va, X_te4, M_te, in_channels=4, seed=s, device=device)
        print(f"  4-ch Seed {s:4d} (ep={r4['best_ep']:2d}): PR-AUC={r4['pr_auc']:.4f} ({r4['lift']:.1f}x) | Dice={r4['dice']:.4f} | IoU={r4['iou']:.4f}")
        runs_4ch.append(r4)

    sum_3ch = summarize(runs_3ch)
    sum_4ch = summarize(runs_4ch)

    out = {
        "diagnostic_circular": True,
        "seeds": seeds,
        "prevalence": prev,
        "intensity_3ch": {"summary": sum_3ch, "runs": runs_3ch},
        "phase_oracle_4ch": {"summary": sum_4ch, "runs": runs_4ch},
    }
    with open(os.path.join(REPORTS_DIR, "phase_oracle_diagnostic.json"), "w", encoding="utf-8") as jf:
        json.dump(out, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "phase_oracle_diagnostic.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 7 Item 2: 3-Seed Phase Oracle Diagnostic (Circular — NOT a Model Contribution)\n\n")
        mf.write("> **Warning (Circular Diagnostic):** Adding `7×7` multilooked `|HH*conj(VV)|` coherence exposes inter-channel phase stability directly related to `DOP`/`CPR` labeling. This experiment tests the **phase-information ceiling** hypothesis over 3 seeds (`42, 123, 2024`) and is **not** a model contribution.\n\n")
        mf.write("| Model (3 Seeds, Val-Chosen Th) | Inputs | Pixel PR-AUC | Lift vs. 0.0527% | Dice | IoU |\n")
        mf.write("| :--- | :--- | :---: | :---: | :---: | :---: |\n")
        for label, inp, sm in [
            ("IceUNet (3-ch Intensity Only)", "`[HH, HV, VV]`", sum_3ch),
            ("Phase Oracle IceUNet (4-ch, CIRCULAR)", "`[HH, HV, VV, γ_HHVV]`", sum_4ch),
        ]:
            mf.write(
                f"| **{label}** | {inp} | "
                f"{sm['pr_auc']['mean']:.4f} ± {sm['pr_auc']['std']:.4f} | "
                f"{sm['lift']['mean']:.1f}× ± {sm['lift']['std']:.1f}× | "
                f"{sm['dice']['mean']:.4f} ± {sm['dice']['std']:.4f} | "
                f"{sm['iou']['mean']:.4f} ± {sm['iou']['std']:.4f} |\n"
            )
        mf.write("\n## Per-Seed Breakdown\n\n")
        mf.write("| Variant | Seed | Best Ep | Val Th | Pixel PR-AUC | Lift | Dice | IoU |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for vname, rlist in [("3-ch Intensity", runs_3ch), ("4-ch Phase Oracle", runs_4ch)]:
            for r in rlist:
                mf.write(f"| {vname} | {r['seed']} | {r['best_ep']} | {r['val_th']:.2f} | {r['pr_auc']:.4f} | {r['lift']:.1f}× | {r['dice']:.4f} | {r['iou']:.4f} |\n")

    print("Saved 3-seed Phase Oracle results to reports/real/phase_oracle_diagnostic.{json,md}")


if __name__ == "__main__":
    main()
