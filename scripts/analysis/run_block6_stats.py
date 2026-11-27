"""
scripts/analysis/run_block6_stats.py
====================================
Block 6 Item 1: Rigorous Statistical Comparison.
  1. Per-test-strip metrics (Strips 0, 8, 15) across all models on the frozen split:
     CustomCNN, ResNet_scratch, ResNet_pretrained, LogisticRegression, DecisionTree,
     and IceUNet (patch mean_prob).
  2. 5-fold blocked-strip CV over all 20 strips (256-row buffers) for:
     LogisticRegression, CustomCNN, ResNet_scratch, and IceUNet (mean_prob, 1 seed).
  3. Paired bootstrap of metric DIFFERENCES (delta ROC-AUC, delta PR-AUC) over
     256-row sub-blocks (with >=64-row guard gaps between sub-blocks) inside the
     3 test strips. Explicitly labeled: "within-test-set variability only".
     Models are separable ONLY if the paired 95% CI excludes 0.

Outputs:
  - reports/real/block6_stats.json
  - reports/real/block6_stats.md
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, balanced_accuracy_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.preprocessing.data_ops import load_complex_dfsar_sli, compute_cpr_dop_complex
from src.datasets.spatial_split import deterministic_strip_split, assert_no_extent_leakage
from src.datasets.ice_dataset import IcePatchDataset
from src.models.custom_cnn import CustomCNN
from src.models.resnet import IceResNet
from src.models.unet import IceUNet
from src.evaluate import tune_threshold_on_val
from scripts.train_unet import SegPatchDataset, SoftDiceLoss

PROCESSED_REAL = "data/processed_real"
MODELS_DIR = "models_real"
REPORTS_DIR = "reports/real"


def extract_lr_features(imgs):
    feats = []
    for c in range(3):
        ch = imgs[:, c]
        feats.append(ch.mean(axis=(1, 2)))
        feats.append(ch.std(axis=(1, 2)))
        feats.append(np.percentile(ch, 90, axis=(1, 2)))
        feats.append(np.percentile(ch, 10, axis=(1, 2)))
    return np.column_stack(feats)


def predict_cls(model, loader, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _ in loader:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                ps.extend(torch.sigmoid(model(xb.to(device))).cpu().numpy().flatten())
    return np.array(ps, dtype=np.float64)


def predict_unet_mean(model, loader, device):
    model.eval()
    ps = []
    with torch.no_grad():
        for xb, _ in loader:
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                pmap = torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0]
            ps.extend(pmap.mean(axis=(1, 2)))
    return np.array(ps, dtype=np.float64)


def paired_subblock_bootstrap(y_true, coords, scores_dict, pairs, block_len=256, guard=64, n_boot=1000, seed=42):
    """
    Partition test strips into 256-row sub-blocks separated by 64-row guard gaps
    (stride = 320 rows; keep patches fully inside [b_start, b_start + 256)).
    Resample valid sub-blocks with replacement and compute 95% CIs of metric differences.
    """
    rows = coords[:, 0]
    step = block_len + guard
    r_min = int(rows.min())
    r_max = int(rows.max())
    blocks = []
    for b0 in range(r_min, r_max + 1, step):
        b1 = b0 + block_len
        idx = np.where((rows >= b0) & (rows + 64 <= b1))[0]
        if len(idx) > 0:
            blocks.append(idx)

    rng = np.random.default_rng(seed)
    n_blocks = len(blocks)
    results = []
    for m_a, m_b in pairs:
        pa, pb = scores_dict[m_a], scores_dict[m_b]
        d_roc, d_pr = [], []
        for _ in range(n_boot):
            chosen = rng.integers(0, n_blocks, size=n_blocks)
            idx = np.concatenate([blocks[k] for k in chosen])
            yt = y_true[idx]
            if len(np.unique(yt)) < 2:
                continue
            d_roc.append(roc_auc_score(yt, pa[idx]) - roc_auc_score(yt, pb[idx]))
            d_pr.append(average_precision_score(yt, pa[idx]) - average_precision_score(yt, pb[idx]))

        roc_pt = float(roc_auc_score(y_true, pa) - roc_auc_score(y_true, pb))
        pr_pt = float(average_precision_score(y_true, pa) - average_precision_score(y_true, pb))
        roc_ci = [float(np.percentile(d_roc, 2.5)), float(np.percentile(d_roc, 97.5))]
        pr_ci = [float(np.percentile(d_pr, 2.5)), float(np.percentile(d_pr, 97.5))]
        results.append({
            "pair": f"{m_a} - {m_b}",
            "n_subblocks": n_blocks,
            "delta_roc_auc": roc_pt,
            "delta_roc_ci": roc_ci,
            "roc_separable": bool(roc_ci[0] > 0 or roc_ci[1] < 0),
            "delta_pr_auc": pr_pt,
            "delta_pr_ci": pr_ci,
            "pr_separable": bool(pr_ci[0] > 0 or pr_ci[1] < 0),
        })
    return results


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "CUDA required"

    # ── Part 1: Frozen Split Per-Test-Strip Table + Paired Block Bootstrap ─────
    print("=== Part 1: Frozen Split Per-Strip Metrics & Paired Block Bootstrap ===")
    d_tr = np.load(os.path.join(PROCESSED_REAL, "train.npz"))
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    y_tr, y_va, y_te = d_tr["labels"].astype(int), d_va["labels"].astype(int), d_te["labels"].astype(int)
    te_strips = d_te["strip_ids"].astype(int)
    te_coords = d_te["coords"].astype(int)

    va_cls_loader = DataLoader(IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(d_va["images"], y_va)], augment=False), batch_size=64, shuffle=False)
    te_cls_loader = DataLoader(IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(d_te["images"], y_te)], augment=False), batch_size=64, shuffle=False)
    va_seg_loader = DataLoader(SegPatchDataset(d_va["images"], d_va["masks"], augment=False), batch_size=64, shuffle=False)
    te_seg_loader = DataLoader(SegPatchDataset(d_te["images"], d_te["masks"], augment=False), batch_size=64, shuffle=False)

    # Collect 3-seed mean scores (or single run for LR/DT) on frozen test set
    test_scores = {}
    val_scores = {}

    X_tr_tab, X_va_tab, X_te_tab = extract_lr_features(d_tr["images"]), extract_lr_features(d_va["images"]), extract_lr_features(d_te["images"])
    lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42).fit(X_tr_tab, y_tr)
    dt = DecisionTreeClassifier(max_depth=3, class_weight="balanced", random_state=42).fit(X_tr_tab, y_tr)
    val_scores["LogisticRegression"] = lr.predict_proba(X_va_tab)[:, 1]
    test_scores["LogisticRegression"] = lr.predict_proba(X_te_tab)[:, 1]
    val_scores["DecisionTree"] = dt.predict_proba(X_va_tab)[:, 1]
    test_scores["DecisionTree"] = dt.predict_proba(X_te_tab)[:, 1]

    for mname, factory in [
        ("CustomCNN", lambda: CustomCNN(in_channels=3, num_classes=1)),
        ("ResNet_scratch", lambda: IceResNet(in_channels=3, num_classes=1, pretrained=False)),
        ("ResNet_pretrained", lambda: IceResNet(in_channels=3, num_classes=1, pretrained=True)),
    ]:
        vp_list, tp_list = [], []
        for s in [42, 123, 2024]:
            m = factory().to(device)
            st = torch.load(os.path.join(MODELS_DIR, f"{mname}_seed{s}.pth"), map_location=device)
            if isinstance(st, dict) and "model_state_dict" in st:
                st = st["model_state_dict"]
            m.load_state_dict(st)
            vp_list.append(predict_cls(m, va_cls_loader, device))
            tp_list.append(predict_cls(m, te_cls_loader, device))
        val_scores[mname] = np.mean(vp_list, axis=0)
        test_scores[mname] = np.mean(tp_list, axis=0)

    # IceUNet (3-seed mean of patch mean_prob)
    uv_list, ut_list = [], []
    for s in [42, 123, 2024]:
        um = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        um.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"IceUNet_seed{s}.pth"), map_location=device))
        uv_list.append(predict_unet_mean(um, va_seg_loader, device))
        ut_list.append(predict_unet_mean(um, te_seg_loader, device))
    val_scores["IceUNet_mean_prob"] = np.mean(uv_list, axis=0)
    test_scores["IceUNet_mean_prob"] = np.mean(ut_list, axis=0)

    models_order = ["CustomCNN", "IceUNet_mean_prob", "ResNet_scratch", "LogisticRegression", "ResNet_pretrained", "DecisionTree"]
    per_strip_table = []
    for mname in models_order:
        v_th, _ = tune_threshold_on_val(y_va, val_scores[mname])
        tp = test_scores[mname]
        row = {
            "model": mname,
            "val_th": float(v_th),
            "pooled_roc": float(roc_auc_score(y_te, tp)),
            "pooled_pr": float(average_precision_score(y_te, tp)),
            "pooled_f1": float(f1_score(y_te, (tp >= v_th).astype(int), zero_division=0)),
            "strips": {},
        }
        for s in [0, 8, 15]:
            sm = te_strips == s
            yt_s, tp_s = y_te[sm], tp[sm]
            row["strips"][int(s)] = {
                "prev": float(yt_s.mean()),
                "roc_auc": float(roc_auc_score(yt_s, tp_s)),
                "pr_auc": float(average_precision_score(yt_s, tp_s)),
                "f1": float(f1_score(yt_s, (tp_s >= v_th).astype(int), zero_division=0)),
            }
        per_strip_table.append(row)

    pairs = [
        ("CustomCNN", "LogisticRegression"),
        ("CustomCNN", "ResNet_scratch"),
        ("CustomCNN", "ResNet_pretrained"),
        ("IceUNet_mean_prob", "CustomCNN"),
        ("IceUNet_mean_prob", "LogisticRegression"),
        ("ResNet_scratch", "ResNet_pretrained"),
    ]
    paired_res = paired_subblock_bootstrap(y_te, te_coords, test_scores, pairs, block_len=256, guard=64, n_boot=1000, seed=42)

    # ── Part 2: 5-Fold Blocked-Strip CV (LR, CustomCNN, ResNet_scratch, IceUNet) ──
    print("\n=== Part 2: 5-Fold Blocked-Strip CV across All 20 Strips ===")
    with open("config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f).get("real_dfsar", {})
    raw_dir = cfg.get("raw_dir", "data/raw/sar_nxcl")
    ar = cfg.get("active_rows", [90624, 319812])
    start_row, num_rows = int(ar[0]), int(ar[1]) - int(ar[0])
    P, stride, n_strips, buffer_px = 64, 32, 20, 256

    channels = load_complex_dfsar_sli(raw_dir, start_row=start_row, num_rows=num_rows)
    Shh, Shv, Svh, Svv = channels["hh"], channels["hv"], channels["vh"], channels["vv"]
    valid_mask = np.isfinite(Shh) & np.isfinite(Shv) & np.isfinite(Svh) & np.isfinite(Svv) & (np.abs(Shh) > 0) & (np.abs(Svv) > 0)
    Shv_sym = 0.5 * (Shv + Svh)
    eps = 1e-5
    radar_inputs = np.stack([
        np.log10(np.abs(Shh)**2 + eps).astype(np.float32),
        np.log10(np.abs(Shv_sym)**2 + eps).astype(np.float32),
        np.log10(np.abs(Svv)**2 + eps).astype(np.float32),
    ], axis=0)
    cpr, dop = compute_cpr_dop_complex(Shh, Shv, Svh, Svv, W=7)
    pixel_ice_mask = ((cpr > 1.0) & (dop < 0.13) & valid_mask).astype(np.uint8)

    H, W_dim = Shh.shape
    coords, imgs_l, masks_l, labels_l = [], [], [], []
    for r in range(0, H - P + 1, stride):
        for c in range(0, W_dim - P + 1, stride):
            if valid_mask[r:r+P, c:c+P].mean() < 0.95:
                continue
            m_sub = pixel_ice_mask[r:r+P, c:c+P]
            coords.append((r, c))
            imgs_l.append(radar_inputs[:, r:r+P, c:c+P])
            masks_l.append(m_sub)
            labels_l.append(1 if np.count_nonzero(m_sub) >= 5 else 0)

    imgs_all = np.stack(imgs_l)
    masks_all = np.stack(masks_l)
    labels_all = np.array(labels_l, dtype=np.int64)
    coord_to_idx = {c: i for i, c in enumerate(coords)}

    folds = [
        {"test": (0, 1, 2, 3),     "val": (6, 11, 16)},
        {"test": (4, 5, 6, 7),     "val": (1, 12, 17)},
        {"test": (8, 9, 10, 11),   "val": (2, 14, 18)},
        {"test": (12, 13, 14, 15), "val": (2, 7, 18)},
        {"test": (16, 17, 18, 19), "val": (2, 8, 13)},
    ]

    cv_rows = []
    for f_i, fcfg in enumerate(folds):
        tr_c, va_c, te_c, _ = deterministic_strip_split(coords, n_strips=n_strips, val_strips=fcfg["val"], test_strips=fcfg["test"], patch_size=P, buffer_pixels=buffer_px)
        assert_no_extent_leakage(tr_c, va_c, te_c, patch_size=P)
        tr_idx = [coord_to_idx[c] for c in tr_c]
        va_idx = [coord_to_idx[c] for c in va_c]
        te_idx = [coord_to_idx[c] for c in te_c]

        X_tr, y_tr_f, M_tr_f = imgs_all[tr_idx], labels_all[tr_idx], masks_all[tr_idx]
        X_va, y_va_f, M_va_f = imgs_all[va_idx], labels_all[va_idx], masks_all[va_idx]
        X_te, y_te_f, M_te_f = imgs_all[te_idx], labels_all[te_idx], masks_all[te_idx]
        prev = float(y_te_f.mean())
        print(f"Fold {f_i+1}/5 (Test strips={fcfg['test']}, N={len(y_te_f)}, prev={100*prev:.1f}%)")

        # LR
        clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42).fit(extract_lr_features(X_tr), y_tr_f)
        vp_lr, tp_lr = clf.predict_proba(extract_lr_features(X_va))[:, 1], clf.predict_proba(extract_lr_features(X_te))[:, 1]
        th_lr, _ = tune_threshold_on_val(y_va_f, vp_lr)
        cv_rows.append({"fold": f_i+1, "model": "LogisticRegression", "prev": prev, "roc_auc": float(roc_auc_score(y_te_f, tp_lr)), "pr_auc": float(average_precision_score(y_te_f, tp_lr)), "f1": float(f1_score(y_te_f, (tp_lr >= th_lr).astype(int), zero_division=0))})

        tr_cls = DataLoader(IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(X_tr, y_tr_f)], augment=True), batch_size=32, shuffle=True)
        va_cls = DataLoader(IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(X_va, y_va_f)], augment=False), batch_size=64, shuffle=False)
        te_cls = DataLoader(IcePatchDataset([(im, int(lb), (0, 0)) for im, lb in zip(X_te, y_te_f)], augment=False), batch_size=64, shuffle=False)
        pos_w = float((y_tr_f == 0).sum() / max((y_tr_f == 1).sum(), 1))

        for mname, factory in [
            ("CustomCNN", lambda: CustomCNN(in_channels=3, num_classes=1)),
            ("ResNet_scratch", lambda: IceResNet(in_channels=3, num_classes=1, pretrained=False)),
        ]:
            torch.manual_seed(42)
            np.random.seed(42)
            mod = factory().to(device)
            crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_w], device=device))
            opt = optim.AdamW(mod.parameters(), lr=1e-3, weight_decay=1e-4)
            scaler = torch.amp.GradScaler("cuda")
            best_auc, best_st, pat = -1.0, None, 0
            for ep in range(15):
                mod.train()
                for xb, yb in tr_cls:
                    xb, yb = xb.to(device), yb.to(device).float()
                    opt.zero_grad()
                    with torch.amp.autocast("cuda"):
                        loss = crit(mod(xb), yb)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                vp = predict_cls(mod, va_cls, device)
                vauc = float(roc_auc_score(y_va_f, vp))
                if vauc > best_auc:
                    best_auc, best_st, pat = vauc, {k: v.cpu().clone() for k, v in mod.state_dict().items()}, 0
                else:
                    pat += 1
                    if pat >= 4:
                        break
            mod.load_state_dict({k: v.to(device) for k, v in best_st.items()})
            vp = predict_cls(mod, va_cls, device)
            tp = predict_cls(mod, te_cls, device)
            th, _ = tune_threshold_on_val(y_va_f, vp)
            cv_rows.append({"fold": f_i+1, "model": mname, "prev": prev, "roc_auc": float(roc_auc_score(y_te_f, tp)), "pr_auc": float(average_precision_score(y_te_f, tp)), "f1": float(f1_score(y_te_f, (tp >= th).astype(int), zero_division=0))})

        # IceUNet (patch mean_prob, seed 42)
        torch.manual_seed(42)
        np.random.seed(42)
        tr_seg = DataLoader(SegPatchDataset(X_tr, M_tr_f, augment=True), batch_size=32, shuffle=True)
        va_seg = DataLoader(SegPatchDataset(X_va, M_va_f, augment=False), batch_size=64, shuffle=False)
        te_seg = DataLoader(SegPatchDataset(X_te, M_te_f, augment=False), batch_size=64, shuffle=False)
        umod = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        pw_px = min(float((M_tr_f == 0).sum() / max((M_tr_f == 1).sum(), 1)), 50.0)
        bce_px = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pw_px], device=device))
        dice_px = SoftDiceLoss()
        uopt = optim.AdamW(umod.parameters(), lr=1e-3, weight_decay=1e-4)
        uscaler = torch.amp.GradScaler("cuda")
        best_pr, best_ust, pat = -1.0, None, 0
        for ep in range(10):
            umod.train()
            for xb, mb in tr_seg:
                xb, mb = xb.to(device), mb.to(device)
                uopt.zero_grad()
                with torch.amp.autocast("cuda"):
                    lg = umod(xb)
                    loss = 0.5 * bce_px(lg, mb) + 0.5 * dice_px(lg, mb)
                uscaler.scale(loss).backward()
                uscaler.step(uopt)
                uscaler.update()
            umod.eval()
            vps = []
            with torch.no_grad():
                for xb, _ in va_seg:
                    with torch.amp.autocast("cuda"):
                        vps.append(torch.sigmoid(umod(xb.to(device))).cpu().numpy()[:, 0])
            vpr = float(average_precision_score(M_va_f.ravel(), np.concatenate(vps, axis=0).ravel()))
            if vpr > best_pr:
                best_pr, best_ust, pat = vpr, {k: v.cpu().clone() for k, v in umod.state_dict().items()}, 0
            else:
                pat += 1
                if pat >= 4:
                    break
        umod.load_state_dict({k: v.to(device) for k, v in best_ust.items()})
        vp_u = predict_unet_mean(umod, va_seg, device)
        tp_u = predict_unet_mean(umod, te_seg, device)
        th_u, _ = tune_threshold_on_val(y_va_f, vp_u)
        cv_rows.append({"fold": f_i+1, "model": "IceUNet_mean_prob", "prev": prev, "roc_auc": float(roc_auc_score(y_te_f, tp_u)), "pr_auc": float(average_precision_score(y_te_f, tp_u)), "f1": float(f1_score(y_te_f, (tp_u >= th_u).astype(int), zero_division=0))})

    out = {"per_strip_frozen": per_strip_table, "paired_subblock_bootstrap": paired_res, "five_fold_cv": cv_rows}
    with open(os.path.join(REPORTS_DIR, "block6_stats.json"), "w", encoding="utf-8") as jf:
        json.dump(out, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "block6_stats.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 6 Item 1: Per-Strip Metrics, 5-Fold Blocked-Strip CV, & Paired Sub-Block Bootstrap\n\n")
        mf.write("> **Statistical Protocol:** 3-strip bootstrap CIs (only 10 distinct resamples) are retired as ranking evidence. Primary comparison uses (1) per-test-strip metrics, (2) 5-fold blocked-strip CV across all 20 strips, and (3) paired bootstrap of metric differences over `256`-row sub-blocks with `64`-row guard gaps (*within-test-set variability only*). Models are separable only if the paired 95% CI excludes `0`.\n\n")

        mf.write("## 1. Per-Test-Strip Metrics (Frozen Split: Strip 0 [20.1%], Strip 8 [10.3%], Strip 15 [19.1%])\n\n")
        mf.write("| Model | Pooled ROC / PR / F1 | Strip 0 ROC / PR / F1 | Strip 8 ROC / PR / F1 | Strip 15 ROC / PR / F1 |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: |\n")
        for r in per_strip_table:
            s0, s8, s15 = r["strips"][0], r["strips"][8], r["strips"][15]
            mf.write(
                f"| **{r['model']}** | {r['pooled_roc']:.3f} / {r['pooled_pr']:.3f} / {r['pooled_f1']:.3f} | "
                f"{s0['roc_auc']:.3f} / {s0['pr_auc']:.3f} / {s0['f1']:.3f} | "
                f"{s8['roc_auc']:.3f} / {s8['pr_auc']:.3f} / {s8['f1']:.3f} | "
                f"{s15['roc_auc']:.3f} / {s15['pr_auc']:.3f} / {s15['f1']:.3f} |\n"
            )

        mf.write("\n## 2. Paired Sub-Block Bootstrap Differences (*Within-Test-Set Variability Only*, 256-Row Blocks + 64-Row Guard)\n\n")
        mf.write("| Model Pair (A - B) | Sub-Blocks | $\\Delta$ROC-AUC [95% CI] | ROC Separable? | $\\Delta$PR-AUC [95% CI] | PR Separable? |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for p in paired_res:
            mf.write(
                f"| {p['pair']} | {p['n_subblocks']} | "
                f"{p['delta_roc_auc']:+.4f} `[{p['delta_roc_ci'][0]:+.4f}, {p['delta_roc_ci'][1]:+.4f}]` | {'**YES**' if p['roc_separable'] else 'NO'} | "
                f"{p['delta_pr_auc']:+.4f} `[{p['delta_pr_ci'][0]:+.4f}, {p['delta_pr_ci'][1]:+.4f}]` | {'**YES**' if p['pr_separable'] else 'NO'} |\n"
            )

        mf.write("\n## 3. 5-Fold Blocked-Strip CV Across All 20 Strips (256-Row Buffer)\n\n")
        mf.write("| Model | Mean ROC-AUC ± Std | Mean PR-AUC ± Std | Mean F1 (Val Th) ± Std |\n")
        mf.write("| :--- | :---: | :---: | :---: |\n")
        for mname in ["LogisticRegression", "CustomCNN", "ResNet_scratch", "IceUNet_mean_prob"]:
            sub = [r for r in cv_rows if r["model"] == mname]
            mf.write(
                f"| **{mname}** | {np.mean([r['roc_auc'] for r in sub]):.4f} ± {np.std([r['roc_auc'] for r in sub]):.4f} | "
                f"{np.mean([r['pr_auc'] for r in sub]):.4f} ± {np.std([r['pr_auc'] for r in sub]):.4f} | "
                f"{np.mean([r['f1'] for r in sub]):.4f} ± {np.std([r['f1'] for r in sub]):.4f} |\n"
            )

        mf.write("\n### Per-Fold Breakdown\n\n")
        mf.write("| Fold | Prev | Model | ROC-AUC | PR-AUC | F1 |\n")
        mf.write("| :---: | :---: | :--- | :---: | :---: | :---: |\n")
        for r in cv_rows:
            mf.write(f"| {r['fold']} | {100*r['prev']:.1f}% | {r['model']} | {r['roc_auc']:.4f} | {r['pr_auc']:.4f} | {r['f1']:.4f} |\n")

    print("Saved reports/real/block6_stats.json and reports/real/block6_stats.md")


if __name__ == "__main__":
    main()
