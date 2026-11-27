"""
scripts/analysis/run_uncertainty_and_stitch.py
==============================================
Block 3a: Uncertainty Quantification (MC Dropout T=20 + 3-Seed Ensemble),
Calibration (ECE + Reliability Diagram), Selective Prediction (Dice/Acc vs Coverage),
Boundary Uncertainty Hypothesis Test, and Stitched Test-Strip TIFF Export.

Outputs:
  - reports/real/uncertainty_report.json
  - reports/real/uncertainty_report.md
  - reports/real/figures/unet_reliability.png
  - reports/real/figures/unet_selective_prediction.png
  - reports/real/maps/test_strips_prob_uncertainty_slant_range.tif
  - reports/real/maps/test_strips_prob_uncertainty_moon_stereo.tif
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from scipy.ndimage import binary_dilation
from scipy.interpolate import RBFInterpolator
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_origin
from rasterio.warp import transform as crs_transform
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.models.unet import IceUNet
from scripts.train_unet import SegPatchDataset, pixel_metrics, tune_pixel_threshold

PROCESSED_REAL = "data/processed_real"
MODELS_DIR = "models_real"
REPORTS_DIR = "reports/real"
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")
MAPS_DIR = os.path.join(REPORTS_DIR, "maps")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)

SLI_HH = "data/raw/sar_nxcl/data/calibrated/20260308/ch2_sar_ncxl_20260308t062017233_d_sli_xx_fp_hh_d18.tif"
MOON_LONLAT_PROJ4 = "+proj=longlat +R=1737400 +no_defs"
MOON_STEREO_PROJ4 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"


def enable_mc_dropout(model: nn.Module):
    """Set model to eval() so BatchNorm uses running stats, then enable Dropout2d."""
    model.eval()
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()


def binary_entropy(p: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    p_c = np.clip(p, eps, 1.0 - eps)
    return -(p_c * np.log(p_c) + (1.0 - p_c) * np.log(1.0 - p_c))


def compute_ece_and_plot(y_true: np.ndarray, probs: np.ndarray, title: str, out_png: str, n_bins: int = 15) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_accs, bin_confs, bin_counts = [], [], []
    ece = 0.0
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (probs >= lo) & (probs <= hi)
        else:
            m = (probs >= lo) & (probs < hi)
        cnt = int(m.sum())
        if cnt > 0:
            acc = float(y_true[m].mean())
            conf = float(probs[m].mean())
            ece += (cnt / n) * abs(acc - conf)
            bin_accs.append(acc)
            bin_confs.append(conf)
            bin_counts.append(cnt)
        else:
            bin_accs.append(np.nan)
            bin_confs.append(0.5 * (lo + hi))
            bin_counts.append(0)

    fig, ax = plt.subplots(figsize=(5.5, 5.0))
    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Perfect Calibration")
    valid = ~np.isnan(bin_accs)
    ax.plot(np.array(bin_confs)[valid], np.array(bin_accs)[valid], "o-", color="#2563eb", lw=2, label=f"ECE = {ece:.5f}")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Empirical Positive Fraction")
    ax.set_title(title)
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return float(ece)


def selective_prediction_curve(y_true: np.ndarray, p_mean: np.ndarray, uncertainty: np.ndarray, th: float, out_png: str):
    coverages = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1]
    order = np.argsort(uncertainty)  # lowest uncertainty first
    n = len(y_true)
    rows = []
    for cov in coverages:
        k = max(1, int(cov * n))
        idx = order[:k]
        yt = y_true[idx]
        yp = p_mean[idx]
        pred = (yp >= th).astype(np.uint8)
        tp = int(((pred == 1) & (yt == 1)).sum())
        fp = int(((pred == 1) & (yt == 0)).sum())
        fn = int(((pred == 0) & (yt == 1)).sum())
        dice = (2.0 * tp) / max(2 * tp + fp + fn, 1)
        acc = float((pred == yt).mean())
        rows.append({"coverage": cov, "n_px": k, "pos_px": int(yt.sum()), "dice": float(dice), "accuracy": acc})

    fig, ax1 = plt.subplots(figsize=(6.5, 4.5))
    covs = [r["coverage"] * 100 for r in rows]
    dices = [r["dice"] for r in rows]
    accs = [r["accuracy"] * 100 for r in rows]
    ax1.plot(covs, dices, "o-", color="#dc2626", lw=2, label="Dice Score")
    ax1.set_xlabel("Coverage (% lowest-uncertainty pixels retained)")
    ax1.set_ylabel("Dice Score", color="#dc2626")
    ax1.invert_xaxis()
    ax1.grid(alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(covs, accs, "s--", color="#2563eb", lw=2, label="Pixel Accuracy (%)")
    ax2.set_ylabel("Pixel Accuracy (%)", color="#2563eb")
    plt.title("Selective Prediction: Dice & Accuracy vs. Coverage (MC Dropout)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()
    return rows


def export_stitched_tiffs(coords_te: np.ndarray, mc_mean: np.ndarray, mc_ent: np.ndarray):
    # Active rows [90624, 319812] -> H_act = 229188, W = 116
    start_row = 90624
    H_act, W_act = 229188, 116
    P = 64

    prob_sum = np.zeros((H_act, W_act), dtype=np.float64)
    ent_sum = np.zeros((H_act, W_act), dtype=np.float64)
    cnt_map = np.zeros((H_act, W_act), dtype=np.float32)

    for idx, (r, c) in enumerate(coords_te):
        prob_sum[r:r+P, c:c+P] += mc_mean[idx]
        ent_sum[r:r+P, c:c+P] += mc_ent[idx]
        cnt_map[r:r+P, c:c+P] += 1.0

    valid = cnt_map > 0
    prob_raster = np.full((H_act, W_act), -9999.0, dtype=np.float32)
    ent_raster = np.full((H_act, W_act), -9999.0, dtype=np.float32)
    prob_raster[valid] = (prob_sum[valid] / cnt_map[valid]).astype(np.float32)
    ent_raster[valid] = (ent_sum[valid] / cnt_map[valid]).astype(np.float32)

    # Read original GCPs from SLI_HH and shift row by -start_row
    with rasterio.open(SLI_HH) as ds:
        orig_gcps, gcp_crs = ds.get_gcps()
        H_full, W_full = ds.height, ds.width

    shifted_gcps = [
        GroundControlPoint(row=g.row - start_row, col=g.col, x=g.x, y=g.y, z=g.z, id=g.id, info=g.info)
        for g in orig_gcps
        if 0 <= (g.row - start_row) <= H_act
    ]

    slant_tif = os.path.join(MAPS_DIR, "test_strips_prob_uncertainty_slant_range.tif")
    with rasterio.open(
        slant_tif, "w", driver="GTiff",
        height=H_act, width=W_act, count=2, dtype="float32",
        nodata=-9999.0, compress="deflate",
    ) as dst:
        dst.write(prob_raster, 1)
        dst.write(ent_raster, 2)
        dst.set_band_description(1, "unet_mc_prob_mean")
        dst.set_band_description(2, "unet_mc_predictive_entropy")
        dst.update_tags(
            COORDINATE_SYSTEM="Radar Slant-Range (azimuth x range); NOT map-projected affine",
            ACTIVE_ROWS=f"[{start_row}, {start_row + H_act}]",
            LIMITATION="Anisotropic pixels (0.5m azimuth x 9.6m range); single scene; strip-held-out generalization only",
        )
        if shifted_gcps:
            dst.gcps = (shifted_gcps, gcp_crs)
    print(f"Exported slant-range TIFF -> {slant_tif} ({len(shifted_gcps)} GCPs attached)")

    # TPS-warped Moon South Polar Stereographic GeoTIFF (25 m/px over test strip bounds)
    rc_norm = np.array([[g.row / H_full, g.col / W_full] for g in orig_gcps], dtype=np.float64)
    xs, ys = crs_transform(MOON_LONLAT_PROJ4, MOON_STEREO_PROJ4, [g.x for g in orig_gcps], [g.y for g in orig_gcps])
    xy_m = np.column_stack([xs, ys])
    tps_fwd = RBFInterpolator(rc_norm, xy_m, kernel="thin_plate_spline", degree=1)
    tps_inv = RBFInterpolator(xy_m, rc_norm, kernel="thin_plate_spline", degree=1)

    # Bounding box of valid test pixels in Moon Stereographic
    vr, vc = np.where(valid)
    sample_rc = np.column_stack([(vr[::50] + start_row) / H_full, vc[::50] / W_full])
    sample_xy = tps_fwd(sample_rc)
    xmin, ymin = float(sample_xy[:, 0].min()) - 100.0, float(sample_xy[:, 1].min()) - 100.0
    xmax, ymax = float(sample_xy[:, 0].max()) + 100.0, float(sample_xy[:, 1].max()) + 100.0

    res_m = 25.0
    out_w = int(np.ceil((xmax - xmin) / res_m))
    out_h = int(np.ceil((ymax - ymin) / res_m))
    gx = xmin + (np.arange(out_w) + 0.5) * res_m
    gy = ymax - (np.arange(out_h) + 0.5) * res_m
    GX, GY = np.meshgrid(gx, gy)
    rc_q = tps_inv(np.column_stack([GX.ravel(), GY.ravel()]))
    rr = np.rint(rc_q[:, 0] * H_full - start_row).astype(int)
    cc = np.rint(rc_q[:, 1] * W_full).astype(int)

    in_bounds = (rr >= 0) & (rr < H_act) & (cc >= 0) & (cc < W_act)
    stereo_prob = np.full(GX.size, -9999.0, dtype=np.float32)
    stereo_ent = np.full(GX.size, -9999.0, dtype=np.float32)
    valid_q = np.zeros(GX.size, dtype=bool)
    valid_q[in_bounds] = valid[rr[in_bounds], cc[in_bounds]]
    stereo_prob[valid_q] = prob_raster[rr[valid_q], cc[valid_q]]
    stereo_ent[valid_q] = ent_raster[rr[valid_q], cc[valid_q]]

    stereo_tif = os.path.join(MAPS_DIR, "test_strips_prob_uncertainty_moon_stereo.tif")
    with rasterio.open(
        stereo_tif, "w", driver="GTiff",
        height=out_h, width=out_w, count=2, dtype="float32",
        crs=MOON_STEREO_PROJ4, transform=from_origin(xmin, ymax, res_m, res_m),
        nodata=-9999.0, compress="deflate",
    ) as dst:
        dst.write(stereo_prob.reshape(out_h, out_w), 1)
        dst.write(stereo_ent.reshape(out_h, out_w), 2)
        dst.set_band_description(1, "unet_mc_prob_mean")
        dst.set_band_description(2, "unet_mc_predictive_entropy")
        dst.update_tags(
            WARP="Normalized Thin-Plate-Spline (676 GCPs)",
            RESOLUTION_M=str(res_m),
            LIMITATION="Anisotropic source pixels (0.5m x 9.6m); single scene; strip-held-out generalization only",
        )
    print(f"Exported Moon South Polar Stereographic GeoTIFF -> {stereo_tif} ({out_w}x{out_h} @ {res_m}m)")
    return slant_tif, stereo_tif, int(valid.sum())


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d_va = np.load(os.path.join(PROCESSED_REAL, "val.npz"))
    d_te = np.load(os.path.join(PROCESSED_REAL, "test.npz"))

    X_va, M_va = d_va["images"], d_va["masks"]
    X_te, M_te = d_te["images"], d_te["masks"]
    coords_te = d_te["coords"]

    va_loader = DataLoader(SegPatchDataset(X_va, M_va, augment=False), batch_size=64, shuffle=False)
    te_loader = DataLoader(SegPatchDataset(X_te, M_te, augment=False), batch_size=64, shuffle=False)

    # ── 1. MC Dropout (T=20) on IceUNet_seed42 ─────────────────────────────────
    print("Running MC Dropout (T=20) on IceUNet_seed42...")
    model_42 = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
    model_42.load_state_dict(torch.load(os.path.join(MODELS_DIR, "IceUNet_seed42.pth"), map_location=device))
    enable_mc_dropout(model_42)

    T = 20
    mc_sum = np.zeros_like(M_te, dtype=np.float64)
    mc_sq_sum = np.zeros_like(M_te, dtype=np.float64)
    with torch.no_grad():
        for t in range(T):
            preds = []
            for xb, _ in te_loader:
                with torch.amp.autocast("cuda"):
                    preds.append(torch.sigmoid(model_42(xb.to(device))).cpu().numpy()[:, 0])
            p_t = np.concatenate(preds, axis=0)
            mc_sum += p_t
            mc_sq_sum += p_t ** 2

    mc_mean = (mc_sum / T).astype(np.float32)
    mc_var = np.maximum(mc_sq_sum / T - mc_mean ** 2, 0.0).astype(np.float32)
    mc_ent = binary_entropy(mc_mean).astype(np.float32)

    # Deterministic val threshold for seed 42
    model_42.eval()
    va_det = []
    with torch.no_grad():
        for xb, _ in va_loader:
            with torch.amp.autocast("cuda"):
                va_det.append(torch.sigmoid(model_42(xb.to(device))).cpu().numpy()[:, 0])
    va_det = np.concatenate(va_det, axis=0)
    val_th, _ = tune_pixel_threshold(M_va.ravel(), va_det.ravel())

    # ── 2. 3-Seed Deep Ensemble (seeds 42, 123, 2024) ──────────────────────────
    print("Running 3-Seed Deep Ensemble (42, 123, 2024)...")
    ens_preds = []
    for s in [42, 123, 2024]:
        m = IceUNet(in_channels=3, base_ch=24, dropout_p=0.15).to(device)
        m.load_state_dict(torch.load(os.path.join(MODELS_DIR, f"IceUNet_seed{s}.pth"), map_location=device))
        m.eval()
        ps = []
        with torch.no_grad():
            for xb, _ in te_loader:
                with torch.amp.autocast("cuda"):
                    ps.append(torch.sigmoid(m(xb.to(device))).cpu().numpy()[:, 0])
        ens_preds.append(np.concatenate(ps, axis=0))
    ens_stack = np.stack(ens_preds, axis=0)  # (3, N, 64, 64)
    ens_mean = ens_stack.mean(axis=0)
    ens_var = ens_stack.var(axis=0)
    ens_ent = binary_entropy(ens_mean)

    mc_met = pixel_metrics(M_te.ravel(), mc_mean.ravel(), val_th)
    ens_met = pixel_metrics(M_te.ravel(), ens_mean.ravel(), val_th)

    # ── 3. ECE & Reliability Diagram ───────────────────────────────────────────
    ece_mc = compute_ece_and_plot(
        M_te.ravel(), mc_mean.ravel(),
        f"IceUNet MC Dropout (T={T}) Reliability Diagram",
        os.path.join(FIGURES_DIR, "unet_reliability.png"),
    )
    ece_ens = compute_ece_and_plot(
        M_te.ravel(), ens_mean.ravel(),
        "IceUNet 3-Seed Ensemble Reliability Diagram",
        os.path.join(FIGURES_DIR, "unet_ensemble_reliability.png"),
    )

    # ── 4. Selective Prediction (Dice & Acc vs Coverage) ───────────────────────
    sel_rows = selective_prediction_curve(
        M_te.ravel(), mc_mean.ravel(), mc_ent.ravel(), val_th,
        os.path.join(FIGURES_DIR, "unet_selective_prediction.png"),
    )

    # ── 5. Boundary Uncertainty Hypothesis Test (Test, Do Not Assume!) ─────────
    gt_interior = M_te == 1
    dilated = np.stack([binary_dilation(m.astype(bool), iterations=2) for m in M_te], axis=0)
    gt_boundary = dilated & (~gt_interior)
    far_bg = ~dilated

    boundary_stats = {}
    for reg_name, reg_mask in [("GT Interior", gt_interior), ("GT Boundary Ring (1-2px)", gt_boundary), ("Far Background", far_bg)]:
        boundary_stats[reg_name] = {
            "n_px": int(reg_mask.sum()),
            "mc_entropy_mean": float(mc_ent[reg_mask].mean()),
            "mc_var_mean": float(mc_var[reg_mask].mean()),
            "ens_entropy_mean": float(ens_ent[reg_mask].mean()),
            "ens_var_mean": float(ens_var[reg_mask].mean()),
        }

    # ── 6. Stitch Test-Strip TIFFs ─────────────────────────────────────────────
    slant_tif, stereo_tif, stitched_px = export_stitched_tiffs(coords_te, mc_mean, mc_ent)

    out_json = {
        "val_th": val_th,
        "mc_dropout": {**mc_met, "ece": ece_mc, "mean_entropy": float(mc_ent.mean()), "mean_var": float(mc_var.mean())},
        "deep_ensemble": {**ens_met, "ece": ece_ens, "mean_entropy": float(ens_ent.mean()), "mean_var": float(ens_var.mean())},
        "selective_prediction": sel_rows,
        "boundary_uncertainty_test": boundary_stats,
        "stitched_maps": {"slant_range_tif": slant_tif, "moon_stereo_tif": stereo_tif, "covered_pixels": stitched_px},
    }
    with open(os.path.join(REPORTS_DIR, "uncertainty_report.json"), "w", encoding="utf-8") as jf:
        json.dump(out_json, jf, indent=2)

    with open(os.path.join(REPORTS_DIR, "uncertainty_report.md"), "w", encoding="utf-8") as mf:
        mf.write("# Block 3a: Uncertainty Quantification & Geospatial Map Export\n\n")
        mf.write("## 1. MC Dropout (T=20) vs. 3-Seed Deep Ensemble (Test Pixels)\n\n")
        mf.write("| Method | Pixel PR-AUC | Pixel ROC-AUC | Dice (Val Th) | IoU (Val Th) | ECE | Mean Entropy | Mean Variance |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for label, d in [("MC Dropout (T=20)", out_json["mc_dropout"]), ("3-Seed Ensemble", out_json["deep_ensemble"])]:
            mf.write(
                f"| **{label}** | {d['pr_auc']:.4f} | {d['roc_auc']:.4f} | {d['dice']:.4f} | "
                f"{d['iou']:.4f} | {d['ece']:.5f} | {d['mean_entropy']:.5f} | {d['mean_var']:.6f} |\n"
            )

        mf.write("\n## 2. Boundary Uncertainty Test (Measured, Not Assumed)\n\n")
        mf.write("| Region | Pixel Count | MC Mean Entropy | MC Mean Variance | Ensemble Mean Entropy | Ensemble Mean Variance |\n")
        mf.write("| :--- | :---: | :---: | :---: | :---: | :---: |\n")
        for rname, st in boundary_stats.items():
            mf.write(
                f"| {rname} | {st['n_px']:,} | {st['mc_entropy_mean']:.5f} | "
                f"{st['mc_var_mean']:.6f} | {st['ens_entropy_mean']:.5f} | {st['ens_var_mean']:.6f} |\n"
            )

        mf.write("\n## 3. Selective Prediction (Dice & Accuracy vs. Coverage)\n\n")
        mf.write("| Coverage | Pixels Retained | Positive Pixels | Dice Score | Pixel Accuracy |\n")
        mf.write("| :---: | :---: | :---: | :---: | :---: |\n")
        for r in sel_rows:
            mf.write(f"| {100*r['coverage']:.0f}% | {r['n_px']:,} | {r['pos_px']:,} | {r['dice']:.4f} | {100*r['accuracy']:.3f}% |\n")

        mf.write("\n## 4. Stitched Test-Strip Maps (QGIS Ready)\n\n")
        mf.write(f"- **Slant-Range TIFF (azimuth × range + 676 shifted GCS_MOON GCPs):** `{slant_tif}`\n")
        mf.write(f"- **TPS-Warped Moon South Polar Stereographic GeoTIFF (25 m/px):** `{stereo_tif}`\n")
        mf.write(f"- **Stitched Test-Strip Coverage:** `{stitched_px:,}` pixels across test strips `[0, 8, 15]`\n\n")
        mf.write("![Reliability Diagram](figures/unet_reliability.png)\n\n")
        mf.write("![Selective Prediction](figures/unet_selective_prediction.png)\n")

    print("Saved reports/real/uncertainty_report.json and reports/real/uncertainty_report.md")


if __name__ == "__main__":
    main()
