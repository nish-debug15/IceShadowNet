"""
scripts/analysis/run_georef_10fold.py
=====================================
Block 6 Item 6: 10-Fold Cross-Validation over all 676 GCS_MOON GCPs for
Normalized Thin-Plate-Spline (TPS) georeferencing into Moon South Polar Stereographic,
reported alongside the 5-GCP hold-out test and the exact 1-sentence definition of
the 92.79% SRI overlap metric.

Outputs:
  - reports/real/georef_10fold.json
  - Appends 10-Fold CV + overlap definition to reports/real/georeferencing_audit.md
"""

import os
import json
import numpy as np
import rasterio
from rasterio.warp import transform as crs_transform
from scipy.interpolate import RBFInterpolator
from sklearn.model_selection import KFold

SLI_HH = "data/raw/sar_nxcl/data/calibrated/20260308/ch2_sar_ncxl_20260308t062017233_d_sli_xx_fp_hh_d18.tif"
MOON_LONLAT_PROJ4 = "+proj=longlat +R=1737400 +no_defs"
MOON_STEREO_PROJ4 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
OUT_JSON = "reports/real/georef_10fold.json"
AUDIT_MD = "reports/real/georeferencing_audit.md"


def main():
    with rasterio.open(SLI_HH) as ds:
        gcps, _ = ds.get_gcps()
        H, W = ds.height, ds.width

    rc_norm = np.array([[g.row / H, g.col / W] for g in gcps], dtype=np.float64)
    xs, ys = crs_transform(MOON_LONLAT_PROJ4, MOON_STEREO_PROJ4, [g.x for g in gcps], [g.y for g in gcps])
    xy_m = np.column_stack([xs, ys])

    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    all_errs = []
    fold_stats = []
    for f_i, (tr_idx, te_idx) in enumerate(kf.split(rc_norm), 1):
        tps = RBFInterpolator(rc_norm[tr_idx], xy_m[tr_idx], kernel="thin_plate_spline", degree=1)
        pred = tps(rc_norm[te_idx])
        errs = np.hypot(pred[:, 0] - xy_m[te_idx, 0], pred[:, 1] - xy_m[te_idx, 1])
        all_errs.extend(errs.tolist())
        fold_stats.append({
            "fold": f_i, "n_test": int(len(te_idx)),
            "mean_m": float(errs.mean()), "max_m": float(errs.max()),
        })

    all_errs = np.array(all_errs)
    res = {
        "n_gcps": len(gcps),
        "ten_fold_mean_m": float(all_errs.mean()),
        "ten_fold_median_m": float(np.median(all_errs)),
        "ten_fold_max_m": float(all_errs.max()),
        "folds": fold_stats,
        "overlap_definition": "The 92.79% overlap metric is the percentage of valid TPS-warped test-strip pixels (61,857 of 66,664) whose nearest-neighbor cell on the official 25 m/px ISRO SRI HH grid has positive finite backscatter (SRI HH > 0).",
    }
    with open(OUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(res, jf, indent=2)

    print(f"10-Fold CV over {len(gcps)} GCPs: mean={res['ten_fold_mean_m']:.2f} m, median={res['ten_fold_median_m']:.2f} m, max={res['ten_fold_max_m']:.2f} m")

    with open(AUDIT_MD, "r", encoding="utf-8") as af:
        existing = af.read()
    if "## 10-Fold GCP Cross-Validation" not in existing:
        with open(AUDIT_MD, "a", encoding="utf-8") as af:
            af.write("\n## 10-Fold GCP Cross-Validation (All 676 GCPs)\n\n")
            af.write(f"- **10-Fold Hold-Out Residual (Mean / Median / Max):** `{res['ten_fold_mean_m']:.2f} m` / `{res['ten_fold_median_m']:.2f} m` / `{res['ten_fold_max_m']:.2f} m` (alongside 5-GCP hold-out max `281.7249 m`)\n")
            af.write(f"- **Overlap Metric Definition:** {res['overlap_definition']}\n")


if __name__ == "__main__":
    main()
