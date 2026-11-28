"""
scripts/analysis/verify_tps_georef.py
=====================================
Amendment C: Verify Thin-Plate-Spline (TPS) georeferencing using the 676 embedded
GCPs in the Chandrayaan-2 DFSAR L1A SLI GeoTIFF (GCS_MOON, radius 1,737,400 m)
warped into Moon South Polar Stereographic CRS.

Note: Raw GDAL GCPTransformer(tps=True) is ill-conditioned on 319812x116 strips
(~2757:1 aspect ratio). Normalizing (row/H, col/W) to [0, 1]^2 before fitting a
Thin-Plate-Spline RBF (scipy.interpolate.RBFInterpolator, kernel='thin_plate_spline')
in Moon South Polar Stereographic metres yields stable sub-metre accuracy.

Samples 5 well-spaced GCPs (both in-fit and held-out from the TPS fit) and reports
max residual in metres. Writes reports/real/georeferencing_audit.md.
"""

import json
import os
import numpy as np
import rasterio
from rasterio.warp import transform as crs_transform
from scipy.interpolate import RBFInterpolator

SLI_HH = "data/raw/sar_nxcl/data/calibrated/20260308/ch2_sar_ncxl_20260308t062017233_d_sli_xx_fp_hh_d18.tif"
MOON_LONLAT_PROJ4 = "+proj=longlat +R=1737400 +no_defs"
MOON_STEREO_PROJ4 = "+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs"
OUT_MD = "reports/real/georeferencing_audit.md"
OUT_JSON = "reports/real/georef_summary.json"


def main():
    with rasterio.open(SLI_HH) as ds:
        gcps, _ = ds.get_gcps()
        H, W = ds.height, ds.width

    assert len(gcps) > 0, "No GCPs found in SLI GeoTIFF!"
    print(f"Loaded {len(gcps)} GCPs from {os.path.basename(SLI_HH)} (H={H}, W={W})")

    rc_norm = np.array([[g.row / H, g.col / W] for g in gcps], dtype=np.float64)
    lons = [g.x for g in gcps]
    lats = [g.y for g in gcps]
    xs, ys = crs_transform(MOON_LONLAT_PROJ4, MOON_STEREO_PROJ4, lons, lats)
    xy_m = np.column_stack([xs, ys])

    # Sample 5 interior/well-spaced GCPs
    sample_idxs = np.linspace(10, len(gcps) - 11, 5, dtype=int)
    holdout_mask = np.ones(len(gcps), dtype=bool)
    holdout_mask[sample_idxs] = False

    # Fit TPS on all 676 GCPs (in-fit) and on 671 GCPs (5-GCP holdout)
    tps_full = RBFInterpolator(rc_norm, xy_m, kernel="thin_plate_spline", degree=1)
    tps_hold = RBFInterpolator(rc_norm[holdout_mask], xy_m[holdout_mask], kernel="thin_plate_spline", degree=1)

    pred_full = tps_full(rc_norm[sample_idxs])
    pred_hold = tps_hold(rc_norm[sample_idxs])

    rows_out = []
    for k, idx in enumerate(sample_idxs):
        g = gcps[idx]
        err_full = float(np.hypot(pred_full[k, 0] - xy_m[idx, 0], pred_full[k, 1] - xy_m[idx, 1]))
        err_hold = float(np.hypot(pred_hold[k, 0] - xy_m[idx, 0], pred_hold[k, 1] - xy_m[idx, 1]))
        rows_out.append({
            "gcp_id": g.id,
            "row": g.row,
            "col": g.col,
            "lon": g.x,
            "lat": g.y,
            "true_x": xy_m[idx, 0],
            "true_y": xy_m[idx, 1],
            "err_full_m": err_full,
            "err_hold_m": err_hold,
        })

    max_full = max(r["err_full_m"] for r in rows_out)
    max_hold = max(r["err_hold_m"] for r in rows_out)
    print(f"5-GCP TPS verification: max in-fit residual = {max_full:.6f} m | max hold-out residual = {max_hold:.4f} m")

    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("# Georeferencing Audit (Thin-Plate-Spline Warp on 676 GCPs)\n\n")
        f.write(f"- **Source Raster:** `{os.path.basename(SLI_HH)}` (`{H} × {W}` slant-range pixels)\n")
        f.write("- **Native Transform:** Identity (no affine map projection in SLI)\n")
        f.write(f"- **GCP Count & CRS:** `{len(gcps)}` embedded `GroundControlPoint`s in `GCS_MOON` (`R = 1,737,400 m`)\n")
        f.write(f"- **Target Projected CRS:** Moon South Polar Stereographic (`{MOON_STEREO_PROJ4}`)\n")
        f.write("- **Warp Method:** Normalized-coordinate Thin-Plate-Spline (`RBFInterpolator(..., kernel='thin_plate_spline')`, resolving raw-pixel conditioning over `319812 × 116`)\n\n")
        f.write("## 5-GCP Residual Verification\n\n")
        f.write("| GCP ID | Row | Col | GCP Lon (°) | GCP Lat (°) | True Stereo X (m) | True Stereo Y (m) | In-Fit Residual (m) | Hold-Out Residual (m) |\n")
        f.write("| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")
        for r in rows_out:
            f.write(
                f"| {r['gcp_id']} | {r['row']:.0f} | {r['col']:.0f} | "
                f"{r['lon']:.6f} | {r['lat']:.6f} | "
                f"{r['true_x']:.2f} | {r['true_y']:.2f} | "
                f"{r['err_full_m']:.6f} | {r['err_hold_m']:.4f} |\n"
            )
        f.write(f"\n- **Max In-Fit Residual (5 GCPs):** `{max_full:.6f} m`\n")
        f.write(f"- **Max Hold-Out Residual (5 GCPs):** `{max_hold:.4f} m`\n\n")
        f.write("## Limitations\n\n")
        f.write(
            "- **Anisotropic pixels (`0.5 m` azimuth × `9.6 m` range):** Slant-range SLI pixels are "
            "~19× finer along-track than across-track (`output_line_spacing = 0.504866 m`, "
            "`output_pixel_spacing = 9.593359 m`); TPS warping to isotropic stereographic pixels "
            "interpolates across-track.\n"
        )
        f.write("- **Single scene, strip-held-out generalization only:** All evaluations and exported maps come from one DFSAR strip (`ch2_sar_ncxl_20260308t062017233`).\n")

    print(f"Saved {OUT_MD}")

    existing = json.load(open(OUT_JSON, "r", encoding="utf-8")) if os.path.isfile(OUT_JSON) else {}
    existing.update({
        "source_rows": int(H),
        "source_cols": int(W),
        "n_gcps": int(len(gcps)),
        "moon_radius_m": 1737400,
        "azimuth_spacing_m": 0.504866,
        "range_spacing_m": 9.593359,
        "five_gcp_sample_count": int(len(rows_out)),
        "five_gcp_max_infit_m": float(max_full),
        "five_gcp_max_holdout_m": float(max_hold),
        "five_gcp_rows": rows_out,
    })
    with open(OUT_JSON, "w", encoding="utf-8") as jf:
        json.dump(existing, jf, indent=2)
    print(f"Saved {OUT_JSON}")


if __name__ == "__main__":
    main()
