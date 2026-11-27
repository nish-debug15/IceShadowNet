# Geospatial Map Exports (`reports/real/maps/`)

Generated from `IceUNet` MC Dropout (`T=20`) predictions stitched over held-out test strips `[0, 8, 15]`:

1. `test_strips_prob_uncertainty_slant_range.tif` (gitignored, ~23 MB): 2-band float32 (`unet_mc_prob_mean`, `unet_mc_predictive_entropy`, NoData `-9999.0`) in radar slant-range `(azimuth, range)` coordinates with 468 shifted `GCS_MOON` GCPs.
2. `test_strips_prob_uncertainty_moon_stereo.tif` (gitignored, ~643 KB): 2-band float32 warped into Moon South Polar Stereographic (`25 m/px`, `R = 1,737,400 m`) via normalized Thin-Plate-Spline over 676 GCPs.

## Accuracy & Limitations

- **TPS Hold-Out Accuracy:** `~282 m` (`281.7 m` max across 5 held-out GCPs; `0.000 m` in-fit).
- **Overlay Check vs. Official ISRO SRI HH (`25 m/px`):** `92.79%` of `moon_stereo` test pixels fall inside valid `SRI` footprint; median swath-centerline lateral offset = `993.8 m` (`mean = 2382.0 m`).
- **Limitations:** Anisotropic source pixels (`0.505 m` azimuth × `9.593 m` range); single DFSAR scene; strip-held-out generalization only; proxy labels (`CPR > 1.0 & DOP < 0.13`) do not confirm water ice.
