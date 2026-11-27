# Georeferencing Audit (Thin-Plate-Spline Warp on 676 GCPs)

- **Source Raster:** `ch2_sar_ncxl_20260308t062017233_d_sli_xx_fp_hh_d18.tif` (`319812 × 116` slant-range pixels)
- **Native Transform:** Identity (no affine map projection in SLI)
- **GCP Count & CRS:** `676` embedded `GroundControlPoint`s in `GCS_MOON` (`R = 1,737,400 m`)
- **Target Projected CRS:** Moon South Polar Stereographic (`+proj=stere +lat_0=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 +R=1737400 +units=m +no_defs`)
- **Warp Method:** Normalized-coordinate Thin-Plate-Spline (`RBFInterpolator(..., kernel='thin_plate_spline')`, resolving raw-pixel conditioning over `319812 × 116`)

## 5-GCP Residual Verification

| GCP ID | Row | Col | GCP Lon (°) | GCP Lat (°) | True Stereo X (m) | True Stereo Y (m) | In-Fit Residual (m) | Hold-Out Residual (m) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 11 | 0 | 46 | 48.693749 | -81.936958 | 183969.02 | 161656.30 | 0.000000 | 21.4265 |
| 174 | 76755 | 79 | 41.975599 | -83.020597 | 141722.36 | 157533.49 | 0.000000 | 18.0566 |
| 338 | 153510 | 115 | 32.990424 | -83.978605 | 99511.04 | 153289.65 | 0.000000 | 281.7249 |
| 502 | 243057 | 32 | 24.273051 | -85.231755 | 59472.85 | 131883.13 | 0.000000 | 253.6921 |
| 666 | 319811 | 70 | 14.353449 | -86.297685 | 27840.90 | 108800.04 | 0.000000 | 164.1513 |

- **Max In-Fit Residual (5 GCPs):** `0.000000 m`
- **Max Hold-Out Residual (5 GCPs):** `281.7249 m`

## Limitations

- **Anisotropic pixels (`0.5 m` azimuth × `9.6 m` range):** Slant-range SLI pixels are ~19× finer along-track than across-track (`output_line_spacing = 0.504866 m`, `output_pixel_spacing = 9.593359 m`); TPS warping to isotropic stereographic pixels interpolates across-track.
- **Single scene, strip-held-out generalization only:** All evaluations and exported maps come from one DFSAR strip (`ch2_sar_ncxl_20260308t062017233`).
