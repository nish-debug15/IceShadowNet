# Data Audit — Chandrayaan-2 DFSAR + OHRC Real Tiles
_Generated: 2026-10-18 (Block 1 of Phase 5)_

---

## SUMMARY (read this first)

| Check | Result |
|---|---|
| **DFSAR scene hemisphere** | **SOUTH POLAR** ✅ (centre lat −84.17°, deepest corner −86.35°) |
| **OHRC scene hemisphere** | **SOUTH POLAR** ✅ (centre lat −85.20° approx, all corners negative) |
| **Note: "ncp" in OHRC filename** | **Misleading folder name** — XML label explicitly states `area = South Pole` |
| **CPR / DOP computable from DFSAR?** | **YES** — 4 linear polarizations HH, HV, VH, VV present in the L2-SELENOREF (SRI) product |
| **OHRC ↔ DFSAR geographic overlap** | **NO OVERLAP** ⛔ — SAR lon 8°–57°E, OHRC lon 92°–96°E; latitudes overlap but longitudes do not |
| **Contains / neighbours doubly-shadowed crater** | Cannot determine — no shadow map; sun elevation in OHRC = 4.14°, solar incidence 85.86°, so shadowing is plausible |

### BLOCKERS

> ⛔ **BLOCKER: No geographic overlap between DFSAR and OHRC tiles.**
> DFSAR centre is at (−84.17°, 33.40°); OHRC centre is at (−85.25°, 93.85°).
> These scenes are ~2800 km apart on the lunar surface (at pole, 1° lon ≈ 30 km).
> **Multimodal fusion (DFSAR + OHRC) is NOT possible with these two tiles.**
> Block 3 will **not** be started.

**Action required (human decision):** Either (a) download a new OHRC tile whose lon range overlaps with the SAR strip (~8°–57°E, lat ~−81° to −86°), or (b) proceed SAR-only (DFSAR-only pipeline). See end of document.

---

## 1. Directory Structure and File Sizes

### `data/raw/ohr_ncp/` — OHRC tile (+ duplicate SAR files, see note)

> **⚠️ NOTE:** The `ohr_ncp/` folder contains SAR files from the `ch2_sar_ncxl` acquisition dated 20260308. This appears to be because the PRADAN zip for the OHRC product bundled the SAR data as a companion dataset. The canonical SAR files live in `sar_nxcl/` (same filenames, identical sizes — duplicates).

| File (relative to `data/raw/ohr_ncp/`) | Size (MB) |
|---|---|
| `ch2_ohr_ncp_20260719T0722178640_d_img_d18.img` | 1072.23 |
| `ch2_ohr_ncp_20260719T0722178640_d_img_d18.xml` | 0.01 |
| `ch2_ohr_ncp_20260719T0722178640_g_grd_d18.xml` | 0.01 |
| `ch2_ohr_ncp_20260719T0722178640_b_brw_d18.xml` | 0.00 |
| `browse/calibrated/20260308/ch2_sar_ncxl_…_b_brw_xx_fp_xx_d18.png` | 0.01 |
| `browse/calibrated/20260719/ch2_ohr_ncp_…_b_brw_d18.png` | 7.07 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_gri_xx_fp_hh_d18.tif` | 4.82 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_gri_xx_fp_hv_d18.tif` | 4.82 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_gri_xx_fp_vh_d18.tif` | 4.82 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_gri_xx_fp_vv_d18.tif` | 4.82 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sli_xx_fp_hh_d18.tif` | 285.51 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sli_xx_fp_hv_d18.tif` | 285.51 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sli_xx_fp_vh_d18.tif` | 285.51 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sli_xx_fp_vv_d18.tif` | 285.51 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sri_xx_fp_hh_d18.tif` | 34.87 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sri_xx_fp_hv_d18.tif` | 34.87 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sri_xx_fp_vh_d18.tif` | 34.87 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sri_xx_fp_vv_d18.tif` | 34.87 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sri_in_fp_xx_d18.tif` | 69.71 |
| `data/calibrated/20260308/ch2_sar_ncxl_…_d_sri_ma_fp_xx_d18.tif` | 17.45 |
| geometry/ CSVs | 53.45 + 7.32 + 2.19 + 0.06 |
| miscellaneous/ (.lbr, .oat, .spm) | 0.16 + 0.40 + 0.16 |

### `data/raw/sar_nxcl/` — DFSAR canonical copy

Same files as the SAR content in `ohr_ncp/` — same filenames, same sizes. Exact duplicates stored separately.

| Product sub-type | Files | Size each |
|---|---|---|
| GRI (Ground Range Intensity) | hh, hv, vh, vv + incidence + xml | 4.82 MB × 4 + 9.47 + 0.02 |
| SLI (Slant Range Intensity, L1A) | hh, hv, vh, vv + xml | 285.51 MB × 4 |
| SRI (Seleno-Referenced Intensity, L2) | hh, hv, vh, vv + mask + incidence + xml | 34.87 MB × 4 + 17.45 + 69.71 |

---

## 2. PDS4 Label Extraction

### 2a. DFSAR (ch2_sar_ncxl_20260308t062017233)

| Field | SRI Product | SLI Product | GRI Product |
|---|---|---|---|
| **Product level** | L2-SELENOREF (calibrated, seleno-georef) | L1A-SLANT-RANGE (calibrated) | L2-ish (seleno-tagged ground range) |
| **Instrument** | Dual frequency L and S band SAR | same | same |
| **Frequency band** | L (1.25 GHz) | L (1.25 GHz) | L (1.25 GHz) |
| **Polarizations** | HH, HV, VH, VV (4 channels) | HH, HV, VH, VV | HH, HV, VH, VV |
| **Pixel dimensions (SRI)** | 3215 rows × 5682 cols | 3215 × 285.51 MB raw | 16834 × 145 |
| **Output pixel spacing** | 25 m/pixel (SRI) | 9.59 m/pixel (SLI) | — |
| **Data type** | UnsignedLSB2 (uint16) | UnsignedLSB2 (uint16) | UnsignedLSB2 |
| **Nodata** | None specified | None specified | None specified |
| **CRS** | Polar Stereographic Moon (lat_origin=−90) | None (slant range) | None |
| **Acquisition date** | 2026-03-08T06:20:17Z – 06:22:02Z | same | same |
| **Centre lat/lon** | −84.168°, 33.403° | −84.058°, 32.230° | −84.058°, 32.248° |
| **Corner lats** | UL −83.747°, UR −81.691°, LR −83.415°, LL −86.348° | similar | similar |
| **Corner lons** | UL 8.003°E, UR 41.874°E, LR 57.441°E, LL 13.800°E | similar | similar |

**DFSAR CPR / DOP analysis:**
- **4 linear polarization channels available: HH, HV, VH, VV**
- This is a **full-polarimetric (quad-pol) L-band product**, NOT compact polarimetry
- CPR (Circular Polarization Ratio) can be computed as:
  - CPR = (HV + VH) / HH *(same-sense / opposite-sense for linear pol proxy)*
  - Or more rigorously: need Stokes parameters S1–S4. From quad-pol: S1 = HH² + HV² + VH² + VV², S2 = HH² − HV² + VH² − VV², etc.
  - Simpler proxy used in lunar ice literature: **CPR ≈ (HH + VV) / (HV + VH)** (co-pol / cross-pol ratio)
- DOP = degree of polarization, can be estimated from quad-pol data
- **ANSWER: YES — CPR and DOP CAN be computed** from this quad-pol product

### 2b. OHRC (ch2_ohr_ncp_20260719T0722178640)

| Field | Value |
|---|---|
| **Product level** | Calibrated (L1-equivalent, radiometrically corrected) |
| **Instrument** | Orbiter High Resolution Camera (OHRC) — panchromatic 500–800 nm |
| **Bands** | 1 (panchromatic) |
| **Polarization** | N/A (optical) |
| **Pixel dimensions** | 93693 rows × 12000 cols |
| **Resolution** | 0.24 m/pixel |
| **Data type** | UnsignedByte (uint8) |
| **Nodata** | None specified |
| **CRS** | None (raw image geometry; projection = Polar stereographic stated in XML but not embedded in GeoTIFF) |
| **Projection** | Polar stereographic, South Pole (stated in Mission_Area XML) |
| **Acquisition date** | 2026-07-19T07:22:17.864Z – 07:22:34.248Z |
| **Area (XML explicit)** | **South Pole** |
| **Corner lats (refined)** | UL −84.933°, UR −84.960°, LL −85.735°, LR −85.766° |
| **Corner lons (refined)** | UL 95.034°E, UR 96.167°E, LL 91.974°E, LR 93.299°E |
| **Spacecraft altitude** | 94.57 km |
| **TDI stages** | TDI64 |

> **"ncp" in filename = NOT North Circular Polar.** The XML Mission_Area tag `<isda:area>South Pole</isda:area>` is unambiguous. All corner latitudes are −84° to −86°. This is a South Polar scene.

---

## 3. verify_real_tile.py Output

### DFSAR SRI — HH channel
```
Verifying tile: data/raw/sar_nxcl/data/calibrated/20260308/ch2_sar_ncxl_20260308t062017233_d_sri_xx_fp_hh_d18.tif
  Shape      : (3215, 5682)
  Dtype      : uint16
  CRS        : PROJCS["POLARSTEREOGRAPHIC MOON",...,PROJECTION["Polar_Stereographic"],PARAMETER["latitude_of_origin",-90],...,UNIT["metre",1]]
  Bands      : 1
  NoData     : (None,)
  Band 1     : Min=0.000, Max=18369.000, Mean=9.848
Verification SUCCESS.
```

### DFSAR SRI — HV channel
```
Verifying tile: data/raw/sar_nxcl/data/calibrated/20260308/ch2_sar_ncxl_20260308t062017233_d_sri_xx_fp_hv_d18.tif
  Shape      : (3215, 5682)
  Dtype      : uint16
  CRS        : PROJCS["POLARSTEREOGRAPHIC MOON",...] 
  Bands      : 1
  NoData     : (None,)
  Band 1     : Min=0.000, Max=1093.000, Mean=3.603
Verification SUCCESS.
```

### DFSAR SRI — VH channel
```
Verifying tile: data/raw/sar_nxcl/data/calibrated/20260308/ch2_sar_ncxl_20260308t062017233_d_sri_xx_fp_vh_d18.tif
  Shape      : (3215, 5682)
  Dtype      : uint16
  CRS        : PROJCS["POLARSTEREOGRAPHIC MOON",...]
  Bands      : 1
  NoData     : (None,)
  Band 1     : Min=0.000, Max=1115.000, Mean=3.431
Verification SUCCESS.
```

### DFSAR SRI — VV channel
```
Verifying tile: data/raw/sar_nxcl/data/calibrated/20260308/ch2_sar_ncxl_20260308t062017233_d_sri_xx_fp_vv_d18.tif
  Shape      : (3215, 5682)
  Dtype      : uint16
  CRS        : PROJCS["POLARSTEREOGRAPHIC MOON",...]
  Bands      : 1
  NoData     : (None,)
  Band 1     : Min=0.000, Max=17255.000, Mean=9.856
Verification SUCCESS.
```

### DFSAR GRI — HH channel
```
Verifying tile: data/raw/sar_nxcl/data/calibrated/20260308/ch2_sar_ncxl_20260308t062017233_d_gri_xx_fp_hh_d18.tif
  Shape      : (16834, 145)
  Dtype      : uint16
  CRS        : None
  Bands      : 1
  NoData     : (None,)
  Band 1     : Min=0.000, Max=19889.000, Mean=254.443
Verification SUCCESS.
```

### OHRC .img
```
Verifying tile: data/raw/ohr_ncp/data/calibrated/20260719/ch2_ohr_ncp_20260719T0722178640_d_img_d18.img
FATAL: Failed to open or parse tile. Error: '...d_img_d18.img' not recognized as being in a supported file format.
```
> **Note:** OHRC `.img` cannot be opened directly by rasterio. Opening the companion `.xml` (PDS4 label) works:
> - `driver=PDS4, dtype=uint8, width=12000, height=93693, count=1, CRS=None`
> - Raw numpy read: Min=0, Max=255, Mean=42.52 ✅
> - `verify_real_tile.py` needs to be updated to try the `.xml` companion when `.img` fails. (Logged as improvement, not blocking.)

---

## 4. Polarimetric Channel Analysis for DFSAR

**Product type:** Quad-polarimetric (HH, HV, VH, VV) — this is full-polarimetry, NOT compact polarimetry.

**CPR computation:**
- Lunar surface ice detection uses: CPR = backscatter_same_sense / backscatter_opposite_sense
- For quad-pol (linear basis): co-pol ratio = (σ°_HH + σ°_VV) / 2 and cross-pol = (σ°_HV + σ°_VH) / 2
- CPR ≈ co-pol / cross-pol: `(HH² + VV²) / (HV² + VH²)` (using intensity values)
- **CPR IS COMPUTABLE. YES.**

**DOP computation:**
- From Stokes: DOP = sqrt(S2² + S3² + S4²) / S1
- From quad-pol intensities (approximation): DOP can be estimated
- **DOP IS COMPUTABLE. YES.**

**CPR > 1 threshold:** Standard for anomalous backscatter (ice proxy). Applicable.
**DOP < 0.13 threshold:** From Heggy et al. — applicable to L-band quad-pol.

---

## 5. Geographic Overlap Assessment

### Bounding boxes

| Scene | Lat range | Lon range |
|---|---|---|
| DFSAR SRI | −86.35° to −81.69° | 8.00°E to 57.44°E |
| OHRC | −85.77° to −84.93° | 91.97°E to 96.17°E |

### Overlap result

```
Latitude ranges overlap : True  (both in −82° to −86° band)
Longitude ranges overlap: False (SAR ~8°–57°E, OHRC ~92°–96°E)
Geographic OVERLAP      : FALSE ⛔
```

The two scenes are separated by approximately **~35° of longitude at this latitude**, corresponding to roughly **~1800 km** on the lunar surface. There is **no overlap**.

---

## 6. Doubly-Shadowed Crater Assessment

Cannot determine from the data alone:
- OHRC scene: sun elevation = 4.14°, solar incidence = 85.86° — oblique illumination, severe shadowing likely in crater walls. Cannot identify a specific PSR (Permanently Shadowed Region) without a DEM.
- DFSAR: SAR penetrates shadows, but crater identification requires comparison with a crater catalog.
- Statement: **Cannot determine** whether a doubly-shadowed crater is present in either scene.

---

## 7. STOP CONDITION Evaluation

Per the master prompt:

> "if the SUMMARY shows any of: scene is north polar ❌ | no polarimetric channels ❌ | no overlap AND multimodal fusion is planned ✅ → DO NOT start Block 3."

**STOP CONDITION TRIGGERED: No geographic overlap, and multimodal fusion is planned.**

**Block 3 will NOT be executed.**

---

## 8. Options for the Human

To proceed with multimodal fusion (DFSAR + OHRC), you need an OHRC tile whose longitude overlaps with the DFSAR strip (~8°–57°E, lat ~−82° to −86°):

1. **Download a new OHRC tile:** On PRADAN chmapbrowse, filter OHRC images with centre longitude ~30°E and latitude ~−84° to −85°. Acquisition date of 2026-03-08 (same orbit as the SAR) would be ideal.
2. **Proceed SAR-only:** Use the DFSAR quad-pol product alone (HH, HV, VH, VV → CPR + DOP labeling). No OHRC. This gives a simpler but still valid single-modality pipeline on real data. This is recommended if no matching OHRC tile is found quickly.
3. **Use a different DFSAR tile:** Download a DFSAR acquisition whose centre longitude is ~93°E to match the OHRC tile.

**Recommendation:** Option 2 (SAR-only with the existing tile) is fastest and unblocks Block 3 immediately. The multimodal aspect would need to be documented as "pending matching OHRC tile" in the report.
