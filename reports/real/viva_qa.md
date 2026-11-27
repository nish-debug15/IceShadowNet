# IceShadowNet 2.0 — Viva & Exam Defense Q&A

### Q1: Why use Deep Learning if Logistic Regression ties `CustomCNN` on patch-level ROC-AUC?
- **Measured fact:** On patch-level binary classification, `CustomCNN` (`0.8124 ± 0.0408`) and `LogisticRegression` (`0.8118 ± 0.0460`) tie across 5-fold blocked-strip CV (`reports/real/five_fold_cv.md`).
- **Why DL still wins here:**
  1. **Low-label regime:** With only 10% labeled patches, `ResNet_scratch` collapses (`0.51–0.54` ROC-AUC), whereas in-domain **SimCLR SSL (`ResNet_SSL`)** achieves `0.78–0.80` ROC-AUC (`reports/real/ssl_pretraining.md`).
  2. **Pixel-level localization:** Patch-level LR only outputs 1 scalar per `64×64` patch. At the pixel level, `Pixel LogReg` gets `0.0012` PR-AUC / `0.0016` Dice, whereas **`IceUNet` gets `0.0093` PR-AUC (`0.0110` ensemble) / `0.0266–0.0334` Dice and `0.9380–0.9518` Pixel ROC-AUC** (`reports/real/unet_segmentation_report.md`, `uncertainty_report.md`).

### Q2: Isn't predicting `CPR > 1.0 & DOP < 0.13` circular?
- **No, by strict input policy:** Models only see **raw log-intensities** (`log10|S_hh|^2, log10|S_hv|^2, log10|S_vv|^2`). Complex phase, `CPR`, and `DOP` are strictly excluded from inputs.
- **Ablation proof (`reports/real/circularity_ablation.md`):** Removing `HV` (`HH+VV` only) still achieves `~0.83` ROC-AUC on `CustomCNN`, showing the network learns spatial co-pol backscatter textures rather than trivially recomputing `HV/HH`.

### Q3: What is the "ambiguity band" and how did U-Net address it?
- **Problem:** `34.73%` of all valid patches (`4,960 / 14,282`) contain `1–4` anomalous pixels and are forced to label `0` under the `>=5 px` patch rule. `80.9%` (`250 / 309`) of `CustomCNN` false positives come from this band.
- **Solution:** `IceUNet` predicts per-pixel probabilities on the `64×64` grid without an arbitrary patch-count cutoff, and when aggregated via `mean_prob` (`0.8335`–`0.8497` ROC-AUC), matches or beats patch classifiers while preserving spatial localization.

### Q4: Do your predictions confirm water ice on the Moon?
- **No.** Labels are **polarimetric radar proxy labels** (`CPR > 1.0 & DOP < 0.13`, consistent with volume scattering / rough terrain / potential ice). Confirming physical water ice requires co-registering Permanently Shadowed Regions (PSRs), LOLA DEM slope/thermal models, and multi-scene overlap.

### Q5: What are the georeferencing & data-split limitations?
- **Georeferencing:** Native `SLI` has an identity transform (`0.505 m` azimuth × `9.593 m` range anisotropic pixels). We use `676` embedded `GCS_MOON` GCPs (`R = 1,737,400 m`) with normalized Thin-Plate-Spline warping (`0.000 m` in-fit, `281.7 m` 5-GCP hold-out residual in `reports/real/georeferencing_audit.md`).
- **Generalization:** Single Chandrayaan-2 DFSAR scene (`ch2_sar_ncxl_20260308t062017233`, rows `90,624–319,812`) with deterministic non-adjacent strip splitting and `256`-row azimuth guard buffers (`reports/real/split_rationale.md`).
