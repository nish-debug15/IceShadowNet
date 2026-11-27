# IceShadowNet 2.0: Research Report

**Detection, Pixel-Level Localization, and Statistical Evaluation of Lunar South Polar Radar Anomalies in Chandrayaan-2 DFSAR L-Band Quad-Pol Data**

---

## Abstract

We evaluate deep learning for detecting anomalous polarimetric radar signatures ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$ on $7 \times 7$ multilooked data) in a single Chandrayaan-2 DFSAR Level-1A slant-range strip (`ch2_sar_ncxl_20260308t062017233`, rows `90,624–319,812`, `13,886` patches with `256`-row azimuth guard buffers; anisotropic `0.505 m × 9.593 m` pixels, strip-held-out generalization only). The deterministic split (`val = [2, 6, 17]`, `test = [0, 8, 15]`) used label prevalence constraints with zero model feedback. Under an intensity-only (`[HH, HV, VV]`) anti-circularity policy, four empirical findings emerge: (1) on patch-level classification, `CustomCNN` (`0.8124 ± 0.0408` 5-fold CV ROC-AUC) is not separable from `LogisticRegression` (`0.8118 ± 0.0460`; within-test-set paired `ΔROC-AUC = +0.0228 [-0.0109, +0.0546]`); (2) `ResNet_pretrained` underperforms, consistent across seeds, and is not separable under strip CV, while `ResNet_SSL` (`0.7892` at 10% labels) beats `ResNet_scratch` (`0.5250`) **but does not beat `LogisticRegression` (`0.8036`) or `CustomCNN` (`0.8080`)**; (3) pixel-level `IceUNet` improves localization over context-free baselines (`0.0093` PR-AUC / `17.6×` lift, `0.0110` / `20.9×` as a 3-seed ensemble vs. `0.0012` for `1×1` LogReg), though context-matched `7×7+15×15` `HistGBDT` already achieves `0.0077` PR-AUC (`14.6×` lift), and low Dice (`~0.03`; `0.0348` even with a circular 4-channel `|HH*conj(VV)|` oracle) reflects extreme `0.0527%` imbalance, per-patch normalization, and complex phase dependence; and (4) normalized TPS georeferencing over `676` `GCS_MOON` GCPs yields `367.98 m` mean (`307.38 m` median, `2,377.70 m` max) 10-fold CV residual and `92.79%` footprint overlap with the official `SRI HH` product.

---

## 1. Statistical Comparison & Negative Results

### 1.1 Per-Test-Strip Metrics, 5-Fold CV, & Paired Sub-Block Bootstrap
Retiring 3-strip bootstrap CIs (only 10 distinct resamples), primary comparisons use per-strip metrics (`Strip 0 / 8 / 15`), 5-fold blocked-strip CV across all 20 strips (`CustomCNN 0.8124`, `LogisticRegression 0.8118`, `ResNet_scratch 0.7978`, `IceUNet_mean_prob 0.7955`), and `104`-sub-block paired bootstrap differences (*within-test-set variability only*): `CustomCNN` vs `LogisticRegression` (`+0.0228 [-0.0109, +0.0546]`), `CustomCNN` vs `ResNet_scratch` (`+0.0032 [-0.0084, +0.0183]`), and `IceUNet_mean_prob` vs `CustomCNN` (`+0.0020 [-0.0092, +0.0108]`) all cross zero and are **not separable**. Only `CustomCNN - ResNet_pretrained` (`+0.0288 [+0.0077, +0.0558]`) and `ResNet_scratch - ResNet_pretrained` (`+0.0256 [+0.0053, +0.0452]`) exclude zero on within-test-set ROC-AUC.

### 1.2 ImageNet Underperformance & Label-Efficiency Fairness
`ResNet_pretrained` underperforms, consistent across seeds, and is not separable under strip CV. `ResNet_SSL` (`11,868` unlabeled Train+Val patches) improves over `ResNet_scratch` at 10–50% labels (`0.7892` vs `0.5250` at 10%), **but does not beat `LogisticRegression` (`0.8036`) or `CustomCNN` (`0.8080`)**.

### 1.3 Pixel Localization vs. Context-Matched Baselines & Phase Oracle
`IceUNet` improves localization over context-free baselines (`0.0093` PR-AUC, `17.6×` lift; `0.0110`, `20.9×` ensemble vs `0.0012` for `1×1` LogReg). However, context-matched `7×7+15×15` `HistGBDT` (`0.0077` PR-AUC, `14.6×` lift, `0.0229` Dice) and `LogReg` (`0.0067` PR-AUC, `12.7×` lift, `0.0189` Dice) recover most of that gain from local spatial statistics alone. Optuna (`12` val-only trials; single test run `0.8497` patch ROC-AUC) is reported as a single run and excluded from ranking. In the circular 4-channel phase-oracle diagnostic (`|HH*conj(VV)|` coherence, 1 seed), Pixel PR-AUC rises `+43%` (`0.0093 → 0.0133`, `25.2×` lift), while Dice edges from `0.0295` to `0.0348`.

### 1.4 Uncertainty, Selective Prediction, & Pruning
- **Calibration:** `ECE` is `0.04307` (MC) / `0.04794` (Ensemble), **but Brier Skill Score is negative (`-13.07` / `-14.64`)** due to `pos_weight=50` + Dice loss probability inflation (`0.2656–0.2886` mean $p$ on positives). MC entropy tracks $p$ near $0.5$ (`GT Interior` highest at `0.51350`, `1–2 px` ring `0.46635`, background `0.14280`).
- **Selective Prediction (Unified 3-Seed Ensemble Mean Predictor):** Neither Ensemble Variance (`AURC 0.1259`, `ErrDet-AUC 0.7879`, `50%` Patch AUC `0.7759`) nor MC Entropy (`AURC 0.1096`, `ErrDet-AUC 0.8282`, `50%` Patch AUC `0.7616`) improves Patch ROC-AUC over full coverage (`0.8456`), and `|p - 0.5|` matches or beats both (`AURC 0.1084`, `ErrDet-AUC 0.8310`). **Uncertainty is not usable for selective rejection on this task.**
- **Pruning (Unified Median of 200 Runs, Warmup 50, CUDA Sync):** Unstructured L1 (`90%`: `110,521` NZ params, `B=1: 2.208 ms`, `B=64: 10.939 ms`) is parameter reduction only. Structured channel pruning at `30%` (`544,834` params, `B=64: 11.441 ms` vs `10.860 ms`) gives **no speedup**; `50%` (`271,909` params) achieves `B=64: 5.359 ms` (`B=1: 2.058 ms`).

---

## 2. Georeferencing & Limitations

- **10-Fold GCP Cross-Validation (`676` GCPs):** `367.98 m` mean, `307.38 m` median, `2,377.70 m` max hold-out residual (alongside `281.7 m` max on 5 interior GCPs).
- **Overlap Metric Definition:** The `92.79%` overlap metric is the percentage of valid TPS-warped test-strip pixels (`61,857` of `66,664`) whose nearest-neighbor cell on the official `25 m/px` ISRO `SRI HH` grid has positive finite backscatter (`SRI HH > 0`).
- **Limitations:** Anisotropic slant-range pixels (`0.505 m` azimuth × `9.593 m` range); single scene, strip-held-out generalization only; proxy labels (`CPR > 1.0 & DOP < 0.13`) do not confirm physical water ice.
