# IceShadowNet 2.0: Research Report

**Detection, Pixel-Level Localization, and Statistical Evaluation of Lunar South Polar Radar Anomalies in Chandrayaan-2 DFSAR L-Band Quad-Pol Data**

---

## Abstract

We evaluate deep learning for detecting anomalous polarimetric radar signatures ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$ on $7 \times 7$ multilooked data) in a single Chandrayaan-2 DFSAR Level-1A slant-range strip (`ch2_sar_ncxl_20260308t062017233`, rows `90,624–319,812`, `13,886` patches with `256`-row azimuth guard buffers; anisotropic `0.505 m × 9.593 m` pixels, strip-held-out generalization only). The deterministic split (`val = [2, 6, 17]`, `test = [0, 8, 15]`) used label prevalence constraints with zero model feedback. Under an intensity-only (`[HH, HV, VV]`) anti-circularity policy: (1) on patch classification, `CustomCNN` (`0.8124 ± 0.0408` 5-fold CV ROC-AUC) is not separable from `LogisticRegression` (`0.8118 ± 0.0460`; within-test-set paired `ΔROC-AUC = +0.0228 [-0.0109, +0.0546]`); (2) `ResNet_pretrained` underperforms (`0.7970 ± 0.0123` 3-seed mean ROC-AUC), consistent across seeds, and is not separable under strip CV, while `ResNet_SSL` (`0.7892` at `10%` labels) beats `ResNet_scratch` (`0.5250`) **but does not beat `LogisticRegression` (`0.8036`) or `CustomCNN` (`0.8080`)**; (3) pixel-level `IceUNet` improves localization over context-free baselines (`0.0093 ± 0.0003` PR-AUC / `17.6×` lift, `0.0110` / `20.9×` as a 3-seed ensemble vs. `0.0012` for `1×1` LogReg), though context-matched `7×7+15×15` `HistGBDT` (`0.0077` PR-AUC, `14.6×` lift, `0.0229` Dice) already recovers **`82.6%`** of single-seed `IceUNet` PR-AUC (`69.8%` of the ensemble), and a circular 3-seed diagnostic (`|HH*conj(VV)|` coherence) raises PR-AUC by `+17.9%` (`0.0097` → `0.0114`) while keeping Dice low (`0.0283` → `0.0330`); and (4) selective rejection using MC Dropout or Ensemble Variance does not improve retained Patch ROC-AUC over full coverage (`0.8456` → `0.7616` / `0.7759` at `50%` coverage vs `0.8491` Random), while normalized TPS georeferencing over `676` GCPs yields `307.38 m` median (`367.98 m` mean, `2,377.70 m` max boundary tail) 10-fold CV residual and `92.79%` `SRI HH` overlap.

---

## Statistical Comparison & Negative Results

### Per-Test-Strip Metrics, 5-Fold CV, & Paired Sub-Block Bootstrap
Retiring strip-level bootstrap CIs, primary comparisons evaluate per-strip performance (`Strip 0 / 8 / 15`), 5-fold blocked-strip CV across all `20` strips (`CustomCNN 0.8124 ± 0.0408`, `LogisticRegression 0.8118 ± 0.0460`, `ResNet_scratch 0.7978 ± 0.0427`, `IceUNet_mean_prob 0.7955 ± 0.0448`), and `104`-sub-block paired bootstrap differences (*within-test-set variability only*): `CustomCNN - LogisticRegression` (`+0.0228 [-0.0109, +0.0546]`), `CustomCNN - ResNet_scratch` (`+0.0032 [-0.0084, +0.0183]`), and `IceUNet_mean_prob - CustomCNN` (`+0.0020 [-0.0092, +0.0108]`) all include zero and are **not separable**. Only `CustomCNN - ResNet_pretrained` (`+0.0288 [+0.0077, +0.0558]`) and `ResNet_scratch - ResNet_pretrained` (`+0.0256 [+0.0053, +0.0452]`) exclude zero on within-test-set ROC-AUC.

### ImageNet Underperformance & Label-Efficiency Fairness
`ResNet_pretrained` underperforms (`0.7970 ± 0.0123` 3-seed mean; `0.815` pooled ensemble), consistent across seeds, and is not separable under strip CV. `ResNet_SSL` (`11,868` unlabeled Train+Val patches) improves over `ResNet_scratch` at `10%` labels (`0.7892` vs `0.5250`), **but does not beat `LogisticRegression` (`0.8036`) or `CustomCNN` (`0.8080`)**.

### Pixel Localization, Context-Matched Baselines, & Phase-Oracle Diagnostic
`IceUNet` outperforms context-free pixel baselines (`0.0093` / `17.6×` 3-seed mean and `0.0110` / `20.9×` ensemble vs `0.0012` / `2.3×` for `1×1` Pixel LogReg). Context-matched `7×7+15×15` `HistGBDT` (`0.0077` PR-AUC, `14.6×` lift, `0.0229` Dice) recovers **`82.6%`** of single-seed `IceUNet` PR-AUC (`69.8%` of the 3-seed ensemble), proving spatial context drives most of the gain while `IceUNet` retains a modest learned margin. Optuna (`12` val-only trials; single test run `0.8497` patch ROC-AUC and `0.0108` pixel PR-AUC) is reported separately as an unranked single run. Under our circular 3-seed Phase Oracle diagnostic (evaluating the information ceiling rather than a proposed model), adding `|HH*conj(VV)|` coherence yields `+17.9%` relative Pixel PR-AUC (`0.0114 ± 0.0011`, `21.7× ± 2.1×` vs `3-ch` `0.0097 ± 0.0003`, `18.4× ± 0.6×`) while Dice (`0.0330 ± 0.0044` vs `0.0283 ± 0.0026`) and IoU (`0.0168 ± 0.0023` vs `0.0144 ± 0.0014`) remain low due to `0.0527%` prevalence, per-patch normalization, and full complex Stokes dependence.

### Calibration, Selective Prediction, & Pruning
- **Calibration:** `ECE` is `0.04307` (MC) / `0.04794` (Ensemble), **but Brier Skill Score is negative (`-13.07` / `-14.64`)** due to weighted loss probability inflation (`0.2656–0.2886` mean $p$ on positives). MC entropy tracks $p$ near midpoint (`GT Interior` highest at `0.51350`, `Boundary Ring` `0.46635`, `Far Background` `0.14280`).
- **Selective Prediction:** Error-detection AUROC exceeds `Random` (`0.5240`) for `|p - 0.5|` (`0.8310`), `MC Entropy` (`0.8282`), and `Ensemble Variance` (`0.7879`), yet retaining top-certainty subsets lowers Patch ROC-AUC from `0.8456` (`100%`) to `0.7759` (`EnsVar`), `0.7616` (`MC`), and `0.7416` (`|p - 0.5|`) at `50%` coverage—worse than `Random` rejection (`0.8491`). By `AURC`, `|p - 0.5|` (`0.1084`) outperforms `MC Entropy` (`0.1096`), `Ensemble Variance` (`0.1259`), and `Random` (`0.1917`). **Uncertainty is not usable for selective rejection here.**
- **Pruning:** Measured speedup occurs **only** at `B=64` with `50%` structured channel pruning (`271,909` params: `10.860 → 5.359 ms`, **`2.03×`**), with negligible change at `B=1` (`2.244 → 2.058 ms`, `1.09×`). Structured `30%` (`544,834` params, `B=64: 11.441 ms`) and unstructured L1 `50%`/`90%` (`10.933 / 10.939 ms`) yield no runtime speedup. Patch ROC-AUC variations across pruned variants (`0.8099–0.8474`) are within single-seed noise.

---

## Georeferencing & Limitations

- **10-Fold GCP Cross-Validation (`676` GCPs):** Median **`307.38 m`**, Mean **`367.98 m`**, Max **`2,377.70 m`** (`~2.4 km` boundary tail; `281.7249 m` max on `5` interior GCPs).
- **Overlap Metric Definition:** `The 92.79% overlap metric is the percentage of valid TPS-warped test-strip pixels (61,857 of 66,664) whose nearest-neighbor cell on the official 25 m/px ISRO SRI HH grid has positive finite backscatter (SRI HH > 0).` (`92.79%`: `61,857` / `66,664`; median lateral swath offset `993.8 m`, mean `2382.0 m`, 90th pct `6023.7 m`).
- **Limitations:** Anisotropic slant-range pixels (`0.505 m` azimuth × `9.593 m` range); single DFSAR scene (`ch2_sar_ncxl_20260308t062017233`), strip-held-out generalization only; proxy labels ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$) do not confirm physical water ice.
