# IceShadowNet 2.0: Research Report

**Detection, Pixel-Level Localization, and Uncertainty Analysis of Lunar South Polar Radar Anomalies in Chandrayaan-2 DFSAR L-Band Quad-Pol Data**

---

## Abstract

We evaluate deep learning for detecting anomalous polarimetric radar signatures ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$ on $7 \times 7$ multilooked data) in Chandrayaan-2 DFSAR Level-1A slant-range imagery (`ch2_sar_ncxl_20260308t062017233`, rows `90,624–319,812`, `13,886` leak-free patches with a `256`-row azimuth guard buffer). Under a strict anti-circularity input policy (raw intensity-only `[HH, HV, VV]` log-intensities), we establish four empirical findings: (1) on patch-level binary classification, `CustomCNN` ties `LogisticRegression` (`0.8124 ± 0.0408` vs `0.8118 ± 0.0460` 5-fold blocked-strip CV ROC-AUC); (2) ImageNet pretraining fails to transfer to SAR (`0.7970` vs `0.8301` ROC-AUC from scratch), while in-domain SimCLR SSL (`0.7892` at 10% labels) beats `ResNet_scratch` (`0.5250`) **but does not beat `LogisticRegression` (`0.8036`) or `CustomCNN` (`0.8080`)**; (3) pixel-level `IceUNet` achieves **`0.0093 ± 0.0003` Pixel PR-AUC (`17.6×` lift over `0.0527%` baseline)** and **`0.0110` (`20.9×` lift)** as a 3-seed ensemble, while absolute Dice (`~0.03`, `0.0539` conditional on positive patches) reflects a physical **phase-information ceiling**; and (4) Thin-Plate-Spline georeferencing over `676` `GCS_MOON` GCPs achieves `~282 m` (`~300 m`) hold-out accuracy with `92.79%` footprint overlap against the official ISRO `SRI` product.

---

## 1. Experimental Findings & Honest Negative Results

### 1.1 Patch-Level Tie Between Deep Learning and Logistic Regression
Across 5-fold blocked-strip CV over all 20 strips (`256`-row buffers), `CustomCNN` (`0.8124 ± 0.0408` ROC-AUC, `0.4760 ± 0.1093` PR-AUC) ties `LogisticRegression` (`0.8118 ± 0.0460` ROC-AUC, `0.4921 ± 0.1197` PR-AUC). Circularity ablation shows `80.9%` (`250 / 309`) of `CustomCNN` false positives fall in the `1–4 px` ambiguity band just below the `>=5 px` patch cutoff.

### 1.2 ImageNet Failure & Label-Efficiency Fairness
`ResNet_pretrained` (`0.7970 ± 0.0123` ROC-AUC) underperforms `ResNet_scratch` (`0.8301 ± 0.0064`). In-domain SimCLR pretraining (`ResNet_SSL`, `11,868` unlabeled Train+Val patches) recovers ResNet performance at 10% labels (`0.7892 ± 0.0094` vs `0.5250 ± 0.0132` `ResNet_scratch`), **but does not outperform `LogisticRegression` (`0.8036 ± 0.0135`) or `CustomCNN` (`0.8080 ± 0.0197`)**.

### 1.3 Pixel-Level U-Net Localization & Phase-Information Ceiling
- **Pixel PR-AUC (Primary):** `IceUNet` reaches `0.0093 ± 0.0003` (`17.6×` lift over `0.0527%` prevalence) and `0.0110` (`20.9×` lift) as a 3-seed ensemble, outperforming `HV/HH Ratio` (`0.0011`, `2.1×`) and `Pixel LogReg` (`0.0012`, `2.3×`).
- **Dice (`~0.03`) & Information Ceiling:** Pooled Dice is `0.0266 ± 0.0045` (`0.0334` ensemble; `0.0539` conditional Dice restricted to positive patches; `>=5 px` connected-component filtering yields `0.0303` on seed 42 and `0.3936` Patch `>=5` F1 on ensemble). Ground-truth `DOP` and `CPR` depend on inter-channel complex phase coherence, which is strictly excluded from the intensity-only `[HH, HV, VV]` inputs—imposing a hard information ceiling on pixel-exact boundary recovery.
- **Patch Aggregation:** `IceUNet (mean_prob: 0.8335)` does **not** beat `CustomCNN (0.8379 ± 0.0007)` on patch ROC-AUC (`Optuna IceUNet` single run: `0.8497`). All Optuna tuning (`12` trials) was performed strictly on the validation split before fixing the configuration for a single test evaluation.

### 1.4 Uncertainty, Calibration, & Pruning
- **Calibration:** `ECE` is `0.04307` (MC Dropout) and `0.04794` (Ensemble), **but Brier Skill Score is negative (`-13.07` / `-14.64`)** because `pos_weight=50` + Dice loss inflates probabilities (`0.2656–0.2886` mean $p$ on positives, `0.0426` on background vs `0.000527` prevalence).
- **Boundary Behavior:** MC entropy tracks $p$ near $0.5$, so `GT Interior` is highest (`0.51350`) followed by the `1–2 px` ring (`0.46635`) and far background (`0.14280`).
- **Selective Prediction vs. `|p - 0.5|`:** At `50%` coverage, Ensemble Variance (`0.7788`) and MC Entropy (`0.7431`) beat `|p - 0.5|` (`0.7055`) on Patch ROC-AUC; Ensemble Variance also improves Pixel Dice (`0.0330 → 0.0368` at 60% coverage), whereas MC Entropy and `|p - 0.5|` drop Pixel Dice to `0.0000` at 90% coverage.
- **Pruning:** Unstructured L1 pruning (`90%`, `110,521` non-zero params) is parameter reduction only (`~11.2 ms` dense CUDA latency). Structured channel pruning (`base_ch` 24 → 17 at `30%`) yields **no speedup** (`15.54 ms` vs `12.94 ms`); `50%` (`base_ch=12`, `271,909` params) achieves `9.81 ms` (`1.32×` speedup).

---

## 2. Georeferencing & Limitations

- **TPS Accuracy:** Normalized Thin-Plate-Spline over `676` `GCS_MOON` GCPs (`R = 1,737,400 m`) yields `0.000 m` in-fit residual, `~282 m` (`~300 m`) 5-GCP hold-out residual, and `92.79%` footprint overlap with the official `SRI HH` product (`25 m/px`).
- **Limitations:** Anisotropic slant-range pixels (`0.505 m` azimuth × `9.593 m` range); single scene, strip-held-out generalization only; polarimetric proxy labels (`CPR > 1.0 & DOP < 0.13`) do not confirm physical water ice.
