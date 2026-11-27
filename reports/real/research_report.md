# IceShadowNet 2.0: Research Report

**Detection, Pixel-Level Segmentation, and Calibrated Uncertainty of Lunar South Polar Radar Anomalies in Chandrayaan-2 DFSAR L-Band Quad-Pol Data**

---

## Abstract

We investigate deep learning for detecting anomalous polarimetric radar signatures ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$ on $7 \times 7$ multilooked data) in Chandrayaan-2 DFSAR Level-1A slant-range imagery (`ch2_sar_ncxl_20260308t062017233`, rows `90,624–319,812`, `13,886` leak-free patches with a `256`-row azimuth guard buffer). Under a strict anti-circularity input policy (raw `[HH, HV, VV]` log-intensities only), we establish four empirical findings: (1) ImageNet pretraining fails to transfer to SAR backscatter (`0.7970` vs `0.8301` ROC-AUC from scratch); (2) in-domain SimCLR contrastive pretraining on `11,868` unlabeled Train+Val patches recovers low-label performance (`0.78–0.80` vs `0.51–0.54` ROC-AUC at `10%` labels); (3) patch-level binary classification ties Logistic Regression (`0.8124` vs `0.8118` 5-fold CV ROC-AUC) due to a `34.7%` ambiguity band (`1–4 px` positive patches); and (4) pixel-level `IceUNet` segmentation resolves this ambiguity, achieving `0.9380` Pixel ROC-AUC (`0.9518` for a 3-seed ensemble) and `0.0093–0.0110` Pixel PR-AUC (`~8×` over pixel baselines), with calibrated uncertainty (`ECE = 0.043–0.048`) and Thin-Plate-Spline georeferencing (`0.000 m` in-fit, `281.7 m` hold-out residual across 676 `GCS_MOON` GCPs).

---

## 1. Experimental Progression & Negative Results

### 1.1 Why ImageNet Fails on SAR
`ResNet_pretrained` (`0.7970 ± 0.0123` ROC-AUC, `0.4529 ± 0.0126` PR-AUC) underperforms `ResNet_scratch` (`0.8301 ± 0.0064` / `0.5203 ± 0.0135`) and `CustomCNN` (`0.8379 ± 0.0007` / `0.5485 ± 0.0102`) on the frozen test split. Optical photographic edge/texture filters do not match coherent microwave speckle statistics.

### 1.2 In-Domain SimCLR SSL Solves Low-Label Scarcity
Pretraining a ResNet-18 backbone via SimCLR (`NT-Xent`, $\tau=0.2$) with SAR-specific augmentations (multiplicative speckle, gain/offset jitter, flips/90° rotations; no color jitter) on `11,868` unlabeled Train+Val patches yields `0.7806–0.8023` ROC-AUC at `10%` labeled data (`ResNet_scratch`: `0.5124–0.5432`).

### 1.3 Patch-Level Tie with Logistic Regression (Honest Negative Result)
Across 5-fold blocked-strip CV over all 20 strips (`256`-row buffers), `CustomCNN` (`0.8124 ± 0.0408` ROC-AUC, `0.4760 ± 0.1093` PR-AUC) ties `LogisticRegression` (`0.8118 ± 0.0460` ROC-AUC, `0.4921 ± 0.1197` PR-AUC). Circularity ablation confirms `80.9%` (`250 / 309`) of `CustomCNN` false positives contain `1–4` genuine anomalous pixels just below the `>=5 px` patch binarization cutoff.

### 1.4 Pixel-Level U-Net & Uncertainty Quantification
- **Pixel Segmentation:** `IceUNet` (`1,084,681` params, Dice+weighted-BCE loss) reaches `0.0093 ± 0.0003` Pixel PR-AUC (`17.6×` above `0.0527%` pixel prevalence) and `0.9380 ± 0.0050` Pixel ROC-AUC, beating `HV/HH Ratio` (`0.0011` / `0.6960`) and `Pixel LogReg` (`0.0012` / `0.7023`).
- **Ensemble & Optuna:** A 3-seed `IceUNet` ensemble reaches `0.0110` Pixel PR-AUC / `0.9518` Pixel ROC-AUC / `0.0334` Dice. A 12-trial val-only Optuna sweep (`base_ch=16, lr=8.57e-4, dropout=0.198, dice_w=0.228`) achieves `0.8497` Patch Mean-Prob ROC-AUC and `0.0108` Pixel PR-AUC on test.
- **Boundary Uncertainty Test (Measured, Not Assumed):** MC Dropout (`T=20`) predictive entropy is highest in `GT Interior` (`0.51384`) and `GT Boundary Ring (1–2 px)` (`0.46674`) vs `Far Background` (`0.14293`). 3-seed ensemble variance peaks slightly on the `1–2 px` boundary ring (`0.009529` vs `0.009468` interior; `0.001904` background).
- **Selective Prediction (Honest Negative Result):** Abstaining on high-entropy pixels raises pixel accuracy (`99.431% → 99.999%`), but drops Dice to `0.0000` at `90%` coverage because under `0.0527%` prevalence, rare positive predictions carry high entropy (`p ~ 0.3–0.5`) and are abstained first.
- **Pruning:** Global L1 unstructured pruning at `90%` sparsity (`110,521` non-zero params) retains `0.8351` Patch ROC-AUC, though dense CUDA latency stays `~11.2 ms / batch of 64` without sparse kernels.

---

## 2. Georeferencing & Limitations

- **TPS Warp:** Normalized-coordinate Thin-Plate-Spline over `676` embedded `GCS_MOON` GCPs (`R = 1,737,400 m`) into Moon South Polar Stereographic yields `0.000000 m` in-fit residual and `281.7249 m` max 5-GCP hold-out residual.
- **Limitations:** Anisotropic slant-range pixels (`0.505 m` azimuth × `9.593 m` range); single-scene, strip-held-out generalization only; radar-derived proxy labels (`CPR > 1.0 & DOP < 0.13`) do not confirm physical water ice without PSR/DEM/thermal corroboration.
