# IceShadowNet 2.0

**Deep Learning-Based Detection, Pixel-Level Segmentation, and Uncertainty Quantification of Lunar South Polar Radar Anomalies Using Chandrayaan-2 DFSAR L-Band Quad-Pol Data**

> Motivated by ISRO's Chandrayaan-2 lunar polar ice characterization problem. **Note:** Labels are radar-derived polarimetric proxy labels ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$ on $7 \times 7$ multilooked data) and do **not** directly confirm physical water ice.

---

## 1. Core Research Narrative

1. **ImageNet Pretraining Fails to Transfer to SAR:** Optical RGB features (`ResNet_pretrained` Test ROC-AUC `0.7970`, PR-AUC `0.4529`) consistently underperform training from scratch (`0.8301` / `0.5203`) due to microwave speckle vs. optical reflectance domain mismatch.
2. **In-Domain SimCLR SSL Solves the Low-Label Regime:** Contrastive pretraining (`ResNet_SSL`) on 11,868 unlabeled Train+Val radar patches (strictly zero test leakage) with physics-aware SAR augmentations boosts 10%-label Test ROC-AUC from `0.51–0.54` (`ResNet_scratch`) to **`0.78–0.80`**.
3. **Patch-Level Classification Hits an Ambiguity Ceiling (Honest Negative Result):** Across both our frozen test split (`CustomCNN 0.8379` vs `LogisticRegression 0.8207` ROC-AUC) and 5-fold blocked-strip CV across all 20 strips (`CustomCNN 0.8124 ± 0.0408` vs `LogisticRegression 0.8118 ± 0.0460`), patch-level CNNs tie Logistic Regression because `34.7%` of patches (`4,960 / 14,282`) fall in the `1–4 px` ambiguity band just below the `>=5 px` cutoff (`80.9%` of CustomCNN FPs are `1–4 px` transition patches).
4. **MC-Dropout U-Net Resolves Threshold Ambiguity:** Pixel-level `IceUNet` on raw `[HH, HV, VV]` log-intensities achieves **`0.9380 ± 0.0050` Pixel ROC-AUC** and **`0.0093 ± 0.0003` Pixel PR-AUC** (`17.6×` over the `0.0527%` pixel baseline, `~8×` over Pixel LogReg `0.0012` and HV/HH Ratio `0.0011`), while a 3-seed U-Net Ensemble reaches **`0.9518` Pixel ROC-AUC** and **`0.0110` Pixel PR-AUC**.

---

## 2. Frozen Dataset & Leak-Free Split (`[90624, 319812]`)

- **Scene:** `ch2_sar_ncxl_20260308t062017233` (Chandrayaan-2 DFSAR L1A Quad-Pol SLI, `319,812 × 116`).
- **Active Rows:** `[90624, 319812]` (`229,188` valid rows = `71.66%` of strip; rows `0..90623` zero-padded).
- **Inputs (Anti-Circularity):** Raw log-intensities `[HH, HV, VV]` only. Derived `CPR` and `DOP` are strictly excluded from model inputs.
- **Deterministic Non-Adjacent Strip Split (`256`-row azimuth guard buffer, zero pixel-extent leakage):**
  - **Train:** `9,828` patches (`1,614` pos, `16.42%`) across 14 strips
  - **Val:** `2,040` patches (`335` pos, `16.42%`) across strips `[2, 6, 17]`
  - **Test:** `2,018` patches (`332` pos, `16.45%`) across strips `[0, 8, 15]` (`[132, 70, 130]` pos)
  - Manifest frozen in [`data/processed_real/split_manifest.json`](data/processed_real/split_manifest.json) and documented in [`reports/real/split_rationale.md`](reports/real/split_rationale.md).

---

## 3. Benchmark Results (Identical Frozen Split)

### 3.1 Patch-Level Classification (Test Set: `N = 2,018`, Prevalence = `16.45%`)

| Model | ROC-AUC | PR-AUC (Baseline 16.45%) | F1 (Val Th) | F1 (Best Test Th) | Clean ROC-AUC (Ablation) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **IceUNet (Optuna, agg: mean_prob)** | **0.8497** | — | — | — | — |
| **CustomCNN (3 seeds)** | `0.8379 ± 0.0007` | `0.5485 ± 0.0102` | `0.5216 ± 0.0054` | `0.5459 ± 0.0040` | `0.9103 ± 0.0021` |
| **IceUNet (3 seeds, agg: mean_prob)** | `0.8335` | `0.5404` | `0.5185` | `0.5394` | `0.9035` |
| **ResNet_scratch (3 seeds)** | `0.8301 ± 0.0064` | `0.5203 ± 0.0135` | `0.5240 ± 0.0057` | `0.5269 ± 0.0047` | `0.8961 ± 0.0119` |
| **LogisticRegression** | `0.8207` | `0.4976` | `0.5086` | `0.5089` | `0.8811` |
| **ResNet_pretrained (3 seeds)** | `0.7970 ± 0.0123` | `0.4529 ± 0.0126` | `0.4687 ± 0.0280` | `0.4864 ± 0.0106` | `0.8633 ± 0.0079` |
| **DecisionTree (depth=3)** | `0.6160` | `0.2133` | `0.3143` | `0.3206` | `0.5986` |

- **5-Fold Blocked-Strip CV (All 20 Strips, 256-row buffer):** `CustomCNN` `0.8124 ± 0.0408` ROC-AUC vs `LogisticRegression` `0.8118 ± 0.0460` ROC-AUC ([`reports/real/five_fold_cv.md`](reports/real/five_fold_cv.md)).

### 3.2 Pixel-Level Segmentation & Uncertainty (`8,265,728` Test Pixels, Prevalence = `0.0527%`)

| Model | Pixel PR-AUC | Pixel ROC-AUC | Dice (Val Th) | IoU (Val Th) | ECE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline: HV/HH Ratio | `0.0011` | `0.6960` | `0.0036` | `0.0018` | — |
| Baseline: Pixel LogReg | `0.0012` | `0.7023` | `0.0016` | `0.0008` | — |
| **IceUNet (3 seeds)** | `0.0093 ± 0.0003` | `0.9380 ± 0.0050` | `0.0266 ± 0.0045` | `0.0135 ± 0.0023` | — |
| **IceUNet MC Dropout (`T=20`)** | `0.0094` | `0.9361` | `0.0302` | `0.0153` | `0.04307` |
| **IceUNet (Optuna Val-Tuned)** | `0.0108` | `0.9484` | `0.0286` | `0.0145` | — |
| **IceUNet 3-Seed Ensemble** | **`0.0110`** | **`0.9518`** | **`0.0334`** | **`0.0170`** | `0.04794` |

### 3.3 Magnitude Pruning (`IceUNet_seed42` + 3-Epoch Fine-Tune)

At **90% unstructured sparsity** (`110,521` vs `1,084,681` non-zero params, `9.8×` parameter reduction), `Patch Mean-Prob ROC-AUC` remains `0.8351` (vs `0.8291` unpruned) and Pixel Dice remains `0.0258` (vs `0.0295`). Dense PyTorch CUDA latency remains `~11.2 ms / batch of 64` because unstructured zero-masks do not trigger sparse tensor kernels.

---

## 4. Key Reports & Artifacts

- **Block 1 Benchmark & Before/After Table:** [`reports/real/block1_benchmark_report.md`](reports/real/block1_benchmark_report.md)
- **Split Rationale & 5-Fold CV:** [`reports/real/split_rationale.md`](reports/real/split_rationale.md), [`reports/real/five_fold_cv.md`](reports/real/five_fold_cv.md)
- **SimCLR SSL & Circularity Ablation:** [`reports/real/ssl_pretraining.md`](reports/real/ssl_pretraining.md), [`reports/real/circularity_ablation.md`](reports/real/circularity_ablation.md)
- **Block 2 U-Net Segmentation:** [`reports/real/unet_segmentation_report.md`](reports/real/unet_segmentation_report.md)
- **Block 3 Uncertainty, Optuna & Pruning:** [`reports/real/uncertainty_report.md`](reports/real/uncertainty_report.md), [`reports/real/efficiency_report.md`](reports/real/efficiency_report.md)
- **TPS Georeferencing Audit:** [`reports/real/georeferencing_audit.md`](reports/real/georeferencing_audit.md)
- **Research Report & Viva Q&A:** [`reports/real/research_report.md`](reports/real/research_report.md), [`reports/real/viva_qa.md`](reports/real/viva_qa.md)
- **Interactive Dashboard:** [`dashboard/`](dashboard/) (includes `/segmentation/` view)

## 5. Acknowledgement

> This research is based partially on the results obtained from the Chandrayaan-2, second lunar mission of the Indian Space Research Organisation (ISRO), archived at the Indian Space Science Data Centre (ISSDC).
