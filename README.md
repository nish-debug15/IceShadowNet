# IceShadowNet 2.0

**Deep Learning Detection, Pixel-Level Localization, and Uncertainty Analysis of Lunar South Polar Radar Anomalies Using Chandrayaan-2 DFSAR L-Band Quad-Pol Data**

> Motivated by ISRO's Chandrayaan-2 lunar polar ice characterization problem. **Note:** Labels are radar-derived polarimetric proxy labels ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$ on $7 \times 7$ multilooked data) and do **not** directly confirm physical water ice.

---

## 1. Empirical Findings (Strictly Measured)

1. **Deep Learning Ties Logistic Regression on the Patch Task (Honest Negative Result):** Across 5-fold blocked-strip CV over all 20 strips (`256`-row azimuth guard buffers), `CustomCNN` (`0.8124 ± 0.0408` ROC-AUC, `0.4760 ± 0.1093` PR-AUC) ties `LogisticRegression` (`0.8118 ± 0.0460` ROC-AUC, `0.4921 ± 0.1197` PR-AUC). On the frozen test split (`N = 2,018`, prevalence `16.45%`), `CustomCNN` reaches `0.8379 ± 0.0007` ROC-AUC vs `LogisticRegression` `0.8207`.
2. **ImageNet Pretraining Fails to Transfer to SAR:** `ResNet_pretrained` (`0.7970 ± 0.0123` ROC-AUC, `0.4529 ± 0.0126` PR-AUC) underperforms `ResNet_scratch` (`0.8301 ± 0.0064` / `0.5203 ± 0.0135`) due to optical vs. coherent microwave speckle mismatch.
3. **In-Domain SimCLR SSL Helps Over ResNet-from-Scratch, Not Over LR/CustomCNN:** Pretraining `ResNet_SSL` on `11,868` unlabeled Train+Val radar patches boosts 10%-label ROC-AUC from `0.5250 ± 0.0132` (`ResNet_scratch`) to `0.7892 ± 0.0094`, **but does not beat `LogisticRegression` (`0.8036 ± 0.0135`) or `CustomCNN` (`0.8080 ± 0.0197`)** at 10–50% labels ([`label_efficiency_fairness.md`](reports/real/label_efficiency_fairness.md)).
4. **Pixel-Level U-Net Localization & Phase-Information Ceiling:** `IceUNet` trained on raw intensity-only `[HH, HV, VV]` predicts pixel-level anomalies without requiring a `>=5 px` patch cutoff, achieving **`0.0093 ± 0.0003` Pixel PR-AUC (`17.6×` lift over `0.0527%` baseline)** and **`0.0110` PR-AUC (`20.9×` lift)** as a 3-seed ensemble (vs. `Pixel LogReg` `0.0012` and `HV/HH Ratio` `0.0011`). Absolute Dice is low (**`~0.03`**: `0.0266 ± 0.0045` single-seed, `0.0334` ensemble, `0.0539` conditional Dice on positive patches) due to a fundamental **information ceiling**: ground-truth `DOP` and `CPR` require inter-channel complex phase coherence, whereas model inputs are strictly intensity-only to prevent circularity. At patch level, `IceUNet (mean_prob: 0.8335)` does **not** beat `CustomCNN (0.8379 ± 0.0007)`.

---

## 2. Frozen Dataset & Leak-Free Split (`[90624, 319812]`)

- **Scene:** `ch2_sar_ncxl_20260308t062017233` (Chandrayaan-2 DFSAR L1A Quad-Pol SLI, `319,812 × 116`).
- **Active Rows:** `[90624, 319812]` (`229,188` valid rows = `71.66%` of strip; rows `0..90623` zero-padded).
- **Deterministic Non-Adjacent Strip Split (`256`-row guard buffer, zero pixel-extent leakage):**
  - **Train:** `9,828` patches (`1,614` pos, `16.42%`) | **Val:** `2,040` (`335` pos, strips `[2, 6, 17]`) | **Test:** `2,018` (`332` pos, strips `[0, 8, 15]`: `[132, 70, 130]` pos).
  - Manifest frozen in [`data/processed_real/split_manifest.json`](data/processed_real/split_manifest.json) and [`reports/real/split_rationale.md`](reports/real/split_rationale.md).

---

## 3. Benchmark Results (Identical Frozen Split)

### 3.1 Patch-Level Classification (`N_test = 2,018`, Prevalence = `16.45%`)

| Model | ROC-AUC (Strip 95% CI / Seed Std) | PR-AUC (Baseline 16.45%) | F1 (Val Th) | F1 (Best Test Th) | Clean ROC-AUC (Ablation) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CustomCNN (3 seeds)** | `0.8379 ± 0.0007` (`[0.6691, 0.9394]` s42) | `0.5485 ± 0.0102` | `0.5216 ± 0.0054` | `0.5459 ± 0.0040` | `0.9103 ± 0.0021` |
| **IceUNet (3 seeds, mean_prob)** | `0.8335` | `0.5404` | `0.5185` | `0.5394` | `0.9035` |
| **IceUNet (Optuna, single run)** | `0.8497` *(val-tuned, 1 seed)* | — | — | — | — |
| **ResNet_scratch (3 seeds)** | `0.8301 ± 0.0064` (`[0.6546, 0.9461]` s42) | `0.5203 ± 0.0135` | `0.5240 ± 0.0057` | `0.5269 ± 0.0047` | `0.8961 ± 0.0119` |
| **LogisticRegression** | `0.8207` (`[0.6054, 0.9161]`) | `0.4976` | `0.5086` | `0.5089` | `0.8811` |
| **ResNet_SSL (100% labels, 3 seeds)** | `0.8093 ± 0.0022` | `0.5231 ± 0.0140` | `0.4733` | — | — |
| **ResNet_pretrained (3 seeds)** | `0.7970 ± 0.0123` (`[0.5899, 0.9198]` s42) | `0.4529 ± 0.0126` | `0.4687 ± 0.0280` | `0.4864 ± 0.0106` | `0.8633 ± 0.0079` |

### 3.2 Pixel-Level Segmentation & Uncertainty (`8,265,728` Test Pixels, Prevalence = `0.0527%`)

| Model | Pixel PR-AUC | Lift vs. Prevalence (`0.0527%`) | Dice (Val Th) | Cond. Dice (Pos Patches) | IoU (Val Th) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Baseline: HV/HH Ratio | `0.0011` | `2.1×` | `0.0036` | — | `0.0018` |
| Baseline: Pixel LogReg | `0.0012` | `2.3×` | `0.0016` | — | `0.0008` |
| **IceUNet (3 seeds)** | `0.0093 ± 0.0003` | `17.6×` | `0.0266 ± 0.0045` | `0.0377–0.0519` | `0.0135 ± 0.0023` |
| **IceUNet MC Dropout (`T=20`)** | `0.0094` | `17.8×` | `0.0302` | — | `0.0153` |
| **IceUNet (Optuna Val-Tuned)** | `0.0108` | `20.5×` | `0.0286` | — | `0.0145` |
| **IceUNet 3-Seed Ensemble** | **`0.0110`** | **`20.9×`** | **`0.0334`** | **`0.0539`** | **`0.0170`** |

- **`>=5 px` Connected-Component Post-Processing:** Preserves or improves pooled Dice (`0.0203 → 0.0230` seed 2024; `0.0334` ensemble) and Patch `>=5` F1 (`0.3902 → 0.3936` ensemble).
- **Uncertainty & Calibration:** Due to `pos_weight=50` + Dice loss, Brier Skill Score is negative (`-13.07` MC, `-14.64` ensemble; mean $p$ on positives `0.2656–0.2886`). MC entropy tracks $p$ near $0.5$ (`GT Interior` highest at `0.51350`, `1–2 px` ring `0.46635`, background `0.14280`). On selective prediction, Ensemble Variance preserves/improves pixel Dice (`0.0330 → 0.0368` at 60% coverage) and outperforms `|p - 0.5|` on Patch ROC-AUC (`0.7788` vs `0.7055` at 50% coverage).

### 3.3 Parameter Reduction & Structured Pruning

- **Unstructured L1 Pruning (Parameter Reduction Only):** `90%` sparsity reduces non-zero weights `9.8×` (`110,521` vs `1,084,681`) while keeping Patch ROC-AUC `0.8351`, with **zero CUDA speedup** (`~11.2 ms / B=64`).
- **Structured Channel Pruning (`base_ch` 24 → 17 / 12):** `30%` channel pruning (`544,834` params) gives **no speedup** (`15.54 ms` vs `12.94 ms`); `50%` (`271,909` params) achieves `9.81 ms` (`~1.32×` speedup) at `0.0201` Dice.

### 3.4 Georeferencing & Reproducibility

- **TPS Warp:** `676` `GCS_MOON` GCPs (`R = 1,737,400 m`) → `0.000 m` in-fit, `~282 m` (`~300 m`) 5-GCP hold-out residual, `92.79%` footprint overlap with official ISRO `SRI HH` (`25 m/px`).
- **Reproduce:** `pip install -r requirements.txt` (pinned) and `make reproduce` (`bash run_all.sh`). See [`reports/real/REPRODUCIBILITY.md`](reports/real/REPRODUCIBILITY.md) for gitignored artifacts (raw tiles, `.npz`, `.pth`, `.tif`).

---

## 4. Key Reports

- [`reports/real/claims_audit.md`](reports/real/claims_audit.md) | [`reports/real/research_report.md`](reports/real/research_report.md) | [`reports/real/viva_qa.md`](reports/real/viva_qa.md)
- [`reports/real/block1_benchmark_report.md`](reports/real/block1_benchmark_report.md) | [`reports/real/five_fold_cv.md`](reports/real/five_fold_cv.md) | [`reports/real/label_efficiency_fairness.md`](reports/real/label_efficiency_fairness.md)
- [`reports/real/unet_segmentation_report.md`](reports/real/unet_segmentation_report.md) | [`reports/real/pixel_metrics_hardened.md`](reports/real/pixel_metrics_hardened.md) | [`reports/real/uncertainty_hardened.md`](reports/real/uncertainty_hardened.md)
- [`reports/real/structured_pruning.md`](reports/real/structured_pruning.md) | [`reports/real/georeferencing_audit.md`](reports/real/georeferencing_audit.md)

## 5. Acknowledgement

> This research is based partially on the results obtained from the Chandrayaan-2, second lunar mission of the Indian Space Research Organisation (ISRO), archived at the Indian Space Science Data Centre (ISSDC).
