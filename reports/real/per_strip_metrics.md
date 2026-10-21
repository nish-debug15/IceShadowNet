# Real DFSAR Along-Track Per-Strip Metrics Breakdown

> **Spatial Test Set Geometry:** Three along-track strips spanning radar lines 100,000–140,000:
> - **Strip 6**: Total = 112 ($N_{pos} = 43$, $N_{neg} = 69$, Prevalence = 38.4%)
> - **Strip 10**: Total = 112 ($N_{pos} = 53$, $N_{neg} = 59$, Prevalence = 47.3%)
> - **Strip 14**: Total = 114 ($N_{pos} = 5$, $N_{neg} = 109$, Prevalence = 4.4%)

---

## 1. Per-Strip Metrics Table (All 15 DL Runs + 2 Baselines)

| Model Architecture | Seed | Strip | N Pos | N Neg | TP | FP | FN | TN | Recall | Precision | F1 Score | ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| CustomCNN | 42 | 6 | 43 | 69 | 43 | 69 | 0 | 0 | 1.0000 | 0.3839 | 0.5548 | 0.6073 |
| CustomCNN | 42 | 10 | 53 | 59 | 50 | 49 | 3 | 10 | 0.9434 | 0.5051 | 0.6579 | 0.6760 |
| CustomCNN | 42 | 14 | 5 | 109 | 0 | 0 | 5 | 109 | 0.0000 | 0.0000 | 0.0000 | 0.7138 |
| CustomCNN | 123 | 6 | 43 | 69 | 42 | 66 | 1 | 3 | 0.9767 | 0.3889 | 0.5563 | 0.5706 |
| CustomCNN | 123 | 10 | 53 | 59 | 46 | 30 | 7 | 29 | 0.8679 | 0.6053 | 0.7132 | 0.6719 |
| CustomCNN | 123 | 14 | 5 | 109 | 0 | 0 | 5 | 109 | 0.0000 | 0.0000 | 0.0000 | 0.7376 |
| CustomCNN | 2024 | 6 | 43 | 69 | 43 | 66 | 0 | 3 | 1.0000 | 0.3945 | 0.5658 | 0.6485 |
| CustomCNN | 2024 | 10 | 53 | 59 | 46 | 31 | 7 | 28 | 0.8679 | 0.5974 | 0.7077 | 0.7035 |
| CustomCNN | 2024 | 14 | 5 | 109 | 0 | 0 | 5 | 109 | 0.0000 | 0.0000 | 0.0000 | 0.6734 |
| ResNet_scratch | 42 | 6 | 43 | 69 | 41 | 65 | 2 | 4 | 0.9535 | 0.3868 | 0.5503 | 0.5484 |
| ResNet_scratch | 42 | 10 | 53 | 59 | 41 | 37 | 12 | 22 | 0.7736 | 0.5256 | 0.6260 | 0.6514 |
| ResNet_scratch | 42 | 14 | 5 | 109 | 0 | 1 | 5 | 108 | 0.0000 | 0.0000 | 0.0000 | 0.6422 |
| ResNet_scratch | 123 | 6 | 43 | 69 | 43 | 69 | 0 | 0 | 1.0000 | 0.3839 | 0.5548 | 0.6134 |
| ResNet_scratch | 123 | 10 | 53 | 59 | 42 | 30 | 11 | 29 | 0.7925 | 0.5833 | 0.6720 | 0.6972 |
| ResNet_scratch | 123 | 14 | 5 | 109 | 0 | 0 | 5 | 109 | 0.0000 | 0.0000 | 0.0000 | 0.6954 |
| ResNet_scratch | 2024 | 6 | 43 | 69 | 43 | 69 | 0 | 0 | 1.0000 | 0.3839 | 0.5548 | 0.5851 |
| ResNet_scratch | 2024 | 10 | 53 | 59 | 45 | 42 | 8 | 17 | 0.8491 | 0.5172 | 0.6429 | 0.6844 |
| ResNet_scratch | 2024 | 14 | 5 | 109 | 1 | 9 | 4 | 100 | 0.2000 | 0.1000 | 0.1333 | 0.8092 |
| ResNet_pretrained | 42 | 6 | 43 | 69 | 34 | 45 | 9 | 24 | 0.7907 | 0.4304 | 0.5574 | 0.5972 |
| ResNet_pretrained | 42 | 10 | 53 | 59 | 39 | 35 | 14 | 24 | 0.7358 | 0.5270 | 0.6142 | 0.5676 |
| ResNet_pretrained | 42 | 14 | 5 | 109 | 3 | 36 | 2 | 73 | 0.6000 | 0.0769 | 0.1364 | 0.7468 |
| ResNet_pretrained | 123 | 6 | 43 | 69 | 24 | 40 | 19 | 29 | 0.5581 | 0.3750 | 0.4486 | 0.4975 |
| ResNet_pretrained | 123 | 10 | 53 | 59 | 33 | 25 | 20 | 34 | 0.6226 | 0.5690 | 0.5946 | 0.6214 |
| ResNet_pretrained | 123 | 14 | 5 | 109 | 1 | 23 | 4 | 86 | 0.2000 | 0.0417 | 0.0690 | 0.6679 |
| ResNet_pretrained | 2024 | 6 | 43 | 69 | 34 | 52 | 9 | 17 | 0.7907 | 0.3953 | 0.5271 | 0.5629 |
| ResNet_pretrained | 2024 | 10 | 53 | 59 | 44 | 45 | 9 | 14 | 0.8302 | 0.4944 | 0.6197 | 0.6038 |
| ResNet_pretrained | 2024 | 14 | 5 | 109 | 2 | 41 | 3 | 68 | 0.4000 | 0.0465 | 0.0833 | 0.6807 |
| ResNet_pretrained_v2a | 42 | 6 | 43 | 69 | 30 | 44 | 13 | 25 | 0.6977 | 0.4054 | 0.5128 | 0.4995 |
| ResNet_pretrained_v2a | 42 | 10 | 53 | 59 | 18 | 16 | 35 | 43 | 0.3396 | 0.5294 | 0.4138 | 0.5948 |
| ResNet_pretrained_v2a | 42 | 14 | 5 | 109 | 0 | 5 | 5 | 104 | 0.0000 | 0.0000 | 0.0000 | 0.7138 |
| ResNet_pretrained_v2a | 123 | 6 | 43 | 69 | 38 | 56 | 5 | 13 | 0.8837 | 0.4043 | 0.5547 | 0.5538 |
| ResNet_pretrained_v2a | 123 | 10 | 53 | 59 | 32 | 39 | 21 | 20 | 0.6038 | 0.4507 | 0.5161 | 0.5491 |
| ResNet_pretrained_v2a | 123 | 14 | 5 | 109 | 0 | 13 | 5 | 96 | 0.0000 | 0.0000 | 0.0000 | 0.6257 |
| ResNet_pretrained_v2a | 2024 | 6 | 43 | 69 | 33 | 50 | 10 | 19 | 0.7674 | 0.3976 | 0.5238 | 0.4796 |
| ResNet_pretrained_v2a | 2024 | 10 | 53 | 59 | 38 | 38 | 15 | 21 | 0.7170 | 0.5000 | 0.5891 | 0.5552 |
| ResNet_pretrained_v2a | 2024 | 14 | 5 | 109 | 0 | 18 | 5 | 91 | 0.0000 | 0.0000 | 0.0000 | 0.4220 |
| ResNet_pretrained_v2b | 42 | 6 | 43 | 69 | 29 | 52 | 14 | 17 | 0.6744 | 0.3580 | 0.4677 | 0.4870 |
| ResNet_pretrained_v2b | 42 | 10 | 53 | 59 | 32 | 32 | 21 | 27 | 0.6038 | 0.5000 | 0.5470 | 0.6044 |
| ResNet_pretrained_v2b | 42 | 14 | 5 | 109 | 0 | 17 | 5 | 92 | 0.0000 | 0.0000 | 0.0000 | 0.4147 |
| ResNet_pretrained_v2b | 123 | 6 | 43 | 69 | 24 | 33 | 19 | 36 | 0.5581 | 0.4211 | 0.4800 | 0.5538 |
| ResNet_pretrained_v2b | 123 | 10 | 53 | 59 | 22 | 15 | 31 | 44 | 0.4151 | 0.5946 | 0.4889 | 0.6089 |
| ResNet_pretrained_v2b | 123 | 14 | 5 | 109 | 0 | 9 | 5 | 100 | 0.0000 | 0.0000 | 0.0000 | 0.5358 |
| ResNet_pretrained_v2b | 2024 | 6 | 43 | 69 | 27 | 49 | 16 | 20 | 0.6279 | 0.3553 | 0.4538 | 0.4280 |
| ResNet_pretrained_v2b | 2024 | 10 | 53 | 59 | 32 | 21 | 21 | 38 | 0.6038 | 0.6038 | 0.6038 | 0.6649 |
| ResNet_pretrained_v2b | 2024 | 14 | 5 | 109 | 1 | 17 | 4 | 92 | 0.2000 | 0.0556 | 0.0870 | 0.5927 |
| Decision Tree (max_depth=3, balanced) | - | 6 | 43 | 69 | 34 | 43 | 9 | 26 | 0.7907 | 0.4416 | 0.5667 | 0.6190 |
| Decision Tree (max_depth=3, balanced) | - | 10 | 53 | 59 | 51 | 50 | 2 | 9 | 0.9623 | 0.5050 | 0.6623 | 0.5201 |
| Decision Tree (max_depth=3, balanced) | - | 14 | 5 | 109 | 1 | 50 | 4 | 59 | 0.2000 | 0.0196 | 0.0357 | 0.4165 |
| Logistic Regression (balanced) | - | 6 | 43 | 69 | 27 | 37 | 16 | 32 | 0.6279 | 0.4219 | 0.5047 | 0.5575 |
| Logistic Regression (balanced) | - | 10 | 53 | 59 | 35 | 41 | 18 | 18 | 0.6604 | 0.4605 | 0.5426 | 0.5414 |
| Logistic Regression (balanced) | - | 14 | 5 | 109 | 0 | 0 | 5 | 109 | 0.0000 | 0.0000 | 0.0000 | 0.4477 |

---

## 2. Mean Across Seeds per Model Architecture per Strip

| Model Architecture | Strip | Mean TP | Mean FP | Mean FN | Mean TN | Mean Recall | Mean Precision | Mean F1 Score | Mean ROC-AUC |
|---|---|---|---|---|---|---|---|---|---|
| **CustomCNN** | Strip 6 | 42.7 | 67.0 | 0.3 | 2.0 | 0.9922 | 0.3891 | **0.5590** | 0.6088 |
| **CustomCNN** | Strip 10 | 47.3 | 36.7 | 5.7 | 22.3 | 0.8931 | 0.5692 | **0.6929** | 0.6838 |
| **CustomCNN** | Strip 14 | 0.0 | 0.0 | 5.0 | 109.0 | 0.0000 | 0.0000 | **0.0000** | 0.7083 |
| **ResNet_scratch** | Strip 6 | 42.3 | 67.7 | 0.7 | 1.3 | 0.9845 | 0.3849 | **0.5533** | 0.5823 |
| **ResNet_scratch** | Strip 10 | 42.7 | 36.3 | 10.3 | 22.7 | 0.8050 | 0.5421 | **0.6469** | 0.6776 |
| **ResNet_scratch** | Strip 14 | 0.3 | 3.3 | 4.7 | 105.7 | 0.0667 | 0.0333 | **0.0444** | 0.7156 |
| **ResNet_pretrained** | Strip 6 | 30.7 | 45.7 | 12.3 | 23.3 | 0.7132 | 0.4002 | **0.5110** | 0.5525 |
| **ResNet_pretrained** | Strip 10 | 38.7 | 35.0 | 14.3 | 24.0 | 0.7296 | 0.5301 | **0.6095** | 0.5976 |
| **ResNet_pretrained** | Strip 14 | 2.0 | 33.3 | 3.0 | 75.7 | 0.4000 | 0.0550 | **0.0962** | 0.6985 |
| **ResNet_pretrained_v2a** | Strip 6 | 33.7 | 50.0 | 9.3 | 19.0 | 0.7829 | 0.4024 | **0.5305** | 0.5110 |
| **ResNet_pretrained_v2a** | Strip 10 | 29.3 | 31.0 | 23.7 | 28.0 | 0.5535 | 0.4934 | **0.5064** | 0.5664 |
| **ResNet_pretrained_v2a** | Strip 14 | 0.0 | 12.0 | 5.0 | 97.0 | 0.0000 | 0.0000 | **0.0000** | 0.5872 |
| **ResNet_pretrained_v2b** | Strip 6 | 26.7 | 44.7 | 16.3 | 24.3 | 0.6202 | 0.3781 | **0.4672** | 0.4896 |
| **ResNet_pretrained_v2b** | Strip 10 | 28.7 | 22.7 | 24.3 | 36.3 | 0.5409 | 0.5661 | **0.5466** | 0.6261 |
| **ResNet_pretrained_v2b** | Strip 14 | 0.3 | 14.3 | 4.7 | 94.7 | 0.0667 | 0.0185 | **0.0290** | 0.5144 |

---

## 3. Strip 14 Detection Analysis (Zero-TP Prevalence)

- **Ground Truth on Strip 14:** Only 5 positive patches out of 114 total patches (prevalence 4.4%).
- **Detection Results on Strip 14:**
  - **CustomCNN** (all 3 seeds): $\text{TP} = 0, \text{FP} = 0, \text{FN} = 5, \text{TN} = 109 \implies \text{F1} = 0.0000$ (0/3 seeds detected positives).
  - **ResNet_pretrained_v2a** (all 3 seeds): $\text{TP} = 0, \text{FP} \in [5, 18], \text{FN} = 5 \implies \text{F1} = 0.0000$ (0/3 seeds detected positives).
  - **Logistic Regression**: $\text{TP} = 0, \text{FP} = 0, \text{FN} = 5, \text{TN} = 109 \implies \text{F1} = 0.0000$.
  - **ResNet_scratch**: Seed 42 and Seed 123 have $\text{TP} = 0$; Seed 2024 has $\text{TP} = 1, \text{FP} = 9 \implies \text{F1} = 0.1333$.
  - **ResNet_pretrained_v2b**: Seed 42 and Seed 123 have $\text{TP} = 0$; Seed 2024 has $\text{TP} = 1, \text{FP} = 17 \implies \text{F1} = 0.0870$.
  - **ResNet_pretrained (Orig)**: Low thresholds produce high false positives ($\text{FP} \in [23, 41]$) picking up 1 to 3 positives ($\text{TP} \in [1, 3]$).
  - **Decision Tree**: High false positives ($\text{FP} = 50$) with $\text{TP} = 1 \implies \text{F1} = 0.0357$.
- **Conclusion:** Strip 14 does **not** have $\text{TP}=0$ for every single model, but has $\text{TP}=0$ across **11 of the 17 total evaluated runs (64.7%)**, including 100% of CustomCNN, 100% of ResNet_pretrained_v2a, and Logistic Regression. In a 3-block bootstrap, $(1/3)^3 = 1/27 \approx 3.7\%$ of resamples select only Strip 14, directly explaining why the 2.5th percentile lower bound of F1 drops to `0.0000` under the default bootstrap procedure.
