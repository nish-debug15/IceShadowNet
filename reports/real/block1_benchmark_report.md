# Block 1 Benchmark Report (Frozen Split `[90624, 319812]`, 256-Row Buffer)

- **Test Set:** `N = 2018`, `Positives = 332`, **Prevalence (PR-AUC baseline) = `16.45%`**
- **Test Strips:** `0` (`132/658`), `8` (`70/680`), `15` (`130/680`)
- **Unambiguous Clean Subset (Ablation, 1–4 px excluded):** `N = 1212`, `Positives = 332` (`27.39%`)

## 1. Pooled Test Metrics (Mean ± Std Across Seeds)

| Model | ROC-AUC | PR-AUC (Baseline 16.45%) | F1 (Val Th) | Acc (Val Th) | F1 (Best Test Th) | Acc (Best Test Th) | Bal Acc (Val Th) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **CustomCNN** | 0.8379 ± 0.0007 | 0.5485 ± 0.0102 | 0.5216 ± 0.0054 | 0.7973 ± 0.0144 | 0.5459 ± 0.0040 | 0.8368 ± 0.0064 | 0.7465 ± 0.0054 |
| **ResNet_scratch** | 0.8301 ± 0.0064 | 0.5203 ± 0.0135 | 0.5240 ± 0.0057 | 0.8051 ± 0.0061 | 0.5269 ± 0.0047 | 0.8171 ± 0.0168 | 0.7439 ± 0.0105 |
| **ResNet_pretrained** | 0.7970 ± 0.0123 | 0.4529 ± 0.0126 | 0.4687 ± 0.0280 | 0.7207 ± 0.0373 | 0.4864 ± 0.0106 | 0.8018 ± 0.0105 | 0.7296 ± 0.0195 |
| **LogisticRegression** | 0.8207 | 0.4976 | 0.5086 | 0.8008 | 0.5089 | 0.8077 | 0.7308 |
| **DecisionTree** | 0.6160 | 0.2133 | 0.3143 | 0.3707 | 0.3206 | 0.6115 | 0.5738 |

## 2. Per-Test-Strip Breakdown & Strip-Block Bootstrap 95% CIs (Seed 42 Representative)

| Model | Pooled ROC-AUC [95% CI] | Pooled PR-AUC [95% CI] | Strip 0 AUC / PR | Strip 8 AUC / PR | Strip 15 AUC / PR |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CustomCNN** | 0.8389 `[0.6691, 0.9394]` | 0.5415 `[0.2178, 0.7689]` | 0.939 / 0.769 | 0.669 / 0.218 | 0.757 / 0.374 |
| **ResNet_scratch** | 0.8376 `[0.6546, 0.9461]` | 0.5012 `[0.1995, 0.7633]` | 0.946 / 0.763 | 0.655 / 0.200 | 0.763 / 0.377 |
| **ResNet_pretrained** | 0.7943 `[0.5899, 0.9198]` | 0.4398 `[0.1410, 0.6420]` | 0.920 / 0.642 | 0.590 / 0.141 | 0.729 / 0.347 |
| **LogisticRegression** | 0.8207 `[0.6054, 0.9161]` | 0.4976 `[0.2282, 0.6432]` | 0.916 / 0.643 | 0.605 / 0.228 | 0.785 / 0.422 |
| **DecisionTree** | 0.6160 `[0.5080, 0.6389]` | 0.2133 `[0.1037, 0.2472]` | 0.602 / 0.241 | 0.508 / 0.104 | 0.630 / 0.247 |

## 3. Unambiguous Subset Ablation (1–4 px Ambiguity Band Excluded)

| Model | Clean ROC-AUC | Clean PR-AUC (Baseline 27.39%) | Clean F1 (Val Th) | Clean Acc (Val Th) | Clean Bal Acc (Val Th) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CustomCNN** | 0.9103 ± 0.0021 | 0.8177 ± 0.0019 | 0.7230 ± 0.0084 | 0.8595 ± 0.0028 | 0.8007 ± 0.0096 |
| **ResNet_scratch** | 0.8961 ± 0.0119 | 0.7707 ± 0.0116 | 0.7042 ± 0.0205 | 0.8501 ± 0.0077 | 0.7886 ± 0.0156 |
| **ResNet_pretrained** | 0.8633 ± 0.0079 | 0.6872 ± 0.0098 | 0.6840 ± 0.0190 | 0.8116 ± 0.0179 | 0.7902 ± 0.0131 |
| **LogisticRegression** | 0.8811 | 0.7012 | 0.6699 | 0.8309 | 0.7672 |
| **DecisionTree** | 0.5986 | 0.3197 | 0.4709 | 0.4604 | 0.5900 |

## 4. Before vs. After Split Comparison (Old `[100000, 140000]` vs. New Frozen `[90624, 319812]`)

| Metric / Model | Old Split (`[100k, 140k]`, 64px buf) | New Frozen Split (`[90.6k, 319.8k]`, 256px buf) |
| :--- | :---: | :---: |
| **Total Post-Buffer Patches** | `2,366` (`1686 / 342 / 338`) | `13,886` (`9828 / 2040 / 2018`) |
| **Test Positives (Prevalence)** | `101` (`29.9%`, min strip `5` pos) | `332` (`16.45%`, min strip `70` pos) |
| **CustomCNN (ROC-AUC / PR-AUC / F1)** | `0.7564 / 0.5096 / 0.6113` | `0.8379 / 0.5485 / 0.5216` |
| **ResNet_scratch (ROC-AUC / PR-AUC / F1)** | `0.7531 / 0.4988 / 0.5814` | `0.8301 / 0.5203 / 0.5240` |
| **ResNet_pretrained (ROC-AUC / PR-AUC / F1)** | `0.6876 / 0.4561 / 0.4871` *(v2a)* | `0.7970 / 0.4529 / 0.4687` |
| **LogisticRegression (ROC-AUC / PR-AUC / F1)** | `0.7314 / 0.4953 / 0.5145` | `0.8207 / 0.4976 / 0.5086` |
