# Label-Efficiency Fairness Benchmark (10% / 25% / 50% / 100%, 3 Seeds)

Includes `LogisticRegression` (tabular raw-intensity stats) and `CustomCNN` alongside `ResNet_SSL`, `ResNet_scratch`, and `ResNet_pretrained` on the identical frozen split (`N_test = 2,018`, prevalence `16.45%`).

| Fraction | N Train | Model | ROC-AUC (Mean ± Std) | 95% Seed CI | PR-AUC (Mean ± Std) | F1 (Val Th) |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: |
| 10% | 982 | **LogisticRegression** | 0.8036 ± 0.0135 | [0.7883, 0.8188] | 0.4807 ± 0.0462 | 0.4461 |
| 10% | 982 | **CustomCNN** | 0.8080 ± 0.0197 | [0.7857, 0.8303] | 0.4828 ± 0.0482 | 0.4898 |
| 10% | 982 | **ResNet_SSL** | 0.7892 ± 0.0094 | [0.7786, 0.7998] | 0.4262 ± 0.0546 | 0.4455 |
| 10% | 982 | **ResNet_scratch** | 0.5250 ± 0.0132 | [0.5101, 0.5399] | 0.1786 ± 0.0120 | 0.2838 |
| 10% | 982 | **ResNet_pretrained** | 0.6476 ± 0.0271 | [0.6170, 0.6783] | 0.2580 ± 0.0281 | 0.3270 |
| 25% | 2,457 | **LogisticRegression** | 0.8190 ± 0.0011 | [0.8177, 0.8202] | 0.4975 ± 0.0139 | 0.4692 |
| 25% | 2,458 | **CustomCNN** | 0.8288 ± 0.0020 | [0.8265, 0.8310] | 0.5360 ± 0.0211 | 0.5033 |
| 25% | 2,458 | **ResNet_SSL** | 0.8012 ± 0.0136 | [0.7858, 0.8166] | 0.4837 ± 0.0096 | 0.4639 |
| 25% | 2,458 | **ResNet_scratch** | 0.5652 ± 0.0352 | [0.5253, 0.6050] | 0.2050 ± 0.0128 | 0.2920 |
| 25% | 2,458 | **ResNet_pretrained** | 0.6623 ± 0.0180 | [0.6419, 0.6827] | 0.2618 ± 0.0234 | 0.3337 |
| 50% | 4,914 | **LogisticRegression** | 0.8246 ± 0.0018 | [0.8225, 0.8266] | 0.5051 ± 0.0032 | 0.4974 |
| 50% | 4,914 | **CustomCNN** | 0.8284 ± 0.0083 | [0.8191, 0.8378] | 0.5396 ± 0.0037 | 0.4684 |
| 50% | 4,914 | **ResNet_SSL** | 0.8057 ± 0.0059 | [0.7991, 0.8124] | 0.4911 ± 0.0446 | 0.4490 |
| 50% | 4,914 | **ResNet_scratch** | 0.7249 ± 0.0254 | [0.6962, 0.7536] | 0.3771 ± 0.0257 | 0.3838 |
| 50% | 4,914 | **ResNet_pretrained** | 0.6985 ± 0.0173 | [0.6789, 0.7181] | 0.2915 ± 0.0253 | 0.3597 |
| 100% | 9,828 | **LogisticRegression** | 0.8207 ± 0.0000 | [0.8207, 0.8207] | 0.4976 ± 0.0000 | 0.5086 |
| 100% | 9,828 | **CustomCNN** | 0.8289 ± 0.0015 | [0.8272, 0.8306] | 0.5214 ± 0.0336 | 0.5068 |
| 100% | 9,828 | **ResNet_SSL** | 0.8093 ± 0.0022 | [0.8069, 0.8118] | 0.5231 ± 0.0140 | 0.4733 |
| 100% | 9,828 | **ResNet_scratch** | 0.7690 ± 0.0084 | [0.7595, 0.7785] | 0.4401 ± 0.0014 | 0.4378 |
| 100% | 9,828 | **ResNet_pretrained** | 0.7156 ± 0.0204 | [0.6925, 0.7386] | 0.3370 ± 0.0416 | 0.3612 |

## Fairness Verdict: Does `ResNet_SSL` Beat `LogisticRegression` & `CustomCNN`?

- **10% labels:** `ResNet_SSL` (0.7892 ROC-AUC, 0.4262 PR-AUC) vs. `LogisticRegression` (0.8036 / 0.4807) vs. `CustomCNN` (0.8080 / 0.4828)
- **25% labels:** `ResNet_SSL` (0.8012 ROC-AUC, 0.4837 PR-AUC) vs. `LogisticRegression` (0.8190 / 0.4975) vs. `CustomCNN` (0.8288 / 0.5360)
- **50% labels:** `ResNet_SSL` (0.8057 ROC-AUC, 0.4911 PR-AUC) vs. `LogisticRegression` (0.8246 / 0.5051) vs. `CustomCNN` (0.8284 / 0.5396)
- **100% labels:** `ResNet_SSL` (0.8093 ROC-AUC, 0.5231 PR-AUC) vs. `LogisticRegression` (0.8207 / 0.4976) vs. `CustomCNN` (0.8289 / 0.5214)
