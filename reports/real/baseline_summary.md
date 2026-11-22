# Classical Baselines on Real Chandrayaan-2 DFSAR L1A (Overhaul Evaluation)

> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI, lines 100,000–140,000)
> **Spatial Test Split:** 338 patches (101 positive, 237 negative, Prevalence=29.9%)
> **Evaluation Protocol:** PR-AUC is headline metric; MCC and Balanced Accuracy replace unweighted accuracy; thresholds tuned strictly on Val; block-bootstrap 95% CIs across strips; ECE calibration computed.

## 1. Primary Test Evaluation Table (Full Spatial Test Set: N=338)

| Model | Tuned Th | PR-AUC (Primary) (95% CI) | ROC-AUC (95% CI) | MCC (95% CI) | Balanced Acc (95% CI) | F1 (95% CI) | Precision | Recall | ECE | Support |
|---|---|---|---|---|---|---|---|---|---|---|
| Trivial Baseline (All Negative) | 0.50 | **0.2988**<br>`[0.0439, 0.4732]` | **0.5000**<br>`[0.5000, 0.5000]` | **0.0000**<br>`[0.0000, 0.0000]` | 0.5000<br>`[0.5000, 0.5000]` | 0.0000<br>`[0.0000, 0.0000]` | 0.0000 | 0.0000 | 0.2988 | TP:0 FP:0 FN:101 TN:237 (Pos:101 Neg:237) |
| Rule Baseline (Circularity Yardstick: CPR>1 & DOP<0.13) | 0.00 | **1.0000**<br>`[1.0000, 1.0000]` | **1.0000**<br>`[1.0000, 1.0000]` | **1.0000**<br>`[1.0000, 1.0000]` | 1.0000<br>`[1.0000, 1.0000]` | 1.0000<br>`[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.2981 | TP:101 FP:0 FN:0 TN:237 (Pos:101 Neg:237) |
| BASELINE: Decision Tree (max_depth=3, balanced) | 0.42 | **0.4280**<br>`[0.2351, 0.4895]` | **0.6922**<br>`[0.4165, 0.7407]` | **0.2430**<br>`[-0.1066, 0.2892]` | 0.6241<br>`[0.3706, 0.6499]` | 0.5212<br>`[0.0357, 0.6623]` | 0.3755 | 0.8515 | 0.1741 | TP:86 FP:143 FN:15 TN:94 (Pos:101 Neg:237) |
| BASELINE: Logistic Regression (balanced) | 0.50 | **0.4953**<br>`[0.2374, 0.5401]` | **0.7314**<br>`[0.4477, 0.8153]` | **0.2646**<br>`[-0.0369, 0.3801]` | 0.6424<br>`[0.4827, 0.7038]` | 0.5145<br>`[0.0000, 0.5426]` | 0.4429 | 0.6139 | 0.1413 | TP:62 FP:78 FN:39 TN:159 (Pos:101 Neg:237) |

---

## 2. Ambiguity-Removed Evaluation (Clean Test Set: N=205)

> Excludes 133 test patches containing between 1 and 4 positive pixels. Leaves 205 clean patches (101 pos, 104 neg).

| Model | Tuned Th | PR-AUC (Primary) (95% CI) | ROC-AUC (95% CI) | MCC (95% CI) | Balanced Acc (95% CI) | F1 (95% CI) | Precision | Recall | ECE | Support |
|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE: Decision Tree (max_depth=3, balanced) | 0.42 | **0.7178**<br>`[0.2533, 0.7953]` | **0.7675**<br>`[0.3886, 0.7875]` | **0.3032**<br>`[-0.1639, 0.3738]` | 0.6373<br>`[0.3357, 0.6551]` | 0.6964<br>`[0.0465, 0.8718]` | 0.5890 | 0.8515 | 0.1090 | TP:86 FP:60 FN:15 TN:44 (Pos:101 Neg:104) |
| BASELINE: Logistic Regression (balanced) | 0.50 | **0.8164**<br>`[0.2587, 0.8721]` | **0.8512**<br>`[0.4657, 0.8760]` | **0.4409**<br>`[0.0000, 0.5506]` | 0.7156<br>`[0.5000, 0.7457]` | 0.6813<br>`[0.0000, 0.7143]` | 0.7654 | 0.6139 | 0.1919 | TP:62 FP:19 FN:39 TN:85 (Pos:101 Neg:104) |
