# Classical Baselines on Real Chandrayaan-2 DFSAR L1A

> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI, lines 100,000–140,000)
> **Evaluated on:** Spatial Test Split (338 patches: 101 positive, 237 negative, Prevalence=29.9%)
> **Input Features:** Strictly raw radar log-intensities [HH, HV, VV] patch means (no circular CPR/DOP inputs)
> **Threshold Protocol:** Decision threshold tuned on Validation set to maximize Val F1, applied unchanged to Test
> **Confidence Intervals:** 95% block-bootstrap CIs computed by resampling entire spatial strips (1000 rounds)

## 1. Primary Test Evaluation (Full Spatial Test Set: N=338)

| Model | Tuned Th | Accuracy | Precision | Recall | F1 (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) | Prevalence | Support |
|---|---|---|---|---|---|---|---|---|---|
| BASELINE: Decision Tree (max_depth=3, balanced) | 0.42 | 0.5325 | 0.3755 | 0.8515 | **0.5212**<br>`[0.0357, 0.6623]` | **0.6922**<br>`[0.4165, 0.7407]` | **0.4280**<br>`[0.2351, 0.4895]` | 29.9% | TP:86 FP:143 FN:15 TN:94 (Pos:101 Neg:237) |
| BASELINE: Logistic Regression (balanced) | 0.50 | 0.6538 | 0.4429 | 0.6139 | **0.5145**<br>`[0.0000, 0.5426]` | **0.7314**<br>`[0.4477, 0.8153]` | **0.4953**<br>`[0.2374, 0.5401]` | 29.9% | TP:62 FP:78 FN:39 TN:159 (Pos:101 Neg:237) |

---

## 2. Ambiguity-Removed Evaluation (Clean Test Set: N=205)

> **NOTE ON AMBIGUITY ABLATION:**
> Excludes 133 test patches containing between 1 and 4 positive pixels (the boundary transition band). Leaves 205 clean patches (101 positive, 104 negative, Prevalence=49.3%).

| Model | Tuned Th | Accuracy | Precision | Recall | F1 (95% CI) | ROC-AUC (95% CI) | PR-AUC (95% CI) | Prevalence | Support |
|---|---|---|---|---|---|---|---|---|---|
| BASELINE: Decision Tree (max_depth=3, balanced) | 0.42 | 0.6341 | 0.5890 | 0.8515 | **0.6964**<br>`[0.0465, 0.8718]` | **0.7675**<br>`[0.3886, 0.7875]` | **0.7178**<br>`[0.2533, 0.7953]` | 49.3% | TP:86 FP:60 FN:15 TN:44 (Pos:101 Neg:104) |
| BASELINE: Logistic Regression (balanced) | 0.50 | 0.7171 | 0.7654 | 0.6139 | **0.6813**<br>`[0.0000, 0.7143]` | **0.8512**<br>`[0.4657, 0.8760]` | **0.8164**<br>`[0.2587, 0.8721]` | 49.3% | TP:62 FP:19 FN:39 TN:85 (Pos:101 Neg:104) |

---

## 3. Ground Truth Reference (Separate Note)

> **ORACLE (label rule, not a model)**
> - **Metrics:** Accuracy=1.0000, Precision=1.0000, Recall=1.0000, F1=1.0000, ROC-AUC=1.0000, PR-AUC=1.0000
> - **Support:** TP:101 FP:0 FN:0 TN:237 (Pos:101 Neg:237)
> - **Explanation:** The Oracle directly evaluates the CPR > 1.0 & DOP < 0.13 ground-truth criteria on polarimetric indicators. It achieves an F1 of 1.0000 by mathematical identity because it defines the labeling rule itself. It is strictly excluded from machine learning comparison tables to prevent false equivalence.
