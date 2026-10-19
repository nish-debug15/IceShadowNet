# Classical Baselines on Real Chandrayaan-2 DFSAR L1A

> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI, lines 100,000–140,000)
> **Evaluated on:** Spatial Test Split (338 patches: 101 positive, 237 negative)
> **Input Features:** Strictly raw radar log-intensities [HH, HV, VV] patch means (no circular CPR/DOP inputs)

## Machine Learning Comparison Table

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Support |
|---|---|---|---|---|---|---|
| BASELINE: Decision Tree (max_depth=3, balanced) | 0.6746 | 0.4681 | 0.6535 | 0.5455 | 0.6922 | TP:66 FP:75 FN:35 TN:162 (Pos:101, Neg:237) |
| BASELINE: Logistic Regression (balanced) | 0.6538 | 0.4429 | 0.6139 | 0.5145 | 0.7314 | TP:62 FP:78 FN:39 TN:159 (Pos:101, Neg:237) |

---

## Ground Truth Reference

> **NOTE: ORACLE (label rule, not a model)**
> - **Evaluation:** Accuracy=1.0000, Precision=1.0000, Recall=1.0000, F1=1.0000, ROC-AUC=1.0000
> - **Support:** TP:101 FP:0 FN:0 TN:237 (Pos:101, Neg:237)
> - **Explanation:** The Oracle directly evaluates the CPR > 1.0 & DOP < 0.13 ground-truth criteria. It achieves an F1 of 1.0000 by mathematical identity because it constitutes the labeling rule itself. It is strictly excluded from machine learning comparison tables to prevent false equivalence.
