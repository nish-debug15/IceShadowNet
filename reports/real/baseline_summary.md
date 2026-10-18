# Classical Polarimetric Baselines (Real DFSAR L1A)

> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI)
> **Evaluated on:** Spatial Test Split (374 patches, 110 positives, 264 negatives)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support |
|---|---|---|---|---|---|---|
| BASELINE: Plain Threshold (CPR>1 & DOP<0.13) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | TP:110 FP:0 FN:0 TN:264 |
| BASELINE: Decision Tree (max_depth=3) | 0.7086 | 0.5045 | 0.5091 | 0.5068 | 0.7582 | TP:56 FP:55 FN:54 TN:209 |
