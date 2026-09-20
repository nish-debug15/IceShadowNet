# Multimodal Training Summary

> **DATA SOURCE:** SYNTHETIC. See reports/data_notes.md.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|---------|--------|----------|
| EarlyFusionCNN | 0.4697±0.0652 | 0.8918±0.0782 | 0.2667±0.1247 | 0.3888±0.1526 | 0.4294±0.0464 | TP:12 FP:2 FN:18 TN:12 (Pos:30 Neg:14) | 618,497 | 11.2 |
| LateFusionResNet | 0.4394±0.1714 | 0.2273±0.3214 | 0.3333±0.4714 | 0.2703±0.3822 | 0.5270±0.0522 | TP:30 FP:14 FN:0 TN:0 (Pos:30 Neg:14) | 22,606,273 | 43.9 |

## [WARNING] Circularity / Low Support Flags


[WARNING] LOW SUPPORT -- METRICS ON THIS SPLIT ARE NOISY
   Val Positives: 0.0, Test Positives: 30.0

