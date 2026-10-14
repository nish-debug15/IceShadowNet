# Training Summary

> **DATA SOURCE:** SYNTHETIC (physically-motivated SAR scene -- no real DFSAR tiles from PRADAN yet). Numbers below demonstrate pipeline correctness only, NOT real ice-detection performance.

> **Trained for 30 epochs, 3 seeds, on SYNTHETIC data**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|---------|--------|----------|
| CustomCNN | 0.7361±0.0196 | 0.4444±0.3143 | 0.1905±0.1347 | 0.2667±0.1886 | 0.6751±0.0555 | TP:2 FP:1 FN:5 TN:16 (Pos:7 Neg:17) | 618,209 | 76.0 |
| ResNet_scratch | 0.8194±0.0393 | 0.7361±0.0856 | 0.6190±0.1347 | 0.6605±0.0915 | 0.8459±0.0914 | TP:3 FP:1 FN:4 TN:16 (Pos:7 Neg:17) | 11,173,889 | 151.1 |
| ResNet_pretrained | 0.7778±0.0196 | 0.6488±0.0748 | 0.5714±0.1166 | 0.5945±0.0521 | 0.8403±0.0562 | TP:4 FP:3 FN:3 TN:14 (Pos:7 Neg:17) | 6,067,201 | 73.9 |

## [WARNING] Circularity / Low Support Flags


[WARNING] LOW SUPPORT -- METRICS ON THIS SPLIT ARE NOISY
   Val Positives: 6, Test Positives: 7

