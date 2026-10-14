# Multimodal Training Summary

> **DATA SOURCE:** SYNTHETIC. See reports/data_notes.md.

> **Trained for 30 epochs, 3 seeds, on SYNTHETIC data**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|---------|--------|----------|
| EarlyFusionCNN | 0.7222±0.0196 | 0.2222±0.3143 | 0.0952±0.1347 | 0.1333±0.1886 | 0.4594±0.0309 | TP:0 FP:0 FN:7 TN:17 (Pos:7 Neg:17) | 618,497 | 54.0 |
| LateFusionResNet | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | TP:7 FP:0 FN:0 TN:17 (Pos:7 Neg:17) | 22,606,273 | 200.1 |

## [WARNING] Circularity / Low Support Flags


[WARNING] LOW SUPPORT -- METRICS ON THIS SPLIT ARE NOISY
   Val Positives: 6.0, Test Positives: 7.0


[WARNING] SUSPICIOUS PERFORMANCE: LateFusionResNet
   accuracy=1.000, F1=1.000
   This may indicate label circularity: the model could be
   re-learning the CPR>1 AND DOP<0.13 threshold rule rather
   than learning spatial/textural features.
   -> Inspect Grad-CAM (Block 3) and check if attention focuses
     on physically plausible spatial structures beyond single pixels.
   -> Report this limitation explicitly in viva and report.


[WARNING] MAJORITY-CLASS COLLAPSE: EarlyFusionCNN
   accuracy=0.708, Precision=0.000, Recall=0.000
   The model failed to learn the minority class and is predicting a single class for everything.

