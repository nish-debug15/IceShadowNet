# Training Summary

> **DATA SOURCE:** SYNTHETIC (physically-motivated SAR scene -- no real DFSAR tiles from PRADAN yet). Numbers below demonstrate pipeline correctness only, NOT real ice-detection performance.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|---------|--------|----------|
| CustomCNN | 0.7281±0.0124 | 0.4444±0.3143 | 0.1212±0.0857 | 0.1905±0.1347 | 0.4882±0.0225 | TP:2 FP:1 FN:9 TN:26 (Pos:11 Neg:27) | 618,209 | 10.3 |
| ResNet_scratch | 0.7105±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.4820±0.0794 | TP:0 FP:0 FN:11 TN:27 (Pos:11 Neg:27) | 11,173,889 | 24.6 |
| ResNet_pretrained | 0.7105±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.4355±0.0894 | TP:0 FP:0 FN:11 TN:27 (Pos:11 Neg:27) | 6,067,201 | 12.0 |

## [WARNING] Circularity / Low Support Flags


[WARNING] MAJORITY-CLASS COLLAPSE: ResNet_scratch
   accuracy=0.711, Precision=0.000, Recall=0.000
   The model failed to learn the minority class and is predicting a single class for everything.


[WARNING] MAJORITY-CLASS COLLAPSE: ResNet_pretrained
   accuracy=0.711, Precision=0.000, Recall=0.000
   The model failed to learn the minority class and is predicting a single class for everything.

