# Training Summary

> **DATA SOURCE:** SYNTHETIC (physically-motivated SAR scene — no real DFSAR tiles from PRADAN yet). Numbers below demonstrate pipeline correctness only, NOT real ice-detection performance.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|--------|----------|
| CustomCNN | 0.8947 | 0.0000 | 0.0000 | 0.0000 | 0.4118 | 618,209 | 18.7 |
| ResNet_scratch | 0.1053 | 0.1053 | 1.0000 | 0.1905 | 0.6103 | 11,173,889 | 33.6 |
| ResNet_pretrained | 0.8421 | 0.0000 | 0.0000 | 0.0000 | 0.4632 | 6,067,201 | 19.3 |
