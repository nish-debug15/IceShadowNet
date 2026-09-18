# Training Summary

> **DATA SOURCE:** SYNTHETIC (physically-motivated SAR scene — no real DFSAR tiles from PRADAN yet). Numbers below demonstrate pipeline correctness only, NOT real ice-detection performance.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|--------|----------|
| CustomCNN | 0.7368 | 0.6667 | 0.1818 | 0.2857 | 0.5522 | 618,209 | 331.8 |
| ResNet_scratch | 0.8421 | 0.8571 | 0.5455 | 0.6667 | 0.8384 | 11,173,889 | 726.7 |
| ResNet_pretrained | 0.8421 | 1.0000 | 0.4545 | 0.6250 | 0.8687 | 6,067,201 | 426.2 |
