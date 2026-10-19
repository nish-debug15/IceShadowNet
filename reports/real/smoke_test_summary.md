# SMOKE TEST SUMMARY (Not Final Results)

> **[SMOKE TEST ONLY - NOT FINAL BENCHMARK RESULTS]:**
> Proves real data flows through model architecture and training pipeline over 3 epochs.
> Do not report these metrics as final research results.

> **DATA SOURCE:** REAL (Real Chandrayaan-2 DFSAR L-band quad-pol data).

> **Trained for 3 epochs, 1 seeds, on REAL data**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|---------|--------|----------|
| CustomCNN (SMOKE TEST) | 0.6509±0.0000 | 0.4577±0.0000 | 0.9109±0.0000 | 0.6093±0.0000 | 0.7493±0.0000 | TP:92 FP:109 FN:9 TN:128 (Pos:101 Neg:237) | 618,497 | 31.1 |
