# Block 3b: Val-Only Optuna Sweep & Magnitude Pruning

## 1. Val-Only Optuna Hyperparameter Sweep (12 Trials, Zero Test Peeking)

- **Winning Config:** `base_ch=16`, `lr=8.5742e-04`, `dropout_p=0.198`, `dice_weight=0.228`
- **Best Val Pixel PR-AUC:** `0.0109`
- **Evaluated Once on Test:** Pixel PR-AUC = `0.0108`, Pixel ROC-AUC = `0.9484`, Dice (val th=0.45) = `0.0286`, Patch Mean-Prob ROC-AUC = `0.8497`

## 2. Global Magnitude Pruning + 3-Epoch Fine-Tune (`IceUNet_seed42`)

| Target Sparsity | Actual Sparsity | Non-Zero Params | Pixel PR-AUC | Dice (Val Th) | Patch Mean-Prob ROC-AUC | CUDA Latency (ms / B=64) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0% | 0.0% | 1,084,681 | 0.0093 | 0.0295 | 0.8291 | 11.10 |
| 30% | 29.9% | 759,961 | 0.0068 | 0.0272 | 0.8400 | 11.24 |
| 50% | 49.9% | 543,481 | 0.0079 | 0.0240 | 0.8382 | 11.31 |
| 70% | 69.9% | 327,001 | 0.0065 | 0.0261 | 0.8335 | 11.20 |
| 90% | 89.8% | 110,521 | 0.0058 | 0.0258 | 0.8351 | 11.18 |
