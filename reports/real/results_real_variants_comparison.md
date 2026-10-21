# Real-Data Benchmark Comparison: Pretrained ResNet Variants vs. Baselines

> **Protocol:** Evaluated on real Chandrayaan-2 DFSAR L1A quad-pol SLI dataset (lines 100,000–140,000). Max 30 epochs, early stopping on Val ROC-AUC (patience 5), validation-tuned decision threshold, 95% block-bootstrap CIs across strips (1000 rounds).
> **Methodological Note on Ranking:** As established in the evaluation protocol, model architectures are ranked strictly by **Validation ROC-AUC** to prevent data leakage and cherry-picking based on test set metrics.

---

## 1. Per-Run Metrics (Seeds 42, 123, 2024)

### Full Spatial Test Set (N=338, Prevalence=29.9%)

| Model Architecture | Seed | Best Ep | Val AUC | Test ROC-AUC (95% CI) | Test PR-AUC (95% CI) | Test F1 (95% CI) | Tuned Th | Prev | Support |
|---|---|---|---|---|---|---|---|---|---|
| **CustomCNN** | 42 | 2 | 0.6712 | 0.7468 `[0.5676, 0.8722]` | 0.4990 `[0.0886, 0.6156]` | 0.5962 `[0.0000, 0.6579]` | 0.17 | 29.9% | TP:93 FP:118 FN:8 TN:119 (Pos:101 Neg:237) |
| **CustomCNN** | 123 | 1 | 0.6706 | 0.7508 `[0.5664, 0.8740]` | 0.4729 `[0.0965, 0.5866]` | 0.6175 `[0.0000, 0.7132]` | 0.05 | 29.9% | TP:88 FP:96 FN:13 TN:141 (Pos:101 Neg:237) |
| **CustomCNN** | 2024 | 12 | 0.6936 | 0.7715 `[0.6186, 0.8712]` | 0.5569 `[0.0735, 0.6646]` | 0.6202 `[0.0000, 0.7077]` | 0.45 | 29.9% | TP:89 FP:97 FN:12 TN:140 (Pos:101 Neg:237) |
| **ResNet_scratch** | 42 | 7 | 0.6504 | 0.7388 `[0.5484, 0.8535]` | 0.4635 `[0.0741, 0.6440]` | 0.5734 `[0.0000, 0.6260]` | 0.77 | 29.9% | TP:82 FP:103 FN:19 TN:134 (Pos:101 Neg:237) |
| **ResNet_scratch** | 123 | 4 | 0.6638 | 0.7551 `[0.5823, 0.8727]` | 0.5290 `[0.1044, 0.6589]` | 0.5965 `[0.0000, 0.6720]` | 0.92 | 29.9% | TP:85 FP:99 FN:16 TN:138 (Pos:101 Neg:237) |
| **ResNet_scratch** | 2024 | 13 | 0.6371 | 0.7655 `[0.5851, 0.8766]` | 0.5039 `[0.1744, 0.6521]` | 0.5742 `[0.1333, 0.6429]` | 0.70 | 29.9% | TP:89 FP:120 FN:12 TN:117 (Pos:101 Neg:237) |
| **ResNet_pretrained (Original)** | 42 | 2 | 0.5520 | 0.6591 `[0.5676, 0.7468]` | 0.4344 `[0.1455, 0.5336]` | 0.5188 `[0.1364, 0.6142]` | 0.41 | 29.9% | TP:76 FP:116 FN:25 TN:121 (Pos:101 Neg:237) |
| **ResNet_pretrained (Original)** | 123 | 9 | 0.5739 | 0.6572 `[0.4975, 0.7334]` | 0.4500 `[0.1114, 0.6026]` | 0.4696 `[0.0690, 0.5946]` | 0.44 | 29.9% | TP:58 FP:88 FN:43 TN:149 (Pos:101 Neg:237) |
| **ResNet_pretrained (Original)** | 2024 | 6 | 0.5736 | 0.6812 `[0.5629, 0.7470]` | 0.5052 `[0.3060, 0.6106]` | 0.5016 `[0.0833, 0.6197]` | 0.17 | 29.9% | TP:80 FP:138 FN:21 TN:99 (Pos:101 Neg:237) |
| **ResNet_pretrained_v2a (Fixed Freeze + Native Conv1)** | 42 | 7 | 0.6729 | 0.7020 `[0.4995, 0.8066]` | 0.4609 `[0.1085, 0.5857]` | 0.4486 `[0.0000, 0.5128]` | 0.53 | 29.9% | TP:48 FP:65 FN:53 TN:172 (Pos:101 Neg:237) |
| **ResNet_pretrained_v2a (Fixed Freeze + Native Conv1)** | 123 | 4 | 0.6612 | 0.6979 `[0.5413, 0.7924]` | 0.4734 `[0.0672, 0.5561]` | 0.5018 `[0.0000, 0.5547]` | 0.50 | 29.9% | TP:70 FP:108 FN:31 TN:129 (Pos:101 Neg:237) |
| **ResNet_pretrained_v2a (Fixed Freeze + Native Conv1)** | 2024 | 2 | 0.6630 | 0.6630 `[0.4220, 0.7291]` | 0.4341 `[0.0449, 0.5801]` | 0.5108 `[0.0000, 0.5891]` | 0.56 | 29.9% | TP:71 FP:106 FN:30 TN:131 (Pos:101 Neg:237) |
| **ResNet_pretrained_v2b (Differential LR: 1e-4 / 1e-3)** | 42 | 1 | 0.5947 | 0.6478 `[0.4147, 0.6917]` | 0.3985 `[0.0431, 0.5751]` | 0.4639 `[0.0000, 0.5470]` | 0.52 | 29.9% | TP:61 FP:101 FN:40 TN:136 (Pos:101 Neg:237) |
| **ResNet_pretrained_v2b (Differential LR: 1e-4 / 1e-3)** | 123 | 4 | 0.6402 | 0.6875 `[0.5358, 0.7515]` | 0.4740 `[0.0577, 0.5999]` | 0.4510 `[0.0000, 0.4889]` | 0.53 | 29.9% | TP:46 FP:57 FN:55 TN:180 (Pos:101 Neg:237) |
| **ResNet_pretrained_v2b (Differential LR: 1e-4 / 1e-3)** | 2024 | 3 | 0.6304 | 0.6700 `[0.4280, 0.7422]` | 0.4425 `[0.1466, 0.6511]` | 0.4839 `[0.0870, 0.6038]` | 0.57 | 29.9% | TP:60 FP:87 FN:41 TN:150 (Pos:101 Neg:237) |

---

## 2. Summary Benchmark: Mean ± Std Across Seeds

| Model Architecture | Trainable Params | Val ROC-AUC | Test ROC-AUC | Test PR-AUC | Test F1 |
|---|---|---|---|---|---|
| **CustomCNN** | 618,497 | **0.6784 ± 0.0131** | 0.7564 ± 0.0132 | 0.5096 ± 0.0430 | 0.6113 ± 0.0132 |
| **ResNet_pretrained_v2a (Fixed Freeze + Native Conv1)** | 10,503,361 | **0.6657 ± 0.0063** | 0.6876 ± 0.0214 | 0.4561 ± 0.0201 | 0.4871 ± 0.0336 |
| **ResNet_scratch** | 11,177,025 | **0.6504 ± 0.0134** | 0.7531 ± 0.0135 | 0.4988 ± 0.0330 | 0.5814 ± 0.0131 |
| **ResNet_pretrained_v2b (Differential LR: 1e-4 / 1e-3)** | 10,503,361 | **0.6217 ± 0.0240** | 0.6684 ± 0.0199 | 0.4383 ± 0.0379 | 0.4662 ± 0.0166 |
| **ResNet_pretrained (Original)** | 6,067,201 | **0.5665 ± 0.0126** | 0.6658 ± 0.0133 | 0.4632 ± 0.0372 | 0.4967 ± 0.0249 |

---

## 3. Architecture Ranking (Strictly by Mean Validation ROC-AUC)

1. **CustomCNN** (`0.6784 ± 0.0131`): Top performer on validation criteria; compact 3-layer representation avoids overfitting to radar speckle.
2. **ResNet_pretrained_v2a** (`0.6657 ± 0.0063`): Substantial improvement over original pretrained (+0.0992 Val AUC). Fixing the substring-freezing bug in `layer3`/`layer4` and maintaining native 3-channel weights on `conv1` elevated validation discriminability above `ResNet_scratch`.
3. **ResNet_scratch** (`0.6504 ± 0.0134`): Solid baseline; learns SAR-specific representations from scratch but exhibits wider variance across seeds on validation splits.
4. **ResNet_pretrained_v2b** (`0.6217 ± 0.0240`): Differential learning rate (backbone `1e-4`, head `1e-3`) underperformed v2a (`1e-3`), indicating that `1e-4` is overly conservative to bridge the cross-domain optical-to-radar distribution shift within 30 epochs.
5. **ResNet_pretrained (Original)** (`0.5665 ± 0.0126`): Severely impaired by the dual issues of channel collapse in `conv1` (averaged RGB weights) and accidental intra-block freezing of `conv1`/`bn1` across `layer3` and `layer4`.
