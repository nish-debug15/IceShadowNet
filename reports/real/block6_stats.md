# Block 6 Item 1: Per-Strip Metrics, 5-Fold Blocked-Strip CV, & Paired Sub-Block Bootstrap

> **Statistical Protocol:** 3-strip bootstrap CIs (only 10 distinct resamples) are retired as ranking evidence. Primary comparison uses (1) per-test-strip metrics, (2) 5-fold blocked-strip CV across all 20 strips, and (3) paired bootstrap of metric differences over `256`-row sub-blocks with `64`-row guard gaps (*within-test-set variability only*). Models are separable only if the paired 95% CI excludes `0`.

## 1. Per-Test-Strip Metrics (Frozen Split: Strip 0 [20.1%], Strip 8 [10.3%], Strip 15 [19.1%])

| Model | Pooled ROC / PR / F1 | Strip 0 ROC / PR / F1 | Strip 8 ROC / PR / F1 | Strip 15 ROC / PR / F1 |
| :--- | :---: | :---: | :---: | :---: |
| **CustomCNN** | 0.844 / 0.566 / 0.537 | 0.942 / 0.783 / 0.771 | 0.662 / 0.222 / 0.262 | 0.763 / 0.380 / 0.431 |
| **IceUNet_mean_prob** | 0.846 / 0.564 / 0.530 | 0.950 / 0.796 / 0.787 | 0.637 / 0.183 / 0.231 | 0.773 / 0.395 / 0.457 |
| **ResNet_scratch** | 0.840 / 0.544 / 0.529 | 0.945 / 0.783 / 0.777 | 0.640 / 0.174 / 0.214 | 0.770 / 0.382 / 0.460 |
| **LogisticRegression** | 0.821 / 0.498 / 0.509 | 0.916 / 0.643 / 0.685 | 0.605 / 0.228 / 0.261 | 0.785 / 0.422 / 0.441 |
| **ResNet_pretrained** | 0.815 / 0.480 / 0.489 | 0.924 / 0.654 / 0.730 | 0.602 / 0.137 / 0.236 | 0.751 / 0.385 / 0.473 |
| **DecisionTree** | 0.616 / 0.213 / 0.314 | 0.602 / 0.241 / 0.336 | 0.508 / 0.104 / 0.184 | 0.630 / 0.247 / 0.386 |

## 2. Paired Sub-Block Bootstrap Differences (*Within-Test-Set Variability Only*, 256-Row Blocks + 64-Row Guard)

| Model Pair (A - B) | Sub-Blocks | $\Delta$ROC-AUC [95% CI] | ROC Separable? | $\Delta$PR-AUC [95% CI] | PR Separable? |
| :--- | :---: | :---: | :---: | :---: | :---: |
| CustomCNN - LogisticRegression | 104 | +0.0228 `[-0.0109, +0.0546]` | NO | +0.0686 `[-0.0422, +0.1520]` | NO |
| CustomCNN - ResNet_scratch | 104 | +0.0032 `[-0.0084, +0.0183]` | NO | +0.0218 `[-0.0223, +0.0490]` | NO |
| CustomCNN - ResNet_pretrained | 104 | +0.0288 `[+0.0077, +0.0558]` | **YES** | +0.0860 `[-0.0035, +0.1746]` | NO |
| IceUNet_mean_prob - CustomCNN | 104 | +0.0020 `[-0.0092, +0.0108]` | NO | -0.0020 `[-0.0319, +0.0307]` | NO |
| IceUNet_mean_prob - LogisticRegression | 104 | +0.0248 `[-0.0140, +0.0568]` | NO | +0.0666 `[-0.0543, +0.1592]` | NO |
| ResNet_scratch - ResNet_pretrained | 104 | +0.0256 `[+0.0053, +0.0452]` | **YES** | +0.0642 `[-0.0214, +0.1552]` | NO |

## 3. 5-Fold Blocked-Strip CV Across All 20 Strips (256-Row Buffer)

| Model | Mean ROC-AUC ± Std | Mean PR-AUC ± Std | Mean F1 (Val Th) ± Std |
| :--- | :---: | :---: | :---: |
| **LogisticRegression** | 0.8118 ± 0.0460 | 0.4921 ± 0.1197 | 0.4804 ± 0.1192 |
| **CustomCNN** | 0.8124 ± 0.0408 | 0.4760 ± 0.1093 | 0.4684 ± 0.1059 |
| **ResNet_scratch** | 0.7978 ± 0.0427 | 0.4641 ± 0.1110 | 0.4228 ± 0.1233 |
| **IceUNet_mean_prob** | 0.7955 ± 0.0448 | 0.4169 ± 0.1328 | 0.1941 ± 0.1738 |

### Per-Fold Breakdown

| Fold | Prev | Model | ROC-AUC | PR-AUC | F1 |
| :---: | :---: | :--- | :---: | :---: | :---: |
| 1 | 22.4% | LogisticRegression | 0.8041 | 0.5082 | 0.5531 |
| 1 | 22.4% | CustomCNN | 0.8183 | 0.5570 | 0.5190 |
| 1 | 22.4% | ResNet_scratch | 0.8051 | 0.5270 | 0.5222 |
| 1 | 22.4% | IceUNet_mean_prob | 0.8250 | 0.5607 | 0.4581 |
| 2 | 8.2% | LogisticRegression | 0.7748 | 0.2746 | 0.3101 |
| 2 | 8.2% | CustomCNN | 0.7960 | 0.2887 | 0.3044 |
| 2 | 8.2% | ResNet_scratch | 0.7626 | 0.2790 | 0.2935 |
| 2 | 8.2% | IceUNet_mean_prob | 0.7913 | 0.2845 | 0.3155 |
| 3 | 13.4% | LogisticRegression | 0.7758 | 0.5245 | 0.4351 |
| 3 | 13.4% | CustomCNN | 0.7937 | 0.4998 | 0.4663 |
| 3 | 13.4% | ResNet_scratch | 0.7889 | 0.5131 | 0.3991 |
| 3 | 13.4% | IceUNet_mean_prob | 0.7564 | 0.3428 | 0.1712 |
| 4 | 35.4% | LogisticRegression | 0.8040 | 0.6427 | 0.6625 |
| 4 | 35.4% | CustomCNN | 0.7668 | 0.6012 | 0.6260 |
| 4 | 35.4% | ResNet_scratch | 0.7567 | 0.5964 | 0.6036 |
| 4 | 35.4% | IceUNet_mean_prob | 0.7410 | 0.5938 | 0.0257 |
| 5 | 5.0% | LogisticRegression | 0.9000 | 0.5103 | 0.4412 |
| 5 | 5.0% | CustomCNN | 0.8871 | 0.4335 | 0.4265 |
| 5 | 5.0% | ResNet_scratch | 0.8755 | 0.4051 | 0.2955 |
| 5 | 5.0% | IceUNet_mean_prob | 0.8636 | 0.3024 | 0.0000 |
