# Classical Baselines on Real Chandrayaan-2 DFSAR L1A (Overhaul Evaluation)

> **DATA SOURCE:** REAL (Chandrayaan-2 DFSAR L1A quad-pol SLI, lines 100,000–140,000)
> **Spatial Test Split:** 2018 patches (332 positive, 1686 negative, Prevalence=16.5%)
> **Evaluation Protocol:** PR-AUC is headline metric; MCC and Balanced Accuracy replace unweighted accuracy; thresholds tuned strictly on Val; block-bootstrap 95% CIs across strips; ECE calibration computed.

## 1. Primary Test Evaluation Table (Full Spatial Test Set: N=338)

| Model | Tuned Th | PR-AUC (Primary) (95% CI) | ROC-AUC (95% CI) | MCC (95% CI) | Balanced Acc (95% CI) | F1 (95% CI) | Precision | Recall | ECE | Support |
|---|---|---|---|---|---|---|---|---|---|---|
| Trivial Baseline (All Negative) | 0.50 | **0.1645**<br>`[0.1029, 0.2006]` | **0.5000**<br>`[0.5000, 0.5000]` | **0.0000**<br>`[0.0000, 0.0000]` | 0.5000<br>`[0.5000, 0.5000]` | 0.0000<br>`[0.0000, 0.0000]` | 0.0000 | 0.0000 | 0.1645 | TP:0 FP:0 FN:332 TN:1686 (Pos:332 Neg:1686) |
| Rule Baseline (Circularity Yardstick: CPR>1 & DOP<0.13) | 0.00 | **1.0000**<br>`[1.0000, 1.0000]` | **1.0000**<br>`[1.0000, 1.0000]` | **1.0000**<br>`[1.0000, 1.0000]` | 1.0000<br>`[1.0000, 1.0000]` | 1.0000<br>`[1.0000, 1.0000]` | 1.0000 | 1.0000 | 0.1640 | TP:332 FP:0 FN:0 TN:1686 (Pos:332 Neg:1686) |
| BASELINE: Decision Tree (max_depth=3, balanced) | 0.31 | **0.1822**<br>`[0.1088, 0.2549]` | **0.5072**<br>`[0.4102, 0.6441]` | **-0.0453**<br>`[-0.2214, 0.1703]` | 0.4760<br>`[0.3736, 0.5792]` | 0.2598<br>`[0.1889, 0.3590]` | 0.1563 | 0.7681 | 0.3287 | TP:255 FP:1376 FN:77 TN:310 (Pos:332 Neg:1686) |
| BASELINE: Logistic Regression (balanced) | 0.55 | **0.5310**<br>`[0.1672, 0.7877]` | **0.8040**<br>`[0.5587, 0.9337]` | **0.2967**<br>`[0.0281, 0.5619]` | 0.6999<br>`[0.5230, 0.8451]` | 0.4153<br>`[0.1867, 0.6318]` | 0.2785 | 0.8163 | 0.3554 | TP:271 FP:702 FN:61 TN:984 (Pos:332 Neg:1686) |

---

## 2. Ambiguity-Removed Evaluation (Clean Test Set: N=205)

> Excludes 806 test patches containing between 1 and 4 positive pixels. Leaves 1212 clean patches (332 pos, 880 neg).

| Model | Tuned Th | PR-AUC (Primary) (95% CI) | ROC-AUC (95% CI) | MCC (95% CI) | Balanced Acc (95% CI) | F1 (95% CI) | Precision | Recall | ECE | Support |
|---|---|---|---|---|---|---|---|---|---|---|
| BASELINE: Decision Tree (max_depth=3, balanced) | 0.31 | **0.3048**<br>`[0.2282, 0.4810]` | **0.5081**<br>`[0.3967, 0.7170]` | **-0.0123**<br>`[-0.2902, 0.3211]` | 0.4943<br>`[0.3519, 0.6426]` | 0.4006<br>`[0.2688, 0.5727]` | 0.2710 | 0.7681 | 0.2658 | TP:255 FP:686 FN:77 TN:194 (Pos:332 Neg:880) |
| BASELINE: Logistic Regression (balanced) | 0.55 | **0.7379**<br>`[0.3125, 0.9209]` | **0.8672**<br>`[0.5830, 0.9621]` | **0.4932**<br>`[0.0749, 0.7312]` | 0.7729<br>`[0.5449, 0.9009]` | 0.6445<br>`[0.3559, 0.7987]` | 0.5324 | 0.8163 | 0.2124 | TP:271 FP:238 FN:61 TN:642 (Pos:332 Neg:880) |
