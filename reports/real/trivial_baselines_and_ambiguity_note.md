# Trivial Baselines & Ambiguity Band Analysis (Real DFSAR Test Data)

> **DATA SOURCE:** Real Chandrayaan-2 DFSAR L1A quad-pol SLI (lines 100,000–140,000).  
> **Evaluation Split:** Spatial test set ($N=338$, patch size $64 \times 64$, along-track strips 6, 10, 14).

---

## 1. Trivial Baselines Benchmark

### A. Full Spatial Test Set ($N=338$, Prevalence = 29.88%)
- **Class Distribution:** Positive = 101, Negative = 237
- **All-Positive Classifier** (predicts label 1 for every patch):
  - **Precision:** 0.2988 (29.9%)
  - **Recall:** 1.0000 (100.0%)
  - **F1 Score:** **0.4601**
- **Random-Score Classifier** (uniform continuous predictions $\sim U(0, 1)$):
  - **ROC-AUC:** **0.5000**
  - **PR-AUC:** **0.2988** (equal to test set prevalence)

### B. Ambiguity-Removed "Clean" Test Set ($N=205$, Prevalence = 49.27%)
- **Class Distribution:** Positive = 101, Negative = 104
- **All-Positive Classifier** (predicts label 1 for every patch):
  - **Precision:** 0.4927 (49.3%)
  - **Recall:** 1.0000 (100.0%)
  - **F1 Score:** **0.6601**
- **Random-Score Classifier**:
  - **ROC-AUC:** **0.5000**
  - **PR-AUC:** **0.4927** (equal to clean test set prevalence)

---

## 2. Ambiguity Band Breakdown & Structural Impact Note

### Exact Composition of the 1–4 Positive-Pixel Transition Band
- During dataset generation, binary patch labels were assigned with threshold $\ge 5$ positive pixels out of $64 \times 64 = 4096$ pixels ($0.122\%$ area).
- The ambiguity band captures patches containing **between 1 and 4 positive pixels** (boundary transition speckle / marginal scattering).
- Total ambiguous patches in test set: **133 patches**.
  - **Negatives removed:** **133 of 237 test negatives (56.1%)**
  - **Positives removed:** **0 of 101 test positives (0.0%)**

### Methodological Implications
1. **Selective Negative Pruning:** Because the positive threshold is $\ge 5$ pixels, patches with 1–4 positive pixels are labeled as negative in ground truth. Consequently, the ambiguity filter removes **exclusively hard negatives** (sub-threshold crater edge and foreshortening speckle patches).
2. **Prevalence Shift:** Removing 133 negatives while keeping all 101 positives shifts the test prevalence from $29.9\%$ directly to $49.3\%$ (near-balanced).
3. **Baseline Inflation:** The trivial all-positive baseline F1 score automatically shifts from **0.4601** to **0.6601** without any model improvement, simply due to the altered class balance.
4. **Model Performance Context:** While deep learning models display higher F1 on the clean test set ($\sim 0.78 - 0.82$ vs $\sim 0.58 - 0.61$), comparing against the shifted all-positive baseline ($0.6601$) demonstrates that boundary transition speckle accounts for a substantial fraction of false alarms in lunar SAR interpretation.
