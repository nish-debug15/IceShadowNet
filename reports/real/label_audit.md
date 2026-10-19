# DFSAR Label Consistency Audit & Threshold Sensitivity Analysis
_Generated: 2026-10-19 (Phase 5 - Data Integrity Resolution)_

---

## 1. Executive Summary & Reconciliation

| Quantity | Quicklook Evaluation (Block 2) | Patch Labeler Input (Block 3) | Reconciled Status |
|---|---|---|---|
| **Scene Slice** | Rows [125,000 : 150,000) | Rows [100,000 : 140,000) | **Identified as Root Discrepancy** |
| **Swath Dimensions** | 25,000 rows × 116 cols | 40,000 rows × 116 cols | Extended for robust spatial split |
| **Analyzed Pixels** | 2,875,000 (power > 1e-3) | 4,640,000 (full grid) | Full swath used for patch extraction |
| **CPR > 1.0 Pixels** | 18,136 (0.6308%) | 80,707 (1.7394%) | Consistent with broader scene |
| **DOP < 0.13 Pixels** | 2,791 (0.0971%) | 8,959 (0.1931%) | Consistent with broader scene |
| **Both (CPR > 1.0 & DOP < 0.13)** | **898 pixels (0.0312%)** | **3,033 pixels (0.0654%)** | **Mask consumed by patch labeler** |
| **Total Patches ($64 \times 64$, stride 32)** | — | 2,498 | Swath width 116 yields 2 columns |
| **Positive Patches ($\ge 5$ px)** | — | 585 (23.42%) | Derived from the 3,033 mask pixels |

### Arithmetic Reconciliation

1. **Patch Overlap Bound:** With patch size $P=64$ and stride $S=32$, any pixel $(r, c)$ in the 2D grid can belong to at most:
   $$\left\lceil \frac{P}{S} \right\rceil \times \left\lceil \frac{P}{S} \right\rceil = 2 \times 2 = 4 \text{ overlapping patches.}$$
2. **Hypothetical 896-pixel Mask Bound:**
   - If the patch labeler had consumed the 896-pixel mask from the 25,000-line quicklook slice, the theoretical maximum sum of positive pixels over all patches would be:
     $$\sum_{\text{patches}} \text{pos\_px} \le 4 \times 896 = 3,584 \text{ pixel-instances.}$$
   - For 585 positive patches with $\ge 5$ pixels each, the minimum positive-pixel sum among positive patches alone is $585 \times 5 = 2,925 \le 3,584$.
   - **However**, the actual measured sum of positive pixels across all 2,498 patches extracted in Block 3 is **6,584**!
   - Because $6,584 > 3,584$, it is **arithmetically impossible** for the 2,498 patches to have been produced by an 896-pixel mask.
3. **True Mask (3,033 pixels) Arithmetic Check:**
   - The patch labeler consumed rows 100,000 to 140,000, where the exact mask contains **3,033 positive pixels**.
   - Maximum possible patch sum: $4 \times 3,033 = 12,132$.
   - Observed patch sum: **6,584** ($6,584 \le 12,132$).
   - The observed average overlap multiplicity is $6,584 / 3,033 \approx 2.17$, exactly matching expectations for a narrow 2-column strip where edge pixels are covered by only 2 patches and interior pixels by 4.
4. **Resolution:**
   - The discrepancy was caused by comparing the Block 2 quicklook script slice (rows 125,000–150,000, 898 positive pixels) against the Block 3 patch ingestion slice (rows 100,000–140,000, 3,033 positive pixels). Specifically, rows 100,000–125,000 contain an additional dense cluster of 2,561 positive pixels.
   - `reports/real/method_notes.md` and quicklook visualizations have been updated to document the exact 40,000-row slice (`start_row=100000, num_rows=40000`) consumed by the patch labeler.
   - A typographical discrepancy in `reports/real/method_notes.md` (§3, stating `patch_label_fraction = 0.05` / 5% instead of `min_pos_pixels = 5` / 0.12%) is corrected.

---

## 2. Positive-Pixel Distribution per Patch (All 2,498 Patches)

The patch labeler evaluates $64 \times 64$ patches with stride 32 over the $40,000 \times 116$ radar swath:

- **Total Patches:** 2,498
- **Minimum positive pixels:** 0
- **Median positive pixels:** 1.0
- **Mean positive pixels:** 2.6357
- **Maximum positive pixels:** 21 (0.51% of patch area)
- **Patches exactly at threshold ($\text{pos\_px} = 5$):** 135 patches (5.40% of all patches, 23.08% of positive patches)

### Full Histogram of Positive-Pixel Counts

| Positive Pixels in Patch | Patch Count | Percentage of Total | Cumulative % ($\le k$) |
|---|---|---|---|
| **0 px** | 1,019 | 40.79% | 40.79% |
| **1 px** | 294 | 11.77% | 52.56% |
| **2 px** | 261 | 10.45% | 63.01% |
| **3 px** | 201 | 8.05% | 71.06% |
| **4 px** | 138 | 5.52% | 76.58% (Negative class threshold) |
| **5 px (Threshold)** | **135** | **5.40%** | **81.99% (First positive bin)** |
| **6 px** | 112 | 4.48% | 86.47% |
| **7 px** | 90 | 3.60% | 90.07% |
| **8 px** | 68 | 2.72% | 92.79% |
| **9 px** | 53 | 2.12% | 94.92% |
| **10 px** | 36 | 1.44% | 96.36% |
| **11 px** | 23 | 0.92% | 97.28% |
| **12 px** | 10 | 0.40% | 97.68% |
| **13 px** | 11 | 0.44% | 98.12% |
| **14 px** | 18 | 0.72% | 98.84% |
| **15 px** | 8 | 0.32% | 99.16% |
| **16 px** | 7 | 0.28% | 99.44% |
| **17 px** | 6 | 0.24% | 99.68% |
| **18 px** | 3 | 0.12% | 99.80% |
| **19 px** | 2 | 0.08% | 99.88% |
| **20 px** | 2 | 0.08% | 99.96% |
| **21 px** | 1 | 0.04% | 100.00% |
| **$\ge 22$ px** | 0 | 0.00% | 100.00% |

---

## 3. Threshold Sensitivity Table

We evaluate the impact of the minimum positive-pixel threshold ($T_{\text{pos}}$) on patch classification:

| Min-Pixel Threshold ($T_{\text{pos}}$) | Patch Fraction ($\%$) | Positive Patches | Negative Patches | Positive Rate ($\%$) | Feasibility for Deep Learning |
|---|---|---|---|---|---|
| **$\ge 1$ pixel** | $0.024\%$ | 1,479 | 1,019 | 59.21% | High risk of false positives from isolated speckle noise |
| **$\ge 5$ pixels (Selected)** | **$0.122\%$** | **585** | **1,913** | **23.42%** | **Balanced class ratio ($\approx 1:3.3$), filters single-pixel noise** |
| **$\ge 10$ pixels** | $0.244\%$ | 127 | 2,371 | 5.08% | Severe class imbalance ($1:18.7$); splits would have $\approx 15$ test positives |
| **$\ge 20$ pixels** | $0.488\%$ | 3 | 2,495 | 0.12% | Insufficient data; impossible to satisfy $\ge 10$ positives per split |
| **$\ge 50$ pixels** | $1.221\%$ | 0 | 2,498 | 0.00% | Zero positive samples across entire scene |

### Conclusion on Threshold Choice
The current choice of $T_{\text{pos}} = 5$ pixels ($0.12\%$ of patch area) strikes the optimal physical and statistical trade-off:
1. It eliminates isolated 1-to-4 pixel speckle noise spikes that can occur in radar polarimetry despite $7 \times 7$ multilooking.
2. It yields 585 positive samples (23.4% positive rate), providing ample statistical support across Train (389), Val (86), and Test (110) splits to train deep neural networks without severe imbalance failure.
