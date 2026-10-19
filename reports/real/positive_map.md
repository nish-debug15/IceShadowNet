# Chandrayaan-2 DFSAR L1A: Along-Track Spatial Strip Anomaly Map
_Generated: 2026-10-19 (Phase 5 Data Integrity Audit)_

---

## 1. Executive Summary: Spatial Clustering & Multi-Split Distribution

> **KEY FINDING ON ANOMALY CLUSTERING:**
> **YES, the dominant polarimetric anomaly clusters span strips from more than one split.**
>
> Rather than being confined to a single contiguous block or isolated split, the high-CPR/low-DOP anomaly clusters extend across adjacent along-track blocks that are assigned to **Train**, **Validation**, and **Test** splits:
> 1. **Cluster Alpha (Strips 4–7, Rows 7,987 to 15,974):** Spans across **Val** (Strip 4: 194 pos px, 31 pos patches), **Train** (Strip 5: 219 pos px, 40 pos patches), **Test** (Strip 6: 205 pos px, 43 pos patches), and **Val** (Strip 7: 193 pos px, 29 pos patches).
> 2. **Cluster Beta (Strips 9–11, Rows 17,971 to 23,961):** Spans across **Train** (Strip 9: 204 pos px, 41 pos patches), **Test** (Strip 10: 265 pos px, 53 pos patches), and **Train** (Strip 11: 247 pos px, 51 pos patches).
> 3. **Non-Anomalous Background (Strips 16–19, Rows 31,948 to 40,000):** Consists almost entirely of non-anomalous surface backscatter (< 35 positive pixels per strip) assigned to the Train set.

This interleave verifies that:
- The test set is evaluated on genuine crater anomaly geological features that share physical generation mechanisms with the training set, rather than an unrepresentative non-anomalous region.
- The 64-pixel boundary buffer completely isolates the pixel extents between these adjacent strips, preventing any pixel-level leakage across the block interfaces.

---

## 2. Per-Strip Anomaly Distribution (All 20 Along-Track Strips)

- **Swath Segment:** Rows 100,000 to 140,000 of `ch2_sar_ncxl_20260308t062017233`
- **Total Along-Track Length:** 40,000 rows ($\approx 1,996.8$ rows per strip)
- **Patch Size:** $64 \times 64$, **Stride:** 32, **Boundary Buffer:** 64 px
- **Label Rule:** $\text{CPR} > 1.0 \land \text{DOP} < 0.13$ ($\ge 5$ candidate pixels per patch)

| Strip | Row Range (Lines) | Split | Positive Pixels | Pre-buffer Patches | Dropped by Buffer | Kept Positive | Kept Negative | Kept Total | Ambiguous (1–4 px) |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **0** | 0 – 1,996 | **Train** | 341 | 126 | 0 | 98 | 28 | 126 | 24 |
| **1** | 1,996 – 3,993 | **Train** | 205 | 124 | 0 | 50 | 74 | 124 | 43 |
| **2** | 3,993 – 5,990 | **Train** | 125 | 126 | 0 | 7 | 119 | 126 | 47 |
| **3** | 5,990 – 7,987 | **Train** | 134 | 124 | 8 | 23 | 93 | 116 | 51 |
| **4** | 7,987 – 9,984 | **Val** | 194 | 126 | 12 | 31 | 83 | 114 | 65 |
| **5** | 9,984 – 11,980 | **Train** | 219 | 124 | 12 | 40 | 72 | 112 | 57 |
| **6** | 11,980 – 13,977 | **Test** | 205 | 124 | 12 | 43 | 69 | 112 | 51 |
| **7** | 13,977 – 15,974 | **Val** | 193 | 126 | 12 | 29 | 85 | 114 | 70 |
| **8** | 15,974 – 17,971 | **Train** | 172 | 124 | 4 | 25 | 95 | 120 | 73 |
| **9** | 17,971 – 19,968 | **Train** | 204 | 126 | 8 | 41 | 77 | 118 | 67 |
| **10** | 19,968 – 21,964 | **Test** | 265 | 124 | 12 | 53 | 59 | 112 | 43 |
| **11** | 21,964 – 23,961 | **Train** | 247 | 124 | 12 | 51 | 61 | 112 | 43 |
| **12** | 23,961 – 25,958 | **Val** | 136 | 126 | 12 | 17 | 97 | 114 | 42 |
| **13** | 25,958 – 27,955 | **Train** | 103 | 124 | 12 | 11 | 101 | 112 | 34 |
| **14** | 27,955 – 29,952 | **Test** | 105 | 126 | 12 | 5 | 109 | 114 | 39 |
| **15** | 29,952 – 31,948 | **Train** | 110 | 124 | 4 | 12 | 108 | 120 | 40 |
| **16** | 31,948 – 33,945 | **Train** | 13 | 124 | 0 | 0 | 124 | 124 | 14 |
| **17** | 33,945 – 35,942 | **Train** | 31 | 126 | 0 | 1 | 125 | 126 | 17 |
| **18** | 35,942 – 37,939 | **Train** | 9 | 124 | 0 | 0 | 124 | 124 | 8 |
| **19** | 37,939 – 40,000 | **Train** | 22 | 126 | 0 | 3 | 123 | 126 | 17 |
| **Total** | **0 – 40,000** | — | **3,033** | **2,498** | **132** | **540** | **1,826** | **2,366** | **845** |

---

## 3. Split-Level Summary & Ambiguity Band Statistics

| Split | Assigned Strips | Total Patches (Kept) | Positive Patches | Negative Patches | Positivity Rate | Ambiguous Patches (1–4 px) | Ambiguous % of Split |
|---|---|---|---|---|---|---|---|
| **TRAIN** | 14 strips (0, 1, 2, 3, 5, 8, 9, 11, 13, 15, 16, 17, 18, 19) | 1,686 | 362 | 1,324 | 21.47% | 535 | 31.73% |
| **VAL** | 3 strips (4, 7, 12) | 342 | 77 | 265 | 22.51% | 177 | 51.75% |
| **TEST** | 3 strips (6, 10, 14) | 338 | 101 | 237 | 29.88% | 133 | 39.35% |
| **Total** | **20 strips** | **2,366** | **540** | **1,826** | **22.82%** | **845** | **35.71%** |

### Ambiguity Band Distribution
- **Definition:** Patches containing between 1 and 4 positive pixels ($0.024\% \le \text{fraction} < 0.122\%$).
- In all splits, these patches are assigned label $0$ under the primary rule ($T_{\text{pos}} \ge 5$).
- In the spatial test set, **133 patches** fall into this ambiguity band. When evaluating clean test performance (ambiguity-removed ablation), removing these 133 edge patches leaves **205 clean test patches** (101 positive, 104 negative).

---

## 4. Visual Anomaly Map Across Strips

The generated bar chart below illustrates the along-track distribution of both positive pixels and post-buffer positive patches across the 20 strips, color-coded by split assignment:

![Along-Track Positive Anomaly Map](positive_map.png)
