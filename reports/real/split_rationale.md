# Split Rationale & Deterministic Strip Selection (`[90624, 319812]`)

**Scene:** `ch2_sar_ncxl_20260308t062017233` (DFSAR L1A Quad-Pol SLI, `319,812 × 116`)  
**Active Rows:** `[90624, 319812]` (`229,188` rows = `71.66%` of strip; rows `0..90623` are zero-padded/invalid)  
**Validity Gate:** Finite, non-zero quad-pol SLI pixels (`>= 95%` valid per `64 × 64` patch, `stride = 32`)  
**Guard Buffer:** `256` rows (azimuth) between adjacent strips assigned to different splits (`4×` patch size, guaranteeing zero pixel-extent leakage).

---

## 1. Per-Strip Audit (Unbuffered, 20 Along-Track Strips)

| Strip | Patches | Positives (`>=5 px`) | Pos Rate | Ambiguity Band (`1–4 px`) |
| :---: | :---: | :---: | :---: | :---: |
| 0 | 678 | 151 | 22.27% | 159 (23.5%) |
| 1 | 716 | 196 | 27.37% | 319 (44.6%) |
| 2 | 716 | 235 | 32.82% | 348 (48.6%) |
| 3 | 716 | 47 | 6.56% | 180 (25.1%) |
| 4 | 716 | 36 | 5.03% | 155 (21.6%) |
| 5 | 716 | 80 | 11.17% | 384 (53.6%) |
| 6 | 716 | 75 | 10.47% | 310 (43.3%) |
| 7 | 716 | 45 | 6.28% | 302 (42.2%) |
| 8 | 716 | 72 | 10.06% | 374 (52.2%) |
| 9 | 716 | 69 | 9.64% | 273 (38.1%) |
| 10 | 716 | 174 | 24.30% | 283 (39.5%) |
| 11 | 716 | 66 | 9.22% | 303 (42.3%) |
| 12 | 716 | 61 | 8.52% | 339 (47.3%) |
| 13 | 716 | 380 | 53.07% | 270 (37.7%) |
| 14 | 716 | 422 | 58.94% | 264 (36.9%) |
| 15 | 716 | 142 | 19.83% | 287 (40.1%) |
| 16 | 716 | 67 | 9.36% | 97 (13.5%) |
| 17 | 716 | 52 | 7.26% | 190 (26.5%) |
| 18 | 716 | 22 | 3.07% | 102 (14.2%) |
| 19 | 716 | 1 | 0.14% | 21 (2.9%) |

**Total unbuffered valid patches:** `14,282` | **Positives:** `2,393 (16.76%)` | **Ambiguity band (`1–4 px`):** `4,960 (34.73% of total, 41.72% of negatives)`

---

## 2. Deterministic Selection Rule (Decided Before Training)

All `100,100` pairs of non-adjacent 3-strip `val` and `test` subsets (`|i - j| >= 2` across all 6 held-out strips) were exhaustively evaluated with `buffer_pixels = 256` under the following constraints:
1. `val_positives >= 300` and `test_positives >= 300`
2. `val_rate / train_rate` and `test_rate / train_rate` both in `[0.70, 1.40]`
3. No single test strip contributes `> 40%` of test positives
4. Objective: minimize $\max\left(\left|\frac{\text{val\_rate}}{\text{train\_rate}} - 1\right|, \left|\frac{\text{test\_rate}}{\text{train\_rate}} - 1\right|\right)$

### Top 5 Valid Candidates (out of `97` valid pairs)

| Rank | Val Strips | Test Strips | Train Pos / N (Rate) | Val Pos / N (Ratio) | Test Pos / N (Ratio) | Test Strip Positives (Max Share) | Max Dev |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1 (Winner)** | `(2, 6, 17)` | `(0, 8, 15)` | `1614 / 9828 (16.42%)` | `335 / 2040 (1.000x)` | `332 / 2018 (1.002x)` | `[132, 70, 130]` (`39.8%`) | **`0.0018`** |
| 2 | `(2, 8, 17)` | `(0, 6, 15)` | `1614 / 9828 (16.42%)` | `336 / 2040 (1.003x)` | `331 / 2018 (0.999x)` | `[132, 69, 130]` (`39.9%`) | `0.0029` |
| 3 | `(2, 9, 12)` | `(0, 6, 15)` | `1612 / 9828 (16.40%)` | `338 / 2040 (1.010x)` | `331 / 2018 (1.000x)` | `[132, 69, 130]` (`39.9%`) | `0.0102` |
| 4 | `(2, 9, 17)` | `(0, 6, 15)` | `1620 / 9828 (16.48%)` | `329 / 2040 (0.978x)` | `331 / 2018 (0.995x)` | `[132, 69, 130]` (`39.9%`) | `0.0216` |
| 5 | `(2, 9, 17)` | `(0, 5, 15)` | `1614 / 9828 (16.42%)` | `329 / 2040 (0.982x)` | `340 / 2018 (1.026x)` | `[132, 78, 130]` (`38.8%`) | `0.0259` |

---

## 3. Winning Frozen Split
- **Val strips:** `[2, 6, 17]` (`335 / 2,040` positives, `16.42%`)
- **Test strips:** `[0, 8, 15]` (`332 / 2,018` positives, `16.45%`, per-strip positives `[132, 70, 130]`)
- **Train strips:** remaining 14 strips (`1,614 / 9,828` positives, `16.42%`)
- **Leakage check:** `assert_no_extent_leakage(..., patch_size=64)` passed with `buffer_pixels = 256`.
