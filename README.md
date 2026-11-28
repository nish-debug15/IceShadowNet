# IceShadowNet

**Deep Learning Detection, Pixel-Level Localization, and Statistical Evaluation of Lunar South Polar Radar Anomalies Using Chandrayaan-2 DFSAR L-Band Quad-Pol Data**

> **Disclaimer:** Labels are radar-derived polarimetric proxy labels ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$ on $7 \times 7$ multilooked data) and do **not** confirm physical water ice. All results come from a single scene (`ch2_sar_ncxl_20260308t062017233`) with anisotropic slant-range pixels (`0.505 m` azimuth × `9.593 m` range), testing strip-held-out generalization only.

---

## Quickstart & Deterministic Reproduction

```bash
pip install -r requirements.txt
make reproduce   # runs bash run_all.sh (fixed seeds: 42, 123, 2024)
```

- **Live Report & Dashboard:** https://nish-debug15.github.io/IceShadowNet/ and https://nish-debug15.github.io/IceShadowNet/dashboard/
- **Audit & CI checks:** `python -m pytest tests -v`, `python scripts/check_repo_images.py`, and `python scripts/analysis/number_trace_audit.py`.
- **Gitignored Large Artifacts:** Raw DFSAR TIFFs, `.npz` tensor caches, `.pth` checkpoints, and QGIS-ready `.tif` GeoTIFF exports are excluded from Git and documented in [`reports/real/REPRODUCIBILITY.md`](reports/real/REPRODUCIBILITY.md) (release tag `v1.0-endsem`).

---

## Empirical Findings (Strictly Verified Against `reports/real/*.json`)

1. **Patch Task — Deep Learning Is Not Separable From Logistic Regression:** Across 5-fold blocked-strip CV over all 20 strips (`256`-row azimuth guard buffers), `CustomCNN` (`0.8124 ± 0.0408` ROC-AUC, `0.4760 ± 0.1093` PR-AUC) ties `LogisticRegression` (`0.8118 ± 0.0460` / `0.4921 ± 0.1197`), `ResNet_scratch` (`0.7978 ± 0.0427` / `0.4641 ± 0.1110`), and `IceUNet_mean_prob` (`0.7955 ± 0.0448` / `0.4169 ± 0.1328`). Under paired sub-block bootstrap on the frozen test set (`104` sub-blocks, *within-test-set variability only*), `CustomCNN - LogisticRegression` (`ΔROC-AUC = +0.0228 [-0.0109, +0.0546]`) and `IceUNet_mean_prob - CustomCNN` (`+0.0020 [-0.0092, +0.0108]`) both cross zero and are **not separable**.
2. **ImageNet Pretraining Underperforms:** `ResNet_pretrained` (`0.815` pooled ROC-AUC; `0.7970 ± 0.0123` across 3 seeds) underperforms `ResNet_scratch` (`0.840`) and `CustomCNN` (`0.844`), consistent across seeds, and is not separable under strip CV (within-test-set paired `ΔROC-AUC = +0.0256 [+0.0053, +0.0452]` for `ResNet_scratch - ResNet_pretrained`).
3. **In-Domain SimCLR SSL vs. Baselines:** `ResNet_SSL` (`11,868` unlabeled Train+Val patches) improves over `ResNet_scratch` at 10% labels (`0.7892 ± 0.0094` vs `0.5250 ± 0.0132` ROC-AUC), **but does not beat `LogisticRegression` (`0.8036 ± 0.0135`) or `CustomCNN` (`0.8080 ± 0.0197`)** ([`label_efficiency_fairness.md`](reports/real/label_efficiency_fairness.md)).
4. **Pixel Localization vs. Context-Matched Baselines & Circular Phase-Oracle Diagnostic:** `IceUNet` improves pixel localization over context-free baselines (`0.0093 ± 0.0003` PR-AUC, `17.6×` lift over `0.0527%` prevalence; `0.0110` / `20.9×` lift as a 3-seed ensemble vs. `Pixel LogReg 1×1` `0.0012` / `2.3×`). However, **context-matched `7×7+15×15` `HistGBDT`** (`0.0077` PR-AUC, `14.6×` lift, `0.0229` Dice) already recovers **`82.6%`** of single-seed `IceUNet` PR-AUC (and **`69.8%`** of the 3-seed ensemble), showing local spatial context drives most of the improvement. Re-evaluated across 3 seeds under a circular diagnostic of the information ceiling (not a model contribution), adding `|HH*conj(VV)|` coherence lifts Pixel PR-AUC by `+17.9%` (`0.0097 ± 0.0003` / `18.4× ± 0.6×` → `0.0114 ± 0.0011` / `21.7× ± 2.1×`) while Dice moves from `0.0283 ± 0.0026` to `0.0330 ± 0.0044`, confirming extreme `0.0527%` imbalance, per-patch normalization, and complex phase dependence jointly cap Dice.
5. **Selective Prediction Negative Result:** Evaluated on the unified 3-seed `IceUNet` ensemble mean predictor, error-detection AUROC exceeds `Random` (`0.5240`) for `|p - 0.5|` (`0.8310`), `MC Entropy` (`0.8282`), and `Ensemble Variance` (`0.7879`). However, on retained-set performance from `100%` coverage (`0.8456` Patch ROC-AUC) down to `50%` coverage, `Ensemble Variance` (`0.7759`), `MC Entropy` (`0.7616`), and `|p - 0.5|` (`0.7416`) all degrade below `Random` rejection (`0.8491`), and `|p - 0.5|` achieves the best `AURC` (`0.1084` vs `0.1096` MC Entropy, `0.1259` EnsVar, `0.1917` Random). **Uncertainty is not usable for selective rejection on this task.**

---

## Frozen Split & Disclosure

- **Active Rows:** `[90624, 319812]` (`229,188` valid rows = `71.66%` of `319,812 × 116` strip; `13,886` post-buffer patches: `9,828` train [`1,614` pos] / `2,040` val [`335` pos] / `2,018` test [`332` pos]).
- **Split Search Disclosure:** Deterministic non-adjacent strip assignment (`val = [2, 6, 17]`, `test = [0, 8, 15]`, `256`-row azimuth guard buffer) used **label prevalence only** with **zero model feedback**. Manifest frozen in [`data/processed_real/split_manifest.json`](data/processed_real/split_manifest.json) ([`split_rationale.md`](reports/real/split_rationale.md)).

---

## Primary Benchmark Tables

### Per-Test-Strip & 5-Fold Blocked-Strip CV (Patch Level, Prevalence `16.45%`)

| Model | Frozen Pooled ROC / PR / F1 | Strip 0 ROC / PR / F1 | Strip 8 ROC / PR / F1 | Strip 15 ROC / PR / F1 | 5-Fold CV ROC-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **CustomCNN** | `0.844 / 0.566 / 0.537` | `0.942 / 0.783 / 0.771` | `0.662 / 0.222 / 0.262` | `0.763 / 0.380 / 0.431` | `0.8124 ± 0.0408` |
| **IceUNet (mean_prob)** | `0.846 / 0.564 / 0.530` | `0.950 / 0.796 / 0.787` | `0.637 / 0.183 / 0.231` | `0.773 / 0.395 / 0.457` | `0.7955 ± 0.0448` |
| **ResNet_scratch** | `0.840 / 0.544 / 0.529` | `0.945 / 0.783 / 0.777` | `0.640 / 0.174 / 0.214` | `0.770 / 0.382 / 0.460` | `0.7978 ± 0.0427` |
| **LogisticRegression** | `0.821 / 0.498 / 0.509` | `0.916 / 0.643 / 0.685` | `0.605 / 0.228 / 0.261` | `0.785 / 0.422 / 0.441` | `0.8118 ± 0.0460` |
| **ResNet_pretrained** | `0.815 / 0.480 / 0.489` | `0.924 / 0.654 / 0.730` | `0.602 / 0.137 / 0.236` | `0.751 / 0.385 / 0.473` | — |
| **DecisionTree** | `0.616 / 0.213 / 0.314` | `0.602 / 0.241 / 0.336` | `0.508 / 0.104 / 0.184` | `0.630 / 0.247 / 0.386` | — |
| *IceUNet (Optuna, single run, excluded from ranking)* | *`0.8497` ROC (single run)* | — | — | — | — |

### Pixel-Level Localization (`8,265,728` Test Pixels, `4,353` Positives, Prevalence = `0.0527%`)

| Model | Context | Pixel PR-AUC | Lift vs. Prevalence | Dice (Val Th) | Cond. Dice (Pos Patches) | IoU (Val Th) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| HV/HH Ratio (context-free) | `1×1` | `0.0011` | `2.2×` | `0.0036` | — | `0.0018` |
| Pixel LogReg (context-free) | `1×1` | `0.0012` | `2.3×` | `0.0016` | — | `0.0008` |
| Context Pixel LogReg (7x7) | `7x7` | `0.0058` | `11.0×` | `0.0177` | — | `0.0089` |
| Context Pixel HistGBDT (7x7) | `7x7` | `0.0072` | `13.7×` | `0.0215` | — | `0.0109` |
| Context Pixel LogReg (7x7+15x15) | `7x7+15x15` | `0.0067` | `12.7×` | `0.0189` | — | `0.0096` |
| Context Pixel HistGBDT (7x7+15x15) | `7x7+15x15` | `0.0077` | `14.6×` | `0.0229` | — | `0.0116` |
| **IceUNet (3-seed mean)** | `64×64` | `0.0093 ± 0.0003` | `17.6×` | `0.0266` | `0.0377–0.0519` | `0.0135` |
| **IceUNet 3-Seed Ensemble** | `64×64` | **`0.0110`** | **`20.9×`** | **`0.0334`** | **`0.0539`** | **`0.0170`** |
| *Intensity 3-ch Baseline (3 seeds, phase-oracle ctrl)* | `64×64` | *`0.0097 ± 0.0003`* | *`18.4× ± 0.6×`* | *`0.0283 ± 0.0026`* | — | *`0.0144 ± 0.0014`* |
| *Phase Oracle IceUNet (CIRCULAR diagnostic, 3 seeds)* | `64×64` | *`0.0114 ± 0.0011`* | *`21.7× ± 2.1×`* | *`0.0330 ± 0.0044`* | — | *`0.0168 ± 0.0023`* |

### Selective Prediction (Unified Ensemble Mean Predictor)

| Rejection Score | AURC (Lower Better) | Error-Detection AUROC | Retained Patch ROC-AUC (`100%` → `50%`) |
| :--- | :---: | :---: | :---: |
| \|p - 0.5\| | `0.1084` | `0.8310` | `0.8456` → `0.7416` |
| MC Entropy | `0.1096` | `0.8282` | `0.8456` → `0.7616` |
| Ensemble Variance | `0.1259` | `0.7879` | `0.8456` → `0.7759` |
| Random | `0.1917` | `0.5240` | `0.8456` → `0.8491` |

### Unified Pruning & CUDA Latency

| Configuration | Non-Zero Params | Patch ROC-AUC (`mean_prob`) | Pixel PR-AUC | Latency `B=1` (ms) | Latency `B=64` (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base (0%) | `1,084,681` | `0.8291` | `0.0093` | `2.244` | `10.860` |
| Structured Channel 30% (base_ch=17) | `544,834` | `0.8099` | `0.0065` | `2.220` | `11.441` |
| Structured Channel 50% (base_ch=12) | `271,909` | `0.8474` | `0.0100` | `2.058` | `5.359` |
| Unstructured L1 50% (param reduction) | `543,481` | `0.8372` | `0.0074` | `2.220` | `10.933` |
| Unstructured L1 90% (param reduction) | `110,521` | `0.8273` | `0.0058` | `2.208` | `10.939` |

- Across repeated `B=64` CUDA latency sweeps, `50%` structured channel pruning (`base_ch=12`) yields a measured **`1.32×–2.03×`** speedup (`12.939 → 9.808 ms` in `structured_pruning.json`; `10.860 → 5.359 ms` in `pruning_unified.json`), with negligible gain at `B=1` (`2.244 → 2.058 ms`, `1.09×`), no speedup at `30%` structured (`11.441 ms`), and no speedup for unstructured L1 pruning (`10.933 / 10.939 ms`). Pruned Patch ROC-AUC differences (`0.8099–0.8474`) reflect single-seed variability within noise.

### Georeferencing

- **10-Fold GCP Cross-Validation (`676` `GCS_MOON` GCPs, `R = 1,737,400 m`):** Median **`307.38 m`**, Mean **`367.98 m`**, Max **`2,377.70 m`** (`~2.4 km` boundary GCP tail; alongside `281.7249 m` max across `5` interior GCPs).
- **SRI Overlap (`92.79%`):** The 92.79% overlap metric is the percentage of valid TPS-warped test-strip pixels (61,857 of 66,664) whose nearest-neighbor cell on the official 25 m/px ISRO SRI HH grid has positive finite backscatter (SRI HH > 0). Median swath centerline lateral offset = `993.8 m` (mean `2382.0 m`).

---

## Team & Affiliation

- **Nishit Patel & Project Team** (`RV University, Bengaluru`) — Roles: Data & Labeling, Custom CNN, ResNet, Evaluation & Explainability, Integration & Report.

## Acknowledgement

> This research is based partially on the results obtained from the Chandrayaan-2, second lunar mission of the Indian Space Research Organisation (ISRO), archived at the Indian Space Science Data Centre (ISSDC).
