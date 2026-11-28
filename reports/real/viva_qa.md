# IceShadowNet 2.0 — Viva & Exam Defense Q&A

### Why use Deep Learning if Logistic Regression ties CustomCNN on the patch task?
- **Measured reality:** Across 5-fold blocked-strip CV (`20` strips, `256`-row buffer), `CustomCNN` (`0.8124 ± 0.0408`) ties `LogisticRegression` (`0.8118 ± 0.0460`), and within-test-set paired sub-block bootstrap (`ΔROC-AUC = +0.0228 [-0.0109, +0.0546]`) is **not separable**. Likewise, `ResNet_SSL` (`0.7892`) does not beat `LogisticRegression` (`0.8036`) or `CustomCNN` (`0.8080`) at `10%` labels.
- **Where DL adds localized signal:** `IceUNet` improves pixel-level localization over context-free baselines (`0.0093` PR-AUC / `17.6×` lift; `0.0110` / `20.9×` ensemble vs. `0.0012` for `1×1` LogReg), though context-matched `7×7+15×15` `HistGBDT` (`0.0077` PR-AUC, `14.6×` lift) recovers **`82.6%`** of single-seed `IceUNet` PR-AUC (`69.8%` of ensemble).

### Why is pixel Dice low (~0.03), and what did the 3-seed Phase Oracle diagnostic show?
- **Diagnostic (`phase_oracle_diagnostic.md`, circular information-ceiling diagnostic, 3 seeds):** Adding `|HH*conj(VV)|` coherence lifts Pixel PR-AUC by `+17.9%` (`0.0097 ± 0.0003` / `18.4× ± 0.6×` → `0.0114 ± 0.0011` / `21.7× ± 2.1×`), confirming complex phase helps ranking—**yet Dice moves modestly from `0.0283 ± 0.0026` to `0.0330 ± 0.0044`** (IoU `0.0144` → `0.0168`), demonstrating that extreme `0.0527%` pixel imbalance, per-patch normalization, and full Stokes/phase dependence jointly cap Dice.

### Can MC Dropout or Ensemble Variance be used for selective rejection?
- **No (`selective_redone.md`):** While error-detection AUROC beats `Random` (`0.5240`) for `|p - 0.5|` (`0.8310`), `MC Entropy` (`0.8282`), and `Ensemble Variance` (`0.7879`), thresholding to `50%` coverage reduces Patch ROC-AUC from `0.8456` down to `0.7759` (`EnsVar`), `0.7616` (`MC`), and `0.7416` (`|p - 0.5|`)—worse than `Random` rejection (`0.8491`). By `AURC`, `|p - 0.5|` (`0.1084`) is best (`MC 0.1096`, `EnsVar 0.1259`, `Random 0.1917`). **Uncertainty is not usable for selective rejection here.**

### Does pruning speed up inference, and how accurate is TPS georeferencing?
- **Pruning:** Speedup appears **only** for `50%` structured channel pruning (`271,909` params) at `B=64` (`10.860 → 5.359 ms`, **`2.03×`**), with negligible speedup at `B=1` (`2.244 → 2.058 ms`, `1.09×`); `30%` structured (`11.441 ms`) and unstructured L1 (`10.933 / 10.939 ms`) yield no speedup. Pruned AUC differences remain within noise.
- **Georeferencing & Ice Limits:** 10-fold CV over `676` `GCS_MOON` GCPs (`R = 1,737,400 m`) gives **`307.38 m` median**, **`367.98 m` mean**, and **`2,377.70 m` max** (`~2.4 km` tail on boundary GCPs), with `92.79%` `SRI HH` overlap (`61,857` / `66,664`). Anisotropic `0.505 m × 9.593 m` pixels, single-scene strip-held-out evaluation, and radar proxy labels (`CPR > 1.0 & DOP < 0.13`) mean predictions do **not** confirm physical water ice.
