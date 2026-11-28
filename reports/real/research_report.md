# IceShadowNet: Research Report

**Detection, Pixel-Level Localization, and Statistical Evaluation of Lunar South Polar Radar Anomalies in Chandrayaan-2 DFSAR L-Band Quad-Pol Data**

---

## Abstract

We evaluate deep learning for detecting anomalous polarimetric signatures ($\text{CPR} > 1.0 \land \text{DOP} < 0.13$) across a single Chandrayaan-2 DFSAR strip (13,886 patches from valid-row-filtered radar imagery with deterministic non-adjacent strip splits and strict exclusion of derived polarimetric ratios from model inputs). Across twenty along-track strips in five-fold cross-validation, `CustomCNN` (0.8124 ROC-AUC) is statistically indistinguishable from `LogisticRegression` (0.8118), ImageNet pretraining degrades transfer performance, and self-supervised pretraining does not surpass linear or small-CNN baselines. On pixel-level localization under severe class skew, `IceUNet` reaches 0.0110 PR-AUC as a three-seed ensemble, yet context-matched gradient boosting already achieves 0.0077 PR-AUC, while Monte Carlo uncertainty fails selective prediction and a circular phase-oracle diagnostic keeps Dice near the information ceiling.

---

## Statistical Comparison & Empirical Findings

- **5-Fold Blocked-Strip CV & Paired Sub-Block Bootstrap (`104` sub-blocks):** `CustomCNN` (`0.8124 ± 0.0408`), `LogisticRegression` (`0.8118 ± 0.0460`), `ResNet_scratch` (`0.7978 ± 0.0427`), and `IceUNet_mean_prob` (`0.7955 ± 0.0448`) are mutually **not separable** (`CustomCNN - LogisticRegression` `ΔROC-AUC = +0.0228 [-0.0109, +0.0546]`). `ResNet_pretrained` (`0.7970 ± 0.0123` 3-seed mean; `0.815` pooled) underperforms and is not separable under strip CV.
- **SimCLR SSL (`11,868` unlabeled Train+Val patches):** Improves over `ResNet_scratch` at `10%` labels (`0.7892` vs `0.5250`), **but does not beat `LogisticRegression` (`0.8036`) or `CustomCNN` (`0.8080`)**.
- **Pixel Localization, Context-Matched GBDT, & 3-Seed Circular Phase Oracle:** `IceUNet` reaches `0.0093 ± 0.0003` (`17.6×` lift) single-seed mean and `0.0110` (`20.9×` lift) ensemble over `0.0527%` prevalence. Context-matched `7×7+15×15` `HistGBDT` (`0.0077`, `14.6×`, Dice `0.0229`) recovers `82.6%` of single-seed `IceUNet` PR-AUC (`69.8%` of ensemble). In our circular 3-seed Phase Oracle diagnostic, adding `|HH*conj(VV)|` coherence lifts PR-AUC by `+17.9%` (`0.0097 ± 0.0003` → `0.0114 ± 0.0011`) while Dice stays low (`0.0283` → `0.0330`).
- **Ambiguity Band (`1–4 px`):** Comprises `39.94%` of test patches (`806/2018`) and `80.91%` (`250/309`) of `CustomCNN` (seed 42) test false positives; excluding ambiguity patches lifts 3-seed `CustomCNN` ROC-AUC from `0.8379` to `0.9102`.
- **Selective Prediction & Pruning:** Error-detection AUROC exceeds `Random` (`0.5240`) for `|p - 0.5|` (`0.8310`), `MC Entropy` (`0.8282`), and `Ensemble Variance` (`0.7879`), yet `50%` retained Patch ROC-AUC falls below full coverage (`0.8456` → `0.7759` / `0.7616` / `0.7416` vs `0.8491` Random). Structured `50%` pruning (`271,909` params) gives a `1.32×–2.03×` range speedup at `B=64` (`12.939→9.808 ms` and `10.860→5.359 ms`), negligible gain at `B=1` (`1.09×`).
- **Georeferencing & Limitations:** `10`-fold CV over `676` GCPs yields `307.38 m` median, `367.98 m` mean, `2,377.70 m` max (`~2.4 km` boundary tail), with `92.79%` SRI overlap (`61,857/66,664`). See interactive HTML paper at [`docs/index.html`](../../docs/index.html).
