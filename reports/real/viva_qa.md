# IceShadowNet 2.0 — Viva & Exam Defense Q&A

### Q1: Why use Deep Learning if Logistic Regression ties `CustomCNN` on the patch task?
- **Honest baseline:** On patch-level classification, `CustomCNN` (`0.8124 ± 0.0408`) ties `LogisticRegression` (`0.8118 ± 0.0460`) across 5-fold blocked-strip CV (`reports/real/five_fold_cv.md`), and `ResNet_SSL` (`0.7892`) does not beat `LogisticRegression` (`0.8036`) at 10% labels (`reports/real/label_efficiency_fairness.md`).
- **Where DL adds real value:** **Pixel-level localization.** `Pixel LogReg` achieves `0.0012` Pixel PR-AUC (`2.3×` over `0.0527%` prevalence) and `0.0016` Dice, whereas `IceUNet` achieves **`0.0093` Pixel PR-AUC (`17.6×` lift)** and **`0.0110` (`20.9×` lift)** as a 3-seed ensemble (`0.0334` Dice, `0.0539` conditional Dice on positive patches).

### Q2: Why is pixel-level Dice only `~0.03`?
- **Information ceiling:** Ground-truth labels (`CPR > 1.0 & DOP < 0.13`) require **inter-channel complex phase coherence** (for Stokes/covariance `DOP` and circular polarization `CPR`). Our models strictly receive **intensity-only** `[HH, HV, VV]` log-intensities to prevent circularity, so exact pixel-level boundary recovery is fundamentally bounded by missing phase information.

### Q3: Does uncertainty concentrate at mask boundaries?
- **Measured result (`reports/real/uncertainty_hardened.md`):** **No for MC entropy** — predictive entropy tracks $p$ near $0.5$, so `GT Interior` has the highest mean $p$ (`0.2656`) and highest MC entropy (`0.51350`), followed by the `1–2 px` boundary ring (`0.46635`) and far background (`0.14280`). Ensemble variance is slightly higher on the `1–2 px` ring (`0.009526`) than interior (`0.009467`).

### Q4: Do your predictions confirm water ice on the Moon?
- **No.** Labels are polarimetric radar proxy labels (`CPR > 1.0 & DOP < 0.13`) consistent with volume scattering / rough terrain / potential ice. Physical confirmation requires PSR, LOLA DEM slope, and thermal/neutron/optical corroboration.

### Q5: What are the georeferencing & data-split limitations?
- **Georeferencing:** Native `SLI` has an identity transform (`0.505 m` azimuth × `9.593 m` range anisotropic pixels). Normalized TPS over `676` `GCS_MOON` GCPs (`R = 1,737,400 m`) has `~282 m` (`~300 m`) hold-out accuracy and `92.79%` footprint overlap with the official `SRI HH` product.
- **Generalization:** Single Chandrayaan-2 DFSAR scene (`ch2_sar_ncxl_20260308t062017233`, rows `90,624–319,812`) with deterministic non-adjacent strip splitting and `256`-row azimuth guard buffers (`reports/real/split_rationale.md`).
