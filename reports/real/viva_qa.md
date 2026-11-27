# IceShadowNet 2.0 — Viva & Exam Defense Q&A

### Q1: Why use Deep Learning if Logistic Regression is not separable from `CustomCNN` on the patch task?
- **Measured reality:** Across 5-fold blocked-strip CV (`20` strips, `256`-row buffer), `CustomCNN` (`0.8124 ± 0.0408`) ties `LogisticRegression` (`0.8118 ± 0.0460`), and within-test-set paired sub-block bootstrap (`ΔROC-AUC = +0.0228 [-0.0109, +0.0546]`) is **not separable**. Likewise, `ResNet_SSL` (`0.7892`) does not beat `LogisticRegression` (`0.8036`) at 10% labels.
- **Where DL adds value:** `IceUNet` improves pixel-level localization over context-free baselines (`0.0093` PR-AUC / `17.6×` lift; `0.0110` / `20.9×` ensemble vs. `0.0012` for `1×1` LogReg) and modestly outperforms context-matched `7×7+15×15` `HistGBDT` (`0.0077` PR-AUC, `14.6×` lift).

### Q2: Why is pixel Dice only `~0.03`, and did the Phase Oracle test confirm why?
- **Diagnostic (`phase_oracle_diagnostic.md`, circular, 1 seed):** Adding `7×7` `|HH*conj(VV)|` coherence lifts Pixel PR-AUC by `+43%` (`0.0093 → 0.0133`, `25.2×` lift), confirming missing phase hurts ranking—**yet Dice only moves `0.0295 → 0.0348`**, proving that extreme `0.0527%` pixel imbalance, per-patch z-scoring, and full Stokes/phase dependence jointly cap Dice at `~0.03`.

### Q3: Can MC Dropout or Ensemble Variance be used for selective rejection?
- **No (`selective_redone.md`):** On the unified 3-seed ensemble mean predictor, neither Ensemble Variance (`AURC 0.1259`) nor MC Entropy (`AURC 0.1096`) improves Patch ROC-AUC over full coverage (`0.8456`), and `|p - 0.5|` (`AURC 0.1084`, `ErrDet-AUC 0.8310`) performs slightly better. **Uncertainty is not usable for selective rejection here.** Also, MC entropy tracks $p$ near $0.5$ (`GT Interior` highest at `0.51350`).

### Q4: How accurate is the georeferencing and do predictions confirm water ice?
- **Georeferencing (`georeferencing_audit.md`):** Normalized TPS over `676` `GCS_MOON` GCPs (`R = 1,737,400 m`) has `367.98 m` mean (`307.38 m` median, `2,377.70 m` max) 10-fold CV residual and `92.79%` overlap (percentage of valid TPS-warped test-strip pixels whose nearest-neighbor cell on the `25 m/px` `SRI HH` grid has `SRI HH > 0`).
- **Limits & Ice:** Anisotropic `0.505 m × 9.593 m` pixels; single scene, strip-held-out generalization only; radar proxy labels (`CPR > 1.0 & DOP < 0.13`) do **not** confirm physical water ice.
