# Block 6 Item 4: Phase Oracle Diagnostic (Circular — NOT a Headline Model)

> **Warning (Circular Diagnostic):** Adding `7×7` multilooked `|HH*conj(VV)|` coherence exposes inter-channel phase stability directly related to `DOP`/`CPR` labeling. This experiment tests the **phase-information ceiling** hypothesis and is **not** a headline model.

| Model (Seed 42) | Inputs | Pixel PR-AUC | Lift vs. 0.0527% | Dice (Val Th) | Dice (Best Test Th) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **IceUNet (3-ch Intensity Only)** | `[HH, HV, VV]` | 0.0093 | 17.6× | 0.0295 | 0.0295 |
| **Phase Oracle IceUNet (4-ch, CIRCULAR)** | `[HH, HV, VV, γ_HHVV]` | 0.0133 | 25.2× | 0.0348 | 0.0348 |
