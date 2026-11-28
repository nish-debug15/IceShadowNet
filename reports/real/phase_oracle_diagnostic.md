# Block 7 Item 2: 3-Seed Phase Oracle Diagnostic (Circular — NOT a Model Contribution)

> **Warning (Circular Diagnostic):** Adding `7×7` multilooked `|HH*conj(VV)|` coherence exposes inter-channel phase stability directly related to `DOP`/`CPR` labeling. This experiment tests the **phase-information ceiling** hypothesis over 3 seeds (`42, 123, 2024`) and is **not** a model contribution.

| Model (3 Seeds, Val-Chosen Th) | Inputs | Pixel PR-AUC | Lift vs. 0.0527% | Dice | IoU |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **IceUNet (3-ch Intensity Only)** | `[HH, HV, VV]` | 0.0097 ± 0.0003 | 18.4× ± 0.6× | 0.0283 ± 0.0026 | 0.0144 ± 0.0014 |
| **Phase Oracle IceUNet (4-ch, CIRCULAR)** | `[HH, HV, VV, γ_HHVV]` | 0.0114 ± 0.0011 | 21.7× ± 2.1× | 0.0330 ± 0.0044 | 0.0168 ± 0.0023 |

## Per-Seed Breakdown

| Variant | Seed | Best Ep | Val Th | Pixel PR-AUC | Lift | Dice | IoU |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 3-ch Intensity | 42 | 2 | 0.45 | 0.0094 | 17.8× | 0.0281 | 0.0143 |
| 3-ch Intensity | 123 | 2 | 0.45 | 0.0096 | 18.2× | 0.0252 | 0.0128 |
| 3-ch Intensity | 2024 | 2 | 0.40 | 0.0102 | 19.3× | 0.0317 | 0.0161 |
| 4-ch Phase Oracle | 42 | 5 | 0.45 | 0.0115 | 21.9× | 0.0345 | 0.0176 |
| 4-ch Phase Oracle | 123 | 1 | 0.40 | 0.0100 | 19.0× | 0.0270 | 0.0137 |
| 4-ch Phase Oracle | 2024 | 7 | 0.50 | 0.0128 | 24.3× | 0.0375 | 0.0191 |
