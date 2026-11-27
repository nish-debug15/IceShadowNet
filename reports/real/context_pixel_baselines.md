# Block 6 Item 2: Context-Matched Pixel Baselines (`7×7` and `15×15` Windows) vs. `IceUNet`

- **Test Pixel Prevalence (PR-AUC Baseline):** `0.0527%` (`4,353 / 8,265,728` pixels)

| Model | Context | Pixel PR-AUC | Lift vs. 0.0527% | Dice (Val Th) | IoU (Val Th) | Dice (Best Test Th) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| HV/HH Ratio (context-free) | 1×1 | 0.0011 | 2.2× | 0.0036 | 0.0018 | 0.0037 |
| Pixel LogReg (context-free) | 1×1 | 0.0012 | 2.3× | 0.0016 | 0.0008 | 0.0039 |
| **Context Pixel LogReg (7x7)** | 7x7 | 0.0058 | 11.0× | 0.0177 | 0.0089 | 0.0177 |
| **Context Pixel HistGBDT (7x7)** | 7x7 | 0.0072 | 13.7× | 0.0215 | 0.0109 | 0.0215 |
| **Context Pixel LogReg (7x7+15x15)** | 7x7+15x15 | 0.0067 | 12.7× | 0.0189 | 0.0096 | 0.0189 |
| **Context Pixel HistGBDT (7x7+15x15)** | 7x7+15x15 | 0.0077 | 14.6× | 0.0229 | 0.0116 | 0.0229 |
| **IceUNet (3-seed mean)** | 64×64 | 0.0093 | 17.6× | 0.0266 | 0.0135 | 0.0266 |
| **IceUNet 3-Seed Ensemble** | 64×64 | 0.0110 | 20.9× | 0.0334 | 0.0170 | — |
