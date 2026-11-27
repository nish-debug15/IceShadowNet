# Block 5 Item 4: Hardened Pixel Metrics (Conditional Dice & `>=5 px` CC Post-Processing)

- **Test Set:** `2018` patches (`332` positive patches), `8,265,728` pixels (`4,353` positive pixels)

| Model | Val Th | Raw Dice / IoU | Raw Cond. Dice (Pos Patches) | CC>=5 Dice / IoU | CC>=5 Cond. Dice | Raw Patch>=5 F1 | CC>=5 Patch>=5 F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **IceUNet_seed42** | 0.45 | 0.0295 / 0.0150 | 0.0485 | 0.0303 / 0.0154 | 0.0485 | 0.3454 | 0.3514 |
| **IceUNet_seed123** | 0.45 | 0.0300 / 0.0152 | 0.0519 | 0.0303 / 0.0154 | 0.0515 | 0.3743 | 0.3747 |
| **IceUNet_seed2024** | 0.45 | 0.0203 / 0.0103 | 0.0377 | 0.0230 / 0.0117 | 0.0415 | 0.3015 | 0.3117 |
| **IceUNet_3Seed_Ensemble** | 0.45 | 0.0334 / 0.0170 | 0.0539 | 0.0334 / 0.0170 | 0.0527 | 0.3902 | 0.3936 |
