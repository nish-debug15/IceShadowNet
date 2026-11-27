# Block 5 Item 5: Structured Channel Pruning vs. Unstructured Parameter Reduction

- **Note on Unstructured Pruning (`efficiency_report.md`):** L1 unstructured pruning at 30–90% is strictly **parameter reduction** (zeroed weights in dense tensors do not yield CUDA speedup, staying at ~11.1–11.3 ms/batch).
- **Structured Channel Pruning (Below):** Physically removes channels (`base_ch` 24 → 17 at ~30% and 24 → 12 at 50%) and fine-tunes for 3 epochs.

| Configuration | Params | Param Reduction | Pixel PR-AUC | Dice (Val Th) | Patch Mean-Prob ROC-AUC | CUDA Latency (ms / B=64) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 0% (Base base_ch=24) | 1,084,681 | 0.0% | 0.0093 | 0.0295 | 0.8291 | 12.94 |
| 30% Channel-Pruned (base_ch=17) | 544,834 | 49.8% | 0.0081 | 0.0301 | 0.8309 | 15.54 |
| 50% Channel-Pruned (base_ch=12) | 271,909 | 74.9% | 0.0099 | 0.0201 | 0.8456 | 9.81 |
