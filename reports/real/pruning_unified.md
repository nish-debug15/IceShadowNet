# Block 6 Item 5: Unified Pruning & Latency Table (Median of 200 Runs, 50 Warmup, CUDA Sync)

| Configuration | Non-Zero Params | Patch ROC-AUC (`mean_prob`) | Pixel PR-AUC | CUDA Latency B=1 (ms) | CUDA Latency B=64 (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base (0%) | 1,084,681 | 0.8291 | 0.0093 | 2.244 | 10.860 |
| Structured Channel 30% (base_ch=17) | 544,834 | 0.8099 | 0.0065 | 2.220 | 11.441 |
| Structured Channel 50% (base_ch=12) | 271,909 | 0.8474 | 0.0100 | 2.058 | 5.359 |
| Unstructured L1 50% (param reduction) | 543,481 | 0.8372 | 0.0074 | 2.220 | 10.933 |
| Unstructured L1 90% (param reduction) | 110,521 | 0.8273 | 0.0058 | 2.208 | 10.939 |
