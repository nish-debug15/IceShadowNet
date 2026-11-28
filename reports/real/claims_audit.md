# Number-Trace Claims Audit (CI-Verified)

Every numeric token across `README.md`, `reports/real/research_report.md`, `reports/real/viva_qa.md`, and `docs/index.html` is extracted and mapped to tracked `reports/real/*.json`, `data/processed_real/split_manifest.json`, or `config.yaml` within rounding tolerance, alongside a recursive leaf-equality check on `dashboard/public/data/segmentation_uncertainty.json`.

## Summary by Audited Artifact

| Audited Artifact | Extracted Numbers | Matched to JSON/Config | Unmapped | Forbidden Overstatements | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `README.md` | `387` | `387` | `0` | `0` | **PASS** |
| `reports/real/research_report.md` | `224` | `224` | `0` | `0` | **PASS** |
| `reports/real/viva_qa.md` | `93` | `93` | `0` | `0` | **PASS** |
| `docs/index.html` | `308` | `308` | `0` | `0` | **PASS** |
| `dashboard/public/data/segmentation_uncertainty.json` | recursive leaves | exact match | `0` | `0` | **PASS** |

**Overall Number-Trace Audit:** `PASSED` (`1012/1012` doc numbers traced; `0` dashboard JSON leaf mismatches; `0` forbidden phrases).

## Representative Source-Key Traces

| Document | Line | Printed Token | Traced JSON / Config Key | Exact Source Value |
| :--- | :---: | :---: | :--- | :---: |
| `README.md` | 5 | `1.0` | `config.yaml.data.ohrc_channels` | `1` |
| `README.md` | 5 | `0.13` | `config.yaml.real_dfsar.dop_threshold` | `0.13` |
| `README.md` | 5 | `7` | `config.yaml.real_dfsar.multilook_window` | `7` |
| `README.md` | 5 | `0.505` | `label_efficiency_fairness.json.summary[18].pr_auc_mean` | `0.505066` |
| `README.md` | 5 | `9.593` | `georef_summary.json.range_spacing_m` | `9.59336` |
| `README.md` | 13 | `42,` | `config.yaml.split.seed` | `42` |
| `README.md` | 13 | `123,` | `block1_benchmark_report.json[3].seed` | `123` |
| `README.md` | 13 | `2024` | `block1_benchmark_report.json[4].seed` | `2024` |
| `README.md` | 23 | `1` | `config.yaml.data.ohrc_channels` | `1` |
| `README.md` | 23 | `5` | `config.yaml.real_dfsar.min_positive_pixels` | `5` |
| `README.md` | 23 | `20` | `config.yaml.real_dfsar.grid_size[1]` | `20` |
| `README.md` | 23 | `256` | `config.yaml.data.patch_size` | `256` |
| `README.md` | 23 | `0.8124` | `derived: five_fold_cv[CustomCNN].roc_mean` | `0.812389` |
| `README.md` | 23 | `0.0408` | `derived: five_fold_cv[CustomCNN].roc_std` | `0.0407922` |
| `README.md` | 23 | `0.4760` | `baseline_results.json.primary[2].balanced_accuracy` | `0.47597` |
| `README.md` | 23 | `0.1093` | `derived: five_fold_cv[CustomCNN].pr_std` | `0.109281` |
| `README.md` | 23 | `0.8118` | `derived: five_fold_cv[LogisticRegression].roc_mean` | `0.811761` |
| `README.md` | 23 | `0.0460` | `derived: five_fold_cv[LogisticRegression].roc_std` | `0.0459816` |
| `README.md` | 23 | `0.4921` | `derived: five_fold_cv[LogisticRegression].pr_mean` | `0.492053` |
| `README.md` | 23 | `0.1197` | `derived: five_fold_cv[LogisticRegression].pr_std` | `0.119708` |
| `README.md` | 23 | `0.7978` | `derived: five_fold_cv[ResNet_scratch].roc_mean` | `0.797759` |
| `README.md` | 23 | `0.0427` | `derived: five_fold_cv[ResNet_scratch].roc_std` | `0.0426706` |
| `README.md` | 23 | `0.4641` | `derived: five_fold_cv[ResNet_scratch].pr_mean` | `0.46412` |
| `README.md` | 23 | `0.1110` | `derived: five_fold_cv[ResNet_scratch].pr_std` | `0.110978` |
| `README.md` | 23 | `0.7955` | `block6_stats.json.per_strip_frozen[1].strips.0.pr_auc` | `0.795531` |
| `README.md` | 23 | `0.0448` | `derived: five_fold_cv[IceUNet_mean_prob].roc_std` | `0.0447728` |
| `README.md` | 23 | `0.4169` | `derived: five_fold_cv[IceUNet_mean_prob].pr_mean` | `0.41685` |
| `README.md` | 23 | `0.1328` | `derived: five_fold_cv[IceUNet_mean_prob].pr_std` | `0.132753` |
| `README.md` | 23 | `104` | `block6_stats.json.paired_subblock_bootstrap[0].n_subblocks` | `104` |
| `README.md` | 23 | `+0.0228` | `block6_stats.json.paired_subblock_bootstrap[0].delta_roc_auc` | `0.0228405` |
| `README.md` | 23 | `-0.0109` | `block6_stats.json.paired_subblock_bootstrap[0].delta_roc_ci[0]` | `-0.0109476` |
| `README.md` | 23 | `+0.0546` | `label_efficiency_fairness.json.summary[0].pr_auc_std` | `0.0545626` |
| `README.md` | 23 | `+0.0020` | `block6_stats.json.paired_subblock_bootstrap[3].delta_roc_auc` | `0.00199999` |
| `README.md` | 23 | `-0.0092` | `block6_stats.json.paired_subblock_bootstrap[3].delta_roc_ci[0]` | `-0.00923039` |
| `README.md` | 23 | `+0.0108` | `efficiency_report.json.optuna.test_metrics.pr_auc` | `0.0108244` |
