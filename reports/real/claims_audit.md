# Block 6 Item 8: Claims Audit (Verified Against `reports/real/`)

| Claim / Invariant | Audited Files | Verified (Y/N) |
| :--- | :--- | :---: |
| 5-fold CV CustomCNN 0.8124 vs LR 0.8118 | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| Paired sub-block CI CustomCNN-LR [-0.0109, +0.0546] | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| ResNet_pretrained underperforms, not separable under strip CV | README.md, research_report.md | **Y** |
| SSL 10% 0.7892 does not beat LR 0.8036 / CustomCNN 0.8080 | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| Context-matched HistGBDT 7x7+15x15 PR-AUC 0.0077 (14.6x) | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| IceUNet 3-seed PR-AUC 0.0093 (17.6x) / Ensemble 0.0110 (20.9x) | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| Phase Oracle 4-ch PR-AUC 0.0133 (25.2x), Dice 0.0348 (circular) | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| Selective prediction negative verdict (not usable for rejection) | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| Pruning 50% structured B=64 5.359 ms / 30% no speedup 11.441 ms | README.md, research_report.md, docs/index.html | **Y** |
| 10-fold GCP CV mean 367.98 m / median 307.38 m / max 2,377.70 m | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| 92.79% SRI overlap defined | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| Split search prevalence-only disclosure (no model feedback) | README.md, research_report.md, docs/index.html | **Y** |
| Anisotropic 0.505 m x 9.593 m, single scene, proxy labels | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| Forbidden overstatement check: no 'resolves ambiguity' / 'clears ambiguity' | README.md, research_report.md, viva_qa.md, docs/index.html | **Y** |
| Dashboard segmentation_uncertainty.json matches Block 6 JSONs | segmentation_uncertainty.json | **Y** |

**Overall Audit Status:** `PASSED (15/15)`
