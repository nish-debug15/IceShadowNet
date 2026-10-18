# Repository Cleanup Plan — IceShadowNet
_Generated: 2026-10-18 (Block 2 of Phase 5)_

This document classifies every file and directory in the repository with a verdict: **KEEP**, **DELETE**, or **UNSURE**, accompanied by a one-line reason and a dependency analysis.

---

## 1. Summary of Actions
- **KEEP**: 48 items (Core code, tests, configs, documentation, audit reports, real raw tiles)
- **DELETE**: 8 items (Python bytecode `__pycache__` directories, pytest cache, duplicate stray files)
- **UNSURE**: 6 item groups (Cached model weights, cached processed synthetic npz arrays, local plots, duplicate raw SAR subfolder in OHRC directory — flagged for human decision)

---

## 2. Dependency Picture for Python Files

| Module / Script | Inbound Callers / Importers | Purpose / Verdict |
|---|---|---|
| `scripts/ingest_and_split.py` | `scripts/run_gradcam.py`, `scripts/train_all.py` | KEEP — DFSAR patch extraction, CPR/DOP thresholding, spatial-block splitting |
| `scripts/ingest_multimodal.py` | `scripts/train_multimodal.py` | KEEP — Dual-modality DFSAR+OHRC patch generation and spatial-block splitting |
| `scripts/inject_results.py` | Manual / CLI | KEEP — Automates injection of training metrics into `report_skeleton.md` |
| `scripts/run_gradcam.py` | CLI / Analysis | KEEP — Generates Grad-CAM visual interpretability overlays for trained models |
| `scripts/train_all.py` | `scripts/run_gradcam.py`, `scripts/train_multimodal.py`, `tests/test_pipeline.py` | KEEP — Main training pipeline for DFSAR single-modality models (CustomCNN, ResNets) |
| `scripts/train_multimodal.py` | CLI / Evaluation | KEEP — Main training pipeline for multimodal fusion models (EarlyFusion, LateFusion) |
| `scripts/verify_real_tile.py` | CLI / Data Ingestion | KEEP — Verifies PDS4 `.img`/`.xml` and GeoTIFF tiles using rasterio/GDAL |
| `scripts/__init__.py` | Package marker | KEEP — Enables importing scripts as a Python package |
| `src/__init__.py` | Package marker | KEEP — Package root marker |
| `src/evaluate.py` | `scripts/train_all.py` | KEEP — Computes accuracy, F1, precision, recall, ROC-AUC, PR-AUC, confusion matrix |
| `src/gradcam.py` | `scripts/run_gradcam.py` | KEEP — Grad-CAM heatmap extraction and color overlay implementation |
| `src/train.py` | `scripts/ingest_and_split.py`, `scripts/ingest_multimodal.py`, `scripts/train_all.py`, `scripts/train_multimodal.py`, `tests/test_pipeline.py` | KEEP — Core PyTorch training loop with early stopping, validation, and checkpointing |
| `src/datasets/__init__.py` | Package marker | KEEP — Package marker for datasets |
| `src/datasets/ice_dataset.py` | `scripts/train_all.py`, `src/train.py` | KEEP — PyTorch Dataset for single-modality DFSAR patches |
| `src/datasets/multimodal_dataset.py` | `scripts/train_multimodal.py` | KEEP — PyTorch Dataset for paired DFSAR + OHRC patches |
| `src/datasets/spatial_split.py` | `scripts/ingest_and_split.py`, `scripts/ingest_multimodal.py`, `src/train.py`, `tests/test_pipeline.py` | KEEP — Core spatial-block splitting algorithm ensuring zero spatial leakage |
| `src/models/__init__.py` | Package marker | KEEP — Package marker for models |
| `src/models/custom_cnn.py` | `scripts/run_gradcam.py`, `scripts/train_all.py`, `src/gradcam.py`, `src/train.py`, `src/models/multimodal_fusion.py` | KEEP — Baseline 4-layer custom convolutional network |
| `src/models/multimodal_fusion.py` | `scripts/train_multimodal.py` | KEEP — EarlyFusionCNN and LateFusionResNet architectures |
| `src/models/resnet.py` | `scripts/run_gradcam.py`, `scripts/train_all.py`, `src/gradcam.py`, `src/models/multimodal_fusion.py` | KEEP — ResNet18 adapted for 2-channel SAR inputs (scratch and pretrained) |
| `src/preprocessing/__init__.py` | Package marker | KEEP — Package marker for preprocessing |
| `src/preprocessing/data_ops.py` | `scripts/ingest_and_split.py`, `scripts/ingest_multimodal.py`, `src/datasets/ice_dataset.py`, `src/datasets/multimodal_dataset.py` | KEEP — Tile I/O, CPR/DOP derivation, thresholding, patching, and augmentation |
| `src/preprocessing/register_ohrc.py` | `scripts/ingest_multimodal.py` | KEEP — Coordinate transform and affine registration from OHRC to DFSAR grid |
| `tests/test_pipeline.py` | Pytest / CI | KEEP — Automated safety tests for spatial leakage, class collapse, and asserts |

---

## 3. Detailed Inventory & Verdicts

### A. Root Files and Configuration (KEEP)
- `.github/workflows/tests.yml`: **KEEP** — GitHub Actions CI pipeline executing pytest suite on push.
- `.gitignore`: **KEEP** — Version control ignore rules (updated `/models/` path so `src/models/` is tracked).
- `config.yaml`: **KEEP** — Configuration file defining patch sizes, model parameters, and thresholds.
- `PRD.md`: **KEEP** — Product Requirements Document and risk register.
- `README.md`: **KEEP** — Project overview, architecture specification, and role mapping.
- `requirements.txt`: **KEEP** — Declared Python project dependencies.

### B. Reports & Documentation (KEEP)
- `reports/report_skeleton.md`: **KEEP** — Primary academic project report with 30-epoch results.
- `reports/data_audit.md`: **KEEP** — Comprehensive data audit of real Chandrayaan-2 tiles from Block 1.
- `reports/data_notes.md`: **KEEP** — Reference notes detailing dataset characteristics and processing conventions.
- `reports/class_balance.txt`: **KEEP** — Documented class balance across spatial splits.
- `reports/training_summary.json`: **KEEP** — Machine-readable 30-epoch training metrics for DFSAR models.
- `reports/training_summary.md`: **KEEP** — Markdown table of 30-epoch DFSAR training results.
- `reports/training_summary_multimodal.json`: **KEEP** — Machine-readable multimodal training metrics.
- `reports/training_summary_multimodal.md`: **KEEP** — Markdown table of multimodal training results.

### C. Raw Data Directory (KEEP / UNSURE / DELETE)
- `data/raw/MISSING_DATA.md`: **KEEP** — Master PRADAN data acquisition guide and verification checklist.
- `data/raw/sar_nxcl/`: **KEEP** — Canonical real Chandrayaan-2 DFSAR L2 SRI and L1A SLI files (local only).
- `data/raw/ohr_ncp/ch2_ohr_ncp_20260719T0722178640_*`: **KEEP** — Canonical real Chandrayaan-2 OHRC optical product (local only).
- `data/raw/ohr_ncp/MISSING_DATA.md`: **DELETE** — Stray duplicate copy of `MISSING_DATA.md` inside subfolder.
- `data/raw/ohr_ncp/data/calibrated/20260308/`: **UNSURE** — Duplicate copy of DFSAR files (~1.3 GB) bundled in the OHRC zip. Flagged for human review to delete and reclaim space.
- `data/raw/ohr_ncp/browse/calibrated/20260308/`: **UNSURE** — Duplicate browse files for DFSAR tile in OHRC folder.
- `data/raw/ohr_ncp/geometry/calibrated/20260308/`: **UNSURE** — Duplicate geometry CSVs for DFSAR tile in OHRC folder.

### D. Temporary & Cache Artifacts (DELETE)
- `scripts/__pycache__/`: **DELETE** — Compiled Python bytecode cache.
- `src/__pycache__/`: **DELETE** — Compiled Python bytecode cache.
- `src/datasets/__pycache__/`: **DELETE** — Compiled Python bytecode cache.
- `src/models/__pycache__/`: **DELETE** — Compiled Python bytecode cache.
- `src/preprocessing/__pycache__/`: **DELETE** — Compiled Python bytecode cache.
- `tests/__pycache__/`: **DELETE** — Compiled Python bytecode cache.
- `.pytest_cache/`: **DELETE** — Pytest runtime test cache.

### E. Local Generated Checkpoints and Data (UNSURE — For Human Decision)
- `models/CustomCNN.pth` (2.5 MB): **UNSURE** — Trained model weights from synthetic 30-epoch run. Not git-tracked. Keep if offline Grad-CAM is needed, delete to save disk.
- `models/EarlyFusionCNN.pth` (2.5 MB): **UNSURE** — Trained model weights from synthetic multimodal run. Not git-tracked.
- `models/LateFusionResNet.pth` (90.6 MB): **UNSURE** — Trained model weights from synthetic multimodal run. Not git-tracked.
- `models/ResNet_pretrained.pth` (44.8 MB): **UNSURE** — Trained model weights from synthetic 30-epoch run. Not git-tracked.
- `models/ResNet_scratch.pth` (44.8 MB): **UNSURE** — Trained model weights from synthetic 30-epoch run. Not git-tracked.
- `data/processed/*.npz` (6 files, 258 MB): **UNSURE** — Preprocessed synthetic patch arrays. Not git-tracked. Easily regenerated via `python -m scripts.ingest_and_split`.
- `reports/*.png` & `reports/gradcam/*.png`: **UNSURE** — Visualizations (loss curves, ROC, Grad-CAM overlays) generated from synthetic training. Git-ignored.
