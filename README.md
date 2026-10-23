# IceShadowNet

**Deep Learning-Based Detection of Potential Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 DFSAR Data: A Comparative Study of Custom CNN and ResNet (with Multimodal OHRC as Synthetic Future Work)**

> Motivated by ISRO's Chandrayaan-2 lunar ice characterization problem. This project extracts a focused, defensible Deep Learning research question from that domain rather than reproducing the full hackathon pipeline (GIS + radar processing + rover path planning).

---

## 1. Overview & Scope Clarification

IceShadowNet investigates whether deep learning models can distinguish **anomalous polarimetric radar signatures** from **typical lunar regolith backscatter** in lunar south-polar regions, using real Chandrayaan-2 **DFSAR** (Dual Frequency Synthetic Aperture Radar) L-band quad-polarimetric data.

> **Scope Decision (Real vs. Synthetic Data):**
> - **Real-Data Pipeline:** **DFSAR-only**. Real L1A complex slant-range (SLI) quad-polarimetric data (`ch2_sar_ncxl_20260308`) is processed to synthesize circular polarization channels and derive multilooked CPR and DOP labels.
> - **Multimodal (OHRC) Pipeline:** **Synthetic-only future work** *(synthetic demo, not part of the real-data benchmark)*. The downloaded real OHRC optical tile (`ch2_ohr_ncp_20260719`) does not geographically overlap the DFSAR acquisition strip (~35° longitude separation / ~1,800 km apart). Multimodal fusion architectures (EarlyFusionCNN, LateFusionResNet) are validated on synthetic data only and scoped as future work pending overlapping optical coverage.

## 2. Scientific Framing & Limitations

- **Label Interpretation:** Labels derived from $\text{CPR} > 1.0 \land \text{DOP} < 0.13$ define **"regions with anomalous circular polarization and low degree of polarization"** (consistent with multiple volume scattering / potential subsurface ice in lunar literature). Because no external PSR (Permanently Shadowed Region) or DEM slope mask is applied, labels denote **anomalous CPR regions**, NOT confirmed physical water-ice.
- **Model Input Policy (Anti-Circularity):** Model inputs are strictly **raw radar log-intensities** ($\log_{10}|S_{HH}|^2, \log_{10}|S_{HV}|^2, \log_{10}|S_{VV}|^2$). Derived quantities (CPR, DOP) are strictly excluded from network inputs to prevent the models from trivially memorizing the labeling rule.
- **Multilook Averaging:** Single-look complex radar data mathematically forces $\text{DOP} \equiv 1.0$ at every pixel. A $7 \times 7$ boxcar multilook filter is applied to the polarimetric covariance terms before calculating DOP and CPR.

## 3. Objectives

- Build a leak-free labeled patch dataset from real DFSAR L1A complex imagery using spatial block splitting.
- Implement a custom CNN from scratch and adapt ResNet18 for raw multi-channel SAR backscatter.
- Compare models on Accuracy, Precision, Recall, F1, ROC-AUC, training time, parameter count, and confusion matrix.
- Evaluate classical baselines (Plain Threshold and Decision Tree on CPR/DOP) on the identical spatial test split.
- Use Grad-CAM to evaluate whether models attend to physically plausible radar backscatter features.

## 4. Dataset

| Source | Payload | Processing Level | Status / Use |
|---|---|---|---|
| Chandrayaan-2 | DFSAR (L-band quad-pol) | L1A SLI (Complex float32) | **REAL:** Primary input; quad-pol complex scattering matrix ($S_{HH}, S_{HV}, S_{VH}, S_{VV}$) |
| Chandrayaan-2 | OHRC (Panchromatic optical) | Calibrated L1 | **FUTURE WORK:** Disjoint footprint; preserved for future multimodal alignment *(synthetic demo, not part of the real-data benchmark)* |

- **Real DFSAR tile:** `ch2_sar_ncxl_20260308t062017233` (South Polar strip, −81.7° to −86.3° latitude).
- **Label rule:** $\text{CPR} > 1.0 \land \text{DOP} < 0.13$ with multilook window $W=7$ ($\ge 5$ candidate pixels per $64 \times 64$ patch).
- **Split strategy:** Spatial block split grid $(1, 20)$ along the ground track to prevent adjacent patch leakage.

## 5. Methodology

```
Real DFSAR L1A Complex SLI (HH, HV, VH, VV)
        ↓
Multilook Polarimetric Analysis (W=7 boxcar)
   ├── S_RR = 0.5 * ((Shh - Svv) + 2i*Shv)   [Same-sense circular]
   └── S_RL = 0.5 * (Shh + Svv)              [Opposite-sense circular]
        ↓
Label Generation: CPR > 1.0 & DOP < 0.13 (min 5 px/patch)
        ↓
Model Input Construction: Raw Log-Intensities [HH, HV, VV]
(CPR and DOP strictly excluded from inputs)
        ↓
Spatial Block Split (Grid 1x20, zero spatial leakage)
        ↓
Model Training (CustomCNN, ResNet-scratch, ResNet-pretrained) & Baselines (Threshold, Decision Tree)
        ↓
Evaluation & Grad-CAM Explainability
```

## 6. Results

> **Real-Data Benchmark & Analysis:**  
> For the completed real Chandrayaan-2 DFSAR benchmark results and scientific interpretation, see:
> - **[Section 3 Analysis Draft](reports/real/section3_analysis.md)**: Headline finding (no deep model beats Logistic Regression on pooled metrics), CustomCNN vs ResNet_scratch parameter efficiency tie, ImageNet pretraining transfer breakdown, per-strip prevalence dynamics, and baseline sensitivity.
> - **[Real-Data Benchmark Results (Spatial Block-Bootstrap CIs)](reports/real/results_real_ci_all.md)**: Default and conditional 95% CIs across all 15 deep learning runs (seeds 42, 123, 2024) and classical baselines on full and clean test sets.
> - **[Along-Track Per-Strip Metrics Breakdown](reports/real/per_strip_metrics.md)**: Granular strip-by-strip (strips 6, 10, 14) breakdown and zero-TP detection analysis.
> - **[Architecture Comparison](reports/real/results_real_variants_comparison.md)**: Cross-variant summary including ResNet_pretrained v2a/v2b.
> - **[Interactive Web Dashboard](dashboard/)**: Interactive visual exploration of benchmark metrics, pipeline stages, and patch classifications. Live dashboard: (add URL after deploy)

> **Historical Synthetic Pipeline Benchmark:**  
> *(synthetic demo, not part of the real-data benchmark)*  
> Demonstrates initial pipeline execution across multiple seeds and epochs on synthetic data.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|---|---|---|---|---|---|---|---|---|
| CustomCNN | 0.7361±0.0196 | 0.4444±0.3143 | 0.1905±0.1347 | 0.2667±0.1886 | 0.6751±0.0555 | TP:2 FP:1 FN:5 TN:16 (Pos:7 Neg:17) | 618,209 | 76.0 |
| ResNet (Scratch) | 0.8194±0.0393 | 0.7361±0.0856 | 0.6190±0.1347 | 0.6605±0.0915 | 0.8459±0.0914 | TP:3 FP:1 FN:4 TN:16 (Pos:7 Neg:17) | 11,173,889 | 151.1 |
| ResNet (Pretrained) | 0.7778±0.0196 | 0.6488±0.0748 | 0.5714±0.1166 | 0.5945±0.0521 | 0.8403±0.0562 | TP:4 FP:3 FN:3 TN:14 (Pos:7 Neg:17) | 6,067,201 | 73.9 |
| EarlyFusionCNN *(synthetic demo)* | 0.7222±0.0196 | 0.2222±0.3143 | 0.0952±0.1347 | 0.1333±0.1886 | 0.4594±0.0309 | TP:0 FP:0 FN:7 TN:17 (Pos:7 Neg:17) | 618,497 | 54.0 |
| LateFusionResNet *(synthetic demo)* | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | TP:7 FP:0 FN:0 TN:17 (Pos:7 Neg:17) | 22,606,273 | 200.1 |

## 7. Experiments

| Experiment | Input | Status |
|---|---|---|
| CNN | DFSAR only | Real-data benchmark complete |
| ResNet | DFSAR only | Real-data benchmark complete |
| CNN (Early Fusion) | DFSAR + OHRC (multimodal) | Synthetic demo, not part of the real-data benchmark |
| ResNet (Late Fusion) | DFSAR + OHRC (multimodal) | Synthetic demo, not part of the real-data benchmark |

## 8. Tech Stack

- **Language:** Python
- **DL Framework:** PyTorch
- **Geo/Radar processing:** GDAL, rasterio, NumPy, SciPy
- **Visualization:** Matplotlib, QGIS (for geospatial output)
- **Explainability:** Grad-CAM

## 9. Repository Structure

```
IceShadowNet/
├── data/
│   ├── raw/
│   │   └── MISSING_DATA.md      # PRADAN download guide for real DFSAR L1A tiles
│   └── processed_real/          # Spatial NPZ splits (train, val, test)
├── models_real/                 # PyTorch model weights for 15 real benchmark runs
├── reports/
│   ├── real/                    # Benchmark metrics, comparison tables, and analysis
│   │   ├── section3_analysis.md # Section 3 draft: findings, comparisons, limitations
│   │   ├── results_real_ci_all.md # 3-block bootstrap CIs (default & conditional)
│   │   ├── per_strip_metrics.md # Granular along-track per-strip metrics (strips 6, 10, 14)
│   │   ├── results_real_variants_comparison.md # Side-by-side comparison across all variants
│   │   ├── trivial_baselines_and_ambiguity_note.md # All-positive baselines & ambiguity note
│   │   └── training_summary*.json # Stored metrics for programmatic reproduction
│   └── synthetic_archive/       # Archived placeholder synthetic reports
├── scripts/
│   ├── train_all.py             # Main DFSAR multi-seed training pipeline
│   ├── baseline_real.py         # Decision Tree & Logistic Regression feature baselines
│   ├── ingest_and_split.py      # Real DFSAR L1A tile ingestion & spatial block splitting
│   ├── run_gradcam.py           # Grad-CAM heatmap visualization runner
│   ├── _smoke_gpu.py            # 1-epoch GPU verification smoke test
│   ├── train_multimodal.py      # Multimodal training (synthetic demo, not part of real benchmark)
│   ├── ingest_multimodal.py     # Multimodal ingestion (synthetic demo, not part of real benchmark)
│   ├── verify_real_tile.py      # DFSAR L1A GeoTIFF inspector
│   └── analysis/                # Standalone analysis & metric reproduction tools
│       ├── generate_comparison.py
│       ├── compute_per_strip.py
│       ├── generate_phase4_reports.py
│       ├── compute_trivial.py
│       ├── generate_ci_fixed.py
│       └── audit_bootstrap.py
├── src/
│   ├── datasets/                # IcePatchDataset & spatial block split logic
│   ├── models/                  # CustomCNN, IceResNet, MultimodalFusion
│   ├── preprocessing/           # Multilooking, circular polarization, CPR/DOP
│   ├── train.py                 # Core training loop & optimizer routines
│   ├── evaluate.py              # Spatial block-bootstrap CI & metric evaluation
│   └── gradcam.py               # Grad-CAM implementation
├── tests/
│   └── test_pipeline.py         # Pytest test suite (splitting, circularity, data ops)
├── .github/workflows/tests.yml  # GitHub Actions CI workflow
├── README.md
├── PRD.md
├── requirements.txt
└── config.yaml
```

## 10. Setup

```bash
git clone https://github.com/nish-debug15/IceShadowNet
cd IceShadowNet
pip install -r requirements.txt
```

Data access: register at [PRADAN](https://pradan.issdc.gov.in/ch2/), download DFSAR/OHRC tiles for the target crater, place under `data/raw/`.

## 11. Team

| Role | Responsibility |
|---|---|
| Data & Labeling | DFSAR/OHRC acquisition, CPR/DOP labeling, spatial split |
| Custom CNN | Architecture design, training, tuning |
| ResNet | Adaptation for SAR input, transfer-learning decisions, training |
| Evaluation & Explainability | Metrics, confusion matrix, Grad-CAM |
| Integration & Report | Pipeline glue, geospatial mapping, report/demo assembly |

## 12. Acknowledgement

> This research is based partially on the results obtained from the Chandrayaan-2, second lunar mission of the Indian Space Research Organisation (ISRO), archived at the Indian Space Science Data Centre (ISSDC).

## 13. References

- ISRO/ISSDC Chandrayaan-2 mission data — https://pradan.issdc.gov.in/ch2/
- ISRO Lunar Polar Region overview — https://www.isro.gov.in/Lunar_Polar_Region.html
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), 2015
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, 2017
