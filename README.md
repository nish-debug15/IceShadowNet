# IceShadowNet

**Deep Learning-Based Detection of Potential Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 DFSAR and OHRC Data: A Comparative Study of Custom CNN and ResNet**

> Motivated by ISRO's Chandrayaan-2 lunar ice characterization problem. This project extracts a focused, defensible Deep Learning research question from that domain rather than reproducing the full hackathon pipeline (GIS + radar processing + rover path planning).

---

## 1. Overview

IceShadowNet investigates whether deep learning models can distinguish **potential ice-bearing regions** from **non-ice / rough terrain** inside lunar south-polar "doubly shadowed craters," using Chandrayaan-2 **DFSAR** (Dual Frequency Synthetic Aperture Radar) and **OHRC** (Orbiter High Resolution Camera) data.

We compare a **custom CNN** against **ResNet** on this classification task, and analyze *why* one outperforms the other — not just that it does.

## 2. Motivation

Subsurface water-ice in the Moon's permanently shadowed regions (PSRs) is a high-priority target for future ISRU (In-Situ Resource Utilization) missions. Existing detection relies on hand-crafted radar thresholds (e.g., CPR > 1, DOP < 0.13). This project tests whether a learned model can pick up spatial/textural patterns a pixel-wise threshold misses.

**Important scientific framing:** this project detects and classifies *radar/imagery signatures consistent with subsurface ice*, not ice itself. All ground truth is derived from established scientific radar criteria, not physical sample verification.

## 3. Objectives

- Build a labeled patch dataset from DFSAR (+ optionally OHRC) imagery of a doubly-shadowed crater.
- Implement a custom CNN from scratch.
- Implement/adapt ResNet for the same task.
- Compare both models on accuracy, precision, recall, F1, ROC-AUC, training/validation loss, training time, parameter count, and confusion matrix.
- Use Grad-CAM to check whether the models attend to physically meaningful radar structures.
- (Optional extension) Convert per-patch predictions into a geospatial ice-probability map.

## 4. Dataset

| Source | Payload | Use |
|---|---|---|
| Chandrayaan-2 | DFSAR (Dual Frequency SAR) | Primary radar input — CPR/DOP-derived features |
| Chandrayaan-2 | OHRC (High Resolution Camera) | Optical/terrain context, optional multimodal input |

- Data obtained via [PRADAN / ISSDC](https://pradan.issdc.gov.in/ch2/) (public, requires free registration).
- Scoped to one or more doubly-shadowed craters in the lunar south polar region.
- Labels derived from radar polarimetric thresholds (CPR > 1, DOP < 0.13), not manual annotation.
- **Split strategy:** spatial block split (not random) to avoid leakage between adjacent, spatially-correlated patches.

## 5. Methodology

```
DFSAR + OHRC raw data
        ↓
Preprocessing (radar normalization, polarimetric feature extraction,
                image registration, patch generation, augmentation)
        ↓
Labeling (CPR/DOP threshold → candidate ice / non-ice patches)
        ↓
Model Training
   ├── Custom CNN
   └── ResNet (evaluate scratch vs. adapted pretrained weights —
       ImageNet stats don't transfer cleanly to SAR data)
        ↓
Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC,
            loss curves, training time, parameter count, confusion matrix)
        ↓
Explainability (Grad-CAM)
        ↓
(Optional) Geospatial ice-probability map
```

## 6. Results

> **⚠️ SYNTHETIC DATA — PLACEHOLDER BENCHMARK**  
> Demonstrates full end-to-end pipeline execution. Final real-world detection metrics await PRADAN tile download.  
> **Trained for 30 epochs, 3 seeds, on SYNTHETIC data**

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|---|---|---|---|---|---|---|---|---|
| CustomCNN | 0.7361±0.0196 | 0.4444±0.3143 | 0.1905±0.1347 | 0.2667±0.1886 | 0.6751±0.0555 | TP:2 FP:1 FN:5 TN:16 (Pos:7 Neg:17) | 618,209 | 76.0 |
| ResNet (Scratch) | 0.8194±0.0393 | 0.7361±0.0856 | 0.6190±0.1347 | 0.6605±0.0915 | 0.8459±0.0914 | TP:3 FP:1 FN:4 TN:16 (Pos:7 Neg:17) | 11,173,889 | 151.1 |
| ResNet (Pretrained) | 0.7778±0.0196 | 0.6488±0.0748 | 0.5714±0.1166 | 0.5945±0.0521 | 0.8403±0.0562 | TP:4 FP:3 FN:3 TN:14 (Pos:7 Neg:17) | 6,067,201 | 73.9 |
| EarlyFusionCNN | 0.7222±0.0196 | 0.2222±0.3143 | 0.0952±0.1347 | 0.1333±0.1886 | 0.4594±0.0309 | TP:0 FP:0 FN:7 TN:17 (Pos:7 Neg:17) | 618,497 | 54.0 |
| LateFusionResNet | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | 1.0000±0.0000 | TP:7 FP:0 FN:0 TN:17 (Pos:7 Neg:17) | 22,606,273 | 200.1 |

*Note: The metrics above demonstrate pipeline correctness across multiple seeds and epochs. Real-world physical performance will be established once PRADAN tiles are ingested.*

## 7. Experiments

| Experiment | Input |
|---|---|
| CNN | DFSAR only |
| ResNet | DFSAR only |
| CNN | DFSAR + OHRC (multimodal) |
| ResNet | DFSAR + OHRC (multimodal) |

## 7. Tech Stack

- **Language:** Python
- **DL Framework:** PyTorch / TensorFlow 
- **Geo/Radar processing:** GDAL, rasterio, NumPy, SciPy
- **Visualization:** Matplotlib, QGIS (for geospatial output)
- **Explainability:** Grad-CAM

## 8. Repository Structure

```
IceShadowNet/
├── data/                # raw + processed patches (not committed if large)
├── notebooks/           # EDA, preprocessing experiments
├── src/
│   ├── preprocessing/   # radar normalization, CPR/DOP, patch generation
│   ├── datasets/        # dataset + spatial-split logic
│   ├── models/          # custom CNN, ResNet
│   ├── train.py
│   ├── evaluate.py
│   └── gradcam.py
├── reports/             # figures, metrics, final report
├── README.md
└── PRD.md
```

## 9. Setup

```bash
git clone https://github.com/nish-debug15/IceShadowNet
cd IceShadowNet
pip install -r requirements.txt
```

Data access: register at [PRADAN](https://pradan.issdc.gov.in/ch2/), download DFSAR/OHRC tiles for the target crater, place under `data/raw/`.

## 10. Team

| Role | Responsibility |
|---|---|
| Data & Labeling | DFSAR/OHRC acquisition, CPR/DOP labeling, spatial split |
| Custom CNN | Architecture design, training, tuning |
| ResNet | Adaptation for SAR input, transfer-learning decisions, training |
| Evaluation & Explainability | Metrics, confusion matrix, Grad-CAM |
| Integration & Report | Pipeline glue, geospatial mapping, report/demo assembly |

## 11. Acknowledgement

> This research is based partially on the results obtained from the Chandrayaan-2, second lunar mission of the Indian Space Research Organisation (ISRO), archived at the Indian Space Science Data Centre (ISSDC).

## 12. References

- ISRO/ISSDC Chandrayaan-2 mission data — https://pradan.issdc.gov.in/ch2/
- ISRO Lunar Polar Region overview — https://www.isro.gov.in/Lunar_Polar_Region.html
- He et al., *Deep Residual Learning for Image Recognition* (ResNet), 2015
- Selvaraju et al., *Grad-CAM: Visual Explanations from Deep Networks*, 2017
