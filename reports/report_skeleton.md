# IceShadowNet Report Skeleton
# Follows the section order specified in PRD.md §8

---

## 1. Title & Team

**Title:** Deep Learning-Based Detection of Potential Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 DFSAR and OHRC Data: A Comparative Study of Custom CNN and ResNet

| # | Name | Role |
|---|------|------|
| 1 | TODO | Data & Labeling |
| 2 | TODO | Custom CNN |
| 3 | TODO | ResNet |
| 4 | TODO | Evaluation & Explainability |
| 5 | TODO | Integration & Report |

---

## 2. Problem & Objectives

<!-- TODO (team): Write 2–3 paragraphs covering:
     - Why lunar south-polar ice detection matters (ISRU, future missions).
     - What DFSAR/OHRC provide and what current threshold-based detection misses.
     - The specific DL research question: custom CNN vs ResNet for binary patch classification.
     Reference PRD §1–§2. -->

**Placeholder text (replace with team-written prose):**
ISRO's Chandrayaan-2 DFSAR payload provides dual-polarisation SAR data over
permanently-shadowed lunar south-polar craters suspected to contain subsurface
water-ice. Current detection relies on hand-crafted radar criteria (CPR > 1,
DOP < 0.13). This project investigates whether a deep learning model trained on
radar patch images can learn spatial and textural signatures beyond pixel-wise
thresholds, and compares a custom CNN baseline to adapted ResNet architectures.

**Objectives:**
1. Construct a labeled patch dataset using the CPR/DOP threshold as ground truth.
2. Train and evaluate a custom CNN from scratch.
3. Train and evaluate ResNet (scratch + ImageNet-pretrained adaptation).
4. Compare models on Accuracy, Precision, Recall, F1, ROC-AUC, training time, parameter count.
5. Apply Grad-CAM to verify the model attends to spatially meaningful radar structures.
6. Implement DFSAR + OHRC multimodal fusion variants.

---

## 3. Dataset & Preprocessing

### 3.1 Data Source

> **⚠️ NOTE:** All results in this report are computed on a **SYNTHETIC** dataset.
> Real DFSAR tiles from PRADAN/ISSDC are not yet downloaded (access pending).
> See `reports/data_notes.md` for full acquisition status.
> When real tiles are obtained, re-run `scripts/ingest_and_split.py` and
> `scripts/train_all.py` — no code changes required.

| Source | Payload | Status |
|--------|---------|--------|
| PRADAN/ISSDC | DFSAR (dual-pol SAR) | ⏳ Pending download |
| PRADAN/ISSDC | OHRC (optical) | ⏳ Pending download |
| **This report** | **Synthetic (physically motivated)** | ✅ Used here |

### 3.2 Synthetic Scene Description

The synthetic DFSAR scene models:
- Spatially correlated speckle (Rayleigh, L=4 looks, Gaussian σ=20px)
- HH / HV dual-polarisation channels
- A circular ice-candidate region (~15% of pixels) with elevated HV so that CPR > 1 and DOP < 0.13

### 3.3 Class Balance

<!-- Auto-populated from reports/class_balance.txt -->

```
Data source          : SYNTHETIC (no real DFSAR tiles found in data/raw/)
OHRC source          : SYNTHETIC
Patch size / stride  : 256 / 128
Total patches        : 225
  Ice (positive)     : 15  (6.7%)
  Non-ice (negative) : 210 (93.3%)
pos_weight for BCE   : 14.00  (= n_noice / n_ice)

Train / Val / Test   : 155 / 32 / 38
  Train ice rate     : 4.5%
  Val   ice rate     : 12.5%
  Test  ice rate     : 10.5%

Spatial split grid   : (8, 8)
Leakage assertion    : PASSED (no coordinate in >1 split)

RECOMMENDATION:
  pos_weight = 14.00 -> use BCEWithLogitsLoss(pos_weight=tensor([14.00]))
  Focal loss is also viable given strong imbalance if ice_ratio < 0.15
```

### 3.4 Labeling

Binary labels derived by: **CPR > 1.0  AND  DOP < 0.13** (Spudis et al. 2010).

**Limitation (state in viva):** These labels encode the same radar criteria the
model is asked to learn. There is inherent label circularity — see §6 Analysis.

### 3.5 Patch Extraction

- Patch size: 256 × 256 pixels
- Stride: 128 pixels (50% overlap for richer sampling)
- Channels-first layout for PyTorch: (C, H, W)

### 3.6 Spatial Block Split

To prevent data leakage from spatially adjacent patches appearing in both
train and test sets, we split by **geographic block** (4 × 4 grid).
Entire contiguous blocks are assigned exclusively to train / val / test.
A hard Python `assert` verifies zero coordinate overlap before training begins.

| Split | Patches |
|-------|---------|
| Train | TODO (from class_balance.txt) |
| Val   | TODO |
| Test  | TODO |

---

## 4. Methodology & Architecture

### 4.1 Custom CNN

```
Input (B, 2, 256, 256)
  ↓ Conv(3×3, 32) + BN + ReLU + MaxPool(2×2)
  ↓ Conv(3×3, 64) + BN + ReLU + MaxPool(2×2)
  ↓ Conv(3×3, 128) + BN + ReLU + MaxPool(2×2)
  ↓ AdaptiveAvgPool → (B, 128, 4, 4)
  ↓ Flatten → Linear(2048→256) + ReLU + Dropout(0.5)
  ↓ Linear(256→1)  [logit]
```

**Design justification:** Kept deliberately small to avoid memorising the threshold
rule. BatchNorm stabilises SAR's high dynamic range. Dropout prevents overfitting
on a small single-crater dataset.

### 4.2 ResNet (Scratch)

Standard ResNet-18 with first conv modified to accept 2-channel SAR input.
Trained entirely from scratch — avoids ImageNet domain mismatch at the cost of
needing more data to converge (limitation for small datasets).

### 4.3 ResNet (ImageNet-Pretrained, Adapted)

Pre-trained ResNet-18 with first conv weight initialised by averaging RGB channel
weights → adapted for 2-channel SAR input. Layers 1–2 frozen; layers 3–4 + FC
fine-tuned.

**Domain mismatch tradeoff (viva answer):** ImageNet features (edges, textures
from natural photos) partially transfer to SAR — early convolutional filters act
as generic spatial frequency extractors. However, SAR coherent speckle statistics
are radically different from optical noise. This trade-off is empirically evaluated
by comparing scratch vs pretrained results in §5.

### 4.4 Multimodal Fusion (DFSAR + OHRC)

Two fusion strategies:
- **Early Fusion:** concatenate DFSAR (2-ch) + OHRC (1-ch) = 3-ch input → CustomCNN
- **Late Fusion:** separate ResNet18 feature extractors per modality, fused at penultimate layer

### 4.5 Loss Function

<!-- TODO: fill in actual pos_weight from class_balance.txt -->
Class imbalance handled by **BCEWithLogitsLoss(pos_weight=<POS_WEIGHT>)** if
imbalance ratio ≤ 10:1, else **Focal Loss** (α=0.25, γ=2).

Optimiser: Adam, lr=0.001, ReduceLROnPlateau (patience=5, factor=0.5).

---

## 5. Implementation Details

| Item | Value |
|------|-------|
| Framework | PyTorch 2.x |
| Patch size | 256 × 256 |
| Batch size | 16 |
| Epochs | 30 |
| Augmentation | Random H/V flip, 90° rotation |
| Normalisation | Per-channel zero-mean / unit-std |
| Seed | 42 (all random sources fixed) |
| Device | CPU (no GPU available in dev environment) |

---

## 6. Results & Analysis

> **⚠️ ALL NUMBERS BELOW ARE FROM SYNTHETIC DATA.  
> Replace with real-data results once PRADAN tiles are downloaded and pipeline is re-run.**

### 6.1 DFSAR-Only Models

<!-- Auto-populated from reports/training_summary.md -->

# Training Summary

> **DATA SOURCE:** SYNTHETIC (physically-motivated SAR scene -- no real DFSAR tiles from PRADAN yet). Numbers below demonstrate pipeline correctness only, NOT real ice-detection performance.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|---------|--------|----------|
| CustomCNN | 0.7281±0.0124 | 0.4444±0.3143 | 0.1212±0.0857 | 0.1905±0.1347 | 0.4882±0.0225 | TP:2 FP:1 FN:9 TN:26 (Pos:11 Neg:27) | 618,209 | 14.3 |
| ResNet_scratch | 0.7105±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.4820±0.0794 | TP:0 FP:0 FN:11 TN:27 (Pos:11 Neg:27) | 11,173,889 | 54.7 |
| ResNet_pretrained | 0.7105±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.0000±0.0000 | 0.4355±0.0894 | TP:0 FP:0 FN:11 TN:27 (Pos:11 Neg:27) | 6,067,201 | 45.3 |

## [WARNING] Circularity / Low Support Flags


[WARNING] MAJORITY-CLASS COLLAPSE: ResNet_scratch
   accuracy=0.711, Precision=0.000, Recall=0.000
   The model failed to learn the minority class and is predicting a single class for everything.


[WARNING] MAJORITY-CLASS COLLAPSE: ResNet_pretrained
   accuracy=0.711, Precision=0.000, Recall=0.000
   The model failed to learn the minority class and is predicting a single class for everything.


### 6.2 Multimodal Models (DFSAR + OHRC)

<!-- Auto-populated from reports/training_summary_multimodal.md -->

# Multimodal Training Summary

> **DATA SOURCE:** SYNTHETIC. See reports/data_notes.md.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Support | Params | Time (s) |
|-------|----------|-----------|--------|----|---------|---------|--------|----------|
| EarlyFusionCNN | 0.4697±0.0652 | 0.8918±0.0782 | 0.2667±0.1247 | 0.3888±0.1526 | 0.4294±0.0464 | TP:12 FP:2 FN:18 TN:12 (Pos:30 Neg:14) | 618,497 | 11.2 |
| LateFusionResNet | 0.4394±0.1714 | 0.2273±0.3214 | 0.3333±0.4714 | 0.2703±0.3822 | 0.5270±0.0522 | TP:30 FP:14 FN:0 TN:0 (Pos:30 Neg:14) | 22,606,273 | 43.9 |

## [WARNING] Circularity / Low Support Flags


[WARNING] LOW SUPPORT -- METRICS ON THIS SPLIT ARE NOISY
   Val Positives: 0.0, Test Positives: 30.0


### 6.3 Loss Curves
**⚠️ SYNTHETIC DATA — PLACEHOLDER**

<!-- Embed: reports/CustomCNN_loss_curve.png, reports/ResNet_scratch_loss_curve.png, reports/ResNet_pretrained_loss_curve.png -->

### 6.4 Confusion Matrices
**⚠️ SYNTHETIC DATA — PLACEHOLDER**

<!-- Embed: reports/CustomCNN_confusion_matrix.png etc. -->

### 6.5 ROC Curves
**⚠️ SYNTHETIC DATA — PLACEHOLDER**

<!-- Embed: reports/CustomCNN_roc_curve.png etc. -->

### 6.6 Grad-CAM Overlays
**⚠️ SYNTHETIC DATA — PLACEHOLDER**

<!-- Embed: reports/gradcam/*.png — discuss whether attention regions are physically plausible -->

<!-- TODO (team): Write 3–5 sentences interpreting each Grad-CAM result:
     Does the heatmap concentrate on the known ice-candidate region?
     Or does it fire uniformly (suggesting threshold re-learning, not texture learning)? -->

### 6.7 Observation: Train/Val vs Test Generalization Gap
**⚠️ SYNTHETIC DATA — PLACEHOLDER**

Because we explicitly prevent data leakage using a strict geographic **spatial block split** (see §3.6), the validation and test sets come from spatially distinct regions of the crater compared to the training set.

This leads to a measurable generalization gap. For example, in our 30-epoch synthetic runs:
* **CustomCNN**: Val-F1: ~0.26 vs Test-F1: ~0.19
* *(Real numbers to be finalized in Block 4)*

**Methodological Finding:** This gap is *expected and desirable*. Without a spatial split, a model could artificially inflate test performance by memorizing adjacent overlapping pixels from the training set. The observed drop in performance on the geographically disjoint test set reflects the *true* difficulty of out-of-distribution spatial generalization on radar textures.

### 6.8 Analysis: Why Does One Model Outperform the Other?

<!-- TODO (team): This is the core analytical section — required for viva.
     Address the following:
     1. Does the more complex model (ResNet) actually outperform the custom CNN?
        If not, explain why (data size, domain mismatch, overfitting).
     2. Does the pretrained ResNet outperform scratch? Expected tradeoff: domain mismatch
        vs. pre-learned spatial features.
     3. Does multimodal fusion improve F1/ROC-AUC? If not, discuss registration errors
        or modality redundancy.
     4. Any near-perfect accuracy? See §6.8 on circularity. -->

### 6.9 ⚠️ Label Circularity Risk

**This section is mandatory in viva.**

Labels are derived from CPR > 1 AND DOP < 0.13 — the same features input to the
model. A model that achieves near-perfect accuracy may simply be re-learning the
threshold rule pixel-by-pixel, NOT learning spatial texture.

Evidence for / against circularity:
- If Grad-CAM heatmaps fire uniformly over entire ice-labelled patches: **circularity likely**.
- If Grad-CAM focuses on patch boundaries or specific texture patterns beyond the label
  mask: **model may be learning spatial features beyond thresholding**.
- The custom CNN's small capacity reduces (but does not eliminate) this risk.

Mitigation stated in report: this limitation is fully acknowledged. The project's
contribution is a reproducible pipeline and architecture comparison — not a claim
of physically confirmed ice detection.

---

## 7. Conclusion & Future Scope

<!-- TODO (team): Write 2–3 paragraphs summarising:
     - Which model performed best and hypothesised reason.
     - Whether spatial block split prevented measurable leakage.
     - Key limitation: synthetic data, label circularity, single crater.
     - Future scope items from PRD §13: U-Net segmentation, geospatial probability map,
       ice volume estimation. -->

**Explicit future scope (from PRD §13):**
1. Semantic segmentation (U-Net) for pixel-level ice probability maps.
2. Integration with QGIS for geospatial visualisation.
3. Ice volume estimation via backscatter/dielectric modelling.
4. Re-run with real DFSAR/OHRC tiles once PRADAN access is confirmed.

---

## 8. References

1. ISRO/ISSDC Chandrayaan-2 mission data — https://pradan.issdc.gov.in/ch2/
2. Spudis et al. (2010). "Initial results for the north pole of the Moon from Mini-RF." *Geophysical Research Letters.*
3. He et al. (2015). "Deep Residual Learning for Image Recognition." *CVPR.*
4. Selvaraju et al. (2017). "Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization." *ICCV.*
5. Lin et al. (2017). "Focal Loss for Dense Object Detection." *ICCV.*
6. Nozette et al. (1996). "The Clementine Bistatic Radar Experiment." *Science.*
