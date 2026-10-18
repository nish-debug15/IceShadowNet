# PRD — IceShadowNet

**Deep Learning-Based Detection of Potential Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 DFSAR Data (with Multimodal OHRC as Synthetic Future Work)**

---

## 1. Problem Statement

ISRO's Chandrayaan-2 mission provides L-band quad-polarimetric radar (DFSAR) data over lunar south-polar craters believed to host subsurface water-ice — a high-priority target for future ISRU missions. Current detection relies on hand-crafted radar polarimetric thresholds (CPR > 1, DOP < 0.13) applied pixel-by-pixel. This project asks: **can a deep learning model learn to distinguish potential ice-bearing regions from non-ice terrain using raw radar spatial/textural backscatter without feeding it the derived polarimetric thresholds — and if so, which architecture (custom CNN vs. ResNet) does it better?**

**Scope Realignment (Post-Audit):**
Real Chandrayaan-2 DFSAR L1A complex quad-pol data (`ch2_sar_ncxl_20260308`) is used for the primary real-data pipeline. Because the available real OHRC optical tile does not geographically overlap the DFSAR acquisition strip (~35° longitude separation), multimodal fusion (DFSAR + OHRC) is maintained as a validated synthetic pipeline and marked as future work for real data.

## 2. Objectives

1. Construct a leak-free labeled patch dataset from real DFSAR L1A complex imagery using spatial block splitting.
2. Formulate CPR and DOP from quad-pol complex scattering matrix elements using a $7 \times 7$ multilook boxcar filter.
3. Restrict model inputs strictly to raw radar channels [HH, HV, VV] log-intensity (CPR and DOP excluded).
4. Implement and evaluate classical baselines (Plain Threshold and Decision Tree on CPR/DOP) on the identical spatial test split.
5. Train and compare a custom CNN baseline and adapted ResNet architectures.
6. Compare models across accuracy, loss, precision/recall/F1, ROC-AUC, training time, parameter count, and confusion matrix.
7. Apply Grad-CAM to verify the models attend to physically plausible regions.

## 3. Scope

**In scope**
- Binary classification: anomalous polarimetric patch vs. normal regolith patch.
- Real-data pipeline: DFSAR-only (L1A complex quad-pol).
- Classical polarimetric baselines (Threshold & Decision Tree on CPR/DOP).
- Custom CNN vs. ResNet comparison on raw radar inputs.
- Synthetic-only pipeline: Multimodal fusion (EarlyFusionCNN, LateFusionResNet) scoped as future work for real data.

**Out of scope**
- Physical ice ground-truth verification (no ground truth exists on Moon).
- Rover traverse path planning and ice volume estimation.
- Optical fusion on real tiles (disjoint spatial footprints).

## 4. Non-Goals / Explicit Disclaimers

- The project does **not** claim to detect confirmed physical ice. It classifies signatures *consistent with* the scientific criteria for potential ice presence.
- **Scientific limitation:** No external PSR (Permanently Shadowed Region) or DEM slope mask is applied. Labels strictly signify "anomalous CPR/DOP regions", not verified subsurface ice.
- **Anti-Circularity Policy:** Derived polarimetric features (CPR, DOP) define labels and baseline comparisons only; they are strictly barred from being model inputs.

## 5. Users / Stakeholders

- Course faculty (evaluator, approval authority for topic deviation from the standard list).
- Team members (5) — implementation and viva.
- Indirect: framed as a real-world-motivated academic exercise referencing ISRO/ISSDC public data.

## 6. Data Requirements

| Requirement | Detail |
|---|---|
| Source | PRADAN / ISSDC (public, free registration) |
| Payloads | DFSAR (primary), OHRC (optional multimodal) |
| Scope | ≥1 doubly-shadowed crater in lunar south polar region |
| Labeling | CPR > 1 and DOP < 0.13 threshold → candidate ice / non-ice |
| Split | Spatial block split (prevents adjacent-patch leakage) — **not** random split |
| Class balance | Expected imbalance (ice = minority); mitigate via weighted loss / focal loss / oversampling |
| Format | PDS4 (.IMG/.XML) → convert via GDAL/rasterio |

## 7. Technical Requirements

- **Preprocessing:** radar normalization, polarimetric feature (CPR/DOP) extraction, image registration (DFSAR↔OHRC if multimodal), patch generation, augmentation.
- **Models:**
  - Custom CNN (architecture designed and justified by team).
  - ResNet — decision required: train from scratch vs. adapt pretrained ImageNet weights (SAR data has different channel/statistics profile than RGB photos; this must be explicitly addressed, not assumed).
- **Evaluation metrics:** Accuracy, Precision, Recall, F1, ROC-AUC, training/validation loss curves, training time, parameter count, confusion matrix.
- **Explainability:** Grad-CAM overlays on sample patches.
- **Reproducibility:** fixed seeds, documented train/val/test spatial split, requirements.txt.

## 8. Deliverables (mapped to faculty guidelines)

| Deliverable | Maps to rubric |
|---|---|
| Complete, executable source code + dataset/links | Project Implementation (15) |
| Report: title/team, problem & objectives, dataset & preprocessing, methodology & architecture, implementation details, results & analysis, conclusion & future scope, references | Project Report (part of 5+5) |
| Live demo of full workflow | Project Demonstration (5) |
| Viva — explain methodology, architecture, results, individual contribution | Viva components (5+5) |

## 9. Team Roles

| # | Role | Owns |
|---|---|---|
| 1 | Data & Labeling | Data acquisition (PRADAN), CPR/DOP labeling pipeline, spatial split logic |
| 2 | Custom CNN | Architecture, training loop, tuning |
| 3 | ResNet | Input adaptation, scratch-vs-pretrained decision, training |
| 4 | Evaluation & Explainability | All metrics, confusion matrix, Grad-CAM |
| 5 | Integration & Report | Pipeline glue, optional geospatial map, report + demo assembly |

## 10. Milestones

| Phase | Deliverable |
|---|---|
| Data feasibility | Sample DFSAR + OHRC tile downloaded and opened in GDAL/rasterio; faculty approval for topic deviation obtained |
| Preprocessing | Labeled patch dataset with spatial train/val/test split |
| Modeling | Custom CNN trained and evaluated |
| Modeling | ResNet trained and evaluated (scratch + adapted variants if time allows) |
| Analysis | Full metric comparison, confusion matrices, Grad-CAM |
| Wrap-up | Report drafted, demo rehearsed, individual viva prep |

## 11. Risks

### 11.1 Confirmed Risks

| Risk | Mitigation | Empirical Status (updated 2026-10-10) |
|---|---|---|
| Data access delay (admin-gated permissions on PRADAN) | Register and request access on day 1; contact ISSDC admin immediately if gated | **CONFIRMED - NOT RESOLVED.** No tiles downloaded. Pipeline runs on synthetic data. data/raw/ contains MISSING_DATA.md. |
| Small dataset (single crater) | Spatial-block split, heavy augmentation, report this as a limitation | **CONFIRMED AS LIMITATION.** Synthetic scene is single-scene. Spatial block split (8x8 grid) verified with hard Python asserts against zero-positive splits. |
| Scope creep back toward full ISRO hackathon problem | Keep rover path/ice-volume/landing-site work explicitly optional/future-work only | **CONFIRMED CONTROLLED.** Rover path, ice volume, landing-site scoring remain deferred to PRD Section 13 (Future Scope). |

### 11.2 Risks Open Pending Real Data

| Risk | Mitigation | Empirical Status |
|---|---|---|
| Label circularity (CNN just re-learns the threshold rule) | Explicitly frame contribution as spatial/textural generalization beyond pixel-wise thresholding; address directly in report/viva | **REMAINS OPEN PENDING REAL DATA.** Pipeline robustness is proven, but circularity cannot be fully assessed until real spatially-textured SAR data is tested. |
| Class imbalance | Weighted loss / focal loss / oversampling; report F1 & ROC-AUC, not accuracy alone | **MITIGATED IN PIPELINE - OPEN PENDING REAL DATA.** The pipeline dynamically balances loss (using BCE pos_weight) and avoids majority-class collapse across 30 epochs on synthetic data. Final validation awaits real PRADAN tile class distribution. |
| ResNet/ImageNet domain mismatch | Test both scratch-trained and adapted-pretrained ResNet; justify choice in report | **TESTED - RESULTS PENDING REAL DATA.** Both scratch and pretrained variants run flawlessly. Final domain mismatch impact awaits real data. |

## 12. Success Criteria

- Both models trained and evaluated on a properly split, non-leaking dataset.
- Clear, defensible answer to "why does one model outperform the other" — not just a metric table.
- Report and demo directly satisfy every item in the faculty guideline checklist.
- Team can individually defend their component in viva.

## 13. Future Scope (explicitly deferred, not core deliverable)

- Semantic segmentation (U-Net) for pixel-level ice mapping.
- Geospatial ice-probability map integrated into QGIS.
- Simple rover-accessibility overlay around high-probability regions.
- Ice volume estimation via backscatter/dielectric modeling.
