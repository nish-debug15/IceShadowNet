# PRD — IceShadowNet

**Deep Learning-Based Detection of Potential Subsurface Ice in Lunar South Polar Regions Using Chandrayaan-2 DFSAR and OHRC Data**

---

## 1. Problem Statement

ISRO's Chandrayaan-2 mission provides radar (DFSAR) and optical (OHRC) data over lunar south-polar "doubly shadowed craters" believed to host subsurface water-ice — a high-priority target for future ISRU missions. Current detection relies on hand-crafted radar polarimetric thresholds (CPR > 1, DOP < 0.13) applied pixel-by-pixel. This project asks: **can a deep learning model learn to distinguish potential ice-bearing regions from non-ice terrain using spatial/textural patterns that a pixel-wise threshold cannot capture — and if so, which architecture (custom CNN vs. ResNet) does it better?**

This is scoped as a course Deep Learning project, not a reproduction of the full ISRO hackathon problem (landing site selection, rover path planning, and ice-volume estimation are explicitly out of scope for the core deliverable).

## 2. Objectives

1. Construct a labeled patch dataset from DFSAR (± OHRC) imagery using established radar criteria.
2. Implement and train a custom CNN baseline.
3. Implement and train a ResNet-based model on the same task.
4. Compare both models across accuracy, loss, training time, parameter count, precision/recall/F1, ROC-AUC, and confusion matrix.
5. Apply Grad-CAM to verify the models attend to physically plausible regions.
6. Document all findings in a report that satisfies the faculty rubric.

## 3. Scope

**In scope**
- Binary classification: potential-ice vs. non-ice patch.
- Custom CNN vs. ResNet comparison (core experiment).
- Optional multimodal variant (DFSAR + OHRC).
- Optional stretch: geospatial ice-probability map from patch predictions.

**Out of scope**
- Rover traverse path planning.
- Ice volume estimation via dielectric/backscatter modeling.
- Landing site safety scoring.
- Pixel-level segmentation (U-Net) — only as a future-work mention unless data/time allow.

## 4. Non-Goals / Explicit Disclaimers

- The project does **not** claim to detect physical ice. It classifies signatures *consistent with* the scientific criteria for potential ice presence.
- Labels are derived from existing radar thresholds, not independently verified ground truth — this limitation must be stated in the report and be answerable in viva.

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

| Risk | Mitigation |
|---|---|
| Data access delay (admin-gated permissions on PRADAN) | Register and request access on day 1; contact ISSDC admin immediately if gated |
| Label circularity (CNN just re-learns the threshold rule) | Explicitly frame contribution as spatial/textural generalization beyond pixel-wise thresholding; address directly in report/viva |
| Small dataset (single crater) | Spatial-block split, heavy augmentation, report this as a limitation |
| Class imbalance | Weighted loss / focal loss / oversampling; report F1 & ROC-AUC, not accuracy alone |
| ResNet/ImageNet domain mismatch | Test both scratch-trained and adapted-pretrained ResNet; justify choice in report |
| Scope creep back toward full ISRO hackathon problem | Keep rover path/ice-volume/landing-site work explicitly optional/future-work only |

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
