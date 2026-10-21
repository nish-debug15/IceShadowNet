## 3. Analysis

### 3.1 Headline finding
On pooled test metrics, no deep model is clearly better than Logistic Regression.
CustomCNN: ROC-AUC 0.756 ± 0.013, F1 0.611 ± 0.013. ResNet_scratch: 0.753 ± 0.014, F1 0.581 ± 0.013.
Logistic Regression: ROC-AUC 0.731, F1 0.515. (Mean ± std over seeds 42, 123, 2024; full test N=338.)
The AUC gap to LR is under 2 seed-stds, and the test set has only 3 along-track strips,
so the block-bootstrap CIs (3 blocks, 10 distinct resamples) overlap completely.
We report this as a negative result, not a ranking.

### 3.2 CustomCNN vs ResNet_scratch
Test ROC-AUC differs by 0.003 against a seed std of ~0.013: a tie.
CustomCNN has 618K trainable parameters vs 11.2M, so it matches ResNet_scratch at ~18x fewer parameters.
Both stop early (best epoch 1-13 across seeds), and val has only 77 positives, so
the early-stopping signal is noisy.

### 3.3 ImageNet pretraining did not transfer
Original config: val AUC 0.567, test ROC-AUC 0.666 ± 0.013. It had two bugs:
substring freezing also froze conv1/bn1 inside layer3/layer4, and conv1 averaged HH/HV/VV into
identical filters, then froze them.
v2a (exact freeze of bn1/layer1/layer2, native 3-channel conv1, 10.5M trainable): val AUC 0.666,
test ROC-AUC 0.688 ± 0.021, F1 0.487 ± 0.034.
v2b (v2a + backbone LR 1e-4, head 1e-3): val AUC 0.622, test ROC-AUC 0.668 ± 0.020. No gain, so v2a is the fixed config (selected on val AUC).
The fixes raised val AUC by ~0.10 but test AUC by only ~0.02, which is within noise at 3 seeds.
All pretrained variants remain below ResNet_scratch and CustomCNN, and v2a/v2b F1 is at the
all-positive baseline (0.460).
Untested explanations: radar-vs-optical domain mismatch, 64 px patches giving a 2x2 map at layer4,
per-patch standardization, BatchNorm behaviour.

### 3.4 Pooled vs per-strip performance
Pooled AUC overstates within-strip skill. Per-strip ROC-AUC (mean over seeds):
| Model | Strip 6 | Strip 10 | Strip 14 (5 pos) |
|---|---|---|---|
| CustomCNN | 0.609 | 0.684 | 0.708 |
| ResNet_scratch | 0.582 | 0.678 | 0.716 |
| ResNet_pretrained_v2a | 0.511 | 0.566 | 0.587 |
| Logistic Regression | 0.558 | 0.541 | 0.448 |

CustomCNN and ResNet_scratch beat LR within strips 6 and 10 by a small margin; strip 6 is near chance for every model.
Per-strip F1 sits near the all-positive baseline (strip 6: 0.559 vs ~0.555; strip 10: 0.693 vs ~0.642).
Pooled AUC (0.75) exceeds every per-strip AUC on strips 6 and 10, consistent with scores also
separating strips by prevalence (hypothesis, not tested).
Strip 14 has 5 positives: 11 of 17 runs have TP=0 there, so its numbers are anecdotal.
This is also why the F1 CI lower bound is 0.0000 in several rows: a resample drawn entirely from strip 14
has 15 true positives and no detections, so F1 = 0 is the correct value (P ≈ 3.7% > 2.5%).
Dropping those resamples (conditional CI, sensitivity only) would censor the lower tail.

### 3.5 Baselines and the ambiguity band
All-positive F1: 0.460 on the full test, 0.660 on the clean test. Random ROC-AUC 0.5, PR-AUC 0.299 / 0.493.
The clean set removes the 133 patches with 1-4 positive pixels. All 133 are negatives (56% of the 237),
and none are positives, so "clean" removes only hard negatives. Higher F1 there is mechanical
(prevalence 29.9% → 49.3%), not evidence about speckle. Report clean-set results as a secondary view.
That 56% of negatives sit just under the >=5 pixel positive threshold is itself a labeling-sensitivity finding.

### 3.6 Failure modes (hypotheses, not tested)
False positives near crater rims may reflect dihedral scattering or slope foreshortening; false negatives may be
weak or diffuse deposits. No radar-geometry analysis supports either. Patches are illustrative only.

### 3.7 Limitations
3 test strips (one with 5 positives); 77 val positives; labels derived from CPR>1 and DOP<0.13, so circularity applies;
per-patch standardization discards absolute backscatter level; one scene, DFSAR-only; 3 seeds per config;
thresholds tuned on val swing from 0.05 to 0.92 across seeds, so AUC is the primary metric.
