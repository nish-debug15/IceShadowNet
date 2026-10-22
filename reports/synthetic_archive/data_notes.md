# reports/data_notes.md
# Data Acquisition Status — IceShadowNet

**Last updated:** 2026-09-18

---

## DFSAR Data (Chandrayaan-2 / PRADAN)

| Item | Status |
|------|--------|
| PRADAN account registration | Pending / in progress |
| DFSAR tile request submitted | Pending |
| Tiles downloaded to `data/raw/` | **NOT YET** |

### Gap description

No real DFSAR PDS4 (`.IMG` / `.XML`) tiles are available in `data/raw/` at
the time of Block 1 execution (2026-09-18).  The full pipeline is therefore
running on a **physically-motivated synthetic DFSAR scene** generated in
`src/preprocessing/data_ops._synthetic_dfsar()`.

The synthetic scene is NOT random noise:
- Two polarisation channels (HH / HV) with spatially correlated speckle
  (Rayleigh, L=4 looks, Gaussian-filtered backscatter).
- A circular ice-candidate region (~15% of scene area) with elevated HV
  backscatter so that CPR > 1 and DOP < 0.13 in that region, matching the
  scientific ice criteria used on real data.
- The resulting class balance (~15% ice patches) is consistent with the
  minority-class expectation from the PRD.

**Action required:** Register at <https://pradan.issdc.gov.in/ch2/>, request
DFSAR Level-1 data for the Shackleton crater (or another doubly-shadowed
south-polar crater), and place the downloaded `.IMG` file under `data/raw/`.
The pipeline will automatically switch from synthetic to real data without
any code change — `load_dfsar()` detects the file and logs `[REAL]`.

---

## OHRC Data (Chandrayaan-2 / PRADAN)

Same access-gate situation as DFSAR.  OHRC tiles are also absent; the pipeline
falls back to a synthetic optical tile.  Real OHRC will be used for the
multimodal fusion variant in Block 3.

---

## Effect on Results

All metrics produced by Block 2 (training runs) and Block 3 (Grad-CAM, report)
are derived from this synthetic dataset.  They demonstrate that the pipeline
executes correctly but **cannot be cited as real ice-detection performance**.
The README Results table and report skeleton clearly label all numbers as
`[SYNTHETIC DATA]`.

When real DFSAR tiles are obtained:
1. Place `.IMG` file(s) in `data/raw/`.
2. Re-run `python -m scripts.ingest_and_split` to regenerate patches.
3. Re-run `python -m scripts.train_all` to retrain all models.
4. Update README Results table with real metric values.
