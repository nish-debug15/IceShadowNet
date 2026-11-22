# Scene Registry & Split Plan Audit

**Plan Type**: BLOCKED STRIP CROSS-VALIDATION (SINGLE-SCENE FALLBACK)

**Note**: SINGLE SCENE DETECTED: Falling back to blocked along-track strip CV with 64px boundary guard buffers to eliminate pixel-extent leakage. This limitation is stated loudly in all reports.

## Registered SAR Scenes

| Scene ID | Payload | Dimensions | Footprint (Lat / Lon) | Analyzed Swath | Anomalous Px | Patches (Pos / Neg) | Pos Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ch2_sar_ncxl_20260308t062017233` | DFSAR | 319812 rows x 116 cols | -86.27° to -81.95° / 13.95° to 49.27° | Lines 100,000..140,000 (40,000 x 116) | 3,033 (0.0654%) | 585 / 1913 | 23.42% |

## Split Verification Summary

- **Train patches**: 1686 (Pos: 362)
- **Val patches**: 342 (Pos: 77)
- **Test patches**: 338 (Pos: 101)
- **Buffer pixels**: 64 px
- **Gate condition (Test pos >= 30)**: PASSED (101 >= 30)
