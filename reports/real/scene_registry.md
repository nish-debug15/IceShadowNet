# Scene Registry & Split Plan Audit

**Plan Type**: BLOCKED STRIP CROSS-VALIDATION (SINGLE-SCENE FALLBACK)

**Note**: SINGLE SCENE DETECTED: Falling back to blocked along-track strip CV with 64px boundary guard buffers to eliminate pixel-extent leakage. This limitation is stated loudly in all reports.

## Registered SAR Scenes

| Scene ID | Payload | Dimensions | Footprint (Lat / Lon) | Analyzed Swath | Anomalous Px | Patches (Pos / Neg) | Pos Rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `ch2_sar_ncxl_20260308t062017233` | DFSAR | 319812 rows x 116 cols | -86.27° to -81.95° / 13.95° to 49.27° | Lines 100,000..140,000 (40,000 x 116) | 3,033 (0.0654%) | 585 / 1913 | 23.42% |

## Split Verification Summary

- **Train patches**: 9828 (Pos: 1614)
- **Val patches**: 2040 (Pos: 335)
- **Test patches**: 2018 (Pos: 332)
- **Buffer pixels**: 256 px
- **Gate condition (Test pos >= 30)**: PASSED (332 >= 30)
