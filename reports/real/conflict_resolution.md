# Block 7 Item 1: Side-by-Side Conflict Resolution Report

**Source of Truth Policy:** Every number in docs and dashboard is traced strictly to `reports/real/*.json`.

## Side-by-Side Comparison

| Experiment | First (Duplicate) Stdout | `reports/*.json` (On Disk) | `reports/*.md` | `README` / `research_report` / `viva_qa` / `docs/index.html` | `dashboard` JSON | Status / Root Cause |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase Oracle (4-ch)** | `0.1753` PR-AUC / `0.2286` Dice (Ep 15) | `0.0133` PR-AUC / `0.0348` Dice (Ep 7) | `0.0133` / `0.0348` | `0.0133` / `0.0348` | `0.0133` / `0.0348` | **CONFLICT IN STDOUT:** Duplicate harness execution (`task-577`); without `cudnn.deterministic`, one trajectory early-stopped at Ep 7 (`Val 0.0099`) while the other reached Ep 15 (`Val 0.1462`). **Resolved in Item 2** via deterministic 3-seed re-run (both 3-ch and 4-ch). |
| **Selective Redone (AURC)** | Ens `0.1455` / MC `0.1516` / `\|p-0.5\|` `0.1544` / Rand `0.1933` | Ens `0.1259` / MC `0.1096` / `\|p-0.5\|` `0.1084` / Rand `0.1917` | `0.1259` / `0.1096` / `0.1084` / `0.1917` | `0.1259` / `0.1096` / `0.1084` | `0.1259` / `0.1096` / `0.1084` / `0.1917` | **CONFLICT IN STDOUT:** Duplicate notification (`task-569`) from an earlier/parallel pass before final threshold/score normalization; disk JSON & all docs match `0.1259 / 0.1096 / 0.1084 / 0.1917` (`val_patch_th = 0.06`). Re-verified deterministic run in Item 3. |
| **Context HistGBDT (7×7+15×15)** | `0.0083` PR-AUC / `0.0176` Dice | `0.0077` (`0.007670`) / `0.0229` (`0.02285`) | `0.0077` / `0.0229` | `0.0077` / `0.0229` | `0.0077` / `0.0229` | **CONFLICT IN STDOUT:** Duplicate notification (`task-561`); disk JSON & all docs match `0.0077 / 0.0229`. |
| **Pruning (50% Struct B=64 / B=1)** | `11.179→9.035 ms` (`1.24×`) / `B=1: 2.345→2.352 ms` (`1.0×`) | `10.860→5.359 ms` (`2.03×`) / `B=1: 2.244→2.058 ms` (`1.09×`) | `10.860 / 5.359 ms` | `10.860 / 5.359 ms` | `10.860 / 5.359 ms` | **CONFLICT IN STDOUT + DOCS:** Duplicate notification (`task-586`) + stale `1.32×` (`9.81 ms` from `structured_pruning.md`) vs `2.03×` (`5.359 ms` in `pruning_unified.json`). Docs must use `pruning_unified.json` only and state speedup is at `B=64` only, negligible (`~1.0–1.09×`) at `B=1`. |
| **Georef (10-fold CV)** | `367.98 / 307.38 / 2377.70 m` | `367.98 / 307.38 / 2377.70 m` | `367.98 / 307.38 / 2377.70 m` | `367.98 / 307.38 / 2377.70 m` | `367.98 / 307.38 / 2377.70 m` | **CONSISTENT** across stdout, JSON, MD, docs, and dashboard. |
