#!/usr/bin/env bash
set -euo pipefail

# IceShadowNet 2.0 — End-to-End Deterministic Reproduction (Fixed Seeds: 42 123 2024)
PYTHON="${PYTHON:-python}"

echo "[1/11] Ingestion & Deterministic Frozen Split [90624, 319812]..."
"$PYTHON" -u -m scripts.ingest_and_split --real

echo "[2/11] TPS Georeferencing Verification..."
"$PYTHON" -u -m scripts.analysis.verify_tps_georef

echo "[3/11] Patch-Level DL Benchmark (CustomCNN, ResNet_scratch, ResNet_pretrained, 3 seeds)..."
"$PYTHON" -u -m scripts.train_all --real --epochs 30 --seeds 42 123 2024

echo "[4/11] Classical Baselines & 5-Fold Blocked-Strip CV..."
"$PYTHON" -u -m scripts.baseline_real
"$PYTHON" -u -m scripts.analysis.run_5fold_strip_cv
"$PYTHON" -u -m scripts.analysis.generate_block1_report

echo "[5/11] In-Domain SimCLR SSL & Label-Efficiency Fairness..."
"$PYTHON" -u -m scripts.pretrain_ssl
"$PYTHON" -u -m scripts.train_label_efficiency
"$PYTHON" -u -m scripts.analysis.run_label_efficiency_fairness

echo "[6/11] Circularity Ablation..."
"$PYTHON" -u -m scripts.analysis.run_circularity_ablation

echo "[7/11] U-Net Pixel Segmentation & Pixel Baselines (3 seeds)..."
"$PYTHON" -u -m scripts.train_unet

echo "[8/11] Uncertainty (MC Dropout T=20, Ensemble, |p-0.5|, Brier) & Stitched TIFFs..."
"$PYTHON" -u -m scripts.analysis.run_uncertainty_and_stitch
"$PYTHON" -u -m scripts.analysis.run_uncertainty_hardening

echo "[9/11] Hardened Pixel Metrics (Conditional Dice & >=5px CC)..."
"$PYTHON" -u -m scripts.analysis.run_pixel_metrics_hardening

echo "[10/11] Val-Only Optuna Sweep & Pruning (Unstructured + Structured)..."
"$PYTHON" -u -m scripts.analysis.run_optuna_and_pruning
"$PYTHON" -u -m scripts.analysis.run_structured_pruning

echo "[11/12] SRI Overlay Check, Phase-Oracle Diagnostic, & Dashboard Export..."
"$PYTHON" -u -m scripts.analysis.verify_sri_overlay
"$PYTHON" -u -m scripts.analysis.run_phase_oracle
"$PYTHON" -u -m scripts.analysis.export_dashboard_data || true

echo "[12/12] Render Docs from JSON & Run Strict Number-Trace Audit..."
"$PYTHON" -u -m scripts.analysis.render_numbers
"$PYTHON" -u -m scripts.analysis.number_trace_audit

echo "Reproduction complete."
