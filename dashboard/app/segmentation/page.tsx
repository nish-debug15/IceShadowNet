'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';

interface SegData {
  unet?: {
    pixel_prevalence_test: number;
    baselines: {
      hv_hh_ratio: { pooled_val_th: any; pooled_best_th: any };
      pixel_logreg: { pooled_val_th: any; pooled_best_th: any };
    };
    unet_runs: Array<{
      seed: number;
      params: number;
      val_px_th: number;
      pooled_val_th: any;
      pooled_best_th: any;
      cis: any;
      per_strip: Record<string, any>;
      patch_agg: any;
    }>;
  };
  uncertainty?: {
    mc_dropout: any;
    deep_ensemble: any;
    selective_prediction: Array<{ coverage: number; n_px: number; pos_px: number; dice: number; accuracy: number }>;
    boundary_uncertainty_test: Record<string, any>;
    stitched_maps: { slant_range_tif: string; moon_stereo_tif: string; covered_pixels: number };
  };
  efficiency?: {
    optuna: { n_trials: number; best_params: any; best_val_px_pr_auc: number; test_metrics: any; test_patch_mean_roc_auc: number };
    pruning: Array<{
      target_sparsity: number;
      actual_sparsity: number;
      nonzero_params: number;
      px_pr_auc: number;
      dice: number;
      patch_mean_roc_auc: number;
      latency_ms_b64: number;
    }>;
  };
}

export default function SegmentationPage() {
  const [data, setData] = useState<SegData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch('/data/segmentation_uncertainty.json')
      .then((r) => {
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        return r.json();
      })
      .then(setData)
      .catch((e) => setError(String(e)));
  }, []);

  if (error) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12 font-mono text-sm text-red-400">
        Failed to load segmentation_uncertainty.json: {error}
      </div>
    );
  }

  if (!data) {
    return (
      <div className="max-w-7xl mx-auto px-4 py-12 font-mono text-xs text-neutral-400">
        Loading precomputed segmentation & uncertainty results...
      </div>
    );
  }

  const unetRuns = data.unet?.unet_runs || [];
  const meanOf = (fn: (r: any) => number) =>
    unetRuns.length ? unetRuns.reduce((a, r) => a + fn(r), 0) / unetRuns.length : 0;

  const hvhh = data.unet?.baselines?.hv_hh_ratio?.pooled_val_th;
  const pxlr = data.unet?.baselines?.pixel_logreg?.pooled_val_th;
  const mc = data.uncertainty?.mc_dropout;
  const ens = data.uncertainty?.deep_ensemble;
  const boundary = data.uncertainty?.boundary_uncertainty_test || {};
  const pruning = data.efficiency?.pruning || [];
  const optuna = data.efficiency?.optuna;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      <div className="border-b border-neutral-800 pb-6">
        <div className="font-mono text-[11px] text-cyan-400 uppercase tracking-widest mb-2">
          IceShadowNet 2.0 · Blocks 2 & 3
        </div>
        <h1 className="text-2xl sm:text-3xl font-semibold text-neutral-100 tracking-tight">
          Pixel-Level Segmentation & Uncertainty Quantification
        </h1>
        <p className="mt-2 text-sm text-neutral-400 max-w-3xl">
          Lightweight MC-Dropout <code className="text-cyan-300">IceUNet</code> trained on raw [HH, HV, VV]
          log-intensities to predict pixel-level polarimetric anomalies (CPR &gt; 1.0 &amp; DOP &lt; 0.13),
          resolving the 1–4 px patch-threshold ambiguity band. Precomputed static outputs (no live inference).
        </p>
      </div>

      {/* 1. Pixel-Level Segmentation Benchmark */}
      <section className="space-y-4">
        <h2 className="text-lg font-medium text-neutral-100 font-mono">
          1. Pixel-Level Segmentation Benchmark (Test Pixels: 8,265,728 | Prevalence: 0.0527%)
        </h2>
        <div className="overflow-x-auto border border-neutral-800 rounded bg-neutral-900/40">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-neutral-900 text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Pixel PR-AUC</th>
                <th className="px-4 py-3">Pixel ROC-AUC</th>
                <th className="px-4 py-3">Dice (Val Th)</th>
                <th className="px-4 py-3">IoU (Val Th)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/70 text-neutral-200">
              <tr>
                <td className="px-4 py-2.5">Baseline: HV/HH Ratio</td>
                <td className="px-4 py-2.5">{hvhh?.pr_auc?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{hvhh?.roc_auc?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{hvhh?.dice?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{hvhh?.iou?.toFixed(4)}</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5">Baseline: Pixel Logistic Regression</td>
                <td className="px-4 py-2.5">{pxlr?.pr_auc?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{pxlr?.roc_auc?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{pxlr?.dice?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{pxlr?.iou?.toFixed(4)}</td>
              </tr>
              <tr className="bg-cyan-950/20">
                <td className="px-4 py-2.5 font-semibold text-cyan-300">IceUNet (3-seed mean)</td>
                <td className="px-4 py-2.5 text-cyan-300">{meanOf((r) => r.pooled_val_th.pr_auc).toFixed(4)}</td>
                <td className="px-4 py-2.5 text-cyan-300">{meanOf((r) => r.pooled_val_th.roc_auc).toFixed(4)}</td>
                <td className="px-4 py-2.5 text-cyan-300">{meanOf((r) => r.pooled_val_th.dice).toFixed(4)}</td>
                <td className="px-4 py-2.5 text-cyan-300">{meanOf((r) => r.pooled_val_th.iou).toFixed(4)}</td>
              </tr>
              {mc && (
                <tr>
                  <td className="px-4 py-2.5">IceUNet MC Dropout (T=20)</td>
                  <td className="px-4 py-2.5">{mc.pr_auc?.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{mc.roc_auc?.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{mc.dice?.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{mc.iou?.toFixed(4)}</td>
                </tr>
              )}
              {ens && (
                <tr className="bg-emerald-950/20">
                  <td className="px-4 py-2.5 font-semibold text-emerald-300">IceUNet 3-Seed Ensemble</td>
                  <td className="px-4 py-2.5 text-emerald-300">{ens.pr_auc?.toFixed(4)}</td>
                  <td className="px-4 py-2.5 text-emerald-300">{ens.roc_auc?.toFixed(4)}</td>
                  <td className="px-4 py-2.5 text-emerald-300">{ens.dice?.toFixed(4)}</td>
                  <td className="px-4 py-2.5 text-emerald-300">{ens.iou?.toFixed(4)}</td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </section>

      {/* 2. Visual Overlays */}
      <section className="space-y-4">
        <h2 className="text-lg font-medium text-neutral-100 font-mono">
          2. Segmentation Overlays (Input [HH, HV, VV] | Ground Truth | U-Net Prob | Binary Pred)
        </h2>
        <div className="border border-neutral-800 rounded bg-neutral-900/40 p-4">
          <img
            src="/data/segmentation/unet_overlays.png"
            alt="U-Net Segmentation Overlays"
            className="w-full max-w-4xl mx-auto rounded border border-neutral-800"
          />
        </div>
      </section>

      {/* 3. Uncertainty & Boundary Hypothesis Test */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-3 border border-neutral-800 rounded bg-neutral-900/40 p-5">
          <h3 className="font-mono text-sm font-semibold text-neutral-100">
            3a. Boundary Uncertainty Test (Measured, Not Assumed)
          </h3>
          <p className="text-xs text-neutral-400">
            Predictive entropy peaks inside GT positive regions (~0.51) and boundary rings (~0.47) vs. far
            background (~0.14), while 3-seed ensemble variance is slightly higher on 1–2 px boundary rings
            (0.00953 vs 0.00947).
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="text-neutral-400 border-b border-neutral-800">
                <tr>
                  <th className="py-2 pr-3">Region</th>
                  <th className="py-2 pr-3">MC Entropy</th>
                  <th className="py-2 pr-3">Ens Variance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/60 text-neutral-200">
                {Object.entries(boundary).map(([reg, st]) => (
                  <tr key={reg}>
                    <td className="py-2 pr-3">{reg}</td>
                    <td className="py-2 pr-3">{st.mc_entropy_mean?.toFixed(5)}</td>
                    <td className="py-2 pr-3">{st.ens_var_mean?.toFixed(6)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <img
            src="/data/segmentation/unet_reliability.png"
            alt="MC Dropout Reliability Diagram"
            className="w-full rounded border border-neutral-800 mt-3"
          />
        </div>

        <div className="space-y-3 border border-neutral-800 rounded bg-neutral-900/40 p-5">
          <h3 className="font-mono text-sm font-semibold text-neutral-100">
            3b. Selective Prediction (Coverage vs. Dice / Accuracy)
          </h3>
          <p className="text-xs text-neutral-400">
            Filtering high-entropy pixels raises pixel accuracy from 99.43% to 99.999%, but abstains on
            rare 0.05% positive predictions first (causing Dice to drop to 0.0 at 90% coverage).
          </p>
          <img
            src="/data/segmentation/unet_selective_prediction.png"
            alt="Selective Prediction Curve"
            className="w-full rounded border border-neutral-800"
          />
        </div>
      </section>

      {/* 4. Optuna & Magnitude Pruning */}
      <section className="space-y-4">
        <h2 className="text-lg font-medium text-neutral-100 font-mono">
          4. Val-Only Optuna Tuning &amp; Magnitude Pruning
        </h2>
        {optuna && (
          <div className="border border-neutral-800 rounded bg-neutral-900/40 p-4 font-mono text-xs text-neutral-300">
            <span className="text-cyan-300 font-semibold">Optuna Best (12 Val Trials):</span>{' '}
            base_ch={optuna.best_params.base_ch}, lr={optuna.best_params.lr.toExponential(2)},{' '}
            dropout={optuna.best_params.dropout_p.toFixed(2)}, dice_w={optuna.best_params.dice_weight.toFixed(2)} →{' '}
            Test Pixel PR-AUC={optuna.test_metrics.pr_auc.toFixed(4)}, Pixel ROC-AUC={optuna.test_metrics.roc_auc.toFixed(4)},{' '}
            Patch Mean-Prob ROC-AUC={optuna.test_patch_mean_roc_auc.toFixed(4)}
          </div>
        )}
        <div className="overflow-x-auto border border-neutral-800 rounded bg-neutral-900/40">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-neutral-900 text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3">Target Sparsity</th>
                <th className="px-4 py-3">Non-Zero Params</th>
                <th className="px-4 py-3">Pixel PR-AUC</th>
                <th className="px-4 py-3">Dice (Val Th)</th>
                <th className="px-4 py-3">Patch Mean-Prob ROC-AUC</th>
                <th className="px-4 py-3">CUDA Latency (ms / B=64)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/70 text-neutral-200">
              {pruning.map((r) => (
                <tr key={r.target_sparsity}>
                  <td className="px-4 py-2.5">{(r.target_sparsity * 100).toFixed(0)}%</td>
                  <td className="px-4 py-2.5">{r.nonzero_params.toLocaleString()}</td>
                  <td className="px-4 py-2.5">{r.px_pr_auc.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{r.dice.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{r.patch_mean_roc_auc.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{r.latency_ms_b64.toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <div className="pt-4">
        <Link href="/results/" className="font-mono text-xs text-cyan-400 hover:underline">
          ← Back to Patch-Level Benchmark Results
        </Link>
      </div>
    </div>
  );
}
