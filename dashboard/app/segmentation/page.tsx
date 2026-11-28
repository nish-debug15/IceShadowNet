'use client';

import React, { useEffect, useState } from 'react';
import Link from 'next/link';
import { withBasePath } from '../../lib/basePath';

interface SegData {
  unet?: {
    baselines: {
      hv_hh_ratio: { pooled_val_th: any };
      pixel_logreg: { pooled_val_th: any };
    };
    unet_runs: Array<{ pooled_val_th: any }>;
  };
  uncertainty?: {
    mc_dropout: any;
    deep_ensemble: any;
    boundary_uncertainty_test: Record<string, any>;
  };
  efficiency?: {
    optuna: { best_params: any; test_metrics: any; test_patch_mean_roc_auc: number };
  };
  context_baselines?: {
    pixel_prevalence: number;
    context_baselines: Array<{
      model: string;
      windows: string;
      pr_auc: number;
      lift_over_prevalence: number;
      dice_val_th: number;
      iou_val_th: number;
    }>;
  };
  phase_oracle?: {
    intensity_3ch?: { summary: any; runs?: any[] };
    phase_oracle_4ch?: { summary: any; runs?: any[] };
  };
  selective_redone?: {
    results: Record<string, { aurc: number; error_detection_auroc: number; curve: any[] }>;
  };
  pruning_unified?: Array<{
    config: string;
    nonzero_params: number;
    patch_roc_auc: number;
    pixel_pr_auc: number;
    lat_b1_ms: number;
    lat_b64_ms: number;
  }>;
  georef_10fold?: {
    ten_fold_mean_m: number;
    ten_fold_median_m: number;
    ten_fold_max_m: number;
    overlap_definition: string;
  };
}

export default function SegmentationPage() {
  const [data, setData] = useState<SegData | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch(withBasePath('/data/segmentation_uncertainty.json'))
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
        Loading precomputed segmentation &amp; statistical evaluation data...
      </div>
    );
  }

  const unetRuns = data.unet?.unet_runs || [];
  const meanOf = (fn: (r: any) => number) =>
    unetRuns.length ? unetRuns.reduce((a, r) => a + fn(r), 0) / unetRuns.length : 0;

  const hvhh = data.unet?.baselines?.hv_hh_ratio?.pooled_val_th;
  const pxlr = data.unet?.baselines?.pixel_logreg?.pooled_val_th;
  const ens = data.uncertainty?.deep_ensemble;
  const ctx = data.context_baselines?.context_baselines || [];
  const oracle = data.phase_oracle;
  const oracle3 = oracle?.intensity_3ch?.summary;
  const oracle4 = oracle?.phase_oracle_4ch?.summary;
  const pxPrev = data.context_baselines?.pixel_prevalence || 0.0005266323789023786;
  const sel = data.selective_redone?.results || {};
  const pruning = data.pruning_unified || [];
  const optuna = data.efficiency?.optuna;
  const georef = data.georef_10fold;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      <div className="border-b border-neutral-800 pb-6">
        <div className="font-mono text-[11px] text-cyan-400 uppercase tracking-widest mb-2">
          IceShadowNet · Pixel Localization &amp; Hardened Diagnostics
        </div>
        <h1 className="text-2xl sm:text-3xl font-semibold text-neutral-100 tracking-tight">
          Pixel-Level Segmentation, Context Baselines &amp; Uncertainty
        </h1>
        <p className="mt-2 text-sm text-neutral-400 max-w-3xl">
          Lightweight MC-Dropout <code className="text-cyan-300">IceUNet</code> trained on intensity-only [HH, HV, VV]
          log-intensities. Improves localization over context-free baselines (0.0093–0.0110 Pixel PR-AUC, 17.6×–20.9×
          over 0.0527% prevalence), with context-matched 7×7+15×15 HistGBDT reaching 0.0077 (14.6×). Single DFSAR
          scene, anisotropic 0.505 m × 9.593 m pixels, proxy labels (not confirmed water ice).
        </p>
      </div>

      {/* 1. Pixel-Level Localization vs Context-Matched Baselines & Phase Oracle */}
      <section className="space-y-4">
        <h2 className="text-lg font-medium text-neutral-100 font-mono">
          1. Pixel Localization vs. Context-Matched Baselines (8,265,728 Test Pixels)
        </h2>
        <div className="overflow-x-auto border border-neutral-800 rounded bg-neutral-900/40">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-neutral-900 text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3">Model</th>
                <th className="px-4 py-3">Context</th>
                <th className="px-4 py-3">Pixel PR-AUC</th>
                <th className="px-4 py-3">Lift vs. 0.0527%</th>
                <th className="px-4 py-3">Dice (Val Th)</th>
                <th className="px-4 py-3">IoU (Val Th)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/70 text-neutral-200">
              <tr>
                <td className="px-4 py-2.5">HV/HH Ratio (context-free)</td>
                <td className="px-4 py-2.5">1×1</td>
                <td className="px-4 py-2.5">{hvhh?.pr_auc?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{(hvhh?.pr_auc / pxPrev).toFixed(1)}×</td>
                <td className="px-4 py-2.5">{hvhh?.dice?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{hvhh?.iou?.toFixed(4)}</td>
              </tr>
              <tr>
                <td className="px-4 py-2.5">Pixel LogReg (context-free)</td>
                <td className="px-4 py-2.5">1×1</td>
                <td className="px-4 py-2.5">{pxlr?.pr_auc?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{(pxlr?.pr_auc / pxPrev).toFixed(1)}×</td>
                <td className="px-4 py-2.5">{pxlr?.dice?.toFixed(4)}</td>
                <td className="px-4 py-2.5">{pxlr?.iou?.toFixed(4)}</td>
              </tr>
              {ctx.map((r) => (
                <tr key={r.model}>
                  <td className="px-4 py-2.5">{r.model}</td>
                  <td className="px-4 py-2.5">{r.windows}</td>
                  <td className="px-4 py-2.5">{r.pr_auc.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{r.lift_over_prevalence.toFixed(1)}×</td>
                  <td className="px-4 py-2.5">{r.dice_val_th.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{r.iou_val_th.toFixed(4)}</td>
                </tr>
              ))}
              <tr className="bg-cyan-950/20">
                <td className="px-4 py-2.5 font-semibold text-cyan-300">IceUNet (3-seed mean)</td>
                <td className="px-4 py-2.5">64×64</td>
                <td className="px-4 py-2.5 text-cyan-300">{meanOf((r) => r.pooled_val_th.pr_auc).toFixed(4)}</td>
                <td className="px-4 py-2.5 text-cyan-300">{(meanOf((r) => r.pooled_val_th.pr_auc) / pxPrev).toFixed(1)}×</td>
                <td className="px-4 py-2.5 text-cyan-300">{meanOf((r) => r.pooled_val_th.dice).toFixed(4)}</td>
                <td className="px-4 py-2.5 text-cyan-300">{meanOf((r) => r.pooled_val_th.iou).toFixed(4)}</td>
              </tr>
              {ens && (
                <tr className="bg-emerald-950/20">
                  <td className="px-4 py-2.5 font-semibold text-emerald-300">IceUNet 3-Seed Ensemble</td>
                  <td className="px-4 py-2.5">64×64</td>
                  <td className="px-4 py-2.5 text-emerald-300">{ens.pr_auc?.toFixed(4)}</td>
                  <td className="px-4 py-2.5 text-emerald-300">{(ens.pr_auc / pxPrev).toFixed(1)}×</td>
                  <td className="px-4 py-2.5 text-emerald-300">{ens.dice?.toFixed(4)}</td>
                  <td className="px-4 py-2.5 text-emerald-300">{ens.iou?.toFixed(4)}</td>
                </tr>
              )}
              {oracle3 && (
                <tr className="bg-amber-950/10 text-amber-200/80">
                  <td className="px-4 py-2.5 italic">Phase Oracle 3-ch Control (3 seeds)</td>
                  <td className="px-4 py-2.5">64×64</td>
                  <td className="px-4 py-2.5">{oracle3.pr_auc.mean.toFixed(4)} ± {oracle3.pr_auc.std.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{oracle3.lift.mean.toFixed(1)}× ± {oracle3.lift.std.toFixed(1)}×</td>
                  <td className="px-4 py-2.5">{oracle3.dice.mean.toFixed(4)} ± {oracle3.dice.std.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{oracle3.iou.mean.toFixed(4)} ± {oracle3.iou.std.toFixed(4)}</td>
                </tr>
              )}
              {oracle4 && (
                <tr className="bg-amber-950/20 text-amber-200">
                  <td className="px-4 py-2.5 italic">Phase Oracle IceUNet (4-ch γ_HHVV, CIRCULAR diagnostic, 3 seeds)</td>
                  <td className="px-4 py-2.5">64×64</td>
                  <td className="px-4 py-2.5">{oracle4.pr_auc.mean.toFixed(4)} ± {oracle4.pr_auc.std.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{oracle4.lift.mean.toFixed(1)}× ± {oracle4.lift.std.toFixed(1)}×</td>
                  <td className="px-4 py-2.5">{oracle4.dice.mean.toFixed(4)} ± {oracle4.dice.std.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{oracle4.iou.mean.toFixed(4)} ± {oracle4.iou.std.toFixed(4)}</td>
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
            src={withBasePath('/data/segmentation/unet_overlays.png')}
            alt="U-Net Segmentation Overlays"
            className="w-full max-w-4xl mx-auto rounded border border-neutral-800"
          />
        </div>
      </section>

      {/* 3. Selective Prediction (Unified Ensemble Predictor) */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="space-y-3 border border-neutral-800 rounded bg-neutral-900/40 p-5">
          <h3 className="font-mono text-sm font-semibold text-neutral-100">
            3a. Selective Prediction (Unified 3-Seed Ensemble Mean Predictor)
          </h3>
          <p className="text-xs text-neutral-400">
            Neither Ensemble Variance nor MC Entropy improves Patch ROC-AUC over full coverage (0.8456), and
            |p - 0.5| matches or beats both on AURC (0.1084) and error-detection AUROC (0.8310). Uncertainty is
            not usable for selective rejection on this task.
          </p>
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="text-neutral-400 border-b border-neutral-800">
                <tr>
                  <th className="py-2 pr-3">Rejection Score</th>
                  <th className="py-2 pr-3">AURC</th>
                  <th className="py-2 pr-3">ErrDet AUC</th>
                  <th className="py-2 pr-3">50% Patch AUC</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/60 text-neutral-200">
                {Object.entries(sel).map(([name, d]) => (
                  <tr key={name}>
                    <td className="py-2 pr-3">{name}</td>
                    <td className="py-2 pr-3">{d.aurc.toFixed(4)}</td>
                    <td className="py-2 pr-3">{d.error_detection_auroc.toFixed(4)}</td>
                    <td className="py-2 pr-3">{d.curve[d.curve.length - 1].patch_roc_auc.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {georef && (
            <p className="text-xs text-neutral-400 pt-2 border-t border-neutral-800">
              <span className="text-neutral-200 font-semibold">10-Fold GCP TPS CV (676 GCPs):</span> mean{' '}
              {georef.ten_fold_mean_m.toFixed(1)} m / median {georef.ten_fold_median_m.toFixed(1)} m / max{' '}
              {georef.ten_fold_max_m.toFixed(1)} m. {georef.overlap_definition}
            </p>
          )}
        </div>

        <div className="space-y-3 border border-neutral-800 rounded bg-neutral-900/40 p-5">
          <h3 className="font-mono text-sm font-semibold text-neutral-100">
            3b. Patch ROC-AUC &amp; Retained Pixel PR-AUC vs. Coverage
          </h3>
          <img
            src={withBasePath('/data/segmentation/selective_redone.png')}
            alt="Selective Prediction Redone"
            className="w-full rounded border border-neutral-800"
          />
        </div>
      </section>

      {/* 4. Unified Pruning & Latency */}
      <section className="space-y-4">
        <h2 className="text-lg font-medium text-neutral-100 font-mono">
          4. Unified Pruning &amp; CUDA Latency (Median of 200 Runs, 50 Warmup, CUDA Sync)
        </h2>
        {optuna && (
          <div className="border border-neutral-800 rounded bg-neutral-900/40 p-4 font-mono text-xs text-neutral-300">
            <span className="text-neutral-200 font-semibold">Optuna (12 Val Trials, single run, excluded from ranking):</span>{' '}
            base_ch={optuna.best_params.base_ch}, lr={optuna.best_params.lr.toExponential(2)},{' '}
            dropout={optuna.best_params.dropout_p.toFixed(2)}, dice_w={optuna.best_params.dice_weight.toFixed(2)} →{' '}
            Test Pixel PR-AUC={optuna.test_metrics.pr_auc.toFixed(4)}, Patch Mean-Prob ROC-AUC={optuna.test_patch_mean_roc_auc.toFixed(4)}
          </div>
        )}
        <div className="overflow-x-auto border border-neutral-800 rounded bg-neutral-900/40">
          <table className="w-full text-left font-mono text-xs">
            <thead className="bg-neutral-900 text-neutral-400 border-b border-neutral-800">
              <tr>
                <th className="px-4 py-3">Configuration</th>
                <th className="px-4 py-3">Non-Zero Params</th>
                <th className="px-4 py-3">Patch ROC-AUC</th>
                <th className="px-4 py-3">Pixel PR-AUC</th>
                <th className="px-4 py-3">Latency B=1 (ms)</th>
                <th className="px-4 py-3">Latency B=64 (ms)</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-neutral-800/70 text-neutral-200">
              {pruning.map((r) => (
                <tr key={r.config}>
                  <td className="px-4 py-2.5">{r.config}</td>
                  <td className="px-4 py-2.5">{r.nonzero_params.toLocaleString()}</td>
                  <td className="px-4 py-2.5">{r.patch_roc_auc.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{r.pixel_pr_auc.toFixed(4)}</td>
                  <td className="px-4 py-2.5">{r.lat_b1_ms.toFixed(3)}</td>
                  <td className="px-4 py-2.5">{r.lat_b64_ms.toFixed(3)}</td>
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
