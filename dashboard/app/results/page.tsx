'use client';

import React, { useState } from 'react';
import { getResultsData, getTrainingCurvesData } from '../../lib/data';
import { Tooltip } from '../../components/Tooltip';
import { ChevronDown, ChevronRight } from 'lucide-react';

export default function ResultsPage() {
  const results = getResultsData();
  const curves = getTrainingCurvesData();

  const [splitMode, setSplitMode] = useState<'primary' | 'clean'>('primary');
  const [showFullStats, setShowFullStats] = useState(false);

  // Collapsible sections (closed by default)
  const [showAllRuns, setShowAllRuns] = useState(false);
  const [showTrainingCurves, setShowTrainingCurves] = useState(false);
  const [showPretrainedStudy, setShowPretrainedStudy] = useState(false);
  const [showPerStrip, setShowPerStrip] = useState(false);

  const runs = results.runs || [];
  const baselines = (splitMode === 'primary' ? results.baselines.primary : results.baselines.clean) || [];
  const perStripTable = results.per_strip_table || [];

  const lrBaseline = baselines.find((b: any) => b.model.includes('Logistic Regression'));
  const lrAuc = lrBaseline ? lrBaseline.roc_auc : 0.7314;
  const lrF1 = lrBaseline ? lrBaseline.f1 : 0.5145;

  const summary = results.summary_by_config;
  const cnnStat = splitMode === 'primary' ? summary.CustomCNN.primary : summary.CustomCNN.clean;
  const scratchStat = splitMode === 'primary' ? summary.ResNet_scratch.primary : summary.ResNet_scratch.clean;
  const v2aStat = splitMode === 'primary' ? summary.ResNet_pretrained_v2a.primary : summary.ResNet_pretrained_v2a.clean;

  // Chart configs (ORACLE strictly excluded)
  const chartConfigs = [
    {
      name: 'CustomCNN',
      auc: cnnStat.roc_auc.mean,
      std: cnnStat.roc_auc.std,
      f1: cnnStat.f1.mean,
      desc: '618K params',
    },
    {
      name: 'ResNet_scratch',
      auc: scratchStat.roc_auc.mean,
      std: scratchStat.roc_auc.std,
      f1: scratchStat.f1.mean,
      desc: '11.2M params',
    },
    {
      name: 'ResNet_v2a (fixed)',
      auc: v2aStat.roc_auc.mean,
      std: v2aStat.roc_auc.std,
      f1: v2aStat.f1.mean,
      desc: '10.5M params',
    },
    {
      name: 'ResNet_v2b (diff-lr)',
      auc: (splitMode === 'primary' ? summary.ResNet_pretrained_v2b.primary : summary.ResNet_pretrained_v2b.clean).roc_auc.mean,
      std: (splitMode === 'primary' ? summary.ResNet_pretrained_v2b.primary : summary.ResNet_pretrained_v2b.clean).roc_auc.std,
      f1: (splitMode === 'primary' ? summary.ResNet_pretrained_v2b.primary : summary.ResNet_pretrained_v2b.clean).f1.mean,
      desc: '10.5M params',
    },
    {
      name: 'ResNet_orig (buggy)',
      auc: (splitMode === 'primary' ? summary.ResNet_pretrained.primary : summary.ResNet_pretrained.clean).roc_auc.mean,
      std: (splitMode === 'primary' ? summary.ResNet_pretrained.primary : summary.ResNet_pretrained.clean).roc_auc.std,
      f1: (splitMode === 'primary' ? summary.ResNet_pretrained.primary : summary.ResNet_pretrained.clean).f1.mean,
      desc: '11.2M params',
    },
  ];

  return (
    <div className="space-y-8 py-4">
      {/* Header & Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-neutral-800 pb-4">
        <div>
          <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
            Statistical Benchmarking // Section 3 Analysis
          </span>
          <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
            Benchmark Results
          </h1>
          <p className="text-xs font-mono text-neutral-400 mt-1">
            Spatial block-bootstrap 95% CI: <strong className="text-neutral-200">3 test strips, 10 distinct resamples</strong>. Negative result reported.
          </p>
        </div>

        {/* Dataset Split Mode Toggle */}
        <div className="flex bg-neutral-900 border border-neutral-800 p-1 rounded">
          <button
            onClick={() => setSplitMode('primary')}
            className={`px-3 py-1 text-xs font-mono rounded transition ${
              splitMode === 'primary'
                ? 'bg-neutral-800 text-neutral-100 border border-neutral-700 font-medium'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            Full Test (N=338)
          </button>
          <button
            onClick={() => setSplitMode('clean')}
            className={`px-3 py-1 text-xs font-mono rounded transition ${
              splitMode === 'clean'
                ? 'bg-neutral-800 text-neutral-100 border border-neutral-700 font-medium'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            Clean Test (N=205)
          </button>
        </div>
      </div>

      {/* 4 Verdict Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* CustomCNN */}
        <div className="p-4 rounded border border-neutral-800 bg-neutral-900/40 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs uppercase tracking-wider text-cyan-400 font-semibold">
              CustomCNN
            </span>
            <span className="font-mono text-[10px] text-neutral-500">618K params</span>
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-light font-mono text-neutral-100">
              {cnnStat.roc_auc.mean.toFixed(3)}
            </span>
            <span className="text-xs font-mono text-neutral-400">
              ± {cnnStat.roc_auc.std.toFixed(3)}
            </span>
            <Tooltip term="ROC-AUC" />
          </div>
          <div className="text-[11px] font-mono text-neutral-400">
            F1: <strong className="text-neutral-200">{cnnStat.f1.mean.toFixed(3)}</strong> ± {cnnStat.f1.std.toFixed(3)}
          </div>
          <p className="text-xs text-neutral-300 border-t border-neutral-800/80 pt-2 leading-relaxed">
            Matches ResNet at ~18× fewer parameters; early-stopping signal noisy.
          </p>
        </div>

        {/* ResNet_scratch */}
        <div className="p-4 rounded border border-neutral-800 bg-neutral-900/40 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs uppercase tracking-wider text-neutral-300 font-semibold">
              ResNet_scratch
            </span>
            <span className="font-mono text-[10px] text-neutral-500">11.2M params</span>
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-light font-mono text-neutral-100">
              {scratchStat.roc_auc.mean.toFixed(3)}
            </span>
            <span className="text-xs font-mono text-neutral-400">
              ± {scratchStat.roc_auc.std.toFixed(3)}
            </span>
            <Tooltip term="ROC-AUC" />
          </div>
          <div className="text-[11px] font-mono text-neutral-400">
            F1: <strong className="text-neutral-200">{scratchStat.f1.mean.toFixed(3)}</strong> ± {scratchStat.f1.std.toFixed(3)}
          </div>
          <p className="text-xs text-neutral-300 border-t border-neutral-800/80 pt-2 leading-relaxed">
            Statistical tie with CustomCNN; no advantage from 11.2M parameters.
          </p>
        </div>

        {/* ResNet_pretrained v2a */}
        <div className="p-4 rounded border border-neutral-800 bg-neutral-900/40 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs uppercase tracking-wider text-neutral-400 font-semibold">
              ResNet_pretrained v2a
            </span>
            <span className="font-mono text-[10px] text-neutral-500">10.5M params</span>
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-light font-mono text-neutral-100">
              {v2aStat.roc_auc.mean.toFixed(3)}
            </span>
            <span className="text-xs font-mono text-neutral-400">
              ± {v2aStat.roc_auc.std.toFixed(3)}
            </span>
            <Tooltip term="ROC-AUC" />
          </div>
          <div className="text-[11px] font-mono text-neutral-400">
            F1: <strong className="text-neutral-200">{v2aStat.f1.mean.toFixed(3)}</strong> ± {v2aStat.f1.std.toFixed(3)}
          </div>
          <p className="text-xs text-neutral-300 border-t border-neutral-800/80 pt-2 leading-relaxed">
            ImageNet features failed to transfer; test AUC below scratch models.
          </p>
        </div>

        {/* Logistic Regression */}
        <div className="p-4 rounded border border-neutral-800 bg-neutral-900/40 space-y-2">
          <div className="flex items-center justify-between">
            <span className="font-mono text-xs uppercase tracking-wider text-neutral-300 font-semibold">
              Logistic Regression
            </span>
            <span className="font-mono text-[10px] text-neutral-500">Linear baseline</span>
          </div>
          <div className="flex items-baseline space-x-2">
            <span className="text-2xl font-light font-mono text-neutral-100">
              {lrAuc.toFixed(3)}
            </span>
            <span className="text-xs font-mono text-neutral-500">
              (deterministic)
            </span>
            <Tooltip term="ROC-AUC" />
          </div>
          <div className="text-[11px] font-mono text-neutral-400">
            F1: <strong className="text-neutral-200">{lrF1.toFixed(3)}</strong>
          </div>
          <p className="text-xs text-neutral-300 border-t border-neutral-800/80 pt-2 leading-relaxed">
            Simple linear baseline; deep models show no statistically distinct advantage.
          </p>
        </div>
      </div>

      {/* SVG Bar Chart with Horizontal Reference Lines (ORACLE strictly excluded) */}
      <section className="p-6 rounded border border-neutral-800 bg-neutral-900/30 space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-neutral-800/80 pb-3">
          <div>
            <span className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
              Test ROC-AUC Comparison
            </span>
            <span className="text-xs text-neutral-500 ml-2 font-mono">
              (Mean ± Seed-Std across seeds 42, 123, 2024)
            </span>
          </div>
          <div className="flex items-center space-x-4 font-mono text-xs text-neutral-400">
            <div className="flex items-center space-x-1.5">
              <span className="w-4 h-0.5 bg-neutral-400 border-t border-dashed border-neutral-300"></span>
              <span>LR baseline: <strong className="text-neutral-200">{lrAuc.toFixed(3)}</strong></span>
            </div>
            <div className="flex items-center space-x-1.5">
              <span className="w-4 h-0.5 bg-neutral-600 border-t border-dotted border-neutral-500"></span>
              <span>Coin flip: <strong className="text-neutral-400">0.500</strong></span>
            </div>
          </div>
        </div>

        <MinimalBarChart configs={chartConfigs} lrBaseline={lrAuc} yDomain={[0.45, 0.85]} />
      </section>

      {/* 3 Bullet Takeaways from section3_analysis.md */}
      <section className="p-5 rounded border border-neutral-800 bg-neutral-900/20 space-y-3">
        <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
          Key Analysis Takeaways (Section 3)
        </h2>
        <ul className="space-y-2 text-xs text-neutral-300 leading-relaxed font-sans">
          <li className="flex items-start space-x-2">
            <span className="text-cyan-400 font-mono font-bold">1.</span>
            <div>
              <strong className="text-neutral-100">No deep model is clearly better than Logistic Regression:</strong>{' '}
              The AUC gap between CustomCNN (0.756) and Logistic Regression (0.731) is under two seed standard deviations.
              Because the test set has only 3 along-track strips, spatial block-bootstrap confidence intervals overlap completely (3 strips, 10 distinct resamples).
              We report this as a negative result, not a ranking.
            </div>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-cyan-400 font-mono font-bold">2.</span>
            <div>
              <strong className="text-neutral-100">CustomCNN ties ResNet_scratch with 18× fewer parameters:</strong>{' '}
              Test ROC-AUC differs by only 0.003 against a seed standard deviation of ~0.013.
              CustomCNN has 618K trainable parameters versus 11.2M in ResNet-18, achieving the same discriminative skill with
              substantially lower model complexity.
            </div>
          </li>
          <li className="flex items-start space-x-2">
            <span className="text-cyan-400 font-mono font-bold">3.</span>
            <div>
              <strong className="text-neutral-100">ImageNet pretraining failed to transfer:</strong>{' '}
              Fixing freeze and channel-averaging bugs in v2a restored validation AUC to 0.666, but test ROC-AUC reached only
              0.688 ± 0.021, remaining well below scratch models. The v2a and v2b F1 scores hover near the all-positive baseline (0.460).
            </div>
          </li>
        </ul>
      </section>

      {/* Collapsible Section 1: All Runs Table (closed by default) */}
      <section className="rounded border border-neutral-800 bg-neutral-900/30 overflow-hidden">
        <button
          onClick={() => setShowAllRuns(!showAllRuns)}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-neutral-800/30 transition border-b border-neutral-800/80"
        >
          <div className="flex items-center space-x-2">
            {showAllRuns ? (
              <ChevronDown className="w-4 h-4 text-cyan-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-neutral-500" />
            )}
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-neutral-200">
              All Runs & Per-Seed Checkpoint Metrics
            </span>
          </div>
          <span className="text-[11px] font-mono text-neutral-500">
            {showAllRuns ? 'Click to hide table' : 'Click to inspect 15 deep runs + baselines'}
          </span>
        </button>

        {showAllRuns && (
          <div className="p-4 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-neutral-400">
                Spatial block-bootstrap 95% CIs: 3 strips, 10 distinct resamples.
              </span>
              <button
                onClick={() => setShowFullStats(!showFullStats)}
                className="px-2.5 py-1 text-xs font-mono bg-neutral-800 hover:bg-neutral-700 text-neutral-200 border border-neutral-700 rounded transition"
              >
                {showFullStats ? 'Compact View (5 Cols)' : 'Show Full Statistics (CIs, PR-AUC, Support)'}
              </button>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-neutral-950 text-neutral-400 uppercase border-b border-neutral-800 text-[11px]">
                  <tr>
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-2">Seed</th>
                    <th className="py-2.5 px-3">Test ROC-AUC</th>
                    <th className="py-2.5 px-3">Test F1</th>
                    <th className="py-2.5 px-2">Val AUC</th>
                    {showFullStats && (
                      <>
                        <th className="py-2.5 px-3 whitespace-nowrap">ROC-AUC 95% CI</th>
                        <th className="py-2.5 px-3 whitespace-nowrap">PR-AUC (95% CI)</th>
                        <th className="py-2.5 px-3 whitespace-nowrap">F1 95% CI</th>
                        <th className="py-2.5 px-2">Tuned Th</th>
                        <th className="py-2.5 px-3 whitespace-nowrap">Support</th>
                      </>
                    )}
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800/60">
                  {runs.map((r: any) => {
                    const m = splitMode === 'primary' ? r.test_primary : r.test_clean;
                    const isOriginalPretrained = r.model === 'ResNet_pretrained';
                    const displayName = isOriginalPretrained
                      ? 'ResNet_pretrained (original, buggy freeze)'
                      : r.model;

                    return (
                      <tr key={`${r.model}_${r.seed}`} className="hover:bg-neutral-800/20 transition">
                        <td className="py-2 px-3 text-neutral-200 font-medium">
                          {displayName}
                        </td>
                        <td className="py-2 px-2 text-neutral-400">{r.seed}</td>
                        <td className="py-2 px-3 font-semibold text-neutral-100">
                          {m.roc_auc.toFixed(4)}
                        </td>
                        <td className="py-2 px-3 text-neutral-200">
                          {m.f1.toFixed(4)}
                        </td>
                        <td className="py-2 px-2 text-neutral-400">
                          {r.val_auc.toFixed(4)}
                        </td>

                        {showFullStats && (
                          <>
                            <td className="py-2 px-3 whitespace-nowrap text-neutral-400">
                              {formatCI(m.roc_auc, m.roc_auc_ci)}
                            </td>
                            <td className="py-2 px-3 whitespace-nowrap text-neutral-400">
                              {m.pr_auc.toFixed(4)} {m.pr_auc_ci}
                            </td>
                            <td className="py-2 px-3 whitespace-nowrap text-neutral-400">
                              {formatCI(m.f1, m.f1_default_ci)}
                            </td>
                            <td className="py-2 px-2 text-neutral-300">
                              {r.tuned_th.toFixed(2)}
                            </td>
                            <td className="py-2 px-3 whitespace-nowrap text-[10px] text-neutral-500">
                              {m.support}
                            </td>
                          </>
                        )}
                      </tr>
                    );
                  })}

                  {/* Baselines */}
                  {baselines.map((b: any) => (
                    <tr key={b.model} className="bg-neutral-950/70 text-neutral-300 border-t border-neutral-700/60">
                      <td className="py-2 px-3 font-medium text-neutral-100">
                        {b.model}
                      </td>
                      <td className="py-2 px-2 text-neutral-600">—</td>
                      <td className="py-2 px-3 font-semibold text-neutral-100">
                        {b.roc_auc.toFixed(4)}
                      </td>
                      <td className="py-2 px-3 text-neutral-200">
                        {b.f1.toFixed(4)}
                      </td>
                      <td className="py-2 px-2 text-neutral-600">—</td>

                      {showFullStats && (
                        <>
                          <td className="py-2 px-3 whitespace-nowrap text-neutral-400">
                            {formatCI(b.roc_auc, b.roc_auc_ci)}
                          </td>
                          <td className="py-2 px-3 whitespace-nowrap text-neutral-400">
                            {b.pr_auc.toFixed(4)} {b.pr_auc_ci}
                          </td>
                          <td className="py-2 px-3 whitespace-nowrap text-neutral-400">
                            {formatCI(b.f1, b.f1_ci)}
                          </td>
                          <td className="py-2 px-2 text-neutral-300">
                            {b.threshold.toFixed(2)}
                          </td>
                          <td className="py-2 px-3 whitespace-nowrap text-[10px] text-neutral-500">
                            {b.support}
                          </td>
                        </>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>

      {/* Collapsible Section 2: Training Details (closed by default) */}
      <section className="rounded border border-neutral-800 bg-neutral-900/30 overflow-hidden">
        <button
          onClick={() => setShowTrainingCurves(!showTrainingCurves)}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-neutral-800/30 transition border-b border-neutral-800/80"
        >
          <div className="flex items-center space-x-2">
            {showTrainingCurves ? (
              <ChevronDown className="w-4 h-4 text-cyan-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-neutral-500" />
            )}
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-neutral-200">
              Training Details & Convergence Trajectories
            </span>
          </div>
          <span className="text-[11px] font-mono text-neutral-500">
            {showTrainingCurves ? 'Click to hide curves' : 'Click to inspect early stopping & loss curves'}
          </span>
        </button>

        {showTrainingCurves && (
          <div className="p-4 space-y-4">
            <p className="text-xs text-neutral-400 leading-relaxed font-sans">
              Models trained with AdamW and cosine annealing. Early stopping triggered after 5 epochs without validation ROC-AUC improvement.
              Due to small validation set size (77 positives), early-stopping signal was noisy and triggered early (epochs 1–13 across seeds).
            </p>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono text-xs">
              {curves.runs.map((r: any) => (
                <div key={`${r.model}_${r.seed}`} className="p-3 rounded border border-neutral-800 bg-neutral-950/60 space-y-1">
                  <div className="flex justify-between items-center text-neutral-300 font-semibold">
                    <span>{r.model}</span>
                    <span className="text-neutral-500 text-[10px]">Seed {r.seed}</span>
                  </div>
                  <div className="text-neutral-400 text-[11px]">
                    Best Epoch: <strong className="text-cyan-400">{r.best_epoch}</strong> (patience=5)
                  </div>
                  <div className="space-y-0.5 text-[10px] text-neutral-500 pt-1 border-t border-neutral-900">
                    {r.curve.map((pt: any) => (
                      <div key={pt.epoch} className="flex justify-between">
                        <span>Ep {pt.epoch}: loss {pt.val_loss.toFixed(3)}</span>
                        <span className="text-neutral-300">val AUC {pt.val_auc.toFixed(3)}</span>
                      </div>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </section>

      {/* Collapsible Section 3: Pretrained Experiments Breakdown (closed by default) */}
      <section className="rounded border border-neutral-800 bg-neutral-900/30 overflow-hidden">
        <button
          onClick={() => setShowPretrainedStudy(!showPretrainedStudy)}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-neutral-800/30 transition border-b border-neutral-800/80"
        >
          <div className="flex items-center space-x-2">
            {showPretrainedStudy ? (
              <ChevronDown className="w-4 h-4 text-cyan-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-neutral-500" />
            )}
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-neutral-200">
              Pretrained Transfer Experiments (Original vs v2a vs v2b)
            </span>
          </div>
          <span className="text-[11px] font-mono text-neutral-500">
            {showPretrainedStudy ? 'Click to hide breakdown' : 'Click to inspect ImageNet bug fix impact'}
          </span>
        </button>

        {showPretrainedStudy && (
          <div className="p-4 space-y-4 font-sans text-xs text-neutral-300 leading-relaxed">
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 font-mono">
              <div className="p-3 rounded border border-neutral-800 bg-neutral-950/60 space-y-2">
                <span className="text-xs font-bold text-neutral-400 block">Original Pretrained</span>
                <p className="text-[11px] text-neutral-400 font-sans">
                  Had two implementation bugs: substring freezing accidentally froze internal conv1/bn1 inside layer3/layer4,
                  and 1st-layer weights were collapsed to grayscale averages before freeze.
                </p>
                <div className="text-[11px] text-neutral-400 pt-2 border-t border-neutral-800">
                  Val AUC: 0.567 · Test AUC: 0.666 ± 0.013 · F1: 0.497
                </div>
              </div>

              <div className="p-3 rounded border border-cyan-900/50 bg-cyan-950/10 space-y-2">
                <span className="text-xs font-bold text-cyan-400 block">v2a (Bug Fix, Selected)</span>
                <p className="text-[11px] text-neutral-300 font-sans">
                  Exact freeze of bn1/layer1/layer2 with native 3-channel conv1 (10.5M trainable). Restored validation AUC by +0.10,
                  but test ROC-AUC improved by only +0.02 (within 3-seed noise).
                </p>
                <div className="text-[11px] text-neutral-300 pt-2 border-t border-neutral-800">
                  Val AUC: 0.666 · Test AUC: 0.688 ± 0.021 · F1: 0.487
                </div>
              </div>

              <div className="p-3 rounded border border-neutral-800 bg-neutral-950/60 space-y-2">
                <span className="text-xs font-bold text-neutral-400 block">v2b (Differential LR)</span>
                <p className="text-[11px] text-neutral-400 font-sans">
                  v2a architecture with lower backbone learning rate (1e-4) and higher head learning rate (1e-3).
                  Showed no statistical advantage over v2a.
                </p>
                <div className="text-[11px] text-neutral-400 pt-2 border-t border-neutral-800">
                  Val AUC: 0.622 · Test AUC: 0.668 ± 0.020 · F1: 0.466
                </div>
              </div>
            </div>

            <p className="text-neutral-400">
              <strong>Why did ImageNet pretraining fail?</strong> Untested explanations include: severe domain mismatch between optical natural
              photographs and multi-polarimetric SAR backscatter; spatial decimation (64 px patches downsampled to 2×2 feature maps at layer4);
              and per-patch z-score normalization removing absolute backscatter intensity levels.
            </p>
          </div>
        )}
      </section>

      {/* Collapsible Section 4: Per-Strip Results (closed by default) */}
      <section className="rounded border border-neutral-800 bg-neutral-900/30 overflow-hidden">
        <button
          onClick={() => setShowPerStrip(!showPerStrip)}
          className="w-full p-4 flex items-center justify-between text-left hover:bg-neutral-800/30 transition border-b border-neutral-800/80"
        >
          <div className="flex items-center space-x-2">
            {showPerStrip ? (
              <ChevronDown className="w-4 h-4 text-cyan-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-neutral-500" />
            )}
            <span className="font-mono text-xs font-semibold uppercase tracking-wider text-neutral-200">
              Per-Strip Spatial Breakdown (Strips 6, 10, 14)
            </span>
          </div>
          <span className="text-[11px] font-mono text-neutral-500">
            {showPerStrip ? 'Click to hide strip metrics' : 'Click to inspect spatial heterogeneity across strips'}
          </span>
        </button>

        {showPerStrip && (
          <div className="p-4 space-y-3">
            <p className="text-xs text-neutral-400 leading-relaxed font-sans">
              Pooled ROC-AUC overstates within-strip discriminative skill. Strip 14 contains only 5 positives; 11 of 17 runs
              produced TP = 0 on Strip 14, driving the spatial bootstrap CI lower tail to 0.000.
            </p>

            <div className="overflow-x-auto">
              <table className="w-full text-left font-mono text-xs">
                <thead className="bg-neutral-950 text-neutral-400 uppercase border-b border-neutral-800 text-[11px]">
                  <tr>
                    <th className="py-2.5 px-3">Model</th>
                    <th className="py-2.5 px-2">Seed</th>
                    <th className="py-2.5 px-2">Strip</th>
                    <th className="py-2.5 px-3">Pos / Neg</th>
                    <th className="py-2.5 px-2">TP</th>
                    <th className="py-2.5 px-2">FP</th>
                    <th className="py-2.5 px-2">FN</th>
                    <th className="py-2.5 px-2">TN</th>
                    <th className="py-2.5 px-3">Recall</th>
                    <th className="py-2.5 px-3">Precision</th>
                    <th className="py-2.5 px-3">F1</th>
                    <th className="py-2.5 px-3">ROC-AUC</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-neutral-800/60">
                  {perStripTable.map((row: any, i: number) => (
                    <tr key={i} className="hover:bg-neutral-800/20 transition">
                      <td className="py-2 px-3 text-neutral-200">{row.model}</td>
                      <td className="py-2 px-2 text-neutral-500">{row.seed ?? '—'}</td>
                      <td className="py-2 px-2 font-semibold text-neutral-300">Strip {row.strip_id}</td>
                      <td className="py-2 px-3 text-neutral-400">{row.n_pos}/{row.n_neg}</td>
                      <td className="py-2 px-2 text-neutral-300">{row.tp}</td>
                      <td className="py-2 px-2 text-neutral-300">{row.fp}</td>
                      <td className="py-2 px-2 text-neutral-300">{row.fn}</td>
                      <td className="py-2 px-2 text-neutral-300">{row.tn}</td>
                      <td className="py-2 px-3 text-neutral-300">{row.recall.toFixed(3)}</td>
                      <td className="py-2 px-3 text-neutral-300">{row.precision.toFixed(3)}</td>
                      <td className="py-2 px-3 font-semibold text-neutral-100">{row.f1.toFixed(3)}</td>
                      <td className="py-2 px-3 text-neutral-300">{row.roc_auc !== null ? row.roc_auc.toFixed(3) : 'N/A'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </section>
    </div>
  );
}

function formatCI(val: number, ciStr: string) {
  if (!ciStr) return val.toFixed(3);
  const clean = ciStr.replace(/[`\[\]]/g, '').trim();
  const parts = clean.split(',').map((s) => s.trim());
  if (parts.length === 2) {
    const lo = parseFloat(parts[0]);
    const hi = parseFloat(parts[1]);
    if (!isNaN(lo) && !isNaN(hi)) {
      return `${val.toFixed(3)} [${lo.toFixed(2)}–${hi.toFixed(2)}]`;
    }
  }
  return `${val.toFixed(3)} ${ciStr}`;
}

function MinimalBarChart({
  configs,
  lrBaseline,
  yDomain = [0.45, 0.85],
}: {
  configs: { name: string; auc: number; std: number; f1: number; desc: string }[];
  lrBaseline: number;
  yDomain?: [number, number];
}) {
  const [minY, maxY] = yDomain;
  const width = 640;
  const height = 230;
  const padding = { top: 25, right: 35, bottom: 45, left: 45 };

  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const barWidth = 46;
  const gap = (innerW - configs.length * barWidth) / (configs.length + 1);

  const getY = (val: number) => padding.top + innerH - ((val - minY) / (maxY - minY)) * innerH;
  const lrY = getY(lrBaseline);
  const coinY = getY(0.5);

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-2xl mx-auto h-56 font-mono">
        {/* Y-axis grid */}
        {[0.5, 0.6, 0.7, 0.8].map((v) => {
          const y = getY(v);
          return (
            <g key={v}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#27272a" strokeDasharray="3 3" />
              <text x={padding.left - 8} y={y + 3} fill="#71717a" fontSize="9" textAnchor="end">
                {v.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Coin-flip 0.5 Line */}
        <line
          x1={padding.left}
          y1={coinY}
          x2={width - padding.right}
          y2={coinY}
          stroke="#52525b"
          strokeWidth="1"
          strokeDasharray="2 3"
        />
        <text
          x={width - padding.right + 4}
          y={coinY + 3}
          fill="#71717a"
          fontSize="8"
          textAnchor="start"
        >
          0.50 (flip)
        </text>

        {/* Logistic Regression Baseline Line */}
        <line
          x1={padding.left}
          y1={lrY}
          x2={width - padding.right}
          y2={lrY}
          stroke="#a1a1aa"
          strokeWidth="1.5"
          strokeDasharray="4 2"
        />
        <text
          x={width - padding.right + 4}
          y={lrY + 3}
          fill="#a1a1aa"
          fontSize="8"
          textAnchor="start"
        >
          LR: {lrBaseline.toFixed(3)}
        </text>

        {/* Bars */}
        {configs.map((c, i) => {
          const x = padding.left + gap + i * (barWidth + gap);
          const y = getY(c.auc);
          const h = getY(minY) - y;
          const yErrTop = getY(c.auc + c.std);
          const yErrBottom = getY(c.auc - c.std);

          const isCustomCNN = c.name === 'CustomCNN';

          return (
            <g key={c.name}>
              {/* Bar */}
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={Math.max(0, h)}
                fill={isCustomCNN ? '#083344' : '#18181b'}
                stroke={isCustomCNN ? '#22d3ee' : '#52525b'}
                strokeWidth={isCustomCNN ? 1.5 : 1}
                rx="2"
              />

              {/* Error Bar */}
              <line x1={x + barWidth / 2} y1={yErrTop} x2={x + barWidth / 2} y2={yErrBottom} stroke="#a1a1aa" strokeWidth="1.2" />
              <line x1={x + barWidth / 2 - 4} y1={yErrTop} x2={x + barWidth / 2 + 4} y2={yErrTop} stroke="#a1a1aa" strokeWidth="1.2" />
              <line x1={x + barWidth / 2 - 4} y1={yErrBottom} x2={x + barWidth / 2 + 4} y2={yErrBottom} stroke="#a1a1aa" strokeWidth="1.2" />

              {/* Value Label */}
              <text x={x + barWidth / 2} y={y - 8} fill="#fafafa" fontSize="9" textAnchor="middle" fontWeight="bold">
                {c.auc.toFixed(3)}
              </text>

              {/* X label */}
              <text
                x={x + barWidth / 2}
                y={height - padding.bottom + 14}
                fill={isCustomCNN ? '#22d3ee' : '#a1a1aa'}
                fontSize="8"
                textAnchor="middle"
              >
                {c.name.split(' ')[0]}
              </text>
              <text
                x={x + barWidth / 2}
                y={height - padding.bottom + 24}
                fill="#71717a"
                fontSize="7"
                textAnchor="middle"
              >
                {c.desc}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
