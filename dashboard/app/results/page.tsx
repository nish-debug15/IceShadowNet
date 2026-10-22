'use client';

import React, { useState } from 'react';
import { getResultsData } from '../../lib/data';

export default function ResultsPage() {
  const results = getResultsData();
  const [splitMode, setSplitMode] = useState<'primary' | 'clean'>('primary');
  const [viewMode, setViewMode] = useState<'pooled' | 'per_strip'>('pooled');

  const runs = results.runs || [];
  const baselines = (splitMode === 'primary' ? results.baselines.primary : results.baselines.clean) || [];
  const perStripTable = results.per_strip_table || [];

  const lrBaseline = baselines.find((b: any) => b.model.includes('Logistic Regression'));
  const lrAuc = lrBaseline ? lrBaseline.roc_auc : 0.7314;

  // Bar chart data (grouped by config mean, ORACLE strictly excluded)
  const summary = results.summary_by_config;
  const chartConfigs = Object.entries(summary).map(([cfg, data]: [string, any]) => {
    const stat = splitMode === 'primary' ? data.primary : data.clean;
    return {
      name: cfg,
      auc: stat.roc_auc.mean,
      std: stat.roc_auc.std,
      f1: stat.f1.mean,
    };
  });

  return (
    <div className="space-y-8 py-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-neutral-800 pb-4">
        <div>
          <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
            Statistical Benchmarking // Full & Clean Evaluated Sets
          </span>
          <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
            Benchmark Results & Metric Distributions
          </h1>
          <p className="text-xs font-mono text-neutral-400 mt-1">
            Spatial block-bootstrap 95% CI: <strong className="text-neutral-200">3 strips, 10 distinct resamples</strong>, 1000 bootstrap rounds.
          </p>
        </div>

        {/* Toggles */}
        <div className="flex flex-wrap items-center gap-2">
          {/* Split Toggle */}
          <div className="flex bg-neutral-900 border border-neutral-800 p-1 rounded">
            <button
              onClick={() => setSplitMode('primary')}
              className={`px-3 py-1 text-xs font-mono rounded transition ${
                splitMode === 'primary'
                  ? 'bg-neutral-800 text-neutral-100 border border-neutral-700'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              Full Test (N=338)
            </button>
            <button
              onClick={() => setSplitMode('clean')}
              className={`px-3 py-1 text-xs font-mono rounded transition ${
                splitMode === 'clean'
                  ? 'bg-neutral-800 text-neutral-100 border border-neutral-700'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              Clean Test (N=205)
            </button>
          </div>

          {/* View Toggle */}
          <div className="flex bg-neutral-900 border border-neutral-800 p-1 rounded">
            <button
              onClick={() => setViewMode('pooled')}
              className={`px-3 py-1 text-xs font-mono rounded transition ${
                viewMode === 'pooled'
                  ? 'bg-neutral-800 text-neutral-100 border border-neutral-700'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              Pooled Table
            </button>
            <button
              onClick={() => setViewMode('per_strip')}
              className={`px-3 py-1 text-xs font-mono rounded transition ${
                viewMode === 'per_strip'
                  ? 'bg-neutral-800 text-neutral-100 border border-neutral-700'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              Per-Strip View
            </button>
          </div>
        </div>
      </div>

      {/* Minimal Bar Chart (AUC with LR Line, ORACLE excluded) */}
      <section className="p-6 rounded-lg border border-neutral-800 bg-neutral-900/40 space-y-4">
        <div className="flex items-center justify-between">
          <span className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
            Test ROC-AUC by Model Family (Mean ± Seed-Std)
          </span>
          <div className="flex items-center space-x-2 font-mono text-xs text-neutral-400">
            <span className="w-4 h-0.5 bg-neutral-400 border-t border-dashed border-neutral-300"></span>
            <span>Logistic Regression Baseline: <strong className="text-neutral-200">{lrAuc.toFixed(4)}</strong></span>
          </div>
        </div>

        <MinimalBarChart
          configs={chartConfigs}
          lrBaseline={lrAuc}
          yDomain={[0.5, 0.9]}
        />
      </section>

      {/* Main Table as Clean Typographic Grid */}
      {viewMode === 'pooled' ? (
        <section className="rounded-lg border border-neutral-800 bg-neutral-900/30 overflow-hidden">
          <div className="p-4 border-b border-neutral-800/80 flex items-center justify-between">
            <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
              {splitMode === 'primary' ? 'Full Spatial Test Split Metrics (N=338)' : 'Ambiguity-Removed Clean Test Split (N=205)'}
            </h2>
            <span className="font-mono text-[11px] text-neutral-500">
              CI Note: 3 strips, 10 distinct resamples
            </span>
          </div>

          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-neutral-950 text-neutral-400 uppercase border-b border-neutral-800">
                <tr>
                  <th className="py-3 px-4">Architecture</th>
                  <th className="py-3 px-2">Seed</th>
                  <th className="py-3 px-2">Best Ep</th>
                  <th className="py-3 px-2">Val AUC</th>
                  <th className="py-3 px-4">Test ROC-AUC (95% CI)</th>
                  <th className="py-3 px-4">Test PR-AUC (95% CI)</th>
                  <th className="py-3 px-4">Test F1 (Default 95% CI)</th>
                  <th className="py-3 px-3">Tuned Th</th>
                  <th className="py-3 px-4">Support</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-neutral-800/60">
                {runs.map((r: any) => {
                  const m = splitMode === 'primary' ? r.test_primary : r.test_clean;
                  return (
                    <tr key={`${r.model}_${r.seed}`} className="hover:bg-neutral-800/20 transition">
                      <td className="py-2.5 px-4 font-bold text-neutral-200">{r.model}</td>
                      <td className="py-2.5 px-2 text-neutral-400">{r.seed}</td>
                      <td className="py-2.5 px-2 text-neutral-400">{r.best_epoch}</td>
                      <td className="py-2.5 px-2 text-neutral-300">{r.val_auc.toFixed(4)}</td>
                      <td className="py-2.5 px-4">
                        <span className="font-bold text-neutral-100">{m.roc_auc.toFixed(4)}</span>{' '}
                        <span className="text-[11px] text-neutral-500">{m.roc_auc_ci}</span>
                      </td>
                      <td className="py-2.5 px-4">
                        <span className="font-bold text-neutral-100">{m.pr_auc.toFixed(4)}</span>{' '}
                        <span className="text-[11px] text-neutral-500">{m.pr_auc_ci}</span>
                      </td>
                      <td className="py-2.5 px-4">
                        <span className="font-bold text-neutral-100">{m.f1.toFixed(4)}</span>{' '}
                        <span className="text-[11px] text-neutral-500">{m.f1_default_ci}</span>
                      </td>
                      <td className="py-2.5 px-3 text-neutral-300">{r.tuned_th.toFixed(2)}</td>
                      <td className="py-2.5 px-4 text-[11px] text-neutral-500 whitespace-nowrap">{m.support}</td>
                    </tr>
                  );
                })}

                {/* Baselines */}
                {baselines.map((b: any) => (
                  <tr key={b.model} className="bg-neutral-950/70 border-t border-neutral-700/60 text-neutral-300">
                    <td className="py-2.5 px-4 font-bold text-neutral-100">{b.model}</td>
                    <td className="py-2.5 px-2 text-neutral-600">—</td>
                    <td className="py-2.5 px-2 text-neutral-600">—</td>
                    <td className="py-2.5 px-2 text-neutral-600">—</td>
                    <td className="py-2.5 px-4">
                      <span className="font-bold text-neutral-100">{b.roc_auc.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-neutral-500">{b.roc_auc_ci}</span>
                    </td>
                    <td className="py-2.5 px-4">
                      <span className="font-bold text-neutral-100">{b.pr_auc.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-neutral-500">{b.pr_auc_ci}</span>
                    </td>
                    <td className="py-2.5 px-4">
                      <span className="font-bold text-neutral-100">{b.f1.toFixed(4)}</span>{' '}
                      <span className="text-[11px] text-neutral-500">{b.f1_ci}</span>
                    </td>
                    <td className="py-2.5 px-3 text-neutral-300">{b.threshold.toFixed(2)}</td>
                    <td className="py-2.5 px-4 text-[11px] text-neutral-500 whitespace-nowrap">{b.support}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ) : (
        /* Per-Strip Table */
        <section className="rounded-lg border border-neutral-800 bg-neutral-900/30 overflow-hidden">
          <div className="p-4 border-b border-neutral-800/80">
            <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
              Per-Strip Metrics (Strips 6, 10, 14 across lines 100k–140k)
            </h2>
          </div>
          <div className="overflow-x-auto">
            <table className="w-full text-left font-mono text-xs">
              <thead className="bg-neutral-950 text-neutral-400 uppercase border-b border-neutral-800">
                <tr>
                  <th className="py-2.5 px-4">Model</th>
                  <th className="py-2.5 px-2">Seed</th>
                  <th className="py-2.5 px-3">Strip</th>
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
                    <td className="py-2 px-4 text-neutral-200">{row.model}</td>
                    <td className="py-2 px-2 text-neutral-500">{row.seed ?? '—'}</td>
                    <td className="py-2 px-3 font-bold text-neutral-300">Strip {row.strip_id}</td>
                    <td className="py-2 px-3 text-neutral-400">{row.n_pos}/{row.n_neg}</td>
                    <td className="py-2 px-2 text-neutral-300">{row.tp}</td>
                    <td className="py-2 px-2 text-neutral-300">{row.fp}</td>
                    <td className="py-2 px-2 text-neutral-300">{row.fn}</td>
                    <td className="py-2 px-2 text-neutral-300">{row.tn}</td>
                    <td className="py-2 px-3 text-neutral-300">{row.recall.toFixed(4)}</td>
                    <td className="py-2 px-3 text-neutral-300">{row.precision.toFixed(4)}</td>
                    <td className="py-2 px-3 font-bold text-neutral-100">{row.f1.toFixed(4)}</td>
                    <td className="py-2 px-3 text-neutral-300">{row.roc_auc !== null ? row.roc_auc.toFixed(4) : 'N/A'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}

function MinimalBarChart({
  configs,
  lrBaseline,
  yDomain = [0.5, 0.9],
}: {
  configs: { name: string; auc: number; std: number; f1: number }[];
  lrBaseline: number;
  yDomain?: [number, number];
}) {
  const [minY, maxY] = yDomain;
  const width = 640;
  const height = 220;
  const padding = { top: 20, right: 30, bottom: 40, left: 45 };

  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const barWidth = 44;
  const gap = (innerW - configs.length * barWidth) / (configs.length + 1);

  const getY = (val: number) => padding.top + innerH - ((val - minY) / (maxY - minY)) * innerH;
  const lrY = getY(lrBaseline);

  return (
    <div className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full max-w-2xl mx-auto h-56 font-mono">
        {/* Y-axis grid */}
        {[0.5, 0.6, 0.7, 0.8, 0.9].map((v) => {
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
          fontSize="9"
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

          return (
            <g key={c.name}>
              {/* Bar */}
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={Math.max(0, h)}
                fill="#18181b"
                stroke="#52525b"
                strokeWidth="1"
                rx="2"
              />

              {/* Error Bar */}
              <line x1={x + barWidth / 2} y1={yErrTop} x2={x + barWidth / 2} y2={yErrBottom} stroke="#71717a" strokeWidth="1.2" />
              <line x1={x + barWidth / 2 - 4} y1={yErrTop} x2={x + barWidth / 2 + 4} y2={yErrTop} stroke="#71717a" strokeWidth="1.2" />
              <line x1={x + barWidth / 2 - 4} y1={yErrBottom} x2={x + barWidth / 2 + 4} y2={yErrBottom} stroke="#71717a" strokeWidth="1.2" />

              {/* Value Label */}
              <text x={x + barWidth / 2} y={y - 8} fill="#fafafa" fontSize="9" textAnchor="middle" fontWeight="bold">
                {c.auc.toFixed(3)}
              </text>

              {/* X label */}
              <text
                x={x + barWidth / 2}
                y={height - padding.bottom + 16}
                fill="#a1a1aa"
                fontSize="8"
                textAnchor="middle"
              >
                {c.name.replace('ResNet_', '')}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
