import React from 'react';
import Link from 'next/link';
import { SpotlightCard } from '../components/SpotlightCard';
import { Tooltip } from '../components/Tooltip';
import { getResultsData } from '../lib/data';

export default function OverviewPage() {
  const results = getResultsData();
  const summary = results.summary_by_config;
  const baselines = results.baselines.primary;

  const cnnAuc = summary.CustomCNN.primary.roc_auc.mean.toFixed(3);
  const cnnAucStd = summary.CustomCNN.primary.roc_auc.std.toFixed(3);

  const resnetAuc = summary.ResNet_scratch.primary.roc_auc.mean.toFixed(3);
  const resnetAucStd = summary.ResNet_scratch.primary.roc_auc.std.toFixed(3);

  const lrRow = baselines.find((b: any) => b.model.includes('Logistic Regression'));
  const lrAuc = lrRow ? lrRow.roc_auc.toFixed(3) : '0.731';

  return (
    <div className="space-y-8 py-2">
      {/* 1. Header & Three-Sentence Summary */}
      <section className="space-y-4">
        <div className="inline-flex items-center space-x-2 border border-neutral-800 bg-neutral-900/80 px-2.5 py-1 rounded text-[11px] font-mono text-neutral-400">
          <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
          <span>ISRO CHANDRAYAAN-2 DFSAR L1A BENCHMARK</span>
        </div>

        <h1 className="text-3xl sm:text-4xl md:text-5xl font-light tracking-tight text-neutral-100 font-sans">
          Deep Learning for Lunar Polar Subsurface Ice Detection
        </h1>

        {/* 3 Sentences: What, How, Result */}
        <p className="text-sm sm:text-base text-neutral-300 leading-relaxed max-w-4xl font-sans">
          We evaluate deep convolutional networks for detecting subsurface lunar water ice deposits using Chandrayaan-2 Dual-Frequency Synthetic Aperture Radar (DFSAR) L1A slant-range imagery.
          Models observe raw calibrated radar backscatter intensities across an along-track spatial test split free of geographic coordinate leakage.
          On pooled test metrics, deep learning architectures achieve ROC-AUC ~0.75 and fail to meaningfully outperform classical Logistic Regression (0.731).
        </p>

        {/* Result Banner in Plain Language */}
        <div className="p-4 rounded-md border border-neutral-800 bg-neutral-900/60 flex items-start space-x-3">
          <span className="w-2 h-2 rounded-full bg-amber-400 mt-1.5 flex-shrink-0"></span>
          <p className="text-sm text-neutral-200 font-sans">
            <strong className="text-neutral-100">Result:</strong> The deep models were not clearly better than a simple baseline. The test set has only 3 along-track strips.
          </p>
        </div>
      </section>

      {/* 2. Wide Rotated SAR Hero Image */}
      <section className="space-y-2">
        <div className="w-full rounded-lg overflow-hidden border border-neutral-800 bg-black shadow-2xl">
          <div className="relative w-full aspect-[2048/464]">
            <img
              src="/data/hero_sar_clean.png"
              alt="Chandrayaan-2 DFSAR L1A Grayscale SAR crop rotated for display with cyan candidate ice"
              className="w-full h-full object-contain"
            />
          </div>
        </div>
        <div className="flex flex-col sm:flex-row sm:items-center justify-between text-[11px] font-mono text-neutral-500 gap-1 px-1">
          <span>Chandrayaan-2 DFSAR L1A SLI (lines 100,000–140,000, rotated for display)</span>
          <span className="text-neutral-400">
            <span className="inline-block w-2 h-2 rounded-full bg-cyan-400 mr-1.5 align-middle"></span>
            Cyan pixels = Candidate Ice (CPR <Tooltip term="CPR" /> &gt; 1.0 and DOP <Tooltip term="DOP" /> &lt; 0.13)
          </span>
        </div>
      </section>

      {/* 3. Three Large Typographic Numerals */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
          <div className="flex items-center space-x-1 font-mono text-xs text-neutral-400 uppercase tracking-wider">
            <span>Headline Model Metrics</span>
            <Tooltip term="ROC-AUC" />
          </div>
          <span className="font-mono text-[10px] text-neutral-500">
            SOURCE: results.json
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          <SpotlightCard
            label="CustomCNN ROC-AUC"
            value={cnnAuc}
            subValue={`± ${cnnAucStd} (0.5 = coin flip, 1.0 = perfect)`}
            provenance="results.json §primary"
            highlight
          />
          <SpotlightCard
            label="ResNet_scratch ROC-AUC"
            value={resnetAuc}
            subValue={`± ${resnetAucStd} (0.5 = coin flip, 1.0 = perfect)`}
            provenance="results.json §primary"
          />
          <SpotlightCard
            label="Logistic Regression AUC"
            value={lrAuc}
            subValue="0.5 = coin flip (simple 3-feature baseline)"
            provenance="baseline_results.json"
          />
        </div>
      </section>

      {/* 4. Quick Action Links */}
      <section className="pt-2 flex flex-wrap gap-3 font-mono text-xs">
        <Link
          href="/pipeline/"
          className="px-4 py-2.5 bg-neutral-100 hover:bg-white text-neutral-950 font-medium rounded-sm transition"
        >
          See How the Pipeline Works →
        </Link>
        <Link
          href="/results/"
          className="px-4 py-2.5 bg-neutral-900 hover:bg-neutral-800 text-neutral-200 border border-neutral-800 rounded-sm transition"
        >
          Inspect Detailed Results Grid
        </Link>
        <Link
          href="/patches/"
          className="px-4 py-2.5 bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-800 rounded-sm transition"
        >
          Explore All 338 Patches
        </Link>
      </section>
    </div>
  );
}
