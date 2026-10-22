import React from 'react';
import Link from 'next/link';
import { SpotlightCard } from '../components/SpotlightCard';
import { getResultsData } from '../lib/data';

export default function OverviewPage() {
  const results = getResultsData();
  const summary = results.summary_by_config;
  const baselines = results.baselines.primary;
  const trivial = results.trivial_baselines.full_test;

  const cnnAuc = summary.CustomCNN.primary.roc_auc.mean.toFixed(3);
  const cnnAucStd = summary.CustomCNN.primary.roc_auc.std.toFixed(3);

  const resnetAuc = summary.ResNet_scratch.primary.roc_auc.mean.toFixed(3);
  const resnetAucStd = summary.ResNet_scratch.primary.roc_auc.std.toFixed(3);

  const lrRow = baselines.find((b: any) => b.model.includes('Logistic Regression'));
  const lrAuc = lrRow ? lrRow.roc_auc.toFixed(3) : '0.731';

  const allPosF1 = trivial.all_positive_f1.toFixed(4);

  return (
    <div className="space-y-16 py-4">
      {/* 1. Full-Bleed Hero SAR Crop with Candidate-Ice Cyan Overlay */}
      <section className="relative overflow-hidden rounded-xl border border-neutral-800 bg-neutral-900/60 p-6 md:p-12">
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-center">
          {/* Left Text */}
          <div className="lg:col-span-7 space-y-6 z-10">
            <div className="inline-flex items-center space-x-2 border border-neutral-700/80 bg-neutral-950/80 px-2.5 py-1 rounded text-[11px] font-mono text-neutral-400">
              <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
              <span>ISRO CHANDRAYAAN-2 DFSAR L1A BENCHMARK</span>
            </div>

            <div className="space-y-3">
              <h1 className="text-3xl sm:text-4xl md:text-5xl font-light tracking-tight text-neutral-100 font-sans leading-tight">
                Deep Learning for Lunar Polar Subsurface Ice Detection
              </h1>
              <p className="text-sm font-mono text-neutral-400 uppercase tracking-wider">
                Full-Scene Slant-Range Complex Imagery (Lines 100,000–140,000)
              </p>
            </div>

            {/* Negative-Result Headline Statement in Plain Type */}
            <div className="border-l-2 border-neutral-600 pl-4 py-1 space-y-1 text-sm text-neutral-300 leading-relaxed font-sans">
              <p className="font-medium text-neutral-100">
                Negative-Result Finding (§3.1): On pooled test metrics, no deep model is clearly superior to classical Logistic Regression.
              </p>
              <p className="text-xs text-neutral-400">
                CustomCNN ROC-AUC 0.756 ± 0.013 vs Logistic Regression 0.731 (gap &lt; 2 seed-std). In along-track strips 6 and 10, deep model F1 remains tethered to the trivial all-positive baseline.
              </p>
            </div>

            <div className="pt-2 flex flex-wrap gap-3 font-mono text-xs">
              <Link
                href="/results/"
                className="px-4 py-2 bg-neutral-100 hover:bg-white text-neutral-950 font-medium rounded-sm transition"
              >
                Inspect Results Grid
              </Link>
              <Link
                href="/patches/"
                className="px-4 py-2 bg-neutral-900 hover:bg-neutral-800 text-neutral-300 border border-neutral-700 rounded-sm transition"
              >
                Explore 338 Patches
              </Link>
            </div>
          </div>

          {/* Right Hero Image: Real Grayscale SAR with Candidate Ice Mask in Pure Cyan */}
          <div className="lg:col-span-5 flex flex-col items-center">
            <div className="relative w-full max-w-[280px] aspect-[116/512] rounded-lg overflow-hidden border border-neutral-700 shadow-2xl bg-black">
              {/* Clean high-contrast SAR speckle with candidate-ice pixels in cyan */}
              <img
                src="/data/hero_sar_clean.png"
                alt="Chandrayaan-2 DFSAR L1A Grayscale SAR with Cyan Candidate Ice"
                className="w-full h-full object-cover filter contrast-125"
              />
              <div className="absolute bottom-2 left-2 right-2 bg-neutral-950/90 border border-neutral-800/80 p-2 rounded text-[10px] font-mono text-neutral-300 backdrop-blur-sm">
                <span className="text-cyan-400 font-bold">CYAN:</span> Candidate Ice (CPR &gt; 1.0 and DOP &lt; 0.13)
              </div>
            </div>
            <p className="text-[10px] font-mono text-neutral-500 mt-2 text-center">
              512 × 116 px radar tile crop · Chandrayaan-2 DFSAR L1A SLI
            </p>
          </div>
        </div>
      </section>

      {/* 2. Row of 4 Large Typographic Numerals */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
          <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider">
            Headline Benchmark Metrics (Full Spatial Test Set, N=338)
          </span>
          <span className="font-mono text-[10px] text-neutral-500">
            SOURCE: results.json
          </span>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <SpotlightCard
            label="CustomCNN ROC-AUC"
            value={cnnAuc}
            subValue={`± ${cnnAucStd} (Seeds 42, 123, 2024)`}
            provenance="results.json §primary"
            highlight
          />
          <SpotlightCard
            label="ResNet_scratch ROC-AUC"
            value={resnetAuc}
            subValue={`± ${resnetAucStd} (Seeds 42, 123, 2024)`}
            provenance="results.json §primary"
          />
          <SpotlightCard
            label="Logistic Regression AUC"
            value={lrAuc}
            subValue="Balanced (hh, hv, vv features)"
            provenance="baseline_results.json"
          />
          <SpotlightCard
            label="All-Positive F1 Baseline"
            value={allPosF1}
            subValue="Full Test Prevalence: 29.88%"
            provenance="trivial_baselines"
          />
        </div>
      </section>

      {/* 3. Scientific Framing and Provenance Note */}
      <section className="p-6 rounded-lg border border-neutral-800 bg-neutral-900/30 space-y-3 font-mono text-xs">
        <div className="text-neutral-400 uppercase font-semibold">
          Strict Anti-Circularity & Provenance Guarantee
        </div>
        <p className="text-neutral-300 leading-relaxed font-sans text-xs">
          Physical labels were derived from polarimetric criteria (CPR &gt; 1.0 and DOP &lt; 0.13) using a 7 × 7 multilook boxcar window.
          To prevent trivial circular reasoning, <strong>CPR and DOP are strictly excluded from all model inputs</strong>.
          Models observe only raw radar log-intensities (log10|S_HH|², log10|S_HV|², log10|S_VV|²).
        </p>
      </section>
    </div>
  );
}
