'use client';

import React, { useState } from 'react';
import { getGradCamData, getPatchScoresData, GradCamSlot } from '../../lib/data';
import { ComparisonSlider } from '../../components/ComparisonSlider';
import { SpotlightCard } from '../../components/SpotlightCard';

export default function UntrainedVsTrainedPage() {
  const gradcamManifest = getGradCamData();
  const patchScoresData = getPatchScoresData();
  const slots: GradCamSlot[] = gradcamManifest.slots || [];

  const [activeSlotName, setActiveSlotName] = useState<string>(slots[0]?.slot || 'failure_fp1');
  const activeSlot = slots.find(s => s.slot === activeSlotName) || slots[0];

  // Score histogram data (from 338 test patches)
  const trainedScores = patchScoresData.patches.map((p: any) => p.scores['CustomCNN_seed2024'] ?? 0.5);

  // Compute 10-bin histogram for trained vs theoretical untrained (~Uniform or centered ~0.5)
  const bins = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0];
  const trainedCounts = new Array(10).fill(0);
  for (const s of trainedScores) {
    const idx = Math.min(9, Math.floor(s * 10));
    trainedCounts[idx]++;
  }

  // Untrained scores from the 28 manifest slots
  const untrainedScores = slots.map(s => s.untrained_model.score);
  const untrainedCounts = new Array(10).fill(0);
  for (const s of untrainedScores) {
    const idx = Math.min(9, Math.floor(s * 10));
    untrainedCounts[idx]++;
  }

  return (
    <div className="space-y-12 py-4">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-4">
        <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
          Ablation & Interpretability Verification (§3.6)
        </span>
        <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
          Untrained vs Trained Network Analysis
        </h1>
        <p className="text-sm text-neutral-400 mt-1 max-w-3xl font-sans">
          Sanity-check ablation evaluating whether network representations learn authentic SAR scattering dynamics rather than architectural priors. Compare Grad-CAM activations on identical patches using interactive drag-to-reveal.
        </p>
      </div>

      {/* 1. Large AUC Numerals */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
        <SpotlightCard
          label="Untrained Random Init AUC"
          value="0.500"
          subValue="Seed 2024 Init (Chance Baseline)"
          provenance="Theoretical Expected Baseline"
        />
        <SpotlightCard
          label="Trained CustomCNN AUC"
          value="0.7715"
          subValue="Seed 2024 Trained (Best Val Epoch 12)"
          provenance="results.json §CustomCNN"
          highlight
        />
        <SpotlightCard
          label="Discriminative Separation"
          value="+0.2715"
          subValue="Delta Over Random Baseline"
          provenance="Metric Gain"
        />
      </div>

      {/* 2. Drag-to-Reveal Comparison Slider */}
      <section className="p-6 rounded-lg border border-neutral-800 bg-neutral-900/40 space-y-6">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          <div>
            <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
              Interactive Grad-CAM Reveal Slider
            </h2>
            <p className="text-xs text-neutral-400 font-sans mt-0.5">
              Drag the central divider line horizontally to reveal the difference between untrained random weights (left) and trained weights (right).
            </p>
          </div>

          {/* Patch Selector */}
          <div className="flex items-center space-x-2 font-mono text-xs">
            <span className="text-neutral-500 uppercase">Patch:</span>
            <select
              value={activeSlotName}
              onChange={(e) => setActiveSlotName(e.target.value)}
              className="bg-neutral-950 border border-neutral-700 px-3 py-1.5 rounded text-neutral-200"
            >
              {slots.map(s => (
                <option key={s.slot} value={s.slot}>
                  Patch #{s.patch_idx} ({s.category})
                </option>
              ))}
            </select>
          </div>
        </div>

        {/* Drag to Reveal Component */}
        {activeSlot && (
          <div className="space-y-4">
            <ComparisonSlider
              beforeImage={`/${activeSlot.untrained_model.gradcam_image}`}
              afterImage={`/${activeSlot.trained_model.gradcam_image}`}
              beforeLabel={`UNTRAINED (SCORE: ${activeSlot.untrained_model.score.toFixed(4)})`}
              afterLabel={`TRAINED (SCORE: ${activeSlot.trained_model.score.toFixed(4)})`}
            />

            <div className="flex flex-wrap items-center justify-between text-xs font-mono text-neutral-400 p-3 rounded bg-neutral-950 border border-neutral-800">
              <div>
                <span>Patch #{activeSlot.patch_idx} · Strip {activeSlot.strip_id} · Coords [{activeSlot.coords[0]}, {activeSlot.coords[1]}]</span>
              </div>
              <div>
                <span>Ground Truth: <strong className="text-neutral-200">{activeSlot.true_label === 1 ? 'Candidate ice (1)' : 'Not an ice candidate (0)'}</strong></span>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* 3. Score Distribution Histogram */}
      <section className="p-6 rounded-lg border border-neutral-800 bg-neutral-900/40 space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
              Confidence Score Distributions
            </h2>
            <p className="text-xs text-neutral-400 font-sans mt-0.5">
              Trained network probabilities polarize toward 0.0 and 1.0, reflecting decisive confidence.
            </p>
          </div>
          <div className="flex items-center space-x-4 font-mono text-xs text-neutral-400">
            <span className="flex items-center space-x-1">
              <span className="w-3 h-3 rounded-sm bg-neutral-200"></span>
              <span>Trained CustomCNN</span>
            </span>
          </div>
        </div>

        {/* SVG Histogram */}
        <div className="w-full overflow-x-auto">
          <svg viewBox="0 0 600 180" className="w-full max-w-xl mx-auto h-48 font-mono">
            {/* Grid */}
            {[0, 50, 100, 150].map((count) => {
              const y = 140 - (count / 160) * 120;
              return (
                <g key={count}>
                  <line x1="40" y1={y} x2="560" y2={y} stroke="#27272a" strokeDasharray="3 3" />
                  <text x="32" y={y + 3} fill="#71717a" fontSize="8" textAnchor="end">{count}</text>
                </g>
              );
            })}

            {/* Bars */}
            {trainedCounts.map((count, i) => {
              const x = 50 + i * 50;
              const barH = (count / 160) * 120;
              const y = 140 - barH;
              return (
                <g key={i}>
                  <rect
                    x={x}
                    y={y}
                    width="36"
                    height={barH}
                    fill="#27272a"
                    stroke="#52525b"
                    rx="1"
                  />
                  <text x={x + 18} y={y - 4} fill="#fafafa" fontSize="8" textAnchor="middle">
                    {count}
                  </text>
                  <text x={x + 18} y="154" fill="#a1a1aa" fontSize="8" textAnchor="middle">
                    {(i * 0.1).toFixed(1)}
                  </text>
                </g>
              );
            })}
            <text x="300" y="172" fill="#71717a" fontSize="9" textAnchor="middle">
              Predicted Probability Bin
            </text>
          </svg>
        </div>
      </section>
    </div>
  );
}
