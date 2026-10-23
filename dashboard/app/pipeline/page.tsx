'use client';

import React, { useState } from 'react';
import { ComparisonSlider } from '../../components/ComparisonSlider';
import { Tooltip } from '../../components/Tooltip';
import { getPipelineStagesData, getGradCamData } from '../../lib/data';

const STAGE_DESCRIPTIONS = [
  {
    stage: 1,
    title: 'Raw HH Complex Backscatter Amplitude',
    summary: 'Horizontally transmitted and received radar echo from the lunar surface. High backscatter appears bright, while radar shadows appear dark.',
    math: 'I_{HH} = \\log_{10}(|S_{HH}|^2 + \\epsilon)',
    mathPlain: 'I_HH = log10(|S_HH|² + 1e-5)',
  },
  {
    stage: 2,
    title: 'Raw HV Cross-Polar Symmetrization',
    summary: 'Cross-polarized backscatter is averaged under radar reciprocity to suppress instrumental noise prior to polarimetric analysis.',
    math: 'S_{HV}^{sym} = 0.5 \\cdot (S_{HV} + S_{VH})',
    mathPlain: 'S_HV_sym = 0.5 × (S_HV + S_VH)',
  },
  {
    stage: 3,
    title: 'Raw VV Complex Backscatter Amplitude',
    summary: 'Vertically transmitted and received co-polarized echo, completing the quad-polarization scattering matrix.',
    math: 'I_{VV} = \\log_{10}(|S_{VV}|^2 + \\epsilon)',
    mathPlain: 'I_VV = log10(|S_VV|² + 1e-5)',
  },
  {
    stage: 4,
    title: 'Multilook Radar Averaging (W=7 Boxcar)',
    summary: 'A 7×7 spatial moving window averages raw speckle noise. This enables calculation of polarization degree without mathematical distortion.',
    math: '\\langle \\sigma \\rangle = \\frac{1}{W^2} \\sum_{i,j \\in W} \\sigma(i, j)',
    mathPlain: '⟨σ⟩ = (1 / 49) ∑ σ(i, j) over 7×7 neighborhood',
  },
  {
    stage: 5,
    title: 'Circular Polarization Ratio (CPR) Map',
    summary: 'Measures the ratio of depolarized same-sense to opposite-sense circular echoes. High CPR indicates multiple scattering from rough terrain or subsurface ice.',
    math: 'CPR = \\frac{\\langle \\sigma_{RR} \\rangle}{\\langle \\sigma_{RL} \\rangle}',
    mathPlain: 'CPR = ⟨σ_RR⟩ / ⟨σ_RL⟩',
  },
  {
    stage: 6,
    title: 'Degree of Polarization (DOP) Map',
    summary: 'Quantifies the fraction of scattered radar energy that remains polarized. Low DOP indicates multiple volume scattering characteristic of ice deposits.',
    math: 'DOP = \\frac{\\sqrt{S_1^2 + S_2^2 + S_3^2}}{S_0}',
    mathPlain: 'DOP = √(S1² + S2² + S3²) / S0',
  },
  {
    stage: 7,
    title: 'Binary Candidate Ice Mask (Cyan on Black)',
    summary: 'Pixels satisfying both physical criteria (CPR > 1.0 and DOP < 0.13) are identified as candidate ice. CPR and DOP are strictly excluded from deep learning inputs.',
    math: 'M(r, c) = (CPR > 1.0) \\land (DOP < 0.13)',
    mathPlain: 'M(r, c) = (CPR > 1.0) and (DOP < 0.13)',
  },
  {
    stage: 8,
    title: 'Spatial Patch Extraction Grid (64×64, Stride 32)',
    summary: '64×64 pixel patches are extracted across the along-track strip. A patch is labeled positive if it contains at least 5 candidate ice pixels (0.12% area).',
    math: 'y_i = 1 \\iff \\sum_{(r,c) \\in P_i} M(r,c) \\ge 5',
    mathPlain: 'y_i = 1 if sum of ice pixels in patch >= 5, else 0',
  },
];

export default function HowItWorksPage() {
  const pipelineData = getPipelineStagesData();
  const gradcamData = getGradCamData();
  const stages = pipelineData.stages || [];
  const slots = gradcamData.slots || [];

  const [activeStageIdx, setActiveStageIdx] = useState(0);
  const [openMathIdx, setOpenMathIdx] = useState<number | null>(null);
  const [activeSlotIdx, setActiveSlotIdx] = useState(0);

  const activeStage = stages[activeStageIdx] || stages[0];
  const activeDesc = STAGE_DESCRIPTIONS[activeStageIdx] || STAGE_DESCRIPTIONS[0];
  const activeSlot = slots[activeSlotIdx] || slots[0];

  return (
    <div className="space-y-16 py-2">
      {/* 1. Header */}
      <section className="space-y-2 border-b border-neutral-800 pb-4">
        <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block">
          Radar Processing &amp; Interpretability
        </span>
        <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
          How It Works: From Raw SLI to Classified Patches
        </h1>
        <p className="text-sm text-neutral-400 max-w-3xl font-sans">
          Chandrayaan-2 DFSAR L1A quad-pol imagery undergoes eight precise physical steps to extract leak-free labeled patches, followed by gradient-based visual ablations.
        </p>
      </section>

      {/* 2. Eight Pipeline Stages (Scroll Story + Sticky Image) */}
      <section className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
            Part 1: The 8 Radar Processing Stages
          </h2>
          <span className="text-[11px] font-mono text-neutral-500">
            512×116 crop, rotated for display
          </span>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left Column: 8 Stage Cards */}
          <div className="lg:col-span-7 space-y-3">
            {STAGE_DESCRIPTIONS.map((st, idx) => {
              const isActive = activeStageIdx === idx;
              const isMathOpen = openMathIdx === idx;
              return (
                <div
                  key={st.stage}
                  onClick={() => setActiveStageIdx(idx)}
                  className={`p-4 rounded-lg border cursor-pointer transition-all ${
                    isActive
                      ? 'border-neutral-700 bg-neutral-900/90 shadow-md ring-1 ring-cyan-500/20'
                      : 'border-neutral-800/80 bg-neutral-950/60 opacity-65 hover:opacity-95'
                  }`}
                >
                  <div className="flex items-center justify-between mb-1.5">
                    <span className="font-mono text-[10px] text-neutral-400 uppercase tracking-widest">
                      STAGE {st.stage} OF 8
                    </span>
                    {isActive && (
                      <span className="w-1.5 h-1.5 rounded-full bg-cyan-400"></span>
                    )}
                  </div>

                  <h3 className="text-sm font-medium text-neutral-100 mb-1">
                    {st.title}
                  </h3>

                  <p className="text-xs text-neutral-300 leading-relaxed font-sans mb-2">
                    {st.summary}
                  </p>

                  {/* Collapsible Math Button */}
                  <div className="pt-1">
                    <button
                      type="button"
                      onClick={(e) => {
                        e.stopPropagation();
                        setOpenMathIdx(isMathOpen ? null : idx);
                      }}
                      className="text-[11px] font-mono text-cyan-400 hover:text-cyan-300 flex items-center space-x-1"
                    >
                      <span>{isMathOpen ? '▾ Hide the formula' : '▸ Show the formula'}</span>
                    </button>
                    {isMathOpen && (
                      <div className="mt-2 p-2 rounded bg-neutral-950 border border-neutral-800 font-mono text-xs text-neutral-300">
                        <code>{st.mathPlain}</code>
                      </div>
                    )}
                  </div>
                </div>
              );
            })}
          </div>

          {/* Right Column: Sticky Clean Image Display */}
          <div className="lg:col-span-5 sticky top-20 space-y-3">
            <div className="rounded-lg border border-neutral-800 bg-neutral-900/80 p-4 shadow-2xl flex flex-col items-center">
              <div className="relative w-full aspect-[2048/464] rounded overflow-hidden border border-neutral-800 bg-black">
                {activeStage && (
                  <img
                    src={`/${activeStage.image}`}
                    alt={activeDesc.title}
                    className="w-full h-full object-contain"
                  />
                )}
              </div>

              {/* HTML Color-Scale Legend */}
              <div className="mt-4 text-center space-y-1.5 w-full border-t border-neutral-800 pt-3 font-mono text-xs">
                <div className="font-bold text-neutral-200">
                  Stage {activeStage?.stage}: {activeStage?.name}
                </div>
                <div className="flex items-center justify-between text-[11px] text-neutral-400 px-2">
                  <span>Min: {activeStage?.scale_min?.toFixed(2) ?? '0.00'}</span>
                  <span className="text-neutral-500 uppercase">{activeStage?.units ?? 'Grayscale'}</span>
                  <span>Max: {activeStage?.scale_max?.toFixed(2) ?? '1.00'}</span>
                </div>
                <p className="text-[10px] text-neutral-500">
                  Rotated 90° for display (azimuth horizontal) · Pure pixel data
                </p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* 3. Merged Interpretability Section: Untrained vs Trained Network Slider */}
      <section className="space-y-6 border-t border-neutral-800 pt-10">
        <div className="space-y-2">
          <div className="flex items-center space-x-1 font-mono text-xs text-neutral-400 uppercase tracking-wider">
            <span>Part 2: Network Representation Ablation (§3.6)</span>
            <Tooltip term="ROC-AUC" />
          </div>
          <h2 className="text-2xl font-light text-neutral-100 tracking-tight">
            Before Training vs. After Training
          </h2>
          <p className="text-sm text-neutral-300 max-w-3xl font-sans">
            A sanity-check ablation showing what the network learns. Before training, randomly initialized filters produce diffuse, noisy activations. After training, the network focuses on specific high-backscatter structures.
          </p>
        </div>

        {/* Patch Selector Dropdown */}
        <div className="flex flex-wrap items-center justify-between gap-4 p-3 rounded-lg border border-neutral-800 bg-neutral-900/50">
          <div className="flex items-center space-x-3 font-mono text-xs">
            <span className="text-neutral-400">Select Evaluated Patch:</span>
            <select
              value={activeSlotIdx}
              onChange={(e) => setActiveSlotIdx(Number(e.target.value))}
              className="bg-neutral-950 border border-neutral-700 text-neutral-200 px-3 py-1.5 rounded text-xs focus:outline-none focus:border-cyan-500"
            >
              {slots.map((s: any, idx: number) => (
                <option key={s.slot} value={idx}>
                  {s.category} (Patch #{s.patch_idx}, Strip {s.strip_id})
                </option>
              ))}
            </select>
          </div>

          <div className="flex items-center space-x-4 font-mono text-xs">
            <span className="text-neutral-400">
              Before Score:{' '}
              <strong className="text-neutral-200">
                {activeSlot?.untrained_model?.score?.toFixed(4) ?? '0.5000'}
              </strong>
            </span>
            <span className="text-neutral-400">
              After Score:{' '}
              <strong className="text-cyan-400">
                {activeSlot?.trained_model?.score?.toFixed(4) ?? '0.7500'}
              </strong>
            </span>
          </div>
        </div>

        {/* The Comparison Slider: Before Training vs After Training */}
        <div className="flex justify-center">
          <div className="w-full max-w-md">
            <ComparisonSlider
              beforeImage={`/${activeSlot?.untrained_model?.gradcam_image}`}
              afterImage={`/${activeSlot?.trained_model?.gradcam_image}`}
              beforeLabel="Before training"
              afterLabel="After training"
            />
          </div>
        </div>

        {/* Physical Disclaimer */}
        <div className="p-4 rounded-md border border-neutral-800 bg-neutral-900/40 text-xs text-neutral-400 font-sans leading-relaxed max-w-3xl mx-auto text-center">
          <strong className="text-neutral-300">Interpretability Disclaimer:</strong> Grad-CAM highlights regional spatial correlation with radar speckle variance rather than resolving subsurface dielectric ice volume scattering.
        </div>
      </section>
    </div>
  );
}
