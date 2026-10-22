'use client';

import React, { useState } from 'react';
import { getPipelineStagesData } from '../../lib/data';

const STAGE_DETAILS = [
  {
    stage: 1,
    title: 'Raw HH Complex Amplitude Log-Intensity',
    image: '/data/pipeline_stages/stage1_raw_hh.png',
    summary: 'Single-look complex (SLI) L1A horizontally transmitted, horizontally received backscatter amplitude.',
    math: 'I_{HH} = \\log_{10}(|S_{HH}|^2 + \\epsilon)',
    detail: 'Raw Chandrayaan-2 DFSAR L1A quad-pol data loaded from slant-range complex GeoTIFFs (line 100,000). Highly speckled single-look intensity capturing lunar terrain surface relief.',
  },
  {
    stage: 2,
    title: 'Raw HV Cross-Polar Symmetrization',
    image: '/data/pipeline_stages/stage2_raw_hv.png',
    summary: 'Reciprocally symmetrized cross-polar backscatter intensity.',
    math: 'S_{HV}^{sym} = 0.5 \\cdot (S_{HV} + S_{VH})',
    detail: 'Under radar reciprocity in monostatic radar geometries, cross-polar scattering elements are symmetrized to minimize receiver channel noise before polarimetric decomposition.',
  },
  {
    stage: 3,
    title: 'Raw VV Complex Amplitude Log-Intensity',
    image: '/data/pipeline_stages/stage3_raw_vv.png',
    summary: 'Vertically transmitted, vertically received backscatter amplitude.',
    math: 'I_{VV} = \\log_{10}(|S_{VV}|^2 + \\epsilon)',
    detail: 'Vertical co-polar channel completes the full linear scattering matrix [S_{HH}, S_{HV}, S_{VH}, S_{VV}] required for circular polarization basis transformations.',
  },
  {
    stage: 4,
    title: 'Multilooked Radar Composite (W=7 Boxcar)',
    image: '/data/pipeline_stages/stage4_multilooked.png',
    summary: 'Spatial multilook boxcar filter averaging covariance matrix terms.',
    math: '\\langle \\sigma \\rangle = \\frac{1}{W^2} \\sum_{i,j \\in W} \\sigma(i, j)',
    detail: 'Single-look SAR data mathematically forces Degree of Polarization (DOP) to 1.0 everywhere. A 7x7 spatial multilook averaging filter is applied to covariance terms to enable true depolarized volume scattering estimation.',
  },
  {
    stage: 5,
    title: 'Circular Polarization Ratio (CPR) Map',
    image: '/data/pipeline_stages/stage5_cpr.png',
    summary: 'Ratio of same-sense circular to opposite-sense circular backscatter.',
    math: 'CPR = \\frac{\\langle \\sigma_{RR} \\rangle}{\\langle \\sigma_{RL} \\rangle}',
    detail: 'Surface scattering produces low CPR (<0.2–0.4), whereas double-bounce and coherent backscatter opposition effects in subsurface water-ice produce elevated CPR (>1.0).',
  },
  {
    stage: 6,
    title: 'Degree of Polarization (DOP) Map',
    image: '/data/pipeline_stages/stage6_dop.png',
    summary: 'Normalized polarized power from multilooked Stokes parameters.',
    math: 'DOP = \\frac{\\sqrt{S_1^2 + S_2^2 + S_3^2}}{S_0}',
    detail: 'Diffuse multiple volume scattering in fragmented or porous regolith depolarizes the reflected radar signal, yielding low DOP (<0.13).',
  },
  {
    stage: 7,
    title: 'Binary Candidate Ice Mask (CPR > 1 & DOP < 0.13)',
    image: '/data/pipeline_stages/stage7_label_mask.png',
    summary: 'Physical thresholding defining anomalous depolarizing scatterers.',
    math: 'M(r, c) = (CPR > 1.0) \\land (DOP < 0.13)',
    detail: 'Both physical criteria applied simultaneously identify pixels displaying strong volume-scattering signatures consistent with candidate ice deposits.',
  },
  {
    stage: 8,
    title: 'Spatial Patch Extraction Grid (64x64, Stride 32)',
    image: '/data/pipeline_stages/stage8_patch_grid.png',
    summary: 'Sliding-window patch extraction with >=5 pixel label assignment.',
    math: 'y_i = 1 \\iff \\sum_{(r,c) \\in P_i} M(r,c) \\ge 5',
    detail: '64x64 patches are extracted with 32-pixel stride along track. A patch is labeled positive if it contains at least 5 candidate ice pixels (0.12% patch area). Buffer zones are dropped to prevent spatial extent leakage.',
  },
];

export default function PipelinePage() {
  const [activeStageIndex, setActiveStageIndex] = useState(0);
  const current = STAGE_DETAILS[activeStageIndex];

  return (
    <div className="space-y-8 py-4">
      {/* Section Header */}
      <div className="border-b border-neutral-800 pb-4">
        <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
          Pipeline Architecture // Step 0 Ingestion
        </span>
        <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
          DFSAR Radar Processing Pipeline Stages
        </h1>
        <p className="text-sm text-neutral-400 mt-1 max-w-2xl font-sans">
          From raw complex slant-range GeoTIFFs to leak-free labeled spatial patches.
        </p>
      </div>

      {/* Scroll-Pinned Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Stage Text Story & Progress Rail */}
        <div className="lg:col-span-7 space-y-4">
          <div className="flex items-center space-x-1 border border-neutral-800 p-1 rounded bg-neutral-900/60 overflow-x-auto scrollbar-none mb-4">
            {STAGE_DETAILS.map((st, i) => (
              <button
                key={st.stage}
                onClick={() => setActiveStageIndex(i)}
                className={`px-3 py-1 text-xs font-mono rounded transition flex items-center space-x-1.5 shrink-0 ${
                  activeStageIndex === i
                    ? 'bg-neutral-800 text-neutral-100 border border-neutral-700 font-bold'
                    : 'text-neutral-400 hover:text-neutral-200'
                }`}
              >
                <span>S{st.stage}</span>
              </button>
            ))}
          </div>

          <div className="space-y-4">
            {STAGE_DETAILS.map((st, i) => {
              const isActive = activeStageIndex === i;
              return (
                <div
                  key={st.stage}
                  onClick={() => setActiveStageIndex(i)}
                  className={`p-5 rounded-lg border transition cursor-pointer ${
                    isActive
                      ? 'border-neutral-700 bg-neutral-900/90 shadow-md ring-1 ring-white/5'
                      : 'border-neutral-800/80 bg-neutral-950/60 opacity-60 hover:opacity-90'
                  }`}
                >
                  <div className="flex items-center justify-between mb-2">
                    <span className="font-mono text-[10px] text-neutral-400 uppercase tracking-widest">
                      STAGE {st.stage} OF 8
                    </span>
                    {isActive && (
                      <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
                    )}
                  </div>

                  <h3 className="text-base font-medium text-neutral-100 mb-1">
                    {st.title}
                  </h3>

                  <p className="text-xs text-neutral-300 leading-relaxed font-sans mb-3">
                    {st.summary}
                  </p>

                  <div className="p-2 rounded bg-neutral-950 border border-neutral-800 font-mono text-xs text-neutral-300">
                    {st.math}
                  </div>

                  <p className="text-xs text-neutral-400 mt-3 font-sans leading-relaxed">
                    {st.detail}
                  </p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right Column: Sticky Image Display */}
        <div className="lg:col-span-5 sticky top-20 space-y-3">
          <div className="rounded-lg border border-neutral-800 bg-neutral-900 p-4 flex flex-col items-center shadow-2xl">
            <div className="relative w-full flex items-center justify-center min-h-[480px] rounded overflow-hidden border border-neutral-800/80 bg-neutral-950 p-2">
              <img
                src={current.image}
                alt={current.title}
                className="max-h-[500px] w-auto object-contain rounded transition-opacity duration-300"
              />
            </div>

            <div className="mt-4 text-center space-y-1 w-full border-t border-neutral-800 pt-3">
              <div className="font-mono text-xs font-bold text-neutral-200">
                Stage {current.stage}: {current.title}
              </div>
              <p className="text-[11px] font-mono text-neutral-500">
                single 512x116 crop, illustrative
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
