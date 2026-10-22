'use client';

import React from 'react';
import { getGradCamData } from '../../lib/data';

const FAILURES = [
  {
    type: 'False Positive 1',
    patchIdx: 217,
    stripId: 10,
    trueLabel: 0,
    score: 0.7644,
    th: 0.45,
    coords: [32, 64],
    gradcamImg: '/data/gradcam/trained/failure_fp1.png',
    patchImg: '/data/patches/patch_217.png',
    analysis: 'High model confidence (0.7644) on regolith ground truth. Patch displays elevated roughness or localized slope dihedral scattering that mimics volume scattering intensity.',
  },
  {
    type: 'False Positive 2',
    patchIdx: 34,
    stripId: 6,
    trueLabel: 0,
    score: 0.7533,
    th: 0.45,
    coords: [128, 32],
    gradcamImg: '/data/gradcam/trained/failure_fp2.png',
    patchImg: '/data/patches/patch_034.png',
    analysis: 'Confidence 0.7533 against 0.45 threshold. Surface scatterer in strip 6 with bright backscatter return falsely classified as candidate ice.',
  },
  {
    type: 'False Negative 1',
    patchIdx: 277,
    stripId: 14,
    trueLabel: 1,
    score: 0.0011,
    th: 0.45,
    coords: [32, 64],
    gradcamImg: '/data/gradcam/trained/failure_fn1.png',
    patchImg: '/data/patches/patch_277.png',
    analysis: 'Extremely confident miss (score 0.0011 vs 0.45 threshold). True candidate ice deposit containing >=5 depolarizing pixels completely unflagged by the CNN.',
  },
  {
    type: 'False Negative 2',
    patchIdx: 275,
    stripId: 14,
    trueLabel: 1,
    score: 0.0035,
    th: 0.45,
    coords: [32, 32],
    gradcamImg: '/data/gradcam/trained/failure_fn2.png',
    patchImg: '/data/patches/patch_275.png',
    analysis: 'Extremely confident miss (score 0.0035). Shares 50% spatial overlap with Patch 277 (same underlying radar feature seen twice).',
  },
];

const LIMITATIONS = [
  '3 along-track test strips only (strip 14 contains merely 5 positive patches); validation split contains only 77 positives.',
  'Labels derived from polarimetric criteria (CPR > 1.0, DOP < 0.13, >=5 anomalous pixels per patch); circularity guard strictly applies.',
  'Per-patch standardization discards absolute backscatter levels, forcing reliance solely on intra-patch spatial contrast.',
  'Single radar scene (lines 100,000–140,000), DFSAR-only; no external Permanently Shadowed Region (PSR) or DEM topographic slope mask applied.',
  '3 random seeds per configuration (42, 123, 2024); limits statistical power to detect small effect sizes.',
  'Optimal decision thresholds tuned on validation swing from 0.05 to 0.92 across seeds; ROC-AUC is therefore the primary defensible metric.',
];

export default function FailureCasesPage() {
  return (
    <div className="space-y-12 py-4">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-4">
        <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
          Error Diagnostics & Physical Caveats (§3.6–§3.7)
        </span>
        <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
          Failure Mode Analysis & Benchmark Limitations
        </h1>
        <p className="text-sm text-neutral-400 mt-1 max-w-3xl font-sans">
          Detailed inspection of the four highest-confidence classification failures under fixed selection rules (CustomCNN seed 2024, threshold 0.45), coupled with documented methodological constraints.
        </p>
      </div>

      {/* Overlapping-FN Critical Note */}
      <div className="p-5 rounded-lg border border-amber-500/30 bg-amber-950/20 space-y-2">
        <div className="flex items-center space-x-2 font-mono text-xs text-amber-400 uppercase font-bold tracking-wider">
          <span>Non-Independent False Negatives Note (§3.6)</span>
        </div>
        <p className="text-xs text-neutral-200 leading-relaxed font-sans">
          The two most confident misses are overlapping patches from along-track strip 14 (<strong>Patches 275 and 277</strong>, row coordinates 32 px apart at row offset 32; with 64 px patches this is a <strong>50% spatial overlap</strong>, representing the exact same radar feature seen twice).
          They score <strong>0.0011</strong> and <strong>0.0035</strong> against the 0.45 threshold. These are confident misses, not borderline ones, directly explaining why CustomCNN has TP = 0 on strip 14 across all three seeds. This is a genuine detection failure whose physical mechanism remains unverified.
        </p>
      </div>

      {/* The 4 Failure Patches Grid */}
      <section className="space-y-4">
        <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold border-b border-neutral-800 pb-2">
          Highest-Confidence Failure Cases (Top-2 FP, Top-2 FN)
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {FAILURES.map((f) => (
            <div
              key={f.type}
              className="p-5 rounded-lg border border-neutral-800 bg-neutral-900/40 space-y-4"
            >
              <div className="flex items-center justify-between border-b border-neutral-800 pb-2.5 font-mono text-xs">
                <div>
                  <span className="font-bold text-neutral-200 uppercase">{f.type}</span>
                  <div className="text-[11px] text-neutral-400">Patch #{f.patchIdx} · Strip {f.stripId}</div>
                </div>
                <div className="text-right">
                  <span className={`px-2 py-0.5 rounded text-[10px] font-bold ${
                    f.trueLabel === 1 ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30' : 'bg-neutral-800 text-neutral-400'
                  }`}>
                    Ground Truth: {f.trueLabel === 1 ? 'Candidate Ice' : 'Regolith'}
                  </span>
                  <div className="text-[11px] text-neutral-300 mt-0.5 font-bold">
                    Score: {f.score.toFixed(4)} (Th: {f.th})
                  </div>
                </div>
              </div>

              {/* Images: Raw Composite & Grad-CAM */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1 text-center">
                  <span className="font-mono text-[10px] text-neutral-400 block">RGB Composite</span>
                  <div className="aspect-square rounded border border-neutral-800 overflow-hidden bg-black">
                    <img src={f.patchImg} alt={`Patch ${f.patchIdx}`} className="w-full h-full object-cover" />
                  </div>
                </div>

                <div className="space-y-1 text-center">
                  <span className="font-mono text-[10px] text-neutral-400 block">Grad-CAM Overlay</span>
                  <div className="aspect-square rounded border border-neutral-800 overflow-hidden bg-black">
                    <img src={f.gradcamImg} alt={`Grad-CAM ${f.patchIdx}`} className="w-full h-full object-contain" />
                  </div>
                </div>
              </div>

              <p className="text-xs text-neutral-300 font-sans leading-relaxed pt-1">
                {f.analysis}
              </p>
            </div>
          ))}
        </div>
      </section>

      {/* Limitations List from section3_analysis.md */}
      <section className="p-6 rounded-lg border border-neutral-800 bg-neutral-900/30 space-y-4">
        <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold border-b border-neutral-800 pb-2">
          Documented Scientific Limitations (§3.7)
        </h2>

        <ul className="space-y-2.5 font-sans text-xs text-neutral-300">
          {LIMITATIONS.map((lim, idx) => (
            <li key={idx} className="flex items-start space-x-2.5">
              <span className="font-mono text-neutral-500 text-[10px] mt-0.5">•</span>
              <span className="leading-relaxed">{lim}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
