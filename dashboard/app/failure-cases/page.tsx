'use client';

import React from 'react';
import { Tooltip } from '../../components/Tooltip';

const FAILURES = [
  {
    type: 'False Positive 1',
    patchIdx: 217,
    stripId: 10,
    labelDesc: 'not an ice candidate (0)',
    isPositive: false,
    score: 0.7644,
    th: 0.45,
    hhImg: '/data/failure_cases/patch_217_hh.webp',
    overlayImg: '/data/failure_cases/patch_217_overlay.webp',
  },
  {
    type: 'False Positive 2',
    patchIdx: 34,
    stripId: 6,
    labelDesc: 'not an ice candidate (0)',
    isPositive: false,
    score: 0.7533,
    th: 0.45,
    hhImg: '/data/failure_cases/patch_34_hh.webp',
    overlayImg: '/data/failure_cases/patch_34_overlay.webp',
  },
  {
    type: 'False Negative 1',
    patchIdx: 277,
    stripId: 14,
    labelDesc: 'candidate ice (1)',
    isPositive: true,
    score: 0.0011,
    th: 0.45,
    hhImg: '/data/failure_cases/patch_277_hh.webp',
    overlayImg: '/data/failure_cases/patch_277_overlay.webp',
  },
  {
    type: 'False Negative 2',
    patchIdx: 275,
    stripId: 14,
    labelDesc: 'candidate ice (1)',
    isPositive: true,
    score: 0.0035,
    th: 0.45,
    hhImg: '/data/failure_cases/patch_275_hh.webp',
    overlayImg: '/data/failure_cases/patch_275_overlay.webp',
  },
];

const LIMITATIONS = [
  '3 test strips (one with 5 positives); 77 validation positives.',
  'Labels derived from polarimetric criteria (CPR > 1.0, DOP < 0.13, and >=5 anomalous pixels per patch), so circularity applies.',
  'Per-patch standardization discards absolute backscatter level, forcing reliance solely on intra-patch spatial contrast.',
  'Single radar scene (lines 100,000–140,000), DFSAR-only; no external Permanently Shadowed Region (PSR) or DEM topographic slope mask applied.',
  '3 seeds per configuration; limits statistical power to detect small effect sizes.',
  'Thresholds tuned on validation swing from 0.05 to 0.92 across seeds, so ROC-AUC is the primary defensible metric.',
];

export default function FailureCasesPage() {
  return (
    <div className="space-y-10 py-4">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-4">
        <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
          Diagnostics & Caveats // Section 3.6–3.7
        </span>
        <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
          What Went Wrong: Failure Modes & Limitations
        </h1>
        <p className="text-xs font-mono text-neutral-400 mt-1 max-w-3xl">
          Inspection of the four highest-confidence classification failures under a fixed rule (CustomCNN seed 2024, threshold 0.45).
          Illustrative only; no causal claims.
        </p>
      </div>

      {/* Overlapping-FN Note (strictly per Section 3.6) */}
      <div className="p-4 rounded border border-neutral-800 bg-neutral-900/40 space-y-2">
        <div className="flex items-center space-x-2 font-mono text-xs text-amber-400 uppercase font-semibold">
          <span>Overlapping False Negatives Note (§3.6)</span>
        </div>
        <p className="text-xs text-neutral-300 leading-relaxed font-sans">
          The two most confident misses are overlapping patches from strip 14 (<strong>Patches 275 and 277</strong>, row coordinates 32 px apart at row offset 32; with 64 px patches this is a <strong>50% spatial overlap</strong>, representing the same radar feature seen twice).
          They score <strong>0.0011</strong> and <strong>0.0035</strong> against the 0.45 threshold.
        </p>
        <p className="text-xs text-neutral-400 leading-relaxed font-sans italic">
          &ldquo;The two most confident misses are overlapping patches from strip 14; this illustrates the TP = 0 on strip 14, it does not explain it.&rdquo;
          Four patches cannot support physical claims about crater rims or dihedral scattering, and §3.6 correctly labels that as untested.
        </p>
      </div>

      {/* 4 Failure Patches Side-by-Side (Pure 384x384 WebP, zero letterbox) */}
      <section className="space-y-4">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between border-b border-neutral-800 pb-2 gap-1">
          <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
            Highest-Confidence Failures (Top-2 FP, Top-2 FN)
          </h2>
          <span className="text-[11px] font-mono text-neutral-500">
            Selected from CustomCNN seed 2024 (tuned threshold 0.45) · Grad-CAM overlays: illustrative, not evidence of physical meaning
          </span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {FAILURES.map((f) => (
            <div
              key={f.type}
              className="p-5 rounded border border-neutral-800 bg-neutral-900/30 space-y-4"
            >
              {/* Header: Patch info only */}
              <div className="flex items-start justify-between border-b border-neutral-800 pb-3 font-mono text-xs">
                <div>
                  <span className="font-bold text-neutral-100 uppercase">{f.type}</span>
                  <div className="text-[11px] text-neutral-400 mt-0.5">
                    Patch #{f.patchIdx} · Strip {f.stripId}
                  </div>
                </div>
                <div className="text-right">
                  <span
                    className={`inline-block px-2 py-0.5 rounded text-[10px] font-mono font-semibold ${
                      f.isPositive
                        ? 'bg-cyan-950 text-cyan-400 border border-cyan-800/80'
                        : 'bg-neutral-800 text-neutral-300 border border-neutral-700'
                    }`}
                  >
                    Label: {f.labelDesc}
                  </span>
                  <div className="text-[11px] text-neutral-300 mt-1 font-mono">
                    Score: <strong className="text-neutral-100">{f.score.toFixed(4)}</strong> (Th: {f.th})
                  </div>
                </div>
              </div>

              {/* Side-by-side: HH pure pixel & Grad-CAM pure pixel (aspect-square, no letterbox) */}
              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5 text-center">
                  <span className="font-mono text-[10px] text-neutral-400 block">SAR HH (64×64 px)</span>
                  <div className="aspect-square rounded border border-neutral-800 overflow-hidden bg-black">
                    <img
                      src={f.hhImg}
                      alt={`Patch ${f.patchIdx} HH`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>

                <div className="space-y-1.5 text-center">
                  <span className="font-mono text-[10px] text-neutral-400 block">Grad-CAM Overlay</span>
                  <div className="aspect-square rounded border border-neutral-800 overflow-hidden bg-black">
                    <img
                      src={f.overlayImg}
                      alt={`Grad-CAM Patch ${f.patchIdx}`}
                      className="w-full h-full object-cover"
                    />
                  </div>
                </div>
              </div>

              <div className="pt-1 text-[11px] font-mono text-neutral-500 flex justify-between">
                <span>Resolution: 384×384 pure WebP</span>
                <span>Layer: conv3 activation (illustrative, not evidence of physical meaning)</span>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Limitations List from section3_analysis.md */}
      <section className="p-5 rounded border border-neutral-800 bg-neutral-900/20 space-y-3">
        <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold border-b border-neutral-800 pb-2">
          Methodological Limitations (§3.7)
        </h2>

        <ul className="space-y-2 font-sans text-xs text-neutral-300 leading-relaxed">
          {LIMITATIONS.map((lim, idx) => (
            <li key={idx} className="flex items-start space-x-2.5">
              <span className="font-mono text-cyan-400 text-xs font-bold mt-0.5">•</span>
              <span>{lim}</span>
            </li>
          ))}
        </ul>
      </section>
    </div>
  );
}
