'use client';

import React, { useState, useMemo } from 'react';
import { getPatchScoresData, getGradCamData, PatchRecord } from '../../lib/data';
import { Tooltip } from '../../components/Tooltip';
import { ChevronDown, ChevronRight, X } from 'lucide-react';

export default function PatchesPage() {
  const patchScoresData = getPatchScoresData();
  const gradcamManifest = getGradCamData();

  const allPatches: PatchRecord[] = patchScoresData.patches || [];
  const modelRuns: string[] = patchScoresData.runs || [];

  // Model & Threshold controls (default to CustomCNN seed 2024 with reported 0.45 threshold)
  const [selectedModel, setSelectedModel] = useState<string>('CustomCNN_seed2024');
  const [threshold, setThreshold] = useState<number>(0.45);
  const reportedThreshold = 0.45;

  // Advanced panel toggle (closed by default)
  const [showAdvanced, setShowAdvanced] = useState<boolean>(false);

  // Plain word filters
  const [filterOutcome, setFilterOutcome] = useState<'all' | 'correct' | 'fp' | 'fn'>('all');
  const [filterStrip, setFilterStrip] = useState<'all' | '6' | '10' | '14'>('all');

  // Modal / Detail state
  const [selectedPatch, setSelectedPatch] = useState<PatchRecord | null>(null);
  const [selectedChannel, setSelectedChannel] = useState<'hh' | 'hv' | 'vv'>('hh');
  const [showGradCam, setShowGradCam] = useState<boolean>(false);

  // Grad-CAM map lookup
  const gradcamMap = useMemo(() => {
    const map = new Map<number, { trainedImg: string; untrainedImg: string; baseImg: string }>();
    if (gradcamManifest && gradcamManifest.slots) {
      for (const slot of gradcamManifest.slots) {
        map.set(slot.patch_idx, {
          trainedImg: slot.trained_model.gradcam_image,
          untrainedImg: slot.untrained_model.gradcam_image,
          baseImg: slot.base_image,
        });
      }
    }
    return map;
  }, [gradcamManifest]);

  // Confusion stats at active threshold
  const confusionStats = useMemo(() => {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    let tn = 0;

    for (const p of allPatches) {
      const score = p.scores[selectedModel] ?? 0.5;
      const pred = score >= threshold ? 1 : 0;
      if (p.label === 1 && pred === 1) tp++;
      else if (p.label === 0 && pred === 1) fp++;
      else if (p.label === 1 && pred === 0) fn++;
      else if (p.label === 0 && pred === 0) tn++;
    }

    return { tp, fp, fn, tn, correct: tp + tn };
  }, [allPatches, selectedModel, threshold]);

  // Filtered patch list
  const filteredPatches = useMemo(() => {
    return allPatches.filter((p) => {
      // Strip filter
      if (filterStrip !== 'all' && p.strip_id.toString() !== filterStrip) return false;

      // Outcome filter
      const score = p.scores[selectedModel] ?? 0.5;
      const pred = score >= threshold ? 1 : 0;
      const isTP = p.label === 1 && pred === 1;
      const isFP = p.label === 0 && pred === 1;
      const isFN = p.label === 1 && pred === 0;
      const isTN = p.label === 0 && pred === 0;

      if (filterOutcome === 'correct' && !isTP && !isTN) return false;
      if (filterOutcome === 'fp' && !isFP) return false;
      if (filterOutcome === 'fn' && !isFN) return false;

      return true;
    });
  }, [allPatches, selectedModel, threshold, filterOutcome, filterStrip]);

  return (
    <div className="space-y-6 py-4">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-4">
        <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
          Patch Explorer // 338 Evaluated Extents (64×64 px)
        </span>
        <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
          Spatial Patch Catalog
        </h1>
        <p className="text-xs font-mono text-neutral-400 mt-1 max-w-2xl">
          Click any thumbnail to inspect multi-polarimetric SAR channels (HH, HV, VV) and Grad-CAM overlays.
        </p>
      </div>

      {/* Filter Bar (Plain English Buttons) */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 p-3 rounded border border-neutral-800 bg-neutral-900/30">
        {/* Outcome filters */}
        <div className="flex flex-wrap items-center gap-1.5 font-mono text-xs">
          <span className="text-neutral-500 text-[11px] mr-1">FILTER:</span>
          <button
            onClick={() => setFilterOutcome('all')}
            className={`px-3 py-1 rounded transition ${
              filterOutcome === 'all'
                ? 'bg-neutral-800 text-neutral-100 border border-neutral-700 font-semibold'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            All (338)
          </button>
          <button
            onClick={() => setFilterOutcome('correct')}
            className={`px-3 py-1 rounded transition ${
              filterOutcome === 'correct'
                ? 'bg-neutral-800 text-neutral-100 border border-neutral-700 font-semibold'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            Correct ({confusionStats.correct})
          </button>
          <button
            onClick={() => setFilterOutcome('fp')}
            className={`px-3 py-1 rounded transition ${
              filterOutcome === 'fp'
                ? 'bg-amber-950/60 text-amber-300 border border-amber-800/80 font-semibold'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            False alarm ({confusionStats.fp})
          </button>
          <button
            onClick={() => setFilterOutcome('fn')}
            className={`px-3 py-1 rounded transition ${
              filterOutcome === 'fn'
                ? 'bg-neutral-800 text-neutral-200 border border-neutral-700 font-semibold'
                : 'text-neutral-400 hover:text-neutral-200'
            }`}
          >
            Missed ice ({confusionStats.fn})
          </button>
        </div>

        {/* Strip selector */}
        <div className="flex items-center space-x-1 font-mono text-xs">
          <span className="text-neutral-500 text-[11px] mr-1">STRIP:</span>
          {(['all', '6', '10', '14'] as const).map((strip) => (
            <button
              key={strip}
              onClick={() => setFilterStrip(strip)}
              className={`px-2.5 py-1 rounded transition text-xs ${
                filterStrip === strip
                  ? 'bg-neutral-800 text-cyan-400 border border-neutral-700 font-medium'
                  : 'text-neutral-400 hover:text-neutral-200'
              }`}
            >
              {strip === 'all' ? 'All' : `Strip ${strip}`}
            </button>
          ))}
          <Tooltip term="Strip" />
        </div>
      </div>

      {/* Advanced Drawer: Threshold slider & Model selector (closed by default) */}
      <div className="rounded border border-neutral-800 bg-neutral-900/20 overflow-hidden">
        <button
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="w-full p-3 flex items-center justify-between text-left hover:bg-neutral-800/30 transition border-b border-neutral-800/60"
        >
          <div className="flex items-center space-x-2">
            {showAdvanced ? (
              <ChevronDown className="w-4 h-4 text-cyan-400" />
            ) : (
              <ChevronRight className="w-4 h-4 text-neutral-500" />
            )}
            <span className="font-mono text-xs uppercase tracking-wider text-neutral-300 font-semibold">
              Advanced: Model & Decision Threshold Controls
            </span>
          </div>
          <div className="flex items-center space-x-2 text-[11px] font-mono text-neutral-400">
            <span>Model: <strong className="text-neutral-200">{selectedModel}</strong></span>
            <span>·</span>
            <span>Threshold: <strong className="text-neutral-200">{threshold.toFixed(2)}</strong> {threshold === reportedThreshold && '(reported)'}</span>
          </div>
        </button>

        {showAdvanced && (
          <div className="p-4 space-y-4 font-mono text-xs">
            <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
              {/* Model Selector */}
              <div className="flex items-center space-x-2">
                <span className="text-neutral-400 text-xs">Select Run:</span>
                <select
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="bg-neutral-950 border border-neutral-700 px-3 py-1.5 rounded text-xs text-neutral-100 focus:outline-none"
                >
                  {modelRuns.map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>

              {/* Slider */}
              <div className="flex-1 max-w-md flex items-center space-x-3">
                <span className="text-neutral-400 whitespace-nowrap">
                  Threshold: <strong className="text-neutral-100">{threshold.toFixed(2)}</strong>
                </span>
                <input
                  type="range"
                  min="0.05"
                  max="0.95"
                  step="0.01"
                  value={threshold}
                  onChange={(e) => setThreshold(parseFloat(e.target.value))}
                  className="flex-1 accent-cyan-400 cursor-pointer h-1.5 bg-neutral-700 rounded-lg appearance-none"
                />
                <button
                  onClick={() => setThreshold(reportedThreshold)}
                  className="px-2 py-1 text-[10px] bg-neutral-800 border border-neutral-700 hover:border-neutral-500 rounded text-neutral-300"
                >
                  Reset 0.45
                </button>
              </div>

              {/* Badges */}
              <div className="flex items-center space-x-2 text-xs">
                <span className="px-2 py-1 rounded bg-neutral-950 border border-cyan-500/40 text-cyan-400">
                  TP: {confusionStats.tp}
                </span>
                <span className="px-2 py-1 rounded bg-neutral-950 border border-amber-500/40 text-amber-400">
                  FP: {confusionStats.fp}
                </span>
                <span className="px-2 py-1 rounded bg-neutral-950 border border-neutral-700 text-neutral-300">
                  FN: {confusionStats.fn}
                </span>
                <span className="px-2 py-1 rounded bg-neutral-950 border border-neutral-800 text-neutral-500">
                  TN: {confusionStats.tn}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Contact Sheet Grid */}
      <div className="p-3 rounded border border-neutral-800 bg-neutral-950">
        <div className="flex justify-between items-center pb-2 mb-2 border-b border-neutral-900 font-mono text-[11px] text-neutral-500">
          <span>Showing {filteredPatches.length} of 338 patches (HH polarization)</span>
          <span className="flex items-center space-x-3">
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-sm border border-cyan-400 bg-cyan-950/40"></span>
              <span>True Positive</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-sm border border-amber-500 bg-amber-950/40"></span>
              <span>False Alarm</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-sm border border-neutral-400 bg-neutral-800/40"></span>
              <span>Missed Ice</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2.5 h-2.5 rounded-sm border border-neutral-800"></span>
              <span>True Negative</span>
            </span>
          </span>
        </div>

        <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 lg:grid-cols-12 gap-1.5 max-h-[620px] overflow-y-auto p-1">
          {filteredPatches.map((p) => {
            const score = p.scores[selectedModel] ?? 0.5;
            const pred = score >= threshold ? 1 : 0;
            let outcomeBorder = 'border-neutral-800/60';
            if (p.label === 1 && pred === 1) {
              outcomeBorder = 'border-cyan-400 ring-1 ring-cyan-500/40';
            } else if (p.label === 0 && pred === 1) {
              outcomeBorder = 'border-amber-500/80';
            } else if (p.label === 1 && pred === 0) {
              outcomeBorder = 'border-neutral-400';
            } else {
              outcomeBorder = 'border-neutral-900 opacity-60';
            }

            const isSelected = selectedPatch?.patch_idx === p.patch_idx;
            const paddedIdx = p.patch_idx.toString().padStart(3, '0');
            const thumbPath = `/data/patches/patch_${paddedIdx}_hh.webp`;

            return (
              <button
                key={p.patch_idx}
                onClick={() => {
                  setSelectedPatch(p);
                  setShowGradCam(false);
                }}
                className={`relative aspect-square rounded overflow-hidden border transition bg-neutral-900 ${outcomeBorder} ${
                  isSelected ? 'ring-2 ring-cyan-400 scale-105 z-10' : 'hover:scale-105 hover:z-10'
                }`}
                title={`Patch #${p.patch_idx} (Strip ${p.strip_id}, Score ${score.toFixed(3)})`}
              >
                <img
                  src={thumbPath}
                  alt={`Patch ${p.patch_idx}`}
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
              </button>
            );
          })}
        </div>
      </div>

      {/* Detail Modal Dialog */}
      {selectedPatch && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm">
          <div className="bg-neutral-950 border border-neutral-800 rounded-lg max-w-xl w-full p-5 space-y-4 max-h-[90vh] overflow-y-auto">
            {/* Modal Header */}
            <div className="flex items-center justify-between border-b border-neutral-800 pb-3">
              <div>
                <span className="font-mono text-[10px] text-neutral-500 uppercase block">PATCH INSPECTION</span>
                <h3 className="font-mono text-lg font-bold text-neutral-100">
                  Patch #{selectedPatch.patch_idx} (Strip {selectedPatch.strip_id})
                </h3>
              </div>
              <div className="flex items-center space-x-2">
                <span
                  className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold ${
                    selectedPatch.label === 1
                      ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                      : 'bg-neutral-900 text-neutral-400 border border-neutral-800'
                  }`}
                >
                  {selectedPatch.label === 1 ? 'Candidate ice (1)' : 'Not an ice candidate (0)'}
                </span>
                <button
                  onClick={() => setSelectedPatch(null)}
                  className="p-1 rounded hover:bg-neutral-900 text-neutral-400 hover:text-neutral-100"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
            </div>

            {/* Image Preview with Channel / Grad-CAM toggles */}
            <div className="space-y-3">
              <div className="flex items-center justify-between font-mono text-xs">
                <div className="flex items-center space-x-1 bg-neutral-900 p-0.5 rounded border border-neutral-800">
                  {(['hh', 'hv', 'vv'] as const).map((ch) => (
                    <button
                      key={ch}
                      onClick={() => {
                        setSelectedChannel(ch);
                        setShowGradCam(false);
                      }}
                      className={`px-3 py-1 rounded uppercase text-[11px] transition ${
                        !showGradCam && selectedChannel === ch
                          ? 'bg-neutral-800 text-neutral-100 font-bold border border-neutral-700'
                          : 'text-neutral-400 hover:text-neutral-200'
                      }`}
                    >
                      {ch}
                    </button>
                  ))}
                </div>

                {gradcamMap.has(selectedPatch.patch_idx) && (
                  <button
                    onClick={() => setShowGradCam(!showGradCam)}
                    className={`px-3 py-1 rounded text-[11px] font-mono border transition ${
                      showGradCam
                        ? 'bg-cyan-950 text-cyan-300 border-cyan-700 font-bold'
                        : 'bg-neutral-900 text-neutral-400 border-neutral-800 hover:text-neutral-200'
                    }`}
                  >
                    {showGradCam ? 'Grad-CAM Active' : 'Show Grad-CAM'}
                  </button>
                )}
              </div>

              {/* 384x384 Pure Pixel Display */}
              <div className="w-full aspect-square max-w-[384px] mx-auto rounded border border-neutral-800 overflow-hidden bg-black flex items-center justify-center">
                {showGradCam && gradcamMap.has(selectedPatch.patch_idx) ? (
                  <img
                    src={`/${gradcamMap.get(selectedPatch.patch_idx)?.trainedImg}`}
                    alt={`Grad-CAM Patch ${selectedPatch.patch_idx}`}
                    className="w-full h-full object-contain"
                  />
                ) : (
                  <img
                    src={`/data/patches/patch_${selectedPatch.patch_idx.toString().padStart(3, '0')}_${selectedChannel}.webp`}
                    alt={`Patch ${selectedPatch.patch_idx} ${selectedChannel}`}
                    className="w-full h-full object-contain"
                  />
                )}
              </div>
              <p className="text-[10px] font-mono text-neutral-500 text-center">
                {showGradCam
                  ? 'Grad-CAM Overlay (CustomCNN seed 2024, layer conv3)'
                  : `Polarization: ${selectedChannel.toUpperCase()} log-intensity · 64×64 radar resolution (Lanczos upscaled)`}
              </p>
            </div>

            {/* Metadata Grid */}
            <div className="grid grid-cols-2 gap-2 font-mono text-xs p-3 rounded bg-neutral-900/60 border border-neutral-800 text-neutral-300">
              <div>Coordinates: <span className="text-neutral-100">{selectedPatch.coords[0]}, {selectedPatch.coords[1]}</span></div>
              <div>Strip ID: <span className="text-neutral-100">Strip {selectedPatch.strip_id}</span></div>
              <div>Positive pixels: <span className={selectedPatch.pos_px >= 5 ? 'text-cyan-400 font-bold' : 'text-neutral-100'}>{selectedPatch.pos_px}/4096</span></div>
              <div>Ambiguous: <span className={selectedPatch.is_clean ? 'text-neutral-400' : 'text-amber-400'}>{selectedPatch.is_clean ? 'No' : 'Yes (1–4 px)'}</span></div>
              <div>Mean CPR: <span className="text-neutral-100">{selectedPatch.cpr_mean.toFixed(3)}</span></div>
              <div>Mean DOP: <span className="text-neutral-100">{selectedPatch.dop_mean.toFixed(3)}</span></div>
            </div>

            {/* Model Score vs Threshold */}
            <div className="p-3 rounded bg-neutral-900/40 border border-neutral-800 font-mono text-xs flex items-center justify-between">
              <div>
                <span className="text-neutral-400">Score ({selectedModel}): </span>
                <span className="font-bold text-neutral-100">
                  {(selectedPatch.scores[selectedModel] ?? 0.5).toFixed(4)}
                </span>
              </div>
              <div className="text-neutral-400">
                Threshold: <span className="text-neutral-200">{threshold.toFixed(2)}</span>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
