'use client';

import React, { useState, useMemo } from 'react';
import { getPatchScoresData, getGradCamData, PatchRecord } from '../../lib/data';
import { PatchHoverCard } from '../../components/PatchHoverCard';

export default function PatchesPage() {
  const patchScoresData = getPatchScoresData();
  const gradcamManifest = getGradCamData();

  const allPatches: PatchRecord[] = patchScoresData.patches || [];
  const modelRuns: string[] = patchScoresData.runs || [];

  // Model & Threshold controls
  const [selectedModel, setSelectedModel] = useState<string>('CustomCNN_seed2024');
  const [threshold, setThreshold] = useState<number>(0.45); // 0.45 is the reported val-tuned threshold for CustomCNN_seed2024
  const reportedThreshold = 0.45;

  // Filters
  const [filterLabel, setFilterLabel] = useState<string>('all');
  const [filterStrip, setFilterStrip] = useState<string>('all');
  const [filterAmbiguity, setFilterAmbiguity] = useState<string>('all');
  const [filterOutcome, setFilterOutcome] = useState<string>('all');

  // Selected patch detail
  const [selectedPatch, setSelectedPatch] = useState<PatchRecord>(allPatches[217] || allPatches[0]);

  // Hover state
  const [hoveredPatch, setHoveredPatch] = useState<PatchRecord | null>(null);
  const [hoverPos, setHoverPos] = useState<{ x: number; y: number }>({ x: 0, y: 0 });

  // Map of patch_idx to gradcam image if available
  const gradcamMap = useMemo(() => {
    const map = new Map<number, { trainedImg: string; untrainedImg: string }>();
    if (gradcamManifest && gradcamManifest.slots) {
      for (const slot of gradcamManifest.slots) {
        map.set(slot.patch_idx, {
          trainedImg: slot.trained_model.gradcam_image,
          untrainedImg: slot.untrained_model.gradcam_image,
        });
      }
    }
    return map;
  }, [gradcamManifest]);

  // Confusion matrix at current threshold
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

    return { tp, fp, fn, tn, pos: tp + fn, neg: fp + tn };
  }, [allPatches, selectedModel, threshold]);

  // Filtered patch list
  const filteredPatches = useMemo(() => {
    return allPatches.filter((p) => {
      if (filterLabel !== 'all' && p.label.toString() !== filterLabel) return false;
      if (filterStrip !== 'all' && p.strip_id.toString() !== filterStrip) return false;
      if (filterAmbiguity === 'clean' && !p.is_clean) return false;
      if (filterAmbiguity === 'ambiguous' && p.is_clean) return false;

      const score = p.scores[selectedModel] ?? 0.5;
      const pred = score >= threshold ? 1 : 0;
      let outcome = '';
      if (p.label === 1 && pred === 1) outcome = 'TP';
      else if (p.label === 0 && pred === 1) outcome = 'FP';
      else if (p.label === 1 && pred === 0) outcome = 'FN';
      else if (p.label === 0 && pred === 0) outcome = 'TN';

      if (filterOutcome !== 'all' && outcome !== filterOutcome) return false;

      return true;
    });
  }, [allPatches, selectedModel, threshold, filterLabel, filterStrip, filterAmbiguity, filterOutcome]);

  const handleMouseEnterPatch = (e: React.MouseEvent, p: PatchRecord) => {
    const rect = e.currentTarget.getBoundingClientRect();
    setHoverPos({ x: rect.right + 12, y: rect.top - 10 });
    setHoveredPatch(p);
  };

  const handleMouseLeavePatch = () => {
    setHoveredPatch(null);
  };

  return (
    <div className="space-y-6 py-4">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-4">
        <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
          Patch Contact Sheet // 338 Evaluated Extents
        </span>
        <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
          Spatial Test Patch Explorer
        </h1>
        <p className="text-sm text-neutral-400 mt-1 max-w-3xl font-sans">
          Dense contact sheet of all 338 test patches. Border colour encodes classification outcome at the active threshold. Hover for preview, click for multi-channel decomposition.
        </p>
      </div>

      {/* Live Threshold Slider & Confusion Matrix Strip */}
      <div className="p-4 rounded-lg border border-neutral-800 bg-neutral-900/40 space-y-4">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-4">
          {/* Model Selector */}
          <div className="flex items-center space-x-3">
            <span className="font-mono text-xs uppercase text-neutral-400">Model:</span>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="bg-neutral-950 border border-neutral-700 px-3 py-1.5 rounded text-xs font-mono text-neutral-100 focus:outline-none"
            >
              {modelRuns.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
          </div>

          {/* Threshold Slider */}
          <div className="flex-1 max-w-md flex items-center space-x-3">
            <span className="font-mono text-xs text-neutral-400 whitespace-nowrap">
              Threshold: <strong className="text-neutral-100">{threshold.toFixed(2)}</strong>
            </span>
            <input
              type="range"
              min="0.05"
              max="0.95"
              step="0.01"
              value={threshold}
              onChange={(e) => setThreshold(parseFloat(e.target.value))}
              className="flex-1 accent-neutral-300 cursor-pointer h-1 bg-neutral-700 rounded-lg appearance-none"
            />
            <button
              onClick={() => setThreshold(reportedThreshold)}
              className="px-2 py-1 text-[10px] font-mono bg-neutral-800 border border-neutral-700 hover:border-neutral-500 rounded text-neutral-300"
              title="Reset to validation-tuned threshold"
            >
              Reset 0.45
            </button>
          </div>

          {/* Confusion Matrix Numbers */}
          <div className="flex items-center space-x-2 font-mono text-xs">
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

        {/* Filter Bar */}
        <div className="flex flex-wrap items-center gap-3 pt-3 border-t border-neutral-800/80 font-mono text-xs">
          <div className="flex items-center space-x-2">
            <span className="text-neutral-500 text-[11px]">GROUND TRUTH:</span>
            <select
              value={filterLabel}
              onChange={(e) => setFilterLabel(e.target.value)}
              className="bg-neutral-950 border border-neutral-800 px-2 py-1 rounded text-neutral-300"
            >
              <option value="all">All Labels</option>
              <option value="1">Candidate Ice (Label 1)</option>
              <option value="0">Regolith (Label 0)</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-neutral-500 text-[11px]">STRIP:</span>
            <select
              value={filterStrip}
              onChange={(e) => setFilterStrip(e.target.value)}
              className="bg-neutral-950 border border-neutral-800 px-2 py-1 rounded text-neutral-300"
            >
              <option value="all">All Strips</option>
              <option value="6">Strip 6 (38.4%)</option>
              <option value="10">Strip 10 (47.3%)</option>
              <option value="14">Strip 14 (4.4%)</option>
            </select>
          </div>

          <div className="flex items-center space-x-2">
            <span className="text-neutral-500 text-[11px]">OUTCOME:</span>
            <select
              value={filterOutcome}
              onChange={(e) => setFilterOutcome(e.target.value)}
              className="bg-neutral-950 border border-neutral-800 px-2 py-1 rounded text-neutral-300"
            >
              <option value="all">All Outcomes</option>
              <option value="TP">True Positive (TP)</option>
              <option value="FP">False Positive (FP)</option>
              <option value="FN">False Negative (FN)</option>
              <option value="TN">True Negative (TN)</option>
            </select>
          </div>

          <span className="text-neutral-500 text-[11px] ml-auto">
            Showing {filteredPatches.length} of 338
          </span>
        </div>
      </div>

      {/* Main Content: Contact Sheet Grid + Large Detail Drawer */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Contact Sheet (Left, 8 Cols) */}
        <div className="lg:col-span-8 p-3 rounded-lg border border-neutral-800 bg-neutral-950">
          <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-10 gap-1.5 max-h-[640px] overflow-y-auto p-1">
            {filteredPatches.map((p) => {
              const score = p.scores[selectedModel] ?? 0.5;
              const pred = score >= threshold ? 1 : 0;
              let outcomeBorder = 'border-neutral-800/80';
              if (p.label === 1 && pred === 1) {
                outcomeBorder = 'border-cyan-400 ring-1 ring-cyan-500/40'; // TP
              } else if (p.label === 0 && pred === 1) {
                outcomeBorder = 'border-amber-500/80'; // FP
              } else if (p.label === 1 && pred === 0) {
                outcomeBorder = 'border-neutral-400'; // FN
              } else {
                outcomeBorder = 'border-neutral-900 opacity-70'; // TN
              }

              const isSelected = selectedPatch?.patch_idx === p.patch_idx;

              return (
                <button
                  key={p.patch_idx}
                  onClick={() => setSelectedPatch(p)}
                  onMouseEnter={(e) => handleMouseEnterPatch(e, p)}
                  onMouseLeave={handleMouseLeavePatch}
                  className={`relative aspect-square rounded overflow-hidden border transition ${outcomeBorder} ${
                    isSelected ? 'ring-2 ring-white scale-105 z-10' : 'hover:scale-105 hover:z-10'
                  }`}
                >
                  <img
                    src={`/${p.image_path}`}
                    alt={`Patch ${p.patch_idx}`}
                    className="w-full h-full object-cover"
                    loading="lazy"
                  />
                </button>
              );
            })}
          </div>
        </div>

        {/* Large Detail View (Right, 4 Cols) */}
        <div className="lg:col-span-4 p-5 rounded-lg border border-neutral-800 bg-neutral-900/50 space-y-5 sticky top-20">
          <div className="border-b border-neutral-800 pb-3 flex items-center justify-between">
            <div>
              <span className="font-mono text-[10px] text-neutral-500 uppercase block">SELECTED EXTENT</span>
              <h3 className="font-mono text-lg font-bold text-neutral-100">
                Patch #{selectedPatch.patch_idx}
              </h3>
            </div>
            <span
              className={`px-2 py-0.5 rounded text-[11px] font-mono font-bold ${
                selectedPatch.label === 1
                  ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
                  : 'bg-neutral-800 text-neutral-400'
              }`}
            >
              {selectedPatch.label === 1 ? 'GROUND TRUTH: ICE' : 'REGOLITH'}
            </span>
          </div>

          {/* Composite Display */}
          <div className="space-y-2">
            <div className="w-full aspect-square rounded border border-neutral-800 overflow-hidden bg-black">
              <img
                src={`/${selectedPatch.image_path}`}
                alt={`Patch ${selectedPatch.patch_idx}`}
                className="w-full h-full object-cover"
              />
            </div>
            <p className="text-[10px] font-mono text-neutral-500 text-center">
              HH (Red) · HV (Green) · VV (Blue) Log-Intensity Composite
            </p>
          </div>

          {/* Grad-CAM if available */}
          {gradcamMap.has(selectedPatch.patch_idx) && (
            <div className="space-y-1.5 pt-2 border-t border-neutral-800">
              <span className="font-mono text-[10px] text-neutral-400 uppercase block">Grad-CAM Overlay (CustomCNN seed 2024)</span>
              <div className="w-full aspect-[2/1] rounded border border-neutral-800 overflow-hidden bg-black">
                <img
                  src={`/${gradcamMap.get(selectedPatch.patch_idx)?.trainedImg}`}
                  alt={`Grad-CAM Patch ${selectedPatch.patch_idx}`}
                  className="w-full h-full object-contain"
                />
              </div>
            </div>
          )}

          {/* Extent Metrics */}
          <div className="grid grid-cols-2 gap-2 font-mono text-xs p-3 rounded bg-neutral-950 border border-neutral-800 text-neutral-300">
            <div>Coords: <span className="text-neutral-100">{selectedPatch.coords[0]}, {selectedPatch.coords[1]}</span></div>
            <div>Strip ID: <span className="text-neutral-100">Strip {selectedPatch.strip_id}</span></div>
            <div>Positive px: <span className={selectedPatch.pos_px >= 5 ? 'text-cyan-400 font-bold' : 'text-neutral-100'}>{selectedPatch.pos_px}/4096</span></div>
            <div>Ambiguous: <span className={selectedPatch.is_clean ? 'text-neutral-400' : 'text-amber-400'}>{selectedPatch.is_clean ? 'No' : 'Yes (1–4 px)'}</span></div>
            <div>Mean CPR: <span className="text-neutral-100">{selectedPatch.cpr_mean.toFixed(3)}</span></div>
            <div>Mean DOP: <span className="text-neutral-100">{selectedPatch.dop_mean.toFixed(3)}</span></div>
          </div>

          {/* Model Confidence Scores */}
          <div className="space-y-1.5 pt-2 border-t border-neutral-800 font-mono text-xs">
            <span className="text-[10px] text-neutral-500 uppercase block">MODEL PROBABILITIES</span>
            <div className="space-y-1.5 max-h-40 overflow-y-auto pr-1">
              {Object.entries(selectedPatch.scores).map(([name, score]) => (
                <div key={name} className="flex items-center justify-between p-1 rounded bg-neutral-950 text-[11px]">
                  <span className="text-neutral-400 truncate max-w-[150px]">{name}</span>
                  <span className={score >= threshold ? 'text-neutral-100 font-bold' : 'text-neutral-500'}>
                    {score.toFixed(4)}
                  </span>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>

      {/* Floating Hover Card */}
      {hoveredPatch && (
        <div
          className="fixed z-50 transition-opacity duration-150 hidden md:block"
          style={{
            left: `${Math.min(hoverPos.x, window.innerWidth - 280)}px`,
            top: `${Math.min(hoverPos.y, window.innerHeight - 340)}px`,
          }}
        >
          <PatchHoverCard patch={hoveredPatch} selectedModel={selectedModel} />
        </div>
      )}
    </div>
  );
}
