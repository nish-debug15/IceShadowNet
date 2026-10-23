'use client';

import React from 'react';
import { PatchRecord } from '../lib/data';

interface PatchHoverCardProps {
  patch: PatchRecord;
  selectedModel: string;
}

export function PatchHoverCard({ patch, selectedModel }: PatchHoverCardProps) {
  const score = patch.scores[selectedModel] ?? 0.5;
  const isPos = patch.label === 1;

  return (
    <div className="w-64 p-3 rounded-lg border border-neutral-700 bg-neutral-950/95 shadow-2xl backdrop-blur text-xs font-mono text-neutral-200 z-50 pointer-events-none">
      <div className="flex items-center justify-between mb-2 pb-1.5 border-b border-neutral-800">
        <span className="font-bold text-neutral-100">PATCH #{patch.patch_idx}</span>
        <span
          className={`px-1.5 py-0.5 rounded text-[10px] font-bold ${
            isPos
              ? 'bg-cyan-500/10 text-cyan-400 border border-cyan-500/30'
              : 'bg-neutral-800 text-neutral-400'
          }`}
        >
          {isPos ? 'CANDIDATE ICE (1)' : 'NOT CANDIDATE (0)'}
        </span>
      </div>

      <div className="w-full aspect-square rounded overflow-hidden border border-neutral-800 mb-2.5 bg-neutral-900">
        <img
          src={`/${patch.image_path}`}
          alt={`Patch ${patch.patch_idx}`}
          className="w-full h-full object-cover"
        />
      </div>

      <div className="grid grid-cols-2 gap-x-2 gap-y-1 text-[11px] text-neutral-400 mb-2">
        <div>Strip: <span className="text-neutral-200">{patch.strip_id}</span></div>
        <div>Coords: <span className="text-neutral-200">{patch.coords[0]},{patch.coords[1]}</span></div>
        <div>Ice px: <span className={patch.pos_px >= 5 ? 'text-cyan-400 font-bold' : 'text-neutral-200'}>{patch.pos_px}/4096</span></div>
        <div>CPR: <span className="text-neutral-200">{patch.cpr_mean.toFixed(2)}</span></div>
        <div>DOP: <span className="text-neutral-200">{patch.dop_mean.toFixed(2)}</span></div>
        <div>Clean: <span className={patch.is_clean ? 'text-neutral-200' : 'text-amber-400'}>{patch.is_clean ? 'Yes' : 'Ambiguous'}</span></div>
      </div>

      <div className="pt-2 border-t border-neutral-800 flex items-center justify-between text-[11px]">
        <span className="text-neutral-400 truncate max-w-[120px]" title={selectedModel}>
          {selectedModel}:
        </span>
        <span className="font-bold text-neutral-100">{score.toFixed(4)}</span>
      </div>
    </div>
  );
}
