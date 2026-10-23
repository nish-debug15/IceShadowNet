'use client';

import React, { useState } from 'react';

export const GLOSSARY: Record<string, string> = {
  'ROC-AUC': '0.5 = coin flip, 1.0 = perfect ranking across all decision thresholds.',
  'F1': 'Harmonic mean of precision and recall (1.0 = perfect detection balance).',
  'Threshold': 'Decision boundary probability: scores at or above this value predict candidate ice.',
  'Strip': 'Along-track radar orbital acquisition swath (independent spatial test block).',
  'CPR': 'Circular Polarization Ratio (high values indicate volume scattering from ice or rough surfaces).',
  'DOP': 'Degree of Polarization (low values indicate depolarized backscatter from multiple volume scattering; threshold <0.13).',
};

interface TooltipProps {
  term: keyof typeof GLOSSARY | string;
  customText?: string;
  className?: string;
}

export function Tooltip({ term, customText, className = '' }: TooltipProps) {
  const [isOpen, setIsOpen] = useState(false);
  const text = customText || GLOSSARY[term] || term;

  return (
    <span
      className={`relative inline-flex items-center ml-1 align-baseline select-none ${className}`}
      onMouseEnter={() => setIsOpen(true)}
      onMouseLeave={() => setIsOpen(false)}
      onClick={() => setIsOpen(!isOpen)}
    >
      <span
        aria-label={`Definition of ${term}`}
        className="w-3.5 h-3.5 rounded-full border border-neutral-600 bg-neutral-900 hover:border-neutral-400 hover:bg-neutral-800 text-[9px] font-mono font-medium text-neutral-400 inline-flex items-center justify-center cursor-help transition-colors"
      >
        ?
      </span>
      {isOpen && (
        <span className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-2 w-56 p-2 rounded bg-neutral-950 border border-neutral-700 shadow-xl text-[11px] font-sans font-normal text-neutral-200 leading-snug backdrop-blur-md pointer-events-none">
          <strong className="font-mono text-cyan-400 block mb-0.5">{term}</strong>
          {text}
        </span>
      )}
    </span>
  );
}
