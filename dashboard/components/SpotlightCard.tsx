'use client';

import React, { useRef, useState } from 'react';

interface SpotlightCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  provenance?: string;
  highlight?: boolean;
  className?: string;
  children?: React.ReactNode;
}

export function SpotlightCard({
  label,
  value,
  subValue,
  provenance,
  highlight = false,
  className = '',
  children,
}: SpotlightCardProps) {
  const divRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState<{ x: number; y: number }>({ x: 0, y: 0 });
  const [opacity, setOpacity] = useState<number>(0);

  const handleMouseMove = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!divRef.current) return;
    const rect = divRef.current.getBoundingClientRect();
    setPosition({ x: e.clientX - rect.left, y: e.clientY - rect.top });
  };

  const handleMouseEnter = () => setOpacity(1);
  const handleMouseLeave = () => setOpacity(0);

  return (
    <div
      ref={divRef}
      onMouseMove={handleMouseMove}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      className={`relative overflow-hidden rounded-lg border bg-neutral-950 p-6 transition-all duration-300 ${
        highlight
          ? 'border-neutral-700 hover:border-neutral-500'
          : 'border-neutral-800 hover:border-neutral-700'
      } ${className}`}
    >
      {/* Subtle radial spotlight overlay (max 1 effect) */}
      <div
        className="pointer-events-none absolute -inset-px opacity-0 transition-opacity duration-300"
        style={{
          opacity,
          background: `radial-gradient(400px circle at ${position.x}px ${position.y}px, rgba(255,255,255,0.04), transparent 80%)`,
        }}
      />

      <div className="relative z-10 flex flex-col justify-between h-full">
        <div>
          <div className="flex items-center justify-between gap-2 mb-2">
            <span className="font-mono text-[11px] uppercase tracking-wider text-neutral-400">
              {label}
            </span>
            {provenance && (
              <span className="font-mono text-[9px] text-neutral-500 bg-neutral-900 border border-neutral-800 px-1.5 py-0.5 rounded">
                {provenance}
              </span>
            )}
          </div>

          <div className="numeral text-4xl sm:text-5xl font-light tracking-tight text-neutral-100">
            {value}
          </div>

          {subValue && (
            <div className="mt-1 font-mono text-xs text-neutral-400">
              {subValue}
            </div>
          )}
        </div>

        {children && <div className="mt-4 pt-3 border-t border-neutral-800/80">{children}</div>}
      </div>
    </div>
  );
}
