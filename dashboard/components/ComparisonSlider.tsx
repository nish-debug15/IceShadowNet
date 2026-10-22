'use client';

import React, { useState, useRef, useCallback } from 'react';

interface ComparisonSliderProps {
  beforeImage: string;
  afterImage: string;
  beforeLabel?: string;
  afterLabel?: string;
  className?: string;
}

export function ComparisonSlider({
  beforeImage,
  afterImage,
  beforeLabel = 'Untrained (Random Init)',
  afterLabel = 'Trained (CustomCNN Seed 2024)',
  className = '',
}: ComparisonSliderProps) {
  const [sliderPosition, setSliderPosition] = useState(50);
  const containerRef = useRef<HTMLDivElement>(null);
  const isDragging = useRef(false);

  const handleMove = useCallback((clientX: number) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const x = clientX - rect.left;
    const clampedPos = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPosition(clampedPos);
  }, []);

  const handleTouchMove = (e: React.TouchEvent) => {
    if (isDragging.current) {
      handleMove(e.touches[0].clientX);
    }
  };

  const handleMouseMove = (e: React.MouseEvent) => {
    if (isDragging.current) {
      handleMove(e.clientX);
    }
  };

  const handlePointerDown = () => {
    isDragging.current = true;
  };

  const handlePointerUp = () => {
    isDragging.current = false;
  };

  return (
    <div
      ref={containerRef}
      onMouseMove={handleMouseMove}
      onTouchMove={handleTouchMove}
      onMouseUp={handlePointerUp}
      onMouseLeave={handlePointerUp}
      onTouchEnd={handlePointerUp}
      className={`relative select-none overflow-hidden rounded-lg border border-neutral-800 bg-neutral-950 ${className}`}
    >
      {/* Container aspect ratio */}
      <div className="relative w-full aspect-[2/1] overflow-hidden">
        {/* Trained Image (Base layer - right side / revealed as slider moves left) */}
        <img
          src={afterImage}
          alt={afterLabel}
          className="absolute inset-0 w-full h-full object-contain pointer-events-none"
        />

        {/* Untrained Image (Clipped layer - left side) */}
        <div
          className="absolute inset-0 overflow-hidden"
          style={{ clipPath: `inset(0 ${100 - sliderPosition}% 0 0)` }}
        >
          <img
            src={beforeImage}
            alt={beforeLabel}
            className="absolute inset-0 w-full h-full object-contain pointer-events-none"
          />
        </div>

        {/* Divider Line and Handle */}
        <div
          className="absolute top-0 bottom-0 w-0.5 bg-neutral-100/80 cursor-ew-resize z-20"
          style={{ left: `${sliderPosition}%` }}
          onMouseDown={handlePointerDown}
          onTouchStart={handlePointerDown}
        >
          <div className="absolute top-1/2 -translate-y-1/2 -translate-x-1/2 w-7 h-7 rounded-full bg-neutral-900 border border-neutral-400 flex items-center justify-center shadow-lg">
            <div className="flex space-x-1 text-neutral-300 text-[10px] font-mono select-none">
              <span>‹</span>
              <span>›</span>
            </div>
          </div>
        </div>

        {/* Labels */}
        <div className="absolute top-3 left-3 z-10 pointer-events-none">
          <span className="font-mono text-[10px] uppercase tracking-wider px-2 py-1 bg-neutral-950/80 border border-neutral-800 text-neutral-300 rounded backdrop-blur-sm">
            {beforeLabel}
          </span>
        </div>
        <div className="absolute top-3 right-3 z-10 pointer-events-none">
          <span className="font-mono text-[10px] uppercase tracking-wider px-2 py-1 bg-neutral-950/80 border border-neutral-800 text-neutral-300 rounded backdrop-blur-sm">
            {afterLabel}
          </span>
        </div>
      </div>
    </div>
  );
}
