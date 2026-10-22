'use client';

import React, { useState } from 'react';
import { getTrainingCurvesData } from '../../lib/data';

interface CurvePoint {
  epoch: number;
  train_loss: number;
  val_loss: number;
  val_auc: number;
  val_f1: number;
}

interface RunCurve {
  model: string;
  seed: number;
  best_epoch: number;
  logged_epochs: number[];
  curve: CurvePoint[];
}

export default function TrainingPage() {
  const curvesData = getTrainingCurvesData();
  const allRuns: RunCurve[] = curvesData.runs || [];

  // Grouping for pretrained 3-column comparison
  const pretrainedOrig = allRuns.filter(r => r.model === 'ResNet_pretrained');
  const pretrainedV2a = allRuns.filter(r => r.model === 'ResNet_pretrained_v2a');
  const pretrainedV2b = allRuns.filter(r => r.model === 'ResNet_pretrained_v2b');

  return (
    <div className="space-y-12 py-4">
      {/* Header */}
      <div className="border-b border-neutral-800 pb-4">
        <span className="font-mono text-xs text-neutral-400 uppercase tracking-wider block mb-1">
          Optimization Dynamics // 15 Real Runs
        </span>
        <h1 className="text-3xl font-light text-neutral-100 tracking-tight">
          Training Curves & Checkpoint Trajectories
        </h1>
        <p className="text-sm text-neutral-400 mt-1 max-w-3xl font-sans">
          Logged epoch checkpoints (epoch 1 and every 5 epochs) with early stopping markers (patience=5 on Val ROC-AUC). Values are exact un-interpolated records.
        </p>
      </div>

      {/* 1. Pretrained 3-Column Comparison: Original vs v2a vs v2b */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
          <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
            Pretrained ResNet Adaptation: 3-Column Architectural Comparison
          </h2>
          <span className="font-mono text-[11px] text-neutral-500">Seeds 42, 123, 2024</span>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Column 1: Original */}
          <div className="p-4 rounded-lg border border-neutral-800 bg-neutral-900/40 space-y-4">
            <div className="border-b border-neutral-800 pb-2">
              <span className="font-mono text-[10px] text-neutral-500 block uppercase">VARIANT 1</span>
              <h3 className="font-mono text-sm font-bold text-neutral-200">ResNet_pretrained</h3>
              <p className="text-[11px] text-neutral-400 font-sans mt-0.5">
                Frozen backbone (bn1, layer1, layer2). 10.50M trainable params.
              </p>
            </div>
            <div className="space-y-3">
              {pretrainedOrig.map(run => (
                <SmallMultipleChart key={run.seed} run={run} />
              ))}
            </div>
          </div>

          {/* Column 2: v2a */}
          <div className="p-4 rounded-lg border border-neutral-800 bg-neutral-900/40 space-y-4">
            <div className="border-b border-neutral-800 pb-2">
              <span className="font-mono text-[10px] text-neutral-500 block uppercase">VARIANT 2A</span>
              <h3 className="font-mono text-sm font-bold text-neutral-200">ResNet_pretrained_v2a</h3>
              <p className="text-[11px] text-neutral-400 font-sans mt-0.5">
                Fully fine-tuned, unified learning rate lr = 1e-3.
              </p>
            </div>
            <div className="space-y-3">
              {pretrainedV2a.map(run => (
                <SmallMultipleChart key={run.seed} run={run} />
              ))}
            </div>
          </div>

          {/* Column 3: v2b */}
          <div className="p-4 rounded-lg border border-neutral-800 bg-neutral-900/40 space-y-4">
            <div className="border-b border-neutral-800 pb-2">
              <span className="font-mono text-[10px] text-neutral-500 block uppercase">VARIANT 2B</span>
              <h3 className="font-mono text-sm font-bold text-neutral-200">ResNet_pretrained_v2b</h3>
              <p className="text-[11px] text-neutral-400 font-sans mt-0.5">
                Differential learning rate (backbone lr = 1e-4, head lr = 1e-3).
              </p>
            </div>
            <div className="space-y-3">
              {pretrainedV2b.map(run => (
                <SmallMultipleChart key={run.seed} run={run} />
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* 2. Full 15-Run Small Multiples Grid */}
      <section className="space-y-4">
        <div className="flex items-center justify-between border-b border-neutral-800 pb-2">
          <h2 className="font-mono text-xs text-neutral-300 uppercase tracking-wider font-semibold">
            All 15 Training Runs (CustomCNN, ResNet_scratch, Pretrained Variants)
          </h2>
          <div className="flex items-center space-x-3 text-[10px] font-mono text-neutral-400">
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-neutral-300"></span>
              <span>Val AUC</span>
            </span>
            <span className="flex items-center space-x-1">
              <span className="w-2 h-2 rounded-full bg-cyan-400"></span>
              <span>Best Epoch Checkpoint</span>
            </span>
          </div>
        </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {allRuns.map(run => (
            <div key={`${run.model}_${run.seed}`} className="p-3.5 rounded-lg border border-neutral-800 bg-neutral-900/30 space-y-2">
              <div className="flex items-center justify-between font-mono text-xs">
                <span className="font-bold text-neutral-200">{run.model}</span>
                <span className="text-neutral-500">Seed {run.seed}</span>
              </div>
              <SmallMultipleChart run={run} height={140} />
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

function SmallMultipleChart({ run, height = 120 }: { run: RunCurve; height?: number }) {
  const curve = run.curve || [];
  if (curve.length === 0) return null;

  const width = 280;
  const padding = { top: 18, right: 20, bottom: 24, left: 36 };
  const innerW = width - padding.left - padding.right;
  const innerH = height - padding.top - padding.bottom;

  const minEpoch = 1;
  const maxEpoch = 15;
  const minY = 0.5;
  const maxY = 0.8;

  const getX = (ep: number) => padding.left + ((ep - minEpoch) / (maxEpoch - minEpoch)) * innerW;
  const getY = (v: number) => padding.top + innerH - ((v - minY) / (maxY - minY)) * innerH;

  const stopEpoch = curve[curve.length - 1]?.epoch;
  const bestPoint = curve.find(p => p.epoch === run.best_epoch);

  return (
    <div className="w-full bg-neutral-950 p-2 rounded border border-neutral-800/80">
      <div className="flex items-center justify-between text-[10px] font-mono text-neutral-400 mb-1">
        <span>Best Ep {run.best_epoch} (AUC: {bestPoint ? bestPoint.val_auc.toFixed(4) : '—'})</span>
        <span>Stopped Ep {stopEpoch}</span>
      </div>

      <svg viewBox={`0 0 ${width} ${height}`} className="w-full" style={{ height: `${height}px` }}>
        {/* Grid lines */}
        {[0.5, 0.65, 0.8].map((yVal) => {
          const y = getY(yVal);
          return (
            <g key={yVal}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#1e1e24" strokeDasharray="2 2" />
              <text x={padding.left - 6} y={y + 3} fill="#52525b" fontSize="8" textAnchor="end" fontFamily="monospace">
                {yVal.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* Trajectory line connecting markers */}
        {curve.length > 1 && (
          <polyline
            fill="none"
            stroke="#52525b"
            strokeWidth="1.2"
            strokeDasharray="3 3"
            points={curve.map(d => `${getX(d.epoch)},${getY(d.val_auc)}`).join(' ')}
          />
        )}

        {/* Markers only at logged epochs */}
        {curve.map((d) => {
          const isBest = d.epoch === run.best_epoch;
          return (
            <g key={d.epoch}>
              <circle
                cx={getX(d.epoch)}
                cy={getY(d.val_auc)}
                r={isBest ? 4.5 : 3}
                fill={isBest ? '#22d3ee' : '#27272a'}
                stroke={isBest ? '#0891b2' : '#71717a'}
                strokeWidth={isBest ? 1.5 : 1}
              />
              <text
                x={getX(d.epoch)}
                y={height - padding.bottom + 14}
                fill="#52525b"
                fontSize="8"
                textAnchor="middle"
                fontFamily="monospace"
              >
                {d.epoch}
              </text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
