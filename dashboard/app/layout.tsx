import React from 'react';
import './globals.css';
import { Navbar } from '../components/Navbar';

export const metadata = {
  title: 'IceShadowNet — Chandrayaan-2 DFSAR Lunar Polar Benchmark',
  description: 'Deep Learning vs Classical Baselines for Lunar Polar Subsurface Ice Detection on Chandrayaan-2 DFSAR Quad-Pol Radar Imagery',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col font-sans selection:bg-cyan-500 selection:text-black">
        <Navbar />
        <main className="flex-1 w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
          {children}
        </main>
        <footer className="border-t border-neutral-800/80 bg-neutral-950 py-8 text-xs font-mono text-neutral-500">
          <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
            <div>
              <p className="text-neutral-400">
                ICESHADOWNET // CHANDRAYAAN-2 DFSAR REAL BENCHMARK
              </p>
              <p className="text-neutral-500 text-[11px] mt-1">
                Data Provenance: ISRO Chandrayaan-2 DFSAR L1A quad-pol SLI (lines 100,000–140,000). Model inputs strictly raw radar log-intensities [HH, HV, VV].
              </p>
            </div>
            <div className="text-[11px] text-neutral-400 text-right">
              <span>N=338 SPATIAL TEST SPLIT · ZERO COORDINATE LEAKAGE</span>
            </div>
          </div>
        </footer>
      </body>
    </html>
  );
}
