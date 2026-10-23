'use client';

import React, { useState } from 'react';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { Menu, X } from 'lucide-react';

const NAV_ITEMS = [
  { href: '/', label: 'Overview' },
  { href: '/pipeline/', label: 'How it works' },
  { href: '/results/', label: 'Results' },
  { href: '/patches/', label: 'Explore' },
  { href: '/failure-cases/', label: 'What went wrong' },
];

export function Navbar() {
  const pathname = usePathname();
  const [mobileOpen, setMobileOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-50 border-b border-neutral-800 bg-neutral-950/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between h-14">
          {/* Brand */}
          <div className="flex items-center space-x-3">
            <Link href="/" className="flex items-center space-x-2 group">
              <span className="font-mono text-xs font-semibold text-neutral-100 tracking-wider uppercase group-hover:text-cyan-400 transition">
                ICESHADOWNET
              </span>
              <span className="text-[10px] font-mono text-neutral-500 hidden sm:inline">
                // CH-2 DFSAR L1A
              </span>
            </Link>
          </div>

          {/* Desktop Nav Links */}
          <div className="hidden md:flex items-center space-x-1">
            {NAV_ITEMS.map((item) => {
              const active = pathname === item.href || (item.href !== '/' && pathname.startsWith(item.href));
              return (
                <Link
                  key={item.href}
                  href={item.href}
                  className={`px-3 py-1.5 text-xs font-mono transition rounded-sm ${
                    active
                      ? 'text-neutral-100 bg-neutral-900 border border-neutral-700/80 font-medium'
                      : 'text-neutral-400 hover:text-neutral-200 hover:bg-neutral-900/50'
                  }`}
                >
                  {item.label}
                </Link>
              );
            })}
          </div>

          {/* Provenance Tag */}
          <div className="hidden lg:flex items-center">
            <span className="font-mono text-[10px] text-neutral-400 border border-neutral-800 px-2 py-1 rounded bg-neutral-900/50">
              CH-2 DFSAR L1A · LINES 100,000–140,000
            </span>
          </div>

          {/* Mobile Menu Button */}
          <div className="md:hidden flex items-center">
            <button
              onClick={() => setMobileOpen(!mobileOpen)}
              className="p-1.5 rounded text-neutral-400 hover:text-neutral-200 bg-neutral-900 border border-neutral-800"
              aria-label="Toggle menu"
            >
              {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
            </button>
          </div>
        </div>
      </div>

      {/* Mobile Menu Dropdown */}
      {mobileOpen && (
        <div className="md:hidden border-b border-neutral-800 bg-neutral-950 px-4 py-3 space-y-1">
          {NAV_ITEMS.map((item) => {
            const active = pathname === item.href || (item.href !== '/' && pathname.startsWith(item.href));
            return (
              <Link
                key={item.href}
                href={item.href}
                onClick={() => setMobileOpen(false)}
                className={`block px-3 py-2 text-xs font-mono rounded ${
                  active
                    ? 'text-neutral-100 bg-neutral-900 border border-neutral-700'
                    : 'text-neutral-400 hover:text-neutral-200'
                }`}
              >
                {item.label}
              </Link>
            );
          })}
        </div>
      )}
    </nav>
  );
}
