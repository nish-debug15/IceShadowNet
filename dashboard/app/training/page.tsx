'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function TrainingRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/results');
  }, [router]);

  return (
    <div className="py-16 text-center font-mono text-xs text-neutral-400 space-y-2">
      <p>Redirecting to Results (Training details)...</p>
      <p className="text-[11px] text-neutral-500">
        If you are not redirected automatically,{' '}
        <a href="/results" className="text-cyan-400 hover:underline">
          click here
        </a>
        .
      </p>
    </div>
  );
}
