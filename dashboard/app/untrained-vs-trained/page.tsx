'use client';

import { useEffect } from 'react';
import { useRouter } from 'next/navigation';

export default function UntrainedVsTrainedRedirect() {
  const router = useRouter();

  useEffect(() => {
    router.replace('/pipeline');
  }, [router]);

  return (
    <div className="py-16 text-center font-mono text-xs text-neutral-400 space-y-2">
      <p>Redirecting to How it works (Before vs After training slider)...</p>
      <p className="text-[11px] text-neutral-500">
        If you are not redirected automatically,{' '}
        <a href="/pipeline" className="text-cyan-400 hover:underline">
          click here
        </a>
        .
      </p>
    </div>
  );
}
