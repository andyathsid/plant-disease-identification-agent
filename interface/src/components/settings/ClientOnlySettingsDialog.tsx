'use client';

import { SettingsDialog } from './SettingsDialog';
import { useEffect, useState } from 'react';

export function ClientOnlySettingsDialog() {
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    setIsMounted(true);
  }, []);

  if (!isMounted) {
    // Render a placeholder during SSR to prevent hydration mismatch
    return (
      <button
        className="w-full justify-start gap-2 hover:bg-accent cursor-pointer inline-flex items-center whitespace-nowrap rounded-md text-sm font-medium transition-[color,box-shadow] disabled:pointer-events-none disabled:opacity-50 [&_svg]:pointer-events-none [&_svg:not([class*='size-'])]:size-4 [&_svg]:shrink-0 outline-none focus-visible:border-ring focus-visible:ring-ring/50 focus-visible:ring-[3px] aria-invalid:ring-destructive/20 dark:aria-invalid:ring-destructive/40 aria-invalid:border-destructive h-9 px-4 py-2 has-[>svg]:px-3"
        style={{
          backgroundColor: 'transparent',
          border: 'none',
          padding: '0',
          margin: '0',
          textAlign: 'left',
          width: '100%'
        }}
        aria-hidden="true"
      >
        <svg
          width="20"
          height="20"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="size-5"
        >
          <circle cx="12" cy="12" r="3"></circle>
          <path d="M12 1v6m0 6v6m11-7h-6m-6 0H1"></path>
        </svg>
        <span>Pengaturan</span>
      </button>
    );
  }

  return <SettingsDialog />;
}