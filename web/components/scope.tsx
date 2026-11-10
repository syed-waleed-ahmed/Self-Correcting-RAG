"use client";

import Link from "next/link";
import type { Scope } from "@/lib/api";

const OPTIONS: { value: Scope; label: string; needsSession: boolean }[] = [
  { value: "samples", label: "Sample docs", needsSession: false },
  { value: "session", label: "My documents", needsSession: true },
  { value: "all", label: "Both", needsSession: true },
];

/** Which documents to search. Uploaded documents need a session (and a server with sessions). */
export function ScopeSelector({
  value,
  onChange,
  hasSession,
  sessionsAvailable,
  disabled = false,
}: {
  value: Scope;
  onChange: (scope: Scope) => void;
  hasSession: boolean;
  sessionsAvailable: boolean;
  disabled?: boolean;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2 text-xs">
      <span className="text-slate-500 dark:text-slate-400">Search</span>
      {OPTIONS.map((option) => {
        const unavailable = option.needsSession && (!sessionsAvailable || !hasSession);
        return (
          <button
            key={option.value}
            type="button"
            onClick={() => onChange(option.value)}
            aria-pressed={value === option.value}
            disabled={disabled || unavailable}
            title={unavailable ? "Upload documents first (My documents tab)" : undefined}
            className={`rounded-full border px-3 py-1 disabled:cursor-not-allowed disabled:opacity-40 ${
              value === option.value
                ? "border-brand-500 bg-brand-50 text-brand-700 dark:bg-slate-800 dark:text-brand-100"
                : "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-300"
            }`}
          >
            {option.label}
          </button>
        );
      })}
      {sessionsAvailable && !hasSession && (
        <Link href="/documents/" className="text-brand-600 hover:underline dark:text-brand-100">
          Upload your own documents →
        </Link>
      )}
    </div>
  );
}
