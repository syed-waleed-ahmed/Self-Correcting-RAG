import type { ReactNode } from "react";
import type { ResultStatus } from "@/lib/api";

export function Card({ title, aside, children }: { title: string; aside?: ReactNode; children: ReactNode }) {
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
      <div className="mb-3 flex items-center justify-between gap-3">
        <h2 className="text-sm font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
          {title}
        </h2>
        {aside}
      </div>
      {children}
    </section>
  );
}

const STATUS_STYLE: Record<ResultStatus, { label: string; className: string }> = {
  passed: {
    label: "Passed evaluation",
    className: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
  },
  below_threshold: {
    label: "Below threshold",
    className: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  },
  unavailable: {
    label: "Not evaluated",
    className: "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  },
  skipped: {
    label: "No context found",
    className: "bg-slate-200 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
  },
};

export function StatusBadge({ status }: { status: ResultStatus }) {
  const style = STATUS_STYLE[status];
  return (
    <span className={`rounded-full px-2.5 py-1 text-xs font-medium ${style.className}`}>
      {style.label}
    </span>
  );
}

export function ScoreBar({ value, label }: { value: number | null; label: string }) {
  const pct = value === null ? 0 : Math.round(value * 100);
  return (
    <div className="flex items-center gap-2" title={label}>
      <div className="h-1.5 w-20 overflow-hidden rounded-full bg-slate-200 dark:bg-slate-800">
        <div
          className="h-full rounded-full bg-brand-500"
          style={{ width: `${pct}%` }}
          aria-hidden="true"
        />
      </div>
      <span className="w-10 font-mono text-xs tabular-nums text-slate-600 dark:text-slate-300">
        {value === null ? "n/a" : value.toFixed(2)}
      </span>
    </div>
  );
}

export function Pill({ children, tone = "slate" }: { children: ReactNode; tone?: "slate" | "brand" | "amber" }) {
  const tones = {
    slate: "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-300",
    brand: "border-brand-100 bg-brand-50 text-brand-700 dark:border-brand-700 dark:bg-slate-900 dark:text-brand-100",
    amber: "border-amber-200 bg-amber-50 text-amber-800 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200",
  };
  return <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs ${tones[tone]}`}>{children}</span>;
}

export function formatDate(iso: string | null): string {
  if (!iso) return "";
  const date = new Date(iso);
  return Number.isNaN(date.getTime())
    ? iso
    : date.toLocaleString(undefined, { dateStyle: "medium", timeStyle: "short" });
}
