"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import type { ReactNode } from "react";
import { useApp, type Connection } from "@/app/providers";
import { API_BASE } from "@/lib/api";
import { Pill } from "./ui";

const REPO_URL = "https://github.com/syed-waleed-ahmed/Self-Correcting-RAG";

const NAV = [
  { href: "/", label: "Ask" },
  { href: "/documents/", label: "My documents" },
  { href: "/agent/", label: "Agent" },
  { href: "/about/", label: "How it works" },
];

function normalize(path: string): string {
  return path.replace(/\/+$/, "") || "/";
}

export function Shell({ children }: { children: ReactNode }) {
  const { connection, demo } = useApp();
  const pathname = normalize(usePathname() ?? "/");
  const budget = demo?.budget;

  return (
    <div className="mx-auto flex min-h-screen max-w-6xl flex-col px-4 py-6 sm:py-10">
      <header className="mb-6 flex flex-wrap items-start justify-between gap-4">
        <div>
          <Link href="/" className="text-2xl font-semibold tracking-tight sm:text-3xl">
            SCRAG
          </Link>
          <p className="mt-1 max-w-xl text-sm text-slate-600 dark:text-slate-400">
            Self-correcting retrieval-augmented generation: hybrid search, an LLM guardrail that filters
            context, and answers that are graded and corrected by an evaluator.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ConnectionPill connection={connection} />
          {demo && <Pill>{demo.model}</Pill>}
          {budget && (
            <Pill tone={budget.remaining === 0 || budget.client_remaining === 0 ? "amber" : "slate"}>
              <span title="Live AI runs left today across all visitors (free-tier budget), and your own share of them">
                {budget.remaining}/{budget.limit} live runs left today
                {budget.client_remaining !== undefined && ` · ${budget.client_remaining} for you`}
              </span>
            </Pill>
          )}
          <a href={REPO_URL} className="text-xs font-medium text-brand-600 hover:underline dark:text-brand-100">
            GitHub
          </a>
        </div>
      </header>

      <nav aria-label="Main" className="mb-6 flex flex-wrap gap-1 border-b border-slate-200 dark:border-slate-800">
        {NAV.map((item) => {
          const active = normalize(item.href) === pathname;
          return (
            <Link
              key={item.href}
              href={item.href}
              aria-current={active ? "page" : undefined}
              className={`-mb-px rounded-t-lg border-b-2 px-3 py-2 text-sm font-medium transition ${
                active
                  ? "border-brand-500 text-brand-700 dark:text-brand-100"
                  : "border-transparent text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-200"
              }`}
            >
              {item.label}
            </Link>
          );
        })}
      </nav>

      {connection === "waking" && (
        <Banner tone="amber">
          Waking the server. The free hosting tier sleeps after 15 minutes without visitors; the first request
          takes up to a minute.
        </Banner>
      )}
      {connection === "offline" && (
        <Banner tone="red">
          The API at <span className="font-mono">{API_BASE}</span> is not responding. Please try again later.
        </Banner>
      )}

      <main className="flex-1">{children}</main>

      <footer className="mt-12 border-t border-slate-200 pt-6 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        <p>
          Questions, uploads and answers are sent to the configured LLM provider (Groq). Do not enter personal
          data. Sessions and uploads are deleted after 24 hours. Live AI runs are limited per visitor and per day
          to stay within free tiers.{" "}
          <a href={`${API_BASE}/docs`} className="text-brand-600 hover:underline dark:text-brand-100">
            API docs
          </a>{" "}
          ·{" "}
          <a href={REPO_URL} className="text-brand-600 hover:underline dark:text-brand-100">
            Source (MIT)
          </a>
        </p>
      </footer>
    </div>
  );
}

export function Banner({ tone, children }: { tone: "amber" | "red" | "slate"; children: ReactNode }) {
  const tones = {
    amber: "border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200",
    red: "border-red-200 bg-red-50 text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200",
    slate: "border-slate-200 bg-white text-slate-700 dark:border-slate-800 dark:bg-slate-900 dark:text-slate-300",
  };
  return (
    <div role={tone === "red" ? "alert" : "status"} className={`mb-6 rounded-xl border p-4 text-sm ${tones[tone]}`}>
      {children}
    </div>
  );
}

function ConnectionPill({ connection }: { connection: Connection }) {
  const label = { connecting: "Connecting…", waking: "Waking server…", online: "API online", offline: "API offline" }[
    connection
  ];
  const dot = {
    connecting: "bg-slate-400",
    waking: "bg-amber-500 animate-pulse",
    online: "bg-emerald-500",
    offline: "bg-red-500",
  }[connection];
  return (
    <Pill>
      <span className={`h-2 w-2 rounded-full ${dot}`} aria-hidden="true" />
      {label}
    </Pill>
  );
}
