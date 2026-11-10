"use client";

import type { PipelineEvent } from "@/lib/api";
import { Card, ScoreBar, Spinner } from "./ui";

interface Row {
  key: string;
  label: string;
  detail?: string;
  score?: number | null;
  state: "running" | "done" | "error";
}

/** Turn the pipeline's progress events into one row per stage/attempt. */
export function progressRows(events: PipelineEvent[]): Row[] {
  const rows: Row[] = [];
  const ms = (e: PipelineEvent) => (e.ms !== undefined ? `${Math.round(e.ms)} ms` : undefined);
  for (const e of events) {
    switch (e.stage) {
      case "cache":
        rows.push({ key: "cache", label: "Served a stored result of an earlier real run", state: "done" });
        break;
      case "retrieve":
        rows.push({
          key: "retrieve",
          label: `Retrieved ${e.count ?? 0} chunk${e.count === 1 ? "" : "s"}`,
          detail: ms(e),
          state: "done",
        });
        break;
      case "guardrail":
        rows.push({
          key: "guardrail",
          label: e.fallback
            ? `Guardrail kept none of ${e.of}; using raw retrieval`
            : `Guardrail kept ${e.kept} of ${e.of} chunks`,
          detail: ms(e),
          state: "done",
        });
        break;
      case "generate": {
        const key = `generate-${e.attempt}`;
        const row: Row = {
          key,
          label: `Attempt ${e.attempt}: ${e.status === "started" ? "writing an answer" : e.status === "error" ? "generation failed" : "answer written"}`,
          detail: ms(e),
          state: e.status === "started" ? "running" : e.status === "error" ? "error" : "done",
        };
        const at = rows.findIndex((r) => r.key === key);
        if (at === -1) rows.push(row);
        else rows[at] = row;
        break;
      }
      case "evaluate":
        rows.push({
          key: `evaluate-${e.attempt}`,
          label:
            e.status === "ok"
              ? `Attempt ${e.attempt}: graded ${e.passed ? "(passed)" : "(below threshold, correcting)"}`
              : `Attempt ${e.attempt}: evaluator ${e.status}`,
          detail: ms(e),
          score: e.score ?? null,
          state: e.status === "ok" ? "done" : "error",
        });
        break;
    }
  }
  return rows;
}

export function ProgressCard({ events, busy }: { events: PipelineEvent[]; busy: boolean }) {
  const rows = progressRows(events);
  const waiting = busy && (rows.length === 0 || rows.every((r) => r.state !== "running"));
  return (
    <Card title="Pipeline" aside={busy ? <Spinner label="Running" /> : <span className="text-xs text-slate-400">done</span>}>
      <ol className="space-y-2 text-sm" aria-live="polite">
        {rows.map((row) => (
          <li key={row.key} className="flex flex-wrap items-center gap-2">
            <span
              aria-hidden="true"
              className={`h-2 w-2 rounded-full ${
                row.state === "running" ? "animate-pulse bg-amber-500" : row.state === "error" ? "bg-red-500" : "bg-emerald-500"
              }`}
            />
            <span className="text-slate-700 dark:text-slate-200">{row.label}</span>
            {row.score !== undefined && <ScoreBar value={row.score} label="Evaluator score" />}
            {row.detail && <span className="ml-auto font-mono text-xs text-slate-400">{row.detail}</span>}
          </li>
        ))}
        {waiting && (
          <li className="flex items-center gap-2 text-slate-500">
            <Spinner label="Waiting" /> {rows.length === 0 ? "Starting…" : "Working…"}
          </li>
        )}
      </ol>
    </Card>
  );
}
