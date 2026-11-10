"use client";

import { useState } from "react";
import type { Attempt, ChunkView, QueryResponse } from "@/lib/api";
import { Markdown } from "./markdown";
import { Card, Pill, ScoreBar, StatusBadge, formatDate } from "./ui";

const SCOPE_LABEL = { samples: "sample docs", session: "your documents", all: "your documents + sample docs" } as const;

export function AnswerCard({ result }: { result: QueryResponse }) {
  return (
    <Card
      title="Answer"
      aside={
        result.cached ? (
          <Pill tone="amber">
            {result.cache_source === "sample" ? "Recorded run" : "Cached run"} ·{" "}
            {formatDate(result.cached_at)}
          </Pill>
        ) : (
          <Pill tone="brand">Live run</Pill>
        )
      }
    >
      <Markdown text={result.answer} />
      <p className="mt-4 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-500 dark:text-slate-400">
        <span>Searched {SCOPE_LABEL[result.scope ?? "samples"]}</span>
        {result.history_saved === true && <span>· Saved to your history</span>}
        {result.history_saved === false && <span>· Not saved to history (storage unavailable or full)</span>}
      </p>
      {result.cached && (
        <p className="mt-4 text-xs text-slate-500 dark:text-slate-400">
          This is the stored result of a real earlier execution, shown to save the free-tier
          AI quota. Use <span className="font-medium">Run live</span> to execute it again.
        </p>
      )}
    </Card>
  );
}

export function EvaluationCard({ result }: { result: QueryResponse }) {
  return (
    <Card title="Self-check" aside={<StatusBadge status={result.evaluation_status} />}>
      <div className="flex flex-wrap items-center gap-x-6 gap-y-2 text-sm">
        <div className="flex items-center gap-2">
          <span className="text-slate-500 dark:text-slate-400">Grounding score</span>
          <ScoreBar value={result.evaluation_score} label="Evaluator score for the returned answer" />
        </div>
        <span className="text-slate-500 dark:text-slate-400">
          Attempts <span className="font-medium text-slate-800 dark:text-slate-200">{result.attempts}</span>
          {result.attempts > 1 && <> · returned #{result.best_attempt}</>}
        </span>
      </div>
      {result.evaluation_explanation && (
        <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">{result.evaluation_explanation}</p>
      )}
      <p className="mt-3 text-xs text-slate-500 dark:text-slate-400">
        Scored by an LLM judge against the retrieved context. It is a heuristic signal, not
        verification.
      </p>
      <RunCost result={result} />
      {result.attempt_history.length > 0 && (
        <ol className="mt-4 space-y-2 border-l border-slate-200 pl-4 dark:border-slate-800">
          {result.attempt_history.map((a) => (
            <AttemptItem key={a.number} attempt={a} returned={a.number === result.best_attempt} />
          ))}
        </ol>
      )}
    </Card>
  );
}

const STAGES = ["retrieve", "guardrail", "generate", "evaluate"] as const;

/** What producing this answer cost: LLM calls, tokens and time (for a recorded run: the original run). */
function RunCost({ result }: { result: QueryResponse }) {
  const usage = result.usage;
  const timings = result.timings_ms;
  if (!usage && !timings) return null;
  const parts: string[] = [];
  if (usage) {
    parts.push(`${usage.llm_calls} LLM call${usage.llm_calls === 1 ? "" : "s"}`);
    parts.push(`${usage.total_tokens.toLocaleString()} tokens`);
    if (usage.estimated_cost_usd !== null) parts.push(`≈ ${usage.estimated_cost_usd.toFixed(5)}`);
  }
  if (timings?.total !== undefined) parts.push(`${(timings.total / 1000).toFixed(1)} s`);
  const breakdown = timings
    ? STAGES.filter((s) => timings[s] !== undefined)
        .map((s) => `${s} ${Math.round(timings[s]!)} ms`)
        .join(" · ")
    : "";
  return (
    <p className="mt-3 text-xs text-slate-500 dark:text-slate-400" title={breakdown || undefined}>
      <span className="font-medium">{result.cached ? "Original run" : "This run"}:</span> {parts.join(" · ")}
      {breakdown && <span className="block font-mono text-[0.7rem] text-slate-400">{breakdown}</span>}
    </p>
  );
}

function AttemptItem({ attempt, returned }: { attempt: Attempt; returned: boolean }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="relative">
      <span
        className={`absolute -left-[21px] top-1.5 h-2.5 w-2.5 rounded-full ${
          returned ? "bg-brand-500" : "bg-slate-300 dark:bg-slate-600"
        }`}
        aria-hidden="true"
      />
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full flex-wrap items-center gap-3 text-left text-sm"
        aria-expanded={open}
      >
        <span className="font-medium">Attempt {attempt.number}</span>
        <ScoreBar value={attempt.score} label="Evaluator score for this attempt" />
        {attempt.evaluation_status !== "ok" && <Pill>evaluator {attempt.evaluation_status}</Pill>}
        {returned && <Pill tone="brand">returned</Pill>}
        <span className="ml-auto text-xs text-slate-400">{open ? "Hide" : "Show"}</span>
      </button>
      {open && (
        <div className="mt-2 space-y-2 rounded-lg bg-slate-50 p-3 text-sm dark:bg-slate-950">
          <Markdown text={attempt.answer} />
          {attempt.explanation && (
            <p className="text-xs text-slate-500 dark:text-slate-400">
              <span className="font-medium">Evaluator:</span> {attempt.explanation}
            </p>
          )}
        </div>
      )}
    </li>
  );
}

export function SourcesCard({ chunks, title = "Sources" }: { chunks: ChunkView[]; title?: string }) {
  return (
    <Card title={title} aside={<span className="text-xs text-slate-400">{chunks.length} chunks</span>}>
      {chunks.length === 0 ? (
        <p className="text-sm text-slate-500">No chunks were retrieved.</p>
      ) : (
        <ul className="space-y-3">
          {chunks.map((chunk, i) => (
            <SourceItem key={`${chunk.source}-${i}`} chunk={chunk} rank={i + 1} />
          ))}
        </ul>
      )}
    </Card>
  );
}

function SourceItem({ chunk, rank }: { chunk: ChunkView; rank: number }) {
  const [open, setOpen] = useState(false);
  // Source files hard-wrap lines; keep paragraph breaks but join wrapped lines.
  const text = chunk.text.replace(/([^\n])\n(?!\n)/g, "$1 ");
  const preview = text.length > 220 && !open ? `${text.slice(0, 220)}…` : text;
  return (
    <li className="rounded-xl border border-slate-200 p-3 dark:border-slate-800">
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <span className="font-mono text-slate-400">#{rank}</span>
        <span className="font-medium text-slate-700 dark:text-slate-200">{chunk.source}</span>
        <span className="ml-auto flex flex-wrap gap-1.5">
          <Pill>cosine {chunk.score.toFixed(2)}</Pill>
          {chunk.keyword_score !== null && <Pill>BM25 {chunk.keyword_score.toFixed(2)}</Pill>}
          {chunk.fusion_score !== null && <Pill>RRF {chunk.fusion_score.toFixed(4)}</Pill>}
          <Pill tone={chunk.guardrail_score === null ? "slate" : "brand"}>
            guardrail {chunk.guardrail_score === null ? "n/a" : chunk.guardrail_score.toFixed(2)}
          </Pill>
        </span>
      </div>
      <p className="mt-2 whitespace-pre-line text-sm text-slate-600 dark:text-slate-300">{preview}</p>
      {text.length > 220 && (
        <button
          type="button"
          onClick={() => setOpen((v) => !v)}
          className="mt-1 text-xs font-medium text-brand-600 hover:underline dark:text-brand-100"
        >
          {open ? "Show less" : "Show full chunk"}
        </button>
      )}
    </li>
  );
}
