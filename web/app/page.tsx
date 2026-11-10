"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { API_BASE, ApiError, api, type DemoInfo, type QueryResponse, type SearchResponse } from "@/lib/api";
import { AnswerCard, EvaluationCard, SourcesCard } from "@/components/results";
import { Card, Pill } from "@/components/ui";

type Connection = "connecting" | "waking" | "online" | "offline";
type Mode = "answer" | "search";

const REPO_URL = "https://github.com/syed-waleed-ahmed/Self-Correcting-RAG";

function describeError(err: unknown): string {
  if (!(err instanceof ApiError)) return "Something went wrong. Please try again.";
  if (err.code === "network_error") return "The API could not be reached. It may be waking up; try again in a minute.";
  const wait = err.retryAfter ? ` Try again in about ${Math.max(1, Math.ceil(err.retryAfter / 60))} min.` : "";
  return `${err.message}${err.status === 429 ? wait : ""}${err.requestId ? ` (request ${err.requestId.slice(0, 8)})` : ""}`;
}

export default function Home() {
  const [connection, setConnection] = useState<Connection>("connecting");
  const [demo, setDemo] = useState<DemoInfo | null>(null);
  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("answer");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [search, setSearch] = useState<SearchResponse | null>(null);
  const started = useRef(false);

  const refreshDemo = useCallback(async () => {
    try {
      setDemo(await api.demo());
    } catch {
      /* keep the last known info */
    }
  }, []);

  // Free-tier instances sleep when idle; poll /health until the API is awake.
  useEffect(() => {
    if (started.current) return;
    started.current = true;
    let cancelled = false;
    const deadline = Date.now() + 150_000;
    const poll = async (attempt: number) => {
      try {
        await api.health();
        if (cancelled) return;
        setConnection("online");
        void refreshDemo();
      } catch {
        if (cancelled) return;
        if (Date.now() > deadline) {
          setConnection("offline");
          return;
        }
        setConnection(attempt === 0 ? "connecting" : "waking");
        setTimeout(() => void poll(attempt + 1), 5_000);
      }
    };
    void poll(0);
    return () => {
      cancelled = true;
    };
  }, [refreshDemo]);

  const run = async (text: string, useCache = true) => {
    const query = text.trim();
    if (!query || busy) return;
    setQuestion(query);
    setBusy(true);
    setError(null);
    try {
      if (mode === "search") {
        setResult(null);
        setSearch(await api.search(query));
      } else {
        setSearch(null);
        setResult(await api.query(query, useCache));
        void refreshDemo();
      }
    } catch (err) {
      setError(describeError(err));
    } finally {
      setBusy(false);
    }
  };

  const maxChars = demo?.limits?.max_query_chars ?? 4000;
  const budget = demo?.budget;

  return (
    <main className="mx-auto max-w-5xl px-4 py-8 sm:py-12">
      <header className="mb-8 flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight sm:text-3xl">SCRAG</h1>
          <p className="mt-1 max-w-xl text-sm text-slate-600 dark:text-slate-400">
            Self-correcting retrieval-augmented generation: hybrid search, an LLM guardrail that
            filters context, and a generator whose answers are graded and corrected by an evaluator.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <ConnectionPill connection={connection} />
          {demo && <Pill>{demo.model}</Pill>}
          <a href={REPO_URL} className="text-xs font-medium text-brand-600 hover:underline dark:text-brand-100">
            Source on GitHub
          </a>
        </div>
      </header>

      {connection === "waking" && (
        <div className="mb-6 rounded-xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900 dark:border-amber-800 dark:bg-amber-950 dark:text-amber-200">
          Waking the server. The free hosting tier sleeps after 15 minutes without visitors; the
          first request takes up to a minute.
        </div>
      )}
      {connection === "offline" && (
        <div className="mb-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          The API at <span className="font-mono">{API_BASE}</span> is not responding. Please try
          again later.
        </div>
      )}

      <Card
        title="Ask the knowledge base"
        aside={
          budget ? (
            <span className="text-xs text-slate-500 dark:text-slate-400">
              Live AI queries left today: <span className="font-medium">{budget.remaining}</span> /{" "}
              {budget.limit}
            </span>
          ) : null
        }
      >
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void run(question);
          }}
          className="flex flex-col gap-3 sm:flex-row"
        >
          <label htmlFor="question" className="sr-only">
            Question
          </label>
          <input
            id="question"
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            maxLength={maxChars}
            placeholder="e.g. How does hybrid search combine keyword and vector results?"
            className="min-w-0 flex-1 rounded-xl border border-slate-300 bg-white px-4 py-2.5 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-100 dark:border-slate-700 dark:bg-slate-950 dark:focus:ring-brand-700"
          />
          <button
            type="submit"
            disabled={busy || !question.trim() || connection !== "online"}
            className="rounded-xl bg-brand-600 px-5 py-2.5 text-sm font-medium text-white transition hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {busy ? (mode === "search" ? "Searching…" : "Thinking…") : mode === "search" ? "Search" : "Ask"}
          </button>
        </form>

        <div className="mt-3 flex flex-wrap items-center gap-2 text-xs">
          <span className="text-slate-500 dark:text-slate-400">Mode</span>
          {(["answer", "search"] as const).map((m) => (
            <button
              key={m}
              type="button"
              onClick={() => setMode(m)}
              aria-pressed={mode === m}
              className={`rounded-full border px-3 py-1 ${
                mode === m
                  ? "border-brand-500 bg-brand-50 text-brand-700 dark:bg-slate-800 dark:text-brand-100"
                  : "border-slate-200 text-slate-600 dark:border-slate-700 dark:text-slate-300"
              }`}
            >
              {m === "answer" ? "Answer with self-check" : "Retrieval only (no AI)"}
            </button>
          ))}
        </div>

        {demo && demo.sample_questions.length > 0 && (
          <div className="mt-4">
            <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">Try a sample question</p>
            <div className="flex flex-wrap gap-2">
              {demo.sample_questions.map((q) => (
                <button
                  key={q}
                  type="button"
                  disabled={busy || connection !== "online"}
                  onClick={() => void run(q)}
                  className="rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-left text-xs text-slate-700 transition hover:border-brand-500 hover:text-brand-700 disabled:opacity-50 dark:border-slate-700 dark:bg-slate-950 dark:text-slate-300"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}
      </Card>

      {error && (
        <div role="alert" className="mt-6 rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}

      {result && (
        <div className="mt-6 grid gap-6 lg:grid-cols-[3fr_2fr]">
          <div className="space-y-6">
            <AnswerCard result={result} />
            {result.cached && (
              <button
                type="button"
                onClick={() => void run(result.query, false)}
                disabled={busy}
                className="rounded-xl border border-brand-500 px-4 py-2 text-sm font-medium text-brand-700 hover:bg-brand-50 disabled:opacity-50 dark:text-brand-100 dark:hover:bg-slate-900"
              >
                Run live (uses 1 of today&apos;s queries)
              </button>
            )}
            <SourcesCard chunks={result.used_chunks} />
          </div>
          <EvaluationCard result={result} />
        </div>
      )}

      {search && (
        <div className="mt-6">
          <SourcesCard chunks={search.chunks} title={`Retrieved (${search.retrieval_mode})`} />
        </div>
      )}

      <footer className="mt-12 border-t border-slate-200 pt-6 text-xs text-slate-500 dark:border-slate-800 dark:text-slate-400">
        <p>
          Questions and answers are sent to the configured LLM provider (Groq). Do not enter
          personal data. Live AI queries are limited per visitor and per day to stay within free
          tiers.{" "}
          <a href={`${API_BASE}/docs`} className="text-brand-600 hover:underline dark:text-brand-100">
            API docs
          </a>
        </p>
      </footer>
    </main>
  );
}

function ConnectionPill({ connection }: { connection: Connection }) {
  const label = {
    connecting: "Connecting…",
    waking: "Waking server…",
    online: "API online",
    offline: "API offline",
  }[connection];
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
