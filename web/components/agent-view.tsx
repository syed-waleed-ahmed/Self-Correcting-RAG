"use client";

import { useState } from "react";
import { useApp } from "@/app/providers";
import { evidenceFromSteps } from "@/lib/agent";
import { api, describeError, type AgentRunResult, type Scope, type TraceEvent } from "@/lib/api";
import { Markdown } from "./markdown";
import { ScopeSelector } from "./scope";
import { Card, Pill, ScoreBar, Spinner, buttonClass } from "./ui";

const EXAMPLES = [
  "What RRF constant does hybrid retrieval use, and what is that constant multiplied by 3?",
  "What does the guardrail agent check, and what does the evaluator agent check?",
  "How is a sanitized API error response structured, and why is the request ID included?",
];

const STAGE_LABEL: Record<TraceEvent["stage"], string> = {
  plan: "Plan",
  approval: "Approval",
  tool: "Tool",
  synthesize: "Write",
  verify: "Verify",
  limit: "Limit",
};

export function AgentView() {
  const { connection, demo, refreshDemo, session, handleSessionError } = useApp();
  const [task, setTask] = useState("");
  const [scope, setScope] = useState<Scope>("samples");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [trace, setTrace] = useState<TraceEvent[]>([]);
  const [result, setResult] = useState<AgentRunResult | null>(null);
  const token = session?.token ?? null;
  const available = demo?.features?.agent ?? true;

  const run = async (text: string) => {
    const value = text.trim();
    if (!value || busy) return;
    setTask(value);
    setBusy(true);
    setError(null);
    setTrace([]);
    setResult(null);
    try {
      setResult(
        await api.agentStream(
          { task: value, scope },
          { token, onProgress: (event) => setTrace((prev) => [...prev, event]) },
        ),
      );
    } catch (err) {
      handleSessionError(err);
      setError(describeError(err));
    } finally {
      setBusy(false);
      void refreshDemo();
    }
  };

  const online = connection === "online";
  return (
    <div className="space-y-6">
      <Card title="Agent workspace" aside={<Pill>uses 2 of today&apos;s live runs</Pill>}>
        <p className="mb-4 max-w-3xl text-sm text-slate-600 dark:text-slate-300">
          A bounded agent: a planner picks tools (document search, a safe calculator), the tools run in parallel, a
          writer answers only from the tools&apos; evidence with citations, and the evaluator verifies the answer,
          allowing one correction. Hard limits apply to steps, LLM calls and time, and retrieved text is treated as
          data, never as instructions.
        </p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void run(task);
          }}
          className="flex flex-col gap-3 sm:flex-row"
        >
          <label htmlFor="task" className="sr-only">
            Task
          </label>
          <input
            id="task"
            value={task}
            onChange={(e) => setTask(e.target.value)}
            maxLength={demo?.limits?.max_query_chars ?? 1000}
            placeholder="Give the agent a task that needs looking things up (and maybe some arithmetic)"
            className={buttonClass.input}
          />
          <button type="submit" disabled={busy || !task.trim() || !online || !available} className={buttonClass.primary}>
            {busy ? "Running…" : "Run agent"}
          </button>
        </form>
        <div className="mt-3">
          <ScopeSelector
            value={scope}
            onChange={setScope}
            hasSession={token !== null}
            sessionsAvailable={demo?.features?.sessions ?? false}
            disabled={busy}
          />
        </div>
        <div className="mt-4 flex flex-wrap gap-2">
          {EXAMPLES.map((example) => (
            <button key={example} type="button" disabled={busy || !online} onClick={() => void run(example)} className={buttonClass.chip}>
              {example}
            </button>
          ))}
        </div>
      </Card>

      {error && (
        <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
          {error}
        </div>
      )}

      {(busy || trace.length > 0) && <TraceCard trace={trace} busy={busy} />}
      {result && <AgentResult result={result} />}
    </div>
  );
}

function TraceCard({ trace, busy }: { trace: TraceEvent[]; busy: boolean }) {
  return (
    <Card title="Trace" aside={busy ? <Spinner label="Running" /> : <span className="text-xs text-slate-400">{trace.length} events</span>}>
      <ol className="space-y-1.5 text-sm" aria-live="polite">
        {trace.map((event) => (
          <li key={event.seq} className="flex flex-wrap items-center gap-2">
            <span className="w-16 text-xs font-medium uppercase tracking-wide text-slate-400">{STAGE_LABEL[event.stage]}</span>
            <span className="font-mono text-xs text-slate-700 dark:text-slate-200">{event.name}</span>
            <Pill tone={event.status === "ok" ? "brand" : event.status === "fallback" ? "amber" : "slate"}>{event.status}</Pill>
            {event.detail && <span className="text-xs text-slate-500">{event.detail}</span>}
            <span className="ml-auto font-mono text-xs text-slate-400">{Math.round(event.latency_ms)} ms</span>
          </li>
        ))}
        {busy && (
          <li className="flex items-center gap-2 text-slate-500">
            <Spinner label="Waiting" /> {trace.length === 0 ? "Planning…" : "Working…"}
          </li>
        )}
      </ol>
    </Card>
  );
}

function AgentResult({ result }: { result: AgentRunResult }) {
  const evidence = evidenceFromSteps(result.steps);
  const v = result.verification;
  return (
    <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
      <div className="min-w-0 space-y-6">
        <Card
          title="Answer"
          aside={
            <Pill tone={result.status === "completed" ? "brand" : "amber"}>
              {result.status === "completed" ? "completed" : result.status.replace("_", " ")}
            </Pill>
          }
        >
          {result.answer ? (
            <Markdown text={result.answer} />
          ) : (
            <p className="text-sm text-slate-600 dark:text-slate-300">
              {result.status === "awaiting_approval"
                ? `The plan needs a sensitive tool (${result.pending_approval.map((s) => s.tool).join(", ")}), which only the operator can approve. Nothing ran.`
                : (result.error ?? "No answer was produced.")}
            </p>
          )}
        </Card>

        <Card title="Plan and tool calls" aside={<Pill>{result.plan_source === "planner" ? "planned by the LLM" : "fallback plan"}</Pill>}>
          <ol className="space-y-3">
            {result.plan.map((step, i) => {
              const executed = result.steps[i];
              return (
                <li key={i} className="rounded-xl border border-slate-200 p-3 text-sm dark:border-slate-800">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="font-mono text-xs text-slate-400">#{i + 1}</span>
                    <span className="font-medium">{step.tool}</span>
                    {executed && (
                      <>
                        <Pill tone={executed.call.status === "ok" ? "brand" : "amber"}>{executed.call.status}</Pill>
                        <span className="ml-auto font-mono text-xs text-slate-400">{Math.round(executed.call.latency_ms)} ms</span>
                      </>
                    )}
                  </div>
                  {step.purpose && <p className="mt-1 text-xs text-slate-500">{step.purpose}</p>}
                  <pre className="mt-2 overflow-x-auto rounded-lg bg-slate-50 p-2 font-mono text-xs text-slate-700 dark:bg-slate-950 dark:text-slate-300">
                    {JSON.stringify(step.input)}
                  </pre>
                  {executed?.call.error && <p className="mt-1 text-xs text-red-600">{executed.call.error}</p>}
                </li>
              );
            })}
          </ol>
        </Card>

        {evidence.length > 0 && (
          <Card title="Evidence" aside={<span className="text-xs text-slate-400">cited as S1, S2, …</span>}>
            <ul className="space-y-2">
              {evidence.map((item) => (
                <li key={item.id} className="text-sm">
                  <span className="mr-2 rounded bg-brand-50 px-1 font-mono text-xs text-brand-700 dark:bg-slate-800 dark:text-brand-100">
                    {item.id}
                  </span>
                  <span className="font-medium text-slate-700 dark:text-slate-200">{item.source}</span>
                  <p className="mt-0.5 line-clamp-3 text-xs text-slate-500 dark:text-slate-400">{item.text}</p>
                </li>
              ))}
            </ul>
          </Card>
        )}
      </div>

      <div className="min-w-0 space-y-6">
        <Card title="Verification" aside={v ? <Pill tone={v.passed ? "brand" : "amber"}>{v.passed ? "passed" : v.status}</Pill> : null}>
          {v ? (
            <>
              <div className="flex items-center gap-2 text-sm">
                <span className="text-slate-500">Score</span>
                <ScoreBar value={v.score} label="Evaluator score" />
                <span className="ml-auto text-slate-500">attempts {result.attempts}</span>
              </div>
              {v.explanation && <p className="mt-3 text-sm text-slate-600 dark:text-slate-300">{v.explanation}</p>}
              <p className="mt-3 text-xs text-slate-500">Scored by an LLM judge against the evidence: a heuristic, not proof.</p>
            </>
          ) : (
            <p className="text-sm text-slate-500">Not verified (no answer was written).</p>
          )}
        </Card>

        <Card title="Usage and limits">
          <dl className="grid grid-cols-2 gap-y-1.5 text-sm">
            <dt className="text-slate-500">Tokens</dt>
            <dd className="text-right font-mono">{result.usage.total_tokens ?? "n/a"}</dd>
            <dt className="text-slate-500">LLM calls</dt>
            <dd className="text-right font-mono">
              {result.usage.llm_calls_made ?? "?"} / {result.limits.max_llm_calls ?? "?"}
            </dd>
            <dt className="text-slate-500">Plan steps</dt>
            <dd className="text-right font-mono">
              {result.plan.length} / {result.limits.max_steps ?? "?"}
            </dd>
            <dt className="text-slate-500">Time limit</dt>
            <dd className="text-right font-mono">{result.limits.timeout_s ?? "?"} s</dd>
          </dl>
        </Card>
      </div>
    </div>
  );
}
