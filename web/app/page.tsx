"use client";

import { useCallback, useEffect, useState } from "react";
import { useApp } from "@/app/providers";
import { HistoryPanel, TurnList } from "@/components/history";
import { ProgressCard } from "@/components/progress";
import { AnswerCard, EvaluationCard, SourcesCard } from "@/components/results";
import { ScopeSelector } from "@/components/scope";
import { Card, buttonClass } from "@/components/ui";
import {
  api,
  describeError,
  type ConversationDetail,
  type PipelineEvent,
  type QueryResponse,
  type Scope,
  type SearchResponse,
} from "@/lib/api";
import { turnFromResponse, turnsFromMessages, type Turn } from "@/lib/turns";

type Mode = "answer" | "search";

export default function AskPage() {
  const { connection, demo, refreshDemo, session, sessionLoaded, handleSessionError } = useApp();
  const sessionsAvailable = demo?.features?.sessions ?? false;
  const streaming = demo?.features?.streaming ?? false;
  const token = session?.token ?? null;

  const [question, setQuestion] = useState("");
  const [mode, setMode] = useState<Mode>("answer");
  const [scope, setScope] = useState<Scope>("samples");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [events, setEvents] = useState<PipelineEvent[]>([]);
  const [result, setResult] = useState<QueryResponse | null>(null);
  const [search, setSearch] = useState<SearchResponse | null>(null);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [historyVersion, setHistoryVersion] = useState(0);

  // Deep link from the documents page: /?scope=session
  useEffect(() => {
    const requested = new URLSearchParams(window.location.search).get("scope");
    if (requested === "session" || requested === "all") setScope(requested);
  }, []);

  // No session (never started, deleted or expired): back to the sample docs.
  useEffect(() => {
    if (sessionLoaded && !session) {
      setScope("samples");
      setConversationId(null);
    }
  }, [session, sessionLoaded]);

  const newConversation = () => {
    setConversationId(null);
    setTurns([]);
    setResult(null);
    setSearch(null);
    setEvents([]);
    setError(null);
  };

  const run = async (text: string, { useCache = true, replace = false } = {}) => {
    const query = text.trim();
    if (!query || busy) return;
    setQuestion(query);
    setBusy(true);
    setError(null);
    setEvents([]);
    try {
      if (mode === "search") {
        setResult(null);
        setSearch(await api.search({ query, scope }, token));
        return;
      }
      setSearch(null);
      const body = { query, use_cache: useCache, scope, conversation_id: token ? conversationId : null };
      const response = streaming
        ? await api.queryStream(body, { token, onProgress: (e) => setEvents((prev) => [...prev, e]) })
        : await api.query(body, token);
      if (result && !replace) setTurns((prev) => [...prev, turnFromResponse(result)]);
      setResult(response);
      if (response.conversation_id) {
        setConversationId(response.conversation_id);
        setHistoryVersion((v) => v + 1);
      }
    } catch (err) {
      handleSessionError(err);
      setError(describeError(err));
    } finally {
      setBusy(false);
      void refreshDemo();
    }
  };

  const openConversation = useCallback((detail: ConversationDetail) => {
    setConversationId(detail.id);
    setTurns(turnsFromMessages(detail.messages));
    setResult(null);
    setSearch(null);
    setEvents([]);
    setError(null);
  }, []);

  const onConversationDeleted = useCallback(
    (id: string) => {
      if (id === conversationId) {
        setConversationId(null);
        setTurns([]);
      }
    },
    [conversationId],
  );

  const maxChars = demo?.limits?.max_query_chars ?? 4000;
  const online = connection === "online";
  const showHistory = sessionsAvailable && token !== null;

  return (
    <div className={showHistory ? "grid gap-6 lg:grid-cols-[1fr_260px]" : ""}>
      <div className="min-w-0 space-y-6">
        <Card
          title="Ask the knowledge base"
          aside={
            (turns.length > 0 || result) && (
              <button type="button" onClick={newConversation} className="text-xs font-medium text-brand-600 hover:underline dark:text-brand-100">
                New conversation
              </button>
            )
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
              placeholder={
                scope === "samples"
                  ? "e.g. How does hybrid search combine keyword and vector results?"
                  : "Ask about your uploaded documents"
              }
              className={buttonClass.input}
            />
            <button type="submit" disabled={busy || !question.trim() || !online} className={buttonClass.primary}>
              {busy ? (mode === "search" ? "Searching…" : "Thinking…") : mode === "search" ? "Search" : "Ask"}
            </button>
          </form>

          <div className="mt-3 flex flex-col gap-2">
            <div className="flex flex-wrap items-center gap-2 text-xs">
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
            <ScopeSelector
              value={scope}
              onChange={setScope}
              hasSession={token !== null}
              sessionsAvailable={sessionsAvailable}
              disabled={busy}
            />
          </div>

          {scope === "samples" && demo && demo.sample_questions.length > 0 && (
            <div className="mt-4">
              <p className="mb-2 text-xs text-slate-500 dark:text-slate-400">Try a sample question</p>
              <div className="flex flex-wrap gap-2">
                {demo.sample_questions.map((q) => (
                  <button key={q} type="button" disabled={busy || !online} onClick={() => void run(q)} className={buttonClass.chip}>
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
        </Card>

        {error && (
          <div role="alert" className="rounded-xl border border-red-200 bg-red-50 p-4 text-sm text-red-900 dark:border-red-900 dark:bg-red-950 dark:text-red-200">
            {error}
          </div>
        )}

        <TurnList turns={turns} saved={conversationId !== null} />

        {(busy || events.length > 0) && mode === "answer" && <ProgressCard events={events} busy={busy} />}

        {result && (
          <div className="grid gap-6 lg:grid-cols-[3fr_2fr]">
            <div className="min-w-0 space-y-6">
              <AnswerCard result={result} />
              {result.cached && (
                <button
                  type="button"
                  onClick={() => void run(result.query, { useCache: false, replace: true })}
                  disabled={busy}
                  className={buttonClass.secondary}
                >
                  Run live (uses 1 of today&apos;s queries)
                </button>
              )}
              <SourcesCard chunks={result.used_chunks} />
            </div>
            <EvaluationCard result={result} />
          </div>
        )}

        {search && <SourcesCard chunks={search.chunks} title={`Retrieved (${search.retrieval_mode})`} />}
      </div>

      {showHistory && token && (
        <aside className="min-w-0">
          <HistoryPanel
            token={token}
            version={historyVersion}
            activeId={conversationId}
            onOpen={openConversation}
            onDeleted={onConversationDeleted}
            onSessionError={handleSessionError}
          />
        </aside>
      )}
    </div>
  );
}
