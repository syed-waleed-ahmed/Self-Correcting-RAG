"use client";

import { useEffect, useState } from "react";
import { api, describeError, type ConversationDetail, type ConversationRecord } from "@/lib/api";
import { Markdown } from "./markdown";
import type { Turn } from "@/lib/turns";
import { Card, Pill, StatusBadge, formatDate } from "./ui";

/** The session's saved conversations. Refreshes whenever `version` changes. */
export function HistoryPanel({
  token,
  version,
  activeId,
  onOpen,
  onDeleted,
  onSessionError,
}: {
  token: string;
  version: number;
  activeId: string | null;
  onOpen: (detail: ConversationDetail) => void;
  onDeleted: (id: string) => void;
  onSessionError: (err: unknown) => void;
}) {
  const [items, setItems] = useState<ConversationRecord[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busyId, setBusyId] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    api
      .conversations(token)
      .then((list) => {
        if (!cancelled) {
          setItems(list);
          setError(null);
        }
      })
      .catch((err) => {
        if (cancelled) return;
        onSessionError(err);
        setError(describeError(err));
      });
    return () => {
      cancelled = true;
    };
  }, [token, version, onSessionError]);

  const open = async (id: string) => {
    setBusyId(id);
    try {
      onOpen(await api.conversation(token, id));
    } catch (err) {
      onSessionError(err);
      setError(describeError(err));
    } finally {
      setBusyId(null);
    }
  };

  const remove = async (id: string) => {
    setBusyId(id);
    try {
      await api.deleteConversation(token, id);
      setItems((prev) => prev?.filter((c) => c.id !== id) ?? null);
      onDeleted(id);
    } catch (err) {
      onSessionError(err);
      setError(describeError(err));
    } finally {
      setBusyId(null);
    }
  };

  return (
    <Card title="History">
      {error && <p className="mb-2 text-xs text-red-600 dark:text-red-400">{error}</p>}
      {items === null ? (
        <p className="text-sm text-slate-500">Loading…</p>
      ) : items.length === 0 ? (
        <p className="text-sm text-slate-500 dark:text-slate-400">
          Your conversations appear here. They are kept for 24 hours with your session.
        </p>
      ) : (
        <ul className="space-y-1">
          {items.map((c) => (
            <li key={c.id} className="group flex items-center gap-1">
              <button
                type="button"
                onClick={() => void open(c.id)}
                disabled={busyId !== null}
                aria-current={c.id === activeId ? "true" : undefined}
                className={`min-w-0 flex-1 truncate rounded-lg px-2 py-1.5 text-left text-sm ${
                  c.id === activeId
                    ? "bg-brand-50 text-brand-700 dark:bg-slate-800 dark:text-brand-100"
                    : "text-slate-700 hover:bg-slate-100 dark:text-slate-300 dark:hover:bg-slate-800"
                }`}
                title={`${c.title} · ${formatDate(c.updated_at)}`}
              >
                {c.title}
                <span className="ml-1 text-xs text-slate-400">({Math.floor(c.message_count / 2)})</span>
              </button>
              <button
                type="button"
                onClick={() => void remove(c.id)}
                disabled={busyId !== null}
                aria-label={`Delete conversation: ${c.title}`}
                className="rounded px-1.5 text-xs text-slate-400 opacity-60 hover:text-red-600 group-hover:opacity-100"
              >
                ✕
              </button>
            </li>
          ))}
        </ul>
      )}
    </Card>
  );
}

/** Earlier turns of the current conversation, compact. */
export function TurnList({ turns, saved }: { turns: Turn[]; saved: boolean }) {
  if (turns.length === 0) return null;
  return (
    <Card title={saved ? "This conversation" : "Earlier on this page"} aside={<Pill>{turns.length} earlier</Pill>}>
      <ol className="space-y-4">
        {turns.map((turn, i) => (
          <li key={i} className="border-l-2 border-slate-200 pl-3 dark:border-slate-700">
            <p className="text-sm font-medium text-slate-800 dark:text-slate-100">{turn.question}</p>
            <div className="mt-1 text-sm">
              {turn.answer ? <Markdown text={turn.answer} /> : <p className="text-slate-500">(no answer saved)</p>}
            </div>
            <div className="mt-2 flex flex-wrap items-center gap-2 text-xs text-slate-500">
              {turn.status && <StatusBadge status={turn.status} />}
              {turn.cached && <Pill tone="amber">recorded run</Pill>}
              {turn.sources.length > 0 && <span>Sources: {turn.sources.join(", ")}</span>}
            </div>
          </li>
        ))}
      </ol>
    </Card>
  );
}
