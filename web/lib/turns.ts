// A question and its answer, as shown in a conversation thread.
import type { MessageRecord, QueryResponse, ResultStatus } from "./api";

export interface Turn {
  question: string;
  answer: string;
  status?: ResultStatus;
  score?: number | null;
  sources: string[];
  cached?: boolean;
}

export function turnFromResponse(r: QueryResponse): Turn {
  return {
    question: r.query,
    answer: r.answer,
    status: r.evaluation_status,
    score: r.evaluation_score,
    sources: [...new Set(r.used_chunks.map((c) => c.source))],
    cached: r.cached,
  };
}

/** Pair saved messages (user, assistant, user, assistant, ...) into turns. */
export function turnsFromMessages(messages: MessageRecord[]): Turn[] {
  const turns: Turn[] = [];
  for (let i = 0; i < messages.length; i++) {
    const message = messages[i]!;
    if (message.role !== "user") continue;
    const reply = messages[i + 1]?.role === "assistant" ? messages[i + 1] : undefined;
    turns.push({
      question: message.content,
      answer: reply?.content ?? "",
      status: reply?.metadata.evaluation_status,
      score: reply?.metadata.evaluation_score ?? null,
      sources: reply?.metadata.sources ?? [],
      cached: reply?.metadata.cached,
    });
    if (reply) i++;
  }
  return turns;
}
