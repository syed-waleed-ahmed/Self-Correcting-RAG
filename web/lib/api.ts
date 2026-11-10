// Typed client for the SCRAG API (/api/v1). Mirrors src/scrag/models.py,
// src/scrag/persistence/records.py and src/scrag/orchestration/workflow.py.

import { createSseParser } from "./sse";

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

// --- answers -------------------------------------------------------------------------
export type Scope = "samples" | "session" | "all";
export type EvaluationStatus = "ok" | "error" | "invalid";
export type ResultStatus = "passed" | "below_threshold" | "unavailable" | "skipped";

export interface Attempt {
  number: number;
  answer: string;
  score: number | null;
  explanation: string;
  evaluation_status: EvaluationStatus;
}

export interface ChunkView {
  source: string;
  text: string;
  score: number;
  fusion_score: number | null;
  keyword_score: number | null;
  guardrail_score: number | null;
}

export interface QueryResponse {
  query: string;
  answer: string;
  evaluation_score: number | null;
  evaluation_explanation: string;
  evaluation_status: ResultStatus;
  attempts: number;
  best_attempt: number;
  attempt_history: Attempt[];
  used_chunks: ChunkView[];
  cached: boolean;
  cached_at: string | null;
  cache_source: "sample" | "recent" | null;
  scope?: Scope;
  conversation_id?: string | null;
  history_saved?: boolean | null;
  usage?: UsageSummary | null;
  timings_ms?: Record<string, number> | null;
}

/** What one run cost: provider-reported tokens, and an estimate if prices are configured. */
export interface UsageSummary {
  llm_calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  total_tokens: number;
  by_model: Record<string, { calls: number; prompt_tokens: number; completion_tokens: number }>;
  estimated_cost_usd: number | null;
}

export interface SearchResponse {
  query: string;
  retrieval_mode: string;
  chunks: ChunkView[];
}

/** One progress event from /query/stream (one per pipeline stage). */
export interface PipelineEvent {
  stage: "cache" | "retrieve" | "guardrail" | "generate" | "evaluate";
  status: string;
  attempt?: number;
  score?: number | null;
  passed?: boolean;
  count?: number;
  kept?: number;
  of?: number;
  fallback?: boolean;
  source?: string;
  ms?: number;
}

// --- server info -----------------------------------------------------------------------
export interface Health {
  status: string;
  version: string;
  llm_configured: boolean;
  index_size: number;
  demo_mode: boolean;
  persistence?: "disabled" | "unknown" | "ok" | "unavailable";
}

export interface DemoInfo {
  demo_mode: boolean;
  llm_configured: boolean;
  model: string;
  retrieval_mode: string;
  sample_questions: string[];
  limits: Record<string, number>;
  budget: { limit: number; remaining: number; resets_at: string } | null;
  features?: { sessions?: boolean; agent?: boolean; streaming?: boolean };
}

// --- sessions, documents, history -------------------------------------------------------
export interface SessionCreated {
  session_id: string;
  token: string;
  expires_at: string;
  limits: Record<string, number>;
}

export interface SessionInfo {
  session_id: string;
  kind: string;
  created_at: string;
  expires_at: string;
  usage: { documents: number; conversations: number; messages: number };
  limits: Record<string, number>;
}

export type DocumentStatus = "queued" | "processing" | "ready" | "failed";

export interface DocumentRecord {
  id: string;
  filename: string;
  content_hash: string;
  size_bytes: number;
  status: DocumentStatus;
  error: string | null;
  chunk_count: number;
  created_at: string;
  updated_at: string;
}

export interface DocumentUpload {
  document: DocumentRecord;
  job_id: string | null;
  duplicate: boolean;
}

export interface ConversationRecord {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
  message_count: number;
}

export interface MessageRecord {
  id: number;
  role: "user" | "assistant";
  content: string;
  metadata: {
    scope?: Scope;
    evaluation_status?: ResultStatus;
    evaluation_score?: number | null;
    attempts?: number;
    cached?: boolean;
    sources?: string[];
  };
  created_at: string;
}

export interface ConversationDetail extends ConversationRecord {
  messages: MessageRecord[];
}

// --- agent ---------------------------------------------------------------------------------
export interface PlanStep {
  tool: string;
  input: Record<string, unknown>;
  purpose: string;
}

export interface ToolCall {
  tool: string;
  input: Record<string, unknown>;
  status: "ok" | "denied" | "needs_approval" | "invalid_input" | "timeout" | "error";
  output: unknown;
  error: string | null;
  latency_ms: number;
}

export interface TraceEvent {
  seq: number;
  stage: "plan" | "approval" | "tool" | "synthesize" | "verify" | "limit";
  name: string;
  status: string;
  latency_ms: number;
  detail: string;
}

export interface AgentRunResult {
  task: string;
  status: "completed" | "awaiting_approval" | "failed";
  answer: string | null;
  plan: PlanStep[];
  plan_source: "planner" | "fallback";
  steps: { index: number; purpose: string; call: ToolCall }[];
  pending_approval: PlanStep[];
  verification: { score: number | null; explanation: string; status: string; passed: boolean } | null;
  attempts: number;
  trace: TraceEvent[];
  usage: { total_tokens?: number; llm_calls_made?: number; estimated_cost_usd?: number | null; [key: string]: unknown };
  limits: { max_steps?: number; max_llm_calls?: number; timeout_s?: number };
  error: string | null;
}

// --- transport -----------------------------------------------------------------------------
export class ApiError extends Error {
  constructor(
    readonly status: number,
    readonly code: string,
    message: string,
    readonly requestId: string | null,
    readonly retryAfter: number | null,
  ) {
    super(message);
  }
}

interface RequestOptions {
  method?: string;
  body?: unknown;
  token?: string | null;
  timeoutMs?: number;
  accept?: string;
}

function headersFor(options: RequestOptions): Record<string, string> {
  const headers: Record<string, string> = {};
  if (options.body !== undefined && !(options.body instanceof FormData)) {
    headers["content-type"] = "application/json";
  }
  if (options.token) headers["x-session-token"] = options.token;
  if (options.accept) headers.accept = options.accept;
  return headers;
}

async function send(path: string, options: RequestOptions, signal?: AbortSignal): Promise<Response> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? 120_000);
  const onAbort = () => controller.abort();
  signal?.addEventListener("abort", onAbort);
  try {
    const body =
      options.body === undefined
        ? undefined
        : options.body instanceof FormData
          ? options.body
          : JSON.stringify(options.body);
    const response = await fetch(`${API_BASE}/api/v1${path}`, {
      method: options.method ?? (options.body === undefined ? "GET" : "POST"),
      headers: headersFor(options),
      body,
      signal: controller.signal,
    });
    if (!response.ok) throw await errorFrom(response);
    return response;
  } catch (err) {
    if (err instanceof ApiError) throw err;
    throw new ApiError(0, "network_error", "The API could not be reached.", null, null);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", onAbort);
  }
}

async function errorFrom(response: Response): Promise<ApiError> {
  let code = "http_error";
  let message = `Request failed with status ${response.status}.`;
  try {
    const body = await response.json();
    if (typeof body?.error === "string") code = body.error;
    if (typeof body?.detail === "string") message = body.detail;
    else if (Array.isArray(body?.detail)) message = "The request was not valid.";
  } catch {
    /* non-JSON error body */
  }
  const retry = Number(response.headers.get("retry-after"));
  return new ApiError(
    response.status,
    code,
    message,
    response.headers.get("x-request-id"),
    Number.isFinite(retry) && retry > 0 ? retry : null,
  );
}

async function json<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const response = await send(path, options);
  return response.status === 204 ? (undefined as T) : ((await response.json()) as T);
}

/**
 * POST to a Server-Sent Events endpoint. Calls `onProgress` for each progress
 * event and resolves with the `result` event's body. HTTP errors (checks that
 * fail before streaming) and `error` events both reject with an ApiError.
 */
export async function postStream<T, P>(
  path: string,
  body: unknown,
  { token, onProgress, signal }: { token?: string | null; onProgress: (event: P) => void; signal?: AbortSignal },
): Promise<T> {
  const response = await send(path, { body, token, accept: "text/event-stream", timeoutMs: 180_000 }, signal);
  const requestId = response.headers.get("x-request-id");
  if (!response.body) throw new ApiError(0, "stream_unsupported", "Streaming is not supported here.", requestId, null);
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSseParser();
  let result: T | undefined;
  const handle = (event: string, data: string) => {
    const payload = JSON.parse(data);
    if (event === "progress") onProgress(payload as P);
    else if (event === "result") result = payload as T;
    else if (event === "error") {
      throw new ApiError(
        payload.status ?? 500,
        payload.error ?? "internal_error",
        payload.detail ?? "The request failed.",
        payload.request_id ?? requestId,
        payload.retry_after ?? null,
      );
    }
  };
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      for (const ev of parser.push(decoder.decode(value, { stream: true }))) handle(ev.event, ev.data);
    }
    for (const ev of parser.flush()) handle(ev.event, ev.data);
  } catch (err) {
    void reader.cancel().catch(() => undefined);
    if (err instanceof ApiError) throw err;
    throw new ApiError(0, "stream_interrupted", "The connection was interrupted. Please retry.", requestId, null);
  }
  if (result === undefined) {
    throw new ApiError(0, "stream_incomplete", "The response ended early. Please retry.", requestId, null);
  }
  return result;
}

export const api = {
  health: () => json<Health>("/health", { timeoutMs: 15_000 }),
  demo: () => json<DemoInfo>("/demo", { timeoutMs: 15_000 }),

  query: (
    body: { query: string; use_cache?: boolean; scope?: Scope; conversation_id?: string | null },
    token?: string | null,
  ) => json<QueryResponse>("/query", { body, token }),
  queryStream: (
    body: { query: string; use_cache?: boolean; scope?: Scope; conversation_id?: string | null },
    options: { token?: string | null; onProgress: (event: PipelineEvent) => void; signal?: AbortSignal },
  ) => postStream<QueryResponse, PipelineEvent>("/query/stream", body, options),
  search: (body: { query: string; scope?: Scope }, token?: string | null) =>
    json<SearchResponse>("/search", { body, token }),

  agentStream: (
    body: { task: string; scope?: Scope },
    options: { token?: string | null; onProgress: (event: TraceEvent) => void; signal?: AbortSignal },
  ) => postStream<AgentRunResult, TraceEvent>("/agent/run/stream", body, options),

  createSession: () => json<SessionCreated>("/sessions", { method: "POST" }),
  session: (token: string) => json<SessionInfo>("/sessions/me", { token }),
  deleteSession: (token: string) => json<void>("/sessions/me", { method: "DELETE", token }),

  documents: (token: string) => json<DocumentRecord[]>("/documents", { token }),
  upload: (token: string, file: File) => {
    const form = new FormData();
    form.append("file", file);
    return json<DocumentUpload>("/documents", { body: form, token, timeoutMs: 60_000 });
  },
  deleteDocument: (token: string, id: string) =>
    json<void>(`/documents/${encodeURIComponent(id)}`, { method: "DELETE", token }),

  conversations: (token: string) => json<ConversationRecord[]>("/conversations", { token }),
  conversation: (token: string, id: string) =>
    json<ConversationDetail>(`/conversations/${encodeURIComponent(id)}`, { token }),
  deleteConversation: (token: string, id: string) =>
    json<void>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE", token }),
};

/** A short, human message for any API failure. */
export function describeError(err: unknown): string {
  if (!(err instanceof ApiError)) return "Something went wrong. Please try again.";
  if (err.code === "network_error") return "The API could not be reached. It may be waking up; try again in a minute.";
  const wait = err.retryAfter ? ` Try again in about ${Math.max(1, Math.ceil(err.retryAfter / 60))} min.` : "";
  return `${err.message}${err.status === 429 ? wait : ""}${err.requestId ? ` (request ${err.requestId.slice(0, 8)})` : ""}`;
}
