// Typed client for the SCRAG API (/api/v1). Mirrors src/scrag/models.py.

export const API_BASE = (process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

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
}

export interface SearchResponse {
  query: string;
  retrieval_mode: string;
  chunks: ChunkView[];
}

export interface Health {
  status: string;
  version: string;
  llm_configured: boolean;
  index_size: number;
  demo_mode: boolean;
}

export interface DemoInfo {
  demo_mode: boolean;
  llm_configured: boolean;
  model: string;
  retrieval_mode: string;
  sample_questions: string[];
  limits: Record<string, number>;
  budget: { limit: number; remaining: number; resets_at: string } | null;
}

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

async function request<T>(path: string, init?: RequestInit, timeoutMs = 120_000): Promise<T> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try {
    response = await fetch(`${API_BASE}/api/v1${path}`, {
      ...init,
      headers: { "content-type": "application/json", ...(init?.headers ?? {}) },
      signal: controller.signal,
    });
  } catch {
    throw new ApiError(0, "network_error", "The API could not be reached.", null, null);
  } finally {
    clearTimeout(timer);
  }
  const requestId = response.headers.get("x-request-id");
  if (!response.ok) {
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
    throw new ApiError(response.status, code, message, requestId, Number.isFinite(retry) ? retry : null);
  }
  return (await response.json()) as T;
}

export const api = {
  health: () => request<Health>("/health", undefined, 15_000),
  demo: () => request<DemoInfo>("/demo", undefined, 15_000),
  query: (query: string, useCache: boolean) =>
    request<QueryResponse>("/query", {
      method: "POST",
      body: JSON.stringify({ query, use_cache: useCache }),
    }),
  search: (query: string) =>
    request<SearchResponse>("/search", { method: "POST", body: JSON.stringify({ query }) }),
};
