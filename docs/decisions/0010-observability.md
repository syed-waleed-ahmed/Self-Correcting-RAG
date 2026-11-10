# 0010: Observability: Prometheus metrics, optional OpenTelemetry, real cost per answer

**Status:** Accepted (2026-11-10). Implemented in Phase 6.

**Context.**
- The system makes several LLM calls per answer, on a free tier whose limits
  are tokens per minute and requests per day. Knowing where time and tokens go
  matters more than raw request counts.
- No observability backend is free without another account. The design must
  be useful locally and must cost nothing in production.

**Decision.**
- **Prometheus metrics at `GET /metrics`** (`prometheus-client`, one
  in-process registry):
  - HTTP requests and latency by route template;
  - pipeline stage durations;
  - run outcomes and attempts;
  - LLM calls, latency and tokens by model and outcome;
  - agent and tool outcomes;
  - answer-cache lookups;
  - background-job outcomes.
- **Cardinality is bounded by construction:**
  - routes are templates rebuilt from the matched path parameters
    (`/api/v1/documents/{document_id}`), and unmatched paths share
    `unmatched`;
  - tool names coming from the planner are collapsed to `unknown` unless
    registered;
  - nothing user-supplied is ever a label.
- **On the public demo, `/metrics` is operator-only** (`require_operator`).
  Locally, without operator auth, it is open. `docker compose --profile
  observability` runs a Prometheus that scrapes it.
- **Every answer reports its cost:**
  - `usage` (LLM calls and tokens from the provider's own counts, by model)
    and `timings_ms` per stage;
  - `estimated_cost_usd` only when list prices are configured
    (`SCRAG_LLM_PRICES`). The project never guesses prices: the demo runs on a
    free tier.
  - Nested usage ledgers forward to their parent, so a harness can measure
    total cost across nested runs.
- **OpenTelemetry tracing is optional** (`otel` extra,
  `SCRAG_OTEL_EXPORTER=none|console|otlp`):
  - spans for the HTTP request (named by route, tagged with the request ID),
    the pipeline run and each stage, every LLM call (GenAI semantic-convention
    attributes), and agent runs and tools;
  - off by default: `span()` is then a no-op and the SDK is not imported;
  - a module-level provider, not the process-global one, so configuration in
    tests and multiple apps never conflicts;
  - context flows through `contextvars`, so spans from worker threads nest
    under their request.

**Consequences.**
- Free and dependency-light in production: `prometheus-client` is small, and
  tracing costs nothing when off.
- Shipping traces anywhere needs a collector (self-hosted, or an account
  elsewhere), so it stays opt-in.
- With several worker processes, Prometheus multiprocess mode would be
  needed. The free tier runs one.
