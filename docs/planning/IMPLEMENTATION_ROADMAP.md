# Implementation Roadmap

Last updated: 2026-11-10.

Every phase ends with the same gate:
1. `python scripts/verify.py` is green.
2. The diff has been reviewed.
3. Docs and `CHANGELOG.md` are updated.
4. This file records what was done and what was not.

A phase is not "done" while any acceptance criterion is unverified.

## Recommended changes to the original phase order

The original brief deployed in Phase 8. Two changes were recommended and
adopted (see "Decisions taken" below):

1. **Deploy a walking skeleton right after Phase 2** (Phase 2b below). The
   free-tier constraints shape almost every later decision: 512 MB, cold starts,
   Groq quotas and Supabase pausing. They should be discovered on real
   infrastructure early, not in Phase 8.
2. **Build a minimal evaluation harness in Phase 2, before hybrid retrieval.**
   Without a golden set, "hybrid + rerank is better" would be an assumption.
   Phase 6 extends the harness; it doesn't introduce it.

## Phase status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository audit and planning documents | **Done** |
| 1 | Core reliability defects | **Done** (see below) |
| 2 | Advanced RAG foundation + minimal eval harness | **Done** (pgvector moved to Phase 4, ADR 0006) |
| 2b | Walking-skeleton deployment (Render + Vercel, bundled index) | **Live** on Render (2026-11-10) |
| 3 | Agent framework and tool registry | **Done** (UI for agents is part of Phase 5) |
| 4 | Persistence and identity | **Live** (2026-11-10): Render Free + Neon Free (Frankfurt) |
| 5 | Frontend | **Live** (2026-11-10): Ask, My documents, Agent and How it works, with SSE streaming |
| 6 | Observability and evaluation | **Done** (2026-11-10): metrics, cost per answer, optional tracing, 60-question golden set, answer-quality comparison |
| 7 | Security hardening, quotas, load tests | Not started |
| 8 | Full free-tier deployment | Not started |
| 9 | Live demo verification | Not started |
| 10 | GitHub showcase release | Not started |

## Phase 1: Core reliability (done)

Acceptance criteria, all met and covered by regression tests:

- [x] Evaluator explanations are passed to the generator on retries.
- [x] `attempts` counts the attempts executed; `best_attempt` and
  `attempt_history` were added.
- [x] Evaluator infrastructure failures (`error` / `invalid`) are distinct from
  low scores, with no fabricated `0.0`.
- [x] Scores parse as decimals, percentages and fractions (`8/10`,
  `9 out of 10`).
- [x] Concurrent ingest and query can't corrupt or expose a partial index
  (atomic swap, one ingest per process, generation-based persistence).
- [x] Workers sharing an index directory converge (`refresh_if_stale` on
  `/query`).
- [x] `reset=false` does not duplicate; it upserts by content hash.
- [x] Embedding dimensions are validated on every embed, failing fast with a
  clear message.
- [x] A missing or empty docs directory is a clean `409`, and the CLI shows a
  clear message.
- [x] The API error contract is sanitized, with request IDs; no internal details
  leak.
- [x] The default model was retired by Groq; it is replaced with a verified free
  model (live check on 2026-11-10).
- [x] `mypy src` and `ruff format --check` are clean and enforced in CI; there is
  a single `scripts/verify.py` command.

Known limits accepted in Phase 1:
- Separate hosts without a shared disk still don't share an index; Phase 2
  (pgvector) fixes that.
- Cross-process `reset=false` ingests are last-writer-wins.
- The FAISS tests are skipped locally unless `faiss-cpu` is installed; CI runs
  them in a separate job.

## Phase 2: Advanced RAG foundation (done 2026-11-10)

Delivered:
- golden set (40 questions) and `scrag eval`, with the baseline measured before
  hybrid retrieval;
- `FastEmbedEmbedder` (memory measured);
- hybrid BM25 + RRF retrieval (measured: MRR 0.915 -> 0.961 at k=5);
- PDF, DOCX, HTML, CSV and JSON loaders with limits;
- batched guardrail (verified live).

Collections and `PgVectorStore` moved to Phase 4 (ADR 0006). The reranker is
deferred until it can be measured on the 0.1 CPU instance. Results are in
`docs/evaluation.md`, decisions in `docs/decisions/`.

Original scope:

Scope:
1. **Golden set.** About 30-50 question/answer/source triples over the sample
   corpus, expanded beyond today's 3 documents. Add retrieval metrics (recall@k,
   MRR, nDCG) and a faithfulness proxy, plus a `scrag eval` command that writes
   a JSON report. Record the baseline numbers **before** retrieval changes.
2. **`FastEmbedEmbedder`** (ONNX), selected with `SCRAG_EMBEDDING_BACKEND`.
   Measure memory under a 512 MB limit.
3. **Collections.** Add `collection_id` and metadata filters to the
   `VectorStore` interface.
4. **`PgVectorStore`** (psycopg 3 through the Supabase pooler) with HNSW.
   Integration tests run against a local `pgvector/pgvector` container in CI.
5. **Hybrid retrieval.** BM25 (in-process for small sets; Postgres `tsvector`
   for pgvector) plus Reciprocal Rank Fusion. The reranker is optional and off
   by default, because it is too slow on 0.1 CPU until measured otherwise.
6. **Ingestion formats.** PDF (pypdf), DOCX (python-docx), HTML (stdlib parser),
   CSV and JSON. XLSX and OCR are optional extras. Every format gets size and
   page limits.
7. **Batched guardrail.** Score all chunks in one call (fewer requests against
   RPD/RPM quotas), keeping the per-chunk path as a fallback.

Acceptance:
- Eval report numbers are recorded for the baseline and for hybrid retrieval.
- The memory measurement is recorded.
- pgvector tests are green in CI.
- No regressions.

## Phase 2b: Walking skeleton deployment (built 2026-11-10)

Delivered:
- `/api/v1`, `/search` and `/demo`;
- demo mode (rate limit, daily budget, caps, operator-only ingest);
- recorded sample answers;
- slim Dockerfile with the bundled index;
- `render.yaml`;
- `web/` (Next.js static export);
- `scripts/smoke.py`;
- CI container job (512 MB / 0.1 CPU) and web build job.

**Deployed** on 2026-11-10: Render Free for the API and a static site, with
GitHub Pages as a mirror (Vercel wasn't needed). The smoke test passes against
the live URL; the evidence is in `DEPLOYMENT_PLAN.md`, section 6.

Original scope:

Scope:
- Render Free hosts the API image with fastembed and the bundled sample index.
- Vercel hosts a single-page UI: ask a sample question, see the answer, sources,
  scores and attempts.
- Demo mode:
  - `top_k=3`;
  - a per-IP rate limit;
  - a global daily LLM budget;
  - cached answers for curated questions.
- The cold-start UX is handled.

Acceptance:
- The public URL answers a sample question.
- A smoke-test script passes against the live URL.
- Evidence (date, commit, output) is recorded in `DEPLOYMENT_PLAN.md`.

## Phase 3: Agents and tools (done 2026-11-10)

Delivered:
- tool registry with schemas, permissions, timeouts and audit records;
- three tools: `document_search`, a safe `calculator` and an approval-gated
  `reindex_documents`;
- bounded plan -> execute -> write -> verify workflow with one correction;
- per-run token accounting;
- `/agent/run`, `/agent/tools` and `scrag agent`;
- 26 tests;
- verified live on Groq: a two-tool plan that passed verification and used
  2,109 tokens.

Known limitation: plan steps are independent, so a step can't consume another
step's output. In the live run the planner wrote the calculator input from its
own knowledge, although the final answer was grounded in cited evidence.
Dependent steps (`$ref` to earlier outputs) are the natural next increment.

Original scope:

Scope:
- `Tool` and `Agent` contracts (typed I/O, budgets, timeouts, retries, audit
  events).
- A bounded workflow engine.
- A supervisor/planner.
- Agents: Retrieval, Verification, Summarization.
- Tools: document search and calculator.
- Human-approval hook for tools marked sensitive.
- Prompt-injection defenses: treat retrieved text and tool output as data,
  delimit it, and never let it select tools by itself.

Acceptance:
- Deterministic workflow tests with fake LLMs.
- Step and token budgets are enforced.
- An approval-required tool can't run without approval.

## Phase 4: Persistence and identity (done 2026-11-10)

Delivered (ADR 0007, ADR 0008):
- schema and migrations (plain SQL with downgrades, not Alembic: ADR 0007);
- anonymous sessions with hashed tokens and a 24 h TTL;
- JWT verification for admins (JWKS or HS256; e.g. Supabase Auth);
- conversations and history;
- uploads stored as extracted text in Postgres, not Supabase Storage (less to
  secure and store);
- a durable Postgres job queue with leases, retries and fencing;
- per-session deletion and expiry purge;
- a persistent daily budget;
- search over uploads (pgvector + full text, RRF) with `scope`.

Acceptance, all as tests that run in CI against Postgres 17 with pgvector:
- data survives a backend restart: `test_api_data_survives_a_backend_restart`;
- tenant isolation: `test_api_sessions_are_isolated_from_each_other`;
- a job interrupted by a restart resumes:
  `test_job_interrupted_by_a_restart_resumes_after_its_lease_expires`.

Evidence:
- **CI run 42775432973:** the `test-postgres` job ran all 236 tests against
  Postgres 17 with pgvector; only the 9 FAISS tests were skipped. The CLI
  migration round trip passed.
- **Memory:** the container peaked at 220 MB under 512 MB / 0.1 CPU (up from
  210 MB, because of the new drivers). The image is 307 MB.
- **Real stack, run locally** (fastembed, live Groq, Postgres):
  - an upload was ingested in 1.8 s;
  - the session-scoped answer was correct and passed evaluation;
  - another session got 404 for the document and the conversation;
  - the persisted budget kept its count across a server restart.
- **Live (2026-11-10, commit `c8ef430`):**
  - Setup: Render Free API with a Neon Free database in Frankfurt, created
    through the Vercel Neon integration with Auth and Realtime off.
  - Database startup: the worker connected and migrated on its own, and
    `/health` reported `persistence: ok`.
  - Timings: creating a session took 0.2 s; an upload took 0.2 s and its
    ingestion job finished in 7.9 s on 0.1 CPU; a session-scoped answer was
    correct and passed evaluation in 2.4 s.
  - Results: history was saved; another session got 404 for the document
    and the conversation; admin stats returned 401 without the operator
    token; deleting the session revoked its token.
  - `scripts/smoke.py` passed every check.

Original scope:

Scope:
- Schema and migrations (Alembic).
- Anonymous demo sessions.
- Supabase Auth JWT verification for the admin.
- Conversations and history.
- Uploads to Supabase Storage.
- A durable Postgres job queue with retries.
- Per-session data deletion.

Acceptance:
- Data survives a backend restart.
- Tenant-isolation tests prove that session A cannot read session B.
- A job interrupted by a restart resumes.

## Phase 5: Frontend (done 2026-11-10)

Delivered (ADR 0009):
- **Chat** (Ask page):
  - live stage-by-stage progress over SSE;
  - scope selection (sample docs, my documents or both);
  - conversation threads and saved history;
  - an attempt timeline;
  - sources with all retrieval scores.
- **Knowledge base:** a session, uploads with live ingestion status, deletion,
  and "delete all my data".
- **Agent workspace:** the live trace, plan, tool calls, evidence for `[S#]`
  citations, verification, and usage against the limits.
- **Evaluation view** ("How it works"): measured results from
  `docs/evaluation.md`.

Verified end to end in Chrome against the real API (fastembed, live Groq,
Postgres), served with the production security headers:
- streaming progress arrived incrementally;
- an upload was ingested and answered from;
- a follow-up stayed in the same conversation and reopened from history;
- a session-scoped agent run completed (2-step plan, verification 1.0);
- "delete all my data" removed everything (the old token answered 401);
- 0 CSP violations and no console errors.

Deferred:
- Admin pages behind sign-in. Operators use the API with the operator token
  or an admin JWT.
- Token-level streaming of the answer text. Progress is per stage (ADR 0009).

Original scope: chat (SSE streaming, citations, attempt timeline), knowledge
base (upload, status, delete), agent workspace (plan, steps, tool calls),
evaluation view; admin pages visible to the authenticated owner only.

## Phase 6: Observability and evaluation (done 2026-11-10)

Delivered (ADR 0010):
- Prometheus `/metrics`, with bounded cardinality and operator-only access on
  the demo.
- Per-answer `usage` and `timings_ms`, shown in the UI. A cost estimate is
  reported only when prices are configured.
- Optional OpenTelemetry spans for HTTP requests, pipeline stages, LLM calls,
  the agent and tools.
- The golden set grew from 40 to 60 questions and retrieval was re-measured.
- `scrag eval --compare` measured answers across the baseline, hybrid,
  self-correcting and agent configurations, plus a hard/unanswerable set.
- **Measured and published:** on the sample corpus, self-correction never
  changed an outcome, at about 2.6 times the cost; retrieval depth was the
  bottleneck. The demo's `top_k` went from 3 to 5 as a result.

Original scope:

Scope:
- OpenTelemetry traces (exported locally / to the console by default).
- Prometheus `/metrics`.
- Per-step latency, tokens and estimated cost.
- An expanded golden set.
- Baseline vs hybrid vs self-correcting vs multi-agent comparisons.
- Only real numbers are published.

## Phase 7: Security and quotas

Scope:
- Threat model.
- Upload validation (MIME sniffing, size and page limits, zip-bomb guards).
- SSRF-safe HTTP tool (allow-list, no private IP ranges).
- Read-only SQL tool on an allow-listed schema.
- Rate limits and quotas.
- `pip-audit` in CI.
- Error sanitization review.
- Load tests (locust or k6) against a local container, with results recorded.

## Phases 8-10: Deploy, verify, release

Scope:
- Full deployment per `DEPLOYMENT_PLAN.md`.
- The live smoke-test suite.
- The cost re-verification checklist (`COST_ZERO_VERIFICATION.md`).
- Real screenshots.
- README, ADRs and a tagged release.

## Decisions taken (2026-11-10)

The maintainer adopted the recommended options:

1. **Reordering:** a walking skeleton right after Phase 2, and the eval harness
   inside Phase 2.
2. **fastembed and the bundled index:** D1 and D2, now ADRs 0001 and 0002.
3. **Groq-only LLM routing:** D4, now ADR 0004.
4. **Render card requirement:** Render didn't ask for a payment method when the
   services were created. `docs/deployment.md` says never to add one.
5. **Reduced agent roster** (D7) for Phase 3.

The original questions follow.

## Original open questions (all decided above)

1. Approve the reordering (Phase 2b and the eval harness in Phase 2)?
2. Approve fastembed as the deployment embedder (D1) and the bundled sample index
   (D2)?
3. Approve Groq-only LLM routing for the public demo (no Gemini, because of its
   EU terms)?
4. Can a Render account be created without a card? If Render requires one,
   accept the monitored-bandwidth risk, or choose a different host?
5. Approve the reduced agent roster for the public demo (D7)?
