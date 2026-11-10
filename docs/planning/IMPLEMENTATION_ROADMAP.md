# Implementation Roadmap

Last updated: 2026-11-10.

Every phase ends with the same gate:
1. `python scripts/verify.py` is green.
2. The diff has been reviewed.
3. Docs and `CHANGELOG.md` are updated.
4. This file records what was done and what was not.

A phase is not "done" while any acceptance criterion is unverified.

## Recommended changes to the original phase order

The master prompt deploys in Phase 8. Two changes are recommended, and both are
the **owner's decision**:

1. **Deploy a walking skeleton right after Phase 2** (Phase 2b below). The
   free-tier constraints shape almost every later decision: 512 MB, cold starts,
   Groq quotas and Supabase pausing. They should be discovered on real
   infrastructure early, not in Phase 8. The second planning reply in the master
   prompt makes the same recommendation.
2. **Build a minimal evaluation harness in Phase 2, before hybrid retrieval.**
   Without a golden set, "hybrid + rerank is better" would be an assumption.
   Phase 6 extends the harness; it doesn't introduce it.

## Phase status

| Phase | Scope | Status |
| --- | --- | --- |
| 0 | Repository audit and planning documents | **Done** |
| 1 | Core reliability defects | **Done** (see below) |
| 2 | Advanced RAG foundation + minimal eval harness | Not started |
| 2b | Walking-skeleton deployment (Render + Vercel, bundled index) | Not started |
| 3 | Agent framework and tool registry | Not started |
| 4 | Persistence, auth, sessions, durable jobs | Not started |
| 5 | Frontend | Not started (a minimal page ships in 2b) |
| 6 | Observability and evaluation | Not started |
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

## Phase 2: Advanced RAG foundation

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

## Phase 2b: Walking skeleton deployment

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

## Phase 3: Agents and tools

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

## Phase 4: Persistence and identity

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

## Phase 5: Frontend

Chat (SSE streaming, citations, attempt timeline), knowledge base (upload,
status, delete), agent workspace (plan, steps, tool calls), evaluation view.
Admin pages are visible to the authenticated owner only.

## Phase 6: Observability and evaluation

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

## Open decisions for the owner

1. Approve the reordering (Phase 2b and the eval harness in Phase 2)?
2. Approve fastembed as the deployment embedder (D1) and the bundled sample index
   (D2)?
3. Approve Groq-only LLM routing for the public demo (no Gemini, because of its
   EU terms)?
4. Can a Render account be created without a card? If Render requires one,
   accept the monitored-bandwidth risk, or choose a different host?
5. Approve the reduced agent roster for the public demo (D7)?
