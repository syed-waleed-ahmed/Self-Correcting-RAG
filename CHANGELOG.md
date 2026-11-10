# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Project (open-source readiness)

- Community files:
  - `SECURITY.md`, with private vulnerability reporting, scope and rules for
    testing the live demo safely;
  - `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1);
  - issue forms and a PR template;
  - Dependabot for pip, npm and GitHub Actions.
- `CONTRIBUTING.md` rewritten: extras, the Postgres tests, web UI development,
  migrations and free-tier constraints.
- `docker-compose.yml` gains an optional local Postgres (`--profile db`).
- `docs/deployment.md` adds "Deploy your own copy" for forks.
- The planning docs describe the live deployment and no longer read as notes
  for a single owner.


### Added (Phase 4: persistence and identity)

- **Optional Postgres + pgvector persistence** (`SCRAG_DATABASE_URL`, extra
  `postgres`). Without it the API behaves exactly as before.
- **Anonymous sessions** (`POST/GET/DELETE /api/v1/sessions[/me]`):
  - 256-bit tokens stored only as SHA-256 hashes;
  - a fixed TTL (24 h by default); expired sessions are purged with all of
    their data.
- **Uploads** (`POST /api/v1/documents`, plus list, get and delete):
  - TXT/MD/HTML/CSV/JSON/PDF/DOCX;
  - text extracted at upload, with immediate 413/415/422 errors;
  - only the text is stored, never the original bytes;
  - per-session and global quotas;
  - size checked before the multipart body is parsed.
- **Durable job queue** in Postgres:
  - claimed with `FOR UPDATE SKIP LOCKED` and a lease, and completion is
    fenced, so a stale worker can't overwrite a result;
  - exponential-backoff retries, permanent failures and failure hooks;
  - jobs interrupted by a restart resume once their lease expires;
  - the worker never polls an idle queue, so it creates no keep-alive traffic;
  - `GET /api/v1/jobs/{id}`.
- **Search over uploads:**
  - pgvector cosine and Postgres full text, fused with RRF;
  - `scope` on `/query` and `/search`: `samples`, `session` or `all`.
- **Conversation history:**
  - with a session, `/query` saves each turn and returns `conversation_id`;
  - `GET/DELETE /api/v1/conversations[/{id}]`.
- **Persistent daily demo budget.** It survives restarts, which used to reset
  it after every free-tier sleep. It falls back to the in-process budget during
  a database outage.
- **JWT admin auth** (`SCRAG_ADMIN_SUBJECTS` with `SCRAG_AUTH_JWKS_URL` or
  `SCRAG_AUTH_JWT_SECRET`, e.g. Supabase Auth). Algorithms are pinned by
  configuration.
- **`GET /api/v1/admin/stats`** (operator only).
- **Migrations:** plain SQL, every one with a downgrade (ADR 0007), applied on
  first use or with `scrag db status|upgrade|downgrade`.
- **Tests:** 49 new ones (236 in total), including:
  - 24 against a real Postgres with pgvector, as a new CI job;
  - tenant-isolation, restart-survival and job-resume tests.

### Changed

- Sessions and uploads use `/api/v1` only. The older routes keep their
  unversioned aliases.
- `HealthResponse.persistence` reports `disabled`, `unknown`, `ok` or
  `unavailable`. Health still does no I/O.


### Added (Phase 3: agents and tools)

- **Bounded agent workflow** (`POST /api/v1/agent/run`, `scrag agent`):
  - stages: plan, then approve, then execute (parallel), then write, then
    verify, with at most one correction;
  - hard limits on steps (4), LLM calls (6) and wall time (90 s);
  - the full trace and real provider token usage are returned per run;
  - unusable plans fall back to a knowledge-base search.
- **Tool registry** (`GET /api/v1/agent/tools`):
  - typed pydantic input schemas, `safe`/`sensitive` permissions, timeouts and
    audit records;
  - tool failures are recorded, never raised;
  - tools: `document_search`, `calculator` (whitelisted-AST arithmetic, never
    `eval`) and `reindex_documents` (sensitive: needs explicit approval, and is
    disabled on the public demo).
- **Prompt-injection defenses by construction:**
  - the planner never sees retrieved text;
  - evidence reaches the writer inside delimiters, marked untrusted and
    escaped against tag injection;
  - the writer has no tool access.
- **`UsageLedger`.** `OpenAICompatibleClient` records provider-reported tokens
  for every call made inside a tracked context.
- **Demo budget for agent runs.** An agent run costs `SCRAG_AGENT_BUDGET_COST`
  units (default 2). The units are refunded if the run ends awaiting approval
  or fails.

### Fixed

- Demo mode: the read endpoints (`/search`, `/query`, `/agent/run`) stay public
  when an operator token is configured. The token only protects `/ingest`.
  The first live deploy's smoke test caught this.

### Deployment

- Live on Render's free tier:
  - API https://scrag-api.onrender.com;
  - web UI https://scrag-web.onrender.com (a static site in the same Blueprint).
- A GitHub Pages workflow also publishes the UI from `main`.
- Phase 4 is live with a Neon Free Postgres database in Frankfurt
  (`persistence: ok`), verified end to end on the deployment.

## [0.2.0] - 2026-11-10

Phase 2 (retrieval and evaluation) and Phase 2b (deployable demo) build on
Phase 1 (core reliability). The roadmap is in `docs/planning/`.

### Added (Phase 2 / 2b)

- **Hybrid retrieval** is the new default (`SCRAG_RETRIEVAL_MODE`):
  - in-process BM25 (k1 = 1.5, b = 0.75) plus dense search, fused with
    Reciprocal Rank Fusion (k = 60);
  - chunks report `fusion_score` and `keyword_score` alongside the cosine
    `score`;
  - measured on the golden set: MRR 0.915 -> 0.961 and recall@5
    0.950 -> 0.988 (`docs/evaluation.md`).
- **Evaluation harness.** `scrag eval` reports recall@k, hit rate, MRR and
  nDCG, broken down by question type, with optional answer-level evaluation. It
  uses a 40-question golden set (`data/eval/golden.jsonl`) over a sample corpus
  expanded to 16 documents.
- **fastembed (ONNX) embedder.** `SCRAG_EMBEDDING_BACKEND`, `auto` by default.
  It produces the same vectors as sentence-transformers with about half the
  peak memory (211 vs 428 MB measured).
- **Document formats.** HTML, CSV and JSON; PDF and DOCX with the new `docs`
  extra. Files are subject to size and page limits, and unreadable files are
  skipped.
- **Batched guardrail** (`SCRAG_GUARDRAIL_MODE=batch`, the default). Chunks are
  scored in one call instead of one per chunk. It falls back to per-chunk calls
  if the reply is unusable, and makes no further calls if the provider is down.
- **Versioned API** under `/api/v1`; the unversioned paths still work.
- **New endpoints:**
  - `POST /search` runs retrieval only, without an LLM;
  - `GET /demo` returns sample questions, limits and the remaining budget.
- **Demo mode** (`SCRAG_DEMO_MODE`):
  - per-client token-bucket rate limit (`429 rate_limited`, `Retry-After`);
  - global daily budget of live pipeline runs (`429 demo_budget_exhausted`),
    refunded when a run fails;
  - query-length and `top_k` caps;
  - `/ingest` needs the operator token.
- **Answer cache.** Recorded sample answers (`data/demo/samples.json`) and a TTL
  cache for recent live answers. Cached responses are always labelled
  (`cached`, `cached_at`, `cache_source`), and samples are only served while
  the corpus fingerprint matches.
- **`web/`:** a Next.js + Tailwind demo UI as a static export for Vercel. It
  shows sample questions, live vs recorded runs, the attempt timeline,
  per-chunk scores, retrieval-only mode, the remaining budget and a cold-start
  state.
- **Deployment:**
  - the slim Dockerfile has no torch and bakes in the embedding model and
    sample index;
  - `render.yaml` is a Render Blueprint (Free plan);
  - `scripts/smoke.py` and `scripts/record_samples.py`;
  - `docs/deployment.md`;
  - ADRs in `docs/decisions/`.
- **CI:**
  - the container job runs the image under 512 MB / 0.1 CPU, smoke-tests it and
    records peak memory;
  - a web job builds and type-checks the UI.

### Changed (Phase 2 / 2b)

- The Docker image uses fastembed instead of torch. Vectors are identical, so
  existing indexes stay valid.
- The pipeline keeps retrieval order (fused rank in hybrid mode) for
  `used_chunks`.
- `VectorStore` gained `versioned_snapshot()`, which has a default
  implementation.
- Loaders normalize line endings, so documents hash identically on Windows and
  Linux.

Phase 1 (core reliability):

### Changed (breaking)

- **Default LLM model is now `qwen/qwen3.8-27b`.** Groq retired
  `llama-3.1-8b-instant` for free and developer tiers on 2026-08-16, so the old
  default failed on every query. Verified on Groq's free tier on 2026-11-10.
- **`attempts` now counts the attempts executed.** It used to report the number
  of the winning attempt. The new `best_attempt` field holds that, and
  `attempt_history` lists every attempt with its score and explanation.
- **`evaluation_score` may be `null`.** An evaluator outage or unparseable
  evaluator output is no longer reported as a fabricated `0.0`. The new
  `evaluation_status` field is `passed`, `below_threshold`, `unavailable` or
  `skipped`.
- **`guardrail_score` may be `null`.** When a chunk could not be scored it is
  kept (fail-open, as before) but is no longer reported as `1.0`.
- **Error responses use one shape:** `{"error", "detail", "request_id"}`.
  Provider messages and other internal details are no longer returned to
  clients. The blanket `ValueError -> 400` mapping is gone; unexpected errors
  return a generic `500`.
- **New index layout on disk.** Indexes are saved as immutable generations
  under `generations/<id>/`, with a `CURRENT` pointer. Indexes in the old flat
  layout still load.
- **`VectorStore` gained abstract `replace()` and `snapshot()` methods.** Custom
  backends must implement them.
- `max_self_correct_steps` is capped at 10, `top_k` at 100 and
  `guardrail_max_workers` at 32.

### Fixed

- The evaluator's explanation is now passed to the generator on retries.
  Self-correction previously asked for a better answer without saying what was
  wrong.
- If generation fails on a retry, the earlier successful attempt is returned
  instead of the whole request failing.
- Score parsing understands percentages (`85%`), fractions (`8/10`) and
  "N out of M". These used to clamp to `1.0`.
- `/ingest` with `reset=false` no longer duplicates every chunk. It upserts by
  source using a content hash.
- Concurrent `/query` and `/ingest` requests can't observe a partial index.
  Ingestion builds the full index, then swaps it in atomically, and only one
  ingestion runs per process (`409 ingest_in_progress`).
- Multiple API workers sharing an index directory now converge. `/query` reloads
  when another process has saved a newer index generation.
- Embedding dimension mismatches fail fast with an actionable message instead
  of the model's dimension being silently adopted.
- A missing documents directory returns `409 documents_not_found` (CLI: a clear
  message) instead of a `500`. Files that aren't valid UTF-8 are skipped instead
  of aborting ingestion.
- A non-ASCII API token returns `401` instead of crashing with a `500`.
- 7 mypy errors fixed, including a real type error in the LLM client call.

### Added

- `scrag/errors.py`: a domain error hierarchy, each error with a stable code,
  HTTP status and client-safe message.
- An `X-Request-ID` header on every response; the same ID appears in logs.
- `SCRAG_GUARDRAIL_MAX_TOKENS`. Reasoning models such as `openai/gpt-oss-20b`
  return empty output at the default of 8 and need about 256.
- Inline `<think>...</think>` reasoning blocks are stripped from model output.
- `python scripts/verify.py`, a single command for lint, format check, mypy and
  tests. CI now also runs the format and type checks, plus a FAISS backend job.
- Tests grew from 31 to 112.

## [0.1.0] - 2026-07-05

First production-grade release. Reworks the original prototype into a tested,
deployable package with a REST API and CLI over a shared core.

### Added

- Installable `scrag` package (`src/` layout) with `pyproject.toml` packaging,
  `py.typed` marker, and console entry point.
- **FastAPI** service (`/health`, `/query`, `/ingest`) via an app factory with
  lifespan-managed dependencies, plus a **Typer** CLI (`ingest`, `query`,
  `serve`, `info`).
- **Pluggable vector store**: dependency-free NumPy backend (default) and an
  optional FAISS backend, behind a common `VectorStore` interface.
- **Provider-agnostic** OpenAI-compatible LLM client with configurable base URL,
  timeouts, and retries (defaults to Groq).
- Boundary-aware, overlapping **chunker** and batched ingestion.
- Typed, environment-driven configuration (pydantic-settings) and structured
  logging with an optional JSON formatter.
- Optional shared-secret API auth on `/query` and `/ingest` (constant-time
  comparison).
- Test suite (31 tests) that runs offline with fake embedder/LLM, GitHub Actions
  CI across Python 3.10-3.12, `Dockerfile` (non-root) and `docker-compose`.

### Changed

- Self-correction loop now keeps the **best-scoring** attempt instead of the last
  one, so it can never return a worse answer than one already produced.
- Guardrail agent uses a calibrated scoring rubric and scores chunks in parallel;
  relevance/JSON parsing is defensive against malformed model output.
- Index persistence moved to `vectors.npy` + `chunks.jsonl` (no delimiter-based
  truncation).

### Fixed

- Removed an f-string backslash that raised `SyntaxError` on Python < 3.12; the
  package now runs on Python 3.10+.

[Unreleased]: https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/releases/tag/v0.1.0
