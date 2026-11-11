# Self-Correcting RAG

A production-grade, multi-agent Retrieval-Augmented Generation pipeline that
retrieves context, guardrails it for relevance, generates a grounded answer, and
then evaluates and self-corrects its own output - exposed as a REST API, a CLI
and a web UI.

**Live demo:** https://scrag-web.onrender.com (mirror:
https://syed-waleed-ahmed.github.io/Self-Correcting-RAG/). The API is at
https://scrag-api.onrender.com, with interactive docs at
[`/docs`](https://scrag-api.onrender.com/docs). Everything runs on free tiers
(Render, Neon Postgres, Groq). The API sleeps after 15 idle minutes, so the
first request can take about a minute. Uploads and history are deleted after 24
hours; please don't upload personal data.

![The live SCRAG demo after a live run: the pipeline panel lists retrieval, the
guardrail keeping 3 of 3 chunks, the written answer and its grade of 1.00, next
to the answer and a passed self-check](docs/images/live-demo.jpg)

*The deployed demo, captured 2026-11-10 during a live run: each pipeline stage
streams in as it completes. The My documents and Agent tabs add private uploads
and a tool-using agent.*

[![CI](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/workflows/ci.yml)
[![CodeQL](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/workflows/codeql.yml/badge.svg)](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/workflows/codeql.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
[![Live demo](https://img.shields.io/badge/demo-live-blue)](https://scrag-web.onrender.com)

---

## Contents

- [Why this exists](#why-this-exists)
- [Architecture](#architecture)
- [Features](#features)
- [Install](#install)
- [Quickstart](#quickstart)
- [API reference](#api-reference)
- [Configuration](#configuration)
- [Security](#security)
- [Project structure](#project-structure)
- [Development](#development)
- [Scaling notes](#scaling-notes)
- [Contributing](#contributing)
- [License](#license)

## Why this exists

Naive RAG retrieves some chunks, stuffs them into a prompt, and hopes the answer
is grounded. This project adds two feedback stages so the system can catch and
repair its own mistakes:

- A guardrail agent scores every retrieved chunk and drops off-topic context
  before it can pollute generation.
- An evaluator agent scores the generated answer against the context and, if it
  is poorly grounded, triggers a bounded self-correction loop - always keeping
  the best-scoring attempt.

## Architecture

```text
   query
     |
     v
  +---------------------+
  |      Retriever      |   hybrid: BM25 keywords + local embeddings,
  |   (vector store)    |   fused with Reciprocal Rank Fusion -> top-k
  +----------+----------+
             |
             v
  +---------------------+
  |      Guardrail      |   LLM relevance score per chunk (one batched call)
  |        agent        |   -> keep >= threshold (fallback: raw retrieval)
  +----------+----------+
             |
             v
  +---------------------------------+
  |     Generator <-> Evaluator     |   generate grounded answer, score it,
  |     self-correction loop        |   correct up to N times, keep best
  +----------------+----------------+
                   |
                   v
       final answer + evaluation + citations
```

Each stage is a small, independently testable component wired together by a
composition root (`scrag/container.py`), so backends and models are swappable
without touching orchestration logic.

## Features

- Multi-agent pipeline: retriever, guardrail, generator, evaluator.
- Hybrid retrieval: BM25 + dense vectors fused with Reciprocal Rank Fusion.
  Measured on a 60-question golden set at k=5, against dense only:
  - recall 0.967 -> 0.992 and MRR 0.943 -> 0.966;
  - lexical-question MRR 0.883 -> 0.938.

  See [docs/evaluation.md](docs/evaluation.md).
- Evaluation harness:
  - `scrag eval` reports recall@k, hit rate, MRR and nDCG on a 60-question
    golden dataset;
  - `scrag eval --compare N` compares answers across a dense baseline, hybrid
    search, the self-correcting pipeline and the agent, paced to provider
    limits.
- Formats: TXT, Markdown, HTML, CSV, JSON, and PDF/DOCX with the `docs` extra,
  with size and page limits; uploads are content-sniffed and zip-bomb checked.
- Bounded agent workflow (`POST /agent/run`, `scrag agent`):
  - a planner chooses tools from a typed, permissioned registry: document
    search, a safe calculator, read-only SQL over a catalog of the corpus, an
    optional SSRF-safe web fetch (allow-listed hosts only), and an
    approval-gated re-index;
  - tools run in parallel with timeouts and audit records;
  - a writer answers from delimited, untrusted evidence with citations;
  - the evaluator verifies the answer and can trigger one correction;
  - hard limits on steps, LLM calls and time, with token usage reported per run.
- Sessions, uploads and history (optional Postgres + pgvector, ADR 0008):
  - anonymous sessions with hashed tokens and a fixed TTL; deleting a session
    (or its expiry) deletes everything in it;
  - document uploads ingested by a durable job queue (`SKIP LOCKED` leases,
    retries, resume after a restart);
  - search over your uploads, the sample corpus, or both (`scope`), with
    pgvector + Postgres full text fused by RRF;
  - conversation history and a daily demo budget that survive restarts;
  - tenant isolation tested end to end; the API still answers sample questions
    while the database is down.
- Observability:
  - Prometheus `/metrics` covers request and stage latency, LLM calls and
    tokens per model, and agent, tool, cache and job outcomes;
  - every answer reports its real token usage and per-stage timings;
  - optional OpenTelemetry tracing (console or OTLP) is off by default.
- Public demo mode: per-client rate limits and a flood guard, each visitor's
  daily share of a global budget of live LLM queries, recorded sample answers
  (labelled as cached), a retrieval-only `/search` endpoint, and a Next.js web
  UI (`web/`).
- Load tests (`loadtests/`, Locust with an LLM stand-in) with recorded results,
  run in CI against the real image at the free plan's CPU and memory
  ([docs/load-testing.md](docs/load-testing.md)).
- Web UI (`web/`, static export), with four pages:
  - **Ask:** answers stream live stage by stage (retrieval, guardrail, each
    attempt and its grade), with search over the sample docs, your uploads, or
    both, plus a conversation thread and saved history;
  - **My documents:** a private 24-hour session, drag-and-drop uploads with
    live ingestion status, per-document delete and "delete all my data";
  - **Agent:** the plan, tool calls, cited evidence, verification and token
    usage, streamed as the agent works;
  - **How it works:** the pipeline and the measured retrieval results.
- Real self-correction: a bounded generate/evaluate loop that feeds the
  evaluator's critique back to the generator and never regresses to a worse
  answer than one it already produced. Every attempt is returned for inspection.
- Honest scoring: evaluator outages are reported as `unavailable`, never as a
  fabricated low score.
- Pluggable vector store: dependency-free NumPy backend by default (exact cosine,
  `argpartition` top-k), optional FAISS backend for large corpora.
- Provider-agnostic LLM: any OpenAI-compatible endpoint (Groq by default, or
  OpenAI / vLLM / Ollama) via config, with timeouts and retries.
- Proper ingestion: overlapping, boundary-aware chunking; batched embedding;
  incremental upserts (`reset=false` skips unchanged documents and replaces
  changed ones); atomic index publication that concurrent queries and other
  worker processes pick up safely.
- REST API (FastAPI) and CLI (Typer) over the same core.
- Typed configuration (pydantic-settings), structured logging (optional JSON)
  with per-request IDs, robust output parsing (decimals, percentages,
  fractions), and a sanitized error contract that never leaks internals.
- Fully tested: 236 tests run with fakes (no network, no torch) and in CI across
  Python 3.10-3.12, plus a job against a real Postgres with pgvector, and lint,
  format and type checks. Dockerfile and
  docker-compose included.

## Install

```bash
# 1. Clone and create a virtual environment
git clone https://github.com/syed-waleed-ahmed/Self-Correcting-RAG.git
cd Self-Correcting-RAG
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1

# 2. Install with the lightweight ONNX embedder (no torch) and PDF/DOCX support
pip install -e ".[onnx,docs]"        # add ,dev for tests + lint: ".[onnx,docs,dev]"

# 3. Configure your key
cp .env.example .env                 # Windows: copy .env.example .env
#     then edit .env and set GROQ_API_KEY
```

Optional extras: `.[local]` (torch-based embedder; same vectors, about twice the
memory), `.[faiss]` (FAISS backend), `.[postgres]` (sessions, uploads, history),
`.[auth]` (JWT admin auth), `.[dev]` (tests + lint), `.[all]`.
VS Code users get a pre-wired interpreter/test config in `.vscode/settings.json`.

> Get a free Groq API key at https://console.groq.com. Any OpenAI-compatible
> endpoint works - set `SCRAG_LLM_BASE_URL` and the model names accordingly.

## Quickstart

### CLI

```bash
scrag ingest                                   # build the index from data/docs
scrag query "What triggers self-correction?"   # ask a question
scrag eval --k 5                               # retrieval metrics: dense vs hybrid
scrag info                                     # show config + index status
```

### REST API

```bash
scrag serve                                    # http://localhost:8000 (docs at /docs)
```

```bash
# Build the index, then ask a question.
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"reset": true}'

curl -X POST localhost:8000/query -H 'content-type: application/json' \
     -d '{"query": "What does the guardrail agent do?"}'
```

Example `/query` response:

```json
{
  "query": "What does the guardrail agent do?",
  "answer": "The guardrail agent examines retrieved passages and drops those that are off-topic or potentially harmful before they are used to generate an answer.",
  "evaluation_score": 1.0,
  "evaluation_explanation": "The answer is fully supported by the context.",
  "evaluation_status": "passed",
  "attempts": 1,
  "best_attempt": 1,
  "attempt_history": [
    { "number": 1, "answer": "The guardrail agent examines...", "score": 1.0,
      "explanation": "The answer is fully supported by the context.", "evaluation_status": "ok" }
  ],
  "used_chunks": [
    { "source": "doc2.txt", "text": "Guardrail agents are LLM components...", "score": 0.64, "guardrail_score": 0.9 }
  ]
}
```

- `attempts` is the number of generate/evaluate attempts executed; `best_attempt`
  is the one whose answer was returned.
- `evaluation_status` is `passed` (score cleared `SCRAG_EVAL_THRESHOLD`),
  `below_threshold`, `unavailable` (the evaluator failed, so `evaluation_score`
  is `null`) or `skipped` (no context was retrieved). `passed` means an LLM judge
  scored the answer as grounded; it is a heuristic signal, not verification.
- `guardrail_score` is `null` when a chunk could not be scored (the guardrail
  fails open and keeps it).

### Docker

```bash
export GROQ_API_KEY=your_key_here
docker compose up --build                      # API on http://localhost:8000
```

## API reference

Auth is enforced only when `SCRAG_API_AUTH_TOKEN` (or JWT admins) is set (see
[Security](#security)). All routes are served under `/api/v1`; the original
unversioned paths still work for the routes that predate it.

| Method | Path      | Auth | Description                                       |
| ------ | --------- | :--: | ------------------------------------------------- |
| GET    | `/health` |  no  | Liveness, index size, model + backend info        |
| GET    | `/metrics` | operator on the demo | Prometheus metrics                  |
| GET    | `/demo`   |  no  | Sample questions, demo limits, today's budget     |
| POST   | `/search` | yes  | Retrieval only (no LLM): ranked chunks with scores |
| POST   | `/query`  | yes  | Run the full self-correcting pipeline for a query |
| POST   | `/agent/run` | yes | Bounded agent: plan, tools, write, verify       |
| POST   | `/query/stream`, `/agent/run/stream` | yes | The same, streamed as Server-Sent Events |
| GET    | `/agent/tools` | no | Tool catalog with permissions and input schemas |
| POST   | `/ingest` | yes  | (Re)build the index from the configured docs dir  |
| GET    | `/docs`   |  no  | Interactive OpenAPI (Swagger) UI                  |

With a database configured (`SCRAG_DATABASE_URL`), these session routes exist
under `/api/v1` (send the token as `X-Session-Token`):

| Method | Path | Description |
| ------ | ---- | ----------- |
| POST   | `/sessions` | Start an anonymous session; returns the token once |
| GET / DELETE | `/sessions/me` | Usage and limits / delete the session and all its data now |
| POST   | `/documents` | Upload a file (multipart `file`); `202` with an ingestion job |
| GET    | `/documents`, `/documents/{id}` | Your documents and their status |
| DELETE | `/documents/{id}` | Delete a document and its chunks |
| GET    | `/jobs/{id}` | Ingestion job status |
| GET / DELETE | `/conversations`, `/conversations/{id}` | Saved history |
| GET    | `/admin/stats` | Row counts (operator only) |

The `/stream` variants send `event: progress` once per stage (pipeline) or per
trace event (agent). They end with `event: result`, which carries the same body
as the plain endpoint, or with `event: error`, which uses the usual error shape
plus `status`. Validation, rate-limit and session checks still fail as ordinary
HTTP errors before the stream starts.

`POST /query`, `POST /search` and `POST /agent/run` take `scope`: `samples` (default), `session`
(your uploads) or `all` (both, fused). With a session, `/query` saves the turn
and returns `conversation_id`; pass it back to continue the conversation.

`POST /ingest` takes `{"reset": true}` (default) to rebuild the index from the
docs directory, or `{"reset": false}` to upsert: unchanged documents are skipped,
changed ones re-embedded, new ones added, and nothing is deleted. The response
reports `added_documents`, `updated_documents` and `skipped_documents`. One
ingestion runs at a time per process; a concurrent request gets `409`.

Errors (other than `422` request validation) share one shape, and every
response carries an `X-Request-ID` header that matches the server logs:

```json
{ "error": "llm_error", "detail": "The language model provider request failed. Please retry.", "request_id": "3f9c..." }
```

| Status | `error` codes                                                                    |
| ------ | -------------------------------------------------------------------------------- |
| 400    | `invalid_query`, `query_too_long`                                                |
| 401    | `unauthorized`, `session_required`                                               |
| 403    | `forbidden`                                                                      |
| 404    | `not_found` (also for another session's resources)                               |
| 409    | `index_empty`, `ingest_in_progress`, `documents_not_found`, `no_documents`, `session_quota_exceeded` |
| 411 / 413 / 415 / 422 | `length_required`, `upload_too_large` / `document_too_long`, `unsupported_file_type`, `unreadable_document` |
| 429    | `rate_limited`, `demo_budget_exhausted`                                          |
| 500    | `internal_error`, `configuration_error`, `embedding_dimension_mismatch`, `index_incompatible` |
| 502    | `llm_error`                                                                      |
| 503    | `llm_not_configured`, `llm_rate_limited`, `persistence_disabled`, `database_unavailable`, `storage_full` |

## Configuration

All settings are environment-driven (prefix `SCRAG_`), with a `.env` fallback.
The API key also accepts the conventional `GROQ_API_KEY` / `OPENAI_API_KEY`.
See [`.env.example`](.env.example) for the full list.

> Groq retired `llama-3.1-8b-instant` and `llama-3.3-70b-versatile` for free and
> developer tiers on 2026-08-16. The defaults below were verified against Groq's
> free tier on 2026-11-10. Provider model lists change often; if a model ID
> starts failing, check https://console.groq.com/docs/models.

Highlights:

| Variable                       | Default                          | Meaning                                 |
| ------------------------------ | -------------------------------- | --------------------------------------- |
| `GROQ_API_KEY`                 | (none)                           | LLM API key (required for `/query`)     |
| `SCRAG_LLM_BASE_URL`           | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint              |
| `SCRAG_GENERATOR_MODEL`        | `qwen/qwen3.8-27b`               | Generation model (also `SCRAG_EVALUATOR_MODEL`, `SCRAG_GUARDRAIL_MODEL`) |
| `SCRAG_VECTOR_BACKEND`         | `numpy`                          | `numpy` or `faiss`                      |
| `SCRAG_TOP_K`                  | `5`                              | Chunks retrieved per query              |
| `SCRAG_RETRIEVAL_MODE`         | `hybrid`                         | `hybrid` (BM25 + dense, RRF) or `dense` |
| `SCRAG_EMBEDDING_BACKEND`      | `auto`                           | `fastembed`, `sentence-transformers` or `auto` |
| `SCRAG_GUARDRAIL_MODE`         | `batch`                          | `batch` (one call) or `per_chunk`       |
| `SCRAG_DEMO_MODE`              | `false`                          | Rate limits, daily budget, query caps, operator-only ingest |
| `SCRAG_DEMO_DAILY_QUERY_BUDGET`| `30`                             | Live pipeline runs per UTC day (demo)   |
| `SCRAG_DEMO_CLIENT_DAILY_RUNS` | `10`                             | Each client's share of that budget (demo) |
| `SCRAG_GUARDRAIL_THRESHOLD`    | `0.3`                            | Min relevance to keep a chunk           |
| `SCRAG_GUARDRAIL_MAX_TOKENS`   | `8`                              | Guardrail reply budget (use 256 for reasoning models such as `openai/gpt-oss-20b`) |
| `SCRAG_EVAL_THRESHOLD`         | `0.7`                            | Min grounding score to accept an answer |
| `SCRAG_MAX_SELF_CORRECT_STEPS` | `2`                              | Max generate/evaluate attempts (1-10)   |
| `SCRAG_API_AUTH_TOKEN`         | (unset)                          | If set, `/query` + `/ingest` require it |
| `SCRAG_MAX_REQUEST_BYTES`      | `65536`                          | Largest request body outside uploads    |
| `SCRAG_HTTP_TOOL_ALLOWED_HOSTS`| `[]`                             | Hosts the agent's `web_fetch` may GET over HTTPS (empty = tool off) |
| `SCRAG_DATABASE_URL`           | (unset)                          | Postgres + pgvector for sessions, uploads, history (also `DATABASE_URL`) |
| `SCRAG_SESSION_TTL_HOURS`      | `24`                             | Sessions and their data are deleted after this |
| `SCRAG_SESSION_MAX_DOCUMENTS`  | `5`                              | Uploads per session (`SCRAG_UPLOAD_MAX_BYTES` caps each, default 1 MB) |
| `SCRAG_ADMIN_SUBJECTS`         | `[]`                             | JWT `sub`s with operator rights (with `SCRAG_AUTH_JWKS_URL` or `SCRAG_AUTH_JWT_SECRET`) |
| `SCRAG_OTEL_EXPORTER`          | `none`                           | Tracing: `none`, `console` or `otlp` (needs the `otel` extra) |
| `SCRAG_LLM_PRICES`             | `{}`                             | List prices per model (USD per million tokens) for cost estimates |

## Security

- Secrets are read from the environment / `.env` only - never hard-coded and
  never logged. `.env` is git-ignored; commit `.env.example` instead.
- Optional API auth: set `SCRAG_API_AUTH_TOKEN` to require a shared secret on
  `/query` and `/ingest` (`Authorization: Bearer ...` or `X-API-Key: ...`),
  compared in constant time. `/health` stays open for probes.
- Found a vulnerability? Report it privately, as described in
  [SECURITY.md](SECURITY.md), not in a public issue. The
  [threat model](docs/threat-model.md) lists every control and its residual
  risk; [ADR 0011](docs/decisions/0011-security-hardening.md) explains the
  design.
- Restrict CORS in production via `SCRAG_CORS_ALLOW_ORIGINS` (defaults to all
  origins for easy local use).
- No arbitrary file access: ingestion only reads the server-configured
  `docs_dir`. Uploads are size-checked before the body is parsed, their bytes
  must match their extension, DOCX containers are checked for zip bombs before
  inflating, their names are reduced to a safe base name, and only extracted
  text is stored. Every other request body is capped too.
- Agent tools are read-only by construction:
  - SQL runs on a private in-memory catalog under an authorizer that allows
    only SELECT over two tables, with a time budget;
  - web fetching is off unless hosts are allow-listed, and never reaches
    private or internal addresses (checked after DNS and pinned for the
    connection, so DNS rebinding can't get around it).
- Layered quotas on the demo: a flood guard before routing, per-client rate
  limits, each visitor's daily share, a global daily budget, and caps on
  stored documents and live sessions.
- API responses carry `Cache-Control: no-store`, `nosniff`, a deny-all CSP and
  `X-Frame-Options: DENY`; secrets never appear in `repr(settings)` or logs.
- Questions, uploads and answers are sent to the configured LLM provider
  (Groq on the demo). The UI says so.
- Sessions: tokens are 256-bit random values stored only as SHA-256 hashes;
  every query is scoped by session, and other sessions' data answers 404.
- Optional JWT admin auth (e.g. Supabase Auth): algorithms come from the
  configuration, never from the token, and `exp`, `sub` and `aud` are required.
- The container image runs as a non-root user.
- The web UI is a static export with no secrets. The Render site sends a
  Content-Security-Policy that allows network calls only to the API, plus
  `X-Frame-Options: DENY`, `Referrer-Policy` and `Permissions-Policy`.
- Repository protections:
  - secret scanning with push protection, so a commit containing a credential
    is rejected;
  - CodeQL code scanning;
  - Dependabot alerts and security updates;
  - a CI dependency audit (`pip-audit`, `npm audit`);
  - `main` only changes through pull requests with passing CI.

## Project structure

```text
src/scrag/
  config.py            typed settings (pydantic-settings)
  models.py            pydantic domain + API schemas
  chunking.py          boundary-aware overlapping chunker
  loaders.py           text extraction: txt/md/html/csv/json/pdf/docx
  embeddings.py        Embedder protocol + lazy sentence-transformers backend
  vectorstore/         VectorStore ABC, numpy (default), faiss (optional),
                       atomic generation-based persistence
  llm/                 provider-agnostic OpenAI-compatible ChatClient
  agents/              guardrail, generator, evaluator (+ robust parsing)
  errors.py            domain errors -> HTTP status + client-safe message
  retriever.py         dense and hybrid (BM25 + dense, RRF) retrievers
  sparse.py            in-process BM25 index + Reciprocal Rank Fusion
  evaluation.py        golden-set retrieval/answer metrics (`scrag eval`)
  demo.py              rate limiter, daily budget, answer cache
  observability/       Prometheus metrics + optional OpenTelemetry tracing
  evaluation_compare.py  answer-quality comparison across configurations
  persistence/         Postgres: sessions, uploads, history, job queue,
                       persistent budget, SQL migrations (up + down)
  auth.py              bearer-JWT verification for admins
  orchestration/       tool registry + bounded agent workflow
  usage.py             per-run LLM token accounting
  ingest.py            load -> chunk -> embed -> index
  pipeline.py          self-correcting orchestrator (keeps best attempt)
  container.py         composition root / dependency wiring
  api/                 FastAPI app factory, routes, session routes, deps
  cli.py               Typer CLI (ingest / query / serve / info / agent / eval / db)
scripts/verify.py      single command for every CI quality gate
scripts/smoke.py       smoke test for a running API (local or deployed)
scripts/record_samples.py  record real runs for the demo's sample questions
loadtests/             Locust profiles, an LLM stand-in and a results summarizer
web/                   Next.js + Tailwind UI (static export): ask, documents, agent, about
data/eval/             golden dataset for `scrag eval`
data/demo/             recorded sample answers served by the demo
docs/                  evaluation, deployment, threat model, load testing,
                       decisions (ADRs), planning
tests/                 pytest suite (fakes; no network or torch)
```

## Development

```bash
pip install -e ".[onnx,docs,dev]"
pre-commit install              # ruff, format, secret and whitespace checks on every commit
python scripts/verify.py        # lint + format check + mypy + tests (what CI runs)
python scripts/verify.py --fix  # apply ruff fixes/formatting first
```

CI also runs CodeQL and a dependency audit (`pip-audit` on what the Docker
image ships, `npm audit` on the web UI). Load tests (no API key needed; an LLM
stand-in replaces the provider) are described in
[docs/load-testing.md](docs/load-testing.md):

```bash
pip install -e ".[load]"
python loadtests/mock_llm.py &   # OpenAI-compatible stand-in on :8090
SCRAG_LLM_API_KEY=mock SCRAG_LLM_BASE_URL=http://127.0.0.1:8090/v1 scrag serve &
locust -f loadtests/locustfile.py --headless -u 20 -r 5 -t 2m --host http://127.0.0.1:8000 Visitor
```

The 335 tests run in seconds with no network or API key. Two groups skip
unless their dependency is present, and CI runs both:
- the 9 FAISS tests need `faiss-cpu`;
- the 27 Postgres tests need `TEST_DATABASE_URL` pointing at a disposable
  Postgres with pgvector, e.g.
  `docker run -e POSTGRES_PASSWORD=pg -p 5432:5432 pgvector/pgvector:pg17` and
  `TEST_DATABASE_URL=postgresql://postgres:pg@localhost:5432/postgres`.

The test suite injects a fake embedder and a scriptable fake LLM client, so it
runs with no API key, no network, and no torch - the same path CI uses on Python
3.10, 3.11 and 3.12. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow,
[CHANGELOG.md](CHANGELOG.md) for release notes, and
[docs/planning/](docs/planning/) for the audit and roadmap toward a hosted
platform.

## Scaling notes

- The NumPy backend does exact brute-force cosine search with O(n) top-k
  selection - comfortable into the hundreds-of-thousands-of-chunks range. Beyond
  that, switch `SCRAG_VECTOR_BACKEND=faiss` (swap `IndexFlatIP` for IVF/HNSW for
  approximate search at very large scale - the persistence contract is unchanged).
- Guardrail scoring is fanned out across a thread pool, so per-query latency
  stays roughly flat as `top_k` grows.
- The LLM layer is stateless and provider-agnostic, so the API scales
  horizontally behind a load balancer; the vector index is loaded per worker or
  externalized via FAISS.

## Contributing

Contributions are welcome: bug reports, documentation fixes, new retrieval or
embedding backends, evaluation data, and UI work.
- Start with [CONTRIBUTING.md](CONTRIBUTING.md): setup, the checks CI runs, and
  conventions.
- Read the architecture decisions in [docs/decisions/](docs/decisions/) and the
  roadmap in [docs/planning/](docs/planning/IMPLEMENTATION_ROADMAP.md).
- Taking part means following the [code of conduct](CODE_OF_CONDUCT.md).
- Report security issues privately ([SECURITY.md](SECURITY.md)).

To run your own copy of the demo, follow
[docs/deployment.md](docs/deployment.md#deploy-your-own-copy). It takes a
fork, a free Render account and a free Groq key.

## License

[MIT](LICENSE) (c) Syed Waleed Ahmed
