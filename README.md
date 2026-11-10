# Self-Correcting RAG

A production-grade, multi-agent Retrieval-Augmented Generation pipeline that
retrieves context, guardrails it for relevance, generates a grounded answer, and
then evaluates and self-corrects its own output - exposed as a REST API, a CLI
and a web UI.

**Live demo:** https://scrag-web.onrender.com. The API is at
https://scrag-api.onrender.com (docs at `/docs`). Both run on Render's free
tier. The API sleeps after 15 idle minutes, so the first request can take about
a minute.

![The live SCRAG demo answering a sample question: the answer is labelled as a
recorded run, and the self-check panel shows a passed evaluation](docs/images/live-demo.jpg)

*The deployed demo, captured 2026-11-10. Sample questions are answered from
recorded runs; "Run live" executes the pipeline.*

[![CI](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
[![Ruff](https://img.shields.io/badge/lint-ruff-261230)](https://github.com/astral-sh/ruff)
![Tests](https://img.shields.io/badge/tests-187-brightgreen)

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
  Measured on the golden set: MRR 0.915 -> 0.961 at k=5 versus dense only
  ([docs/evaluation.md](docs/evaluation.md)).
- Evaluation harness: `scrag eval` reports recall@k, hit rate, MRR and nDCG on
  a 40-question golden dataset, with optional answer-level evaluation.
- Formats: TXT, Markdown, HTML, CSV, JSON, and PDF/DOCX with the `docs` extra,
  with size and page limits.
- Bounded agent workflow (`POST /agent/run`, `scrag agent`):
  - a planner chooses tools from a typed, permissioned registry (document
    search, a safe calculator, and an approval-gated re-index);
  - tools run in parallel with timeouts and audit records;
  - a writer answers from delimited, untrusted evidence with citations;
  - the evaluator verifies the answer and can trigger one correction;
  - hard limits on steps, LLM calls and time, with token usage reported per run.
- Public demo mode: per-client rate limits, a global daily budget of live LLM
  queries, recorded sample answers (labelled as cached), a retrieval-only
  `/search` endpoint, and a Next.js web UI (`web/`).
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
- Fully tested: 187 tests run with fakes (no network, no torch) and in CI across
  Python 3.10-3.12, plus lint, format and type checks. Dockerfile and
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
memory), `.[faiss]` (FAISS backend), `.[dev]` (tests + lint), `.[all]`.
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

Auth is enforced only when `SCRAG_API_AUTH_TOKEN` is set (see [Security](#security)).

| Method | Path      | Auth | Description                                       |
| ------ | --------- | :--: | ------------------------------------------------- |
All routes are served under `/api/v1`. The original unversioned paths still
work.

| Method | Path      | Auth | Description                                       |
| ------ | --------- | :--: | ------------------------------------------------- |
| GET    | `/health` |  no  | Liveness, index size, model + backend info        |
| GET    | `/demo`   |  no  | Sample questions, demo limits, today's budget     |
| POST   | `/search` | yes  | Retrieval only (no LLM): ranked chunks with scores |
| POST   | `/query`  | yes  | Run the full self-correcting pipeline for a query |
| POST   | `/agent/run` | yes | Bounded agent: plan, tools, write, verify       |
| GET    | `/agent/tools` | no | Tool catalog with permissions and input schemas |
| POST   | `/ingest` | yes  | (Re)build the index from the configured docs dir  |
| GET    | `/docs`   |  no  | Interactive OpenAPI (Swagger) UI                  |

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
| 400    | `invalid_query`                                                                  |
| 401    | `unauthorized`                                                                   |
| 409    | `index_empty`, `ingest_in_progress`, `documents_not_found`, `no_documents`       |
| 500    | `internal_error`, `configuration_error`, `embedding_dimension_mismatch`, `index_incompatible` |
| 502    | `llm_error`                                                                      |
| 503    | `llm_not_configured`, `llm_rate_limited`                                         |

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
| `SCRAG_GUARDRAIL_THRESHOLD`    | `0.3`                            | Min relevance to keep a chunk           |
| `SCRAG_GUARDRAIL_MAX_TOKENS`   | `8`                              | Guardrail reply budget (use 256 for reasoning models such as `openai/gpt-oss-20b`) |
| `SCRAG_EVAL_THRESHOLD`         | `0.7`                            | Min grounding score to accept an answer |
| `SCRAG_MAX_SELF_CORRECT_STEPS` | `2`                              | Max generate/evaluate attempts (1-10)   |
| `SCRAG_API_AUTH_TOKEN`         | (unset)                          | If set, `/query` + `/ingest` require it |

## Security

- Secrets are read from the environment / `.env` only - never hard-coded and
  never logged. `.env` is git-ignored; commit `.env.example` instead.
- Optional API auth: set `SCRAG_API_AUTH_TOKEN` to require a shared secret on
  `/query` and `/ingest` (`Authorization: Bearer ...` or `X-API-Key: ...`),
  compared in constant time. `/health` stays open for probes.
- Restrict CORS in production via `SCRAG_CORS_ALLOW_ORIGINS` (defaults to all
  origins for easy local use).
- No arbitrary file access: ingestion only reads the server-configured
  `docs_dir`; request bodies never carry file paths.
- The container image runs as a non-root user.

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
  orchestration/       tool registry + bounded agent workflow
  usage.py             per-run LLM token accounting
  ingest.py            load -> chunk -> embed -> index
  pipeline.py          self-correcting orchestrator (keeps best attempt)
  container.py         composition root / dependency wiring
  api/                 FastAPI app factory + routes
  cli.py               Typer CLI (ingest / query / serve / info)
scripts/verify.py      single command for every CI quality gate
scripts/smoke.py       smoke test for a running API (local or deployed)
scripts/record_samples.py  record real runs for the demo's sample questions
web/                   Next.js + Tailwind demo UI (static export, Vercel)
data/eval/             golden dataset for `scrag eval`
data/demo/             recorded sample answers served by the demo
docs/                  evaluation, deployment, decisions (ADRs), planning
tests/                 pytest suite (fakes; no network or torch)
```

## Development

```bash
pip install -e ".[local,dev]"
python scripts/verify.py        # lint + format check + mypy + tests (what CI runs)
python scripts/verify.py --fix  # apply ruff fixes/formatting first
```

The 187 tests run in a few seconds with no network or API key. The 9 FAISS
tests are skipped unless `faiss-cpu` is installed; CI runs them in a separate
job.

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

## License

[MIT](LICENSE) (c) Syed Waleed Ahmed
