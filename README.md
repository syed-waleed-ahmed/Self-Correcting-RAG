# Self-Correcting RAG

A production-grade, multi-agent **Retrieval-Augmented Generation** pipeline that
retrieves context, guardrails it for relevance, generates a grounded answer, and
then **evaluates and self-corrects** its own output — exposed as both a REST API
and a CLI.

[![CI](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/workflows/ci.yml/badge.svg)](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Why this exists

Naive RAG retrieves some chunks, stuffs them into a prompt, and hopes the answer
is grounded. This project adds two feedback stages so the system can catch and
repair its own mistakes:

- A **guardrail agent** scores every retrieved chunk and drops off-topic context
  before it can pollute generation.
- An **evaluator agent** scores the generated answer against the context and, if
  it's poorly grounded, triggers a bounded **self-correction loop** — always
  keeping the best-scoring attempt.

## Architecture

```
                 ┌──────────────┐
   query ───────▶│  Retriever   │  local sentence-transformer embeddings
                 │ (vector store)│  → top-k chunks (cosine)
                 └──────┬───────┘
                        ▼
                 ┌──────────────┐
                 │  Guardrail   │  LLM relevance score per chunk (parallel)
                 │    agent     │  → keep ≥ threshold (fallback: raw retrieval)
                 └──────┬───────┘
                        ▼
          ┌────────────────────────────┐
          │  Generator ⇄ Evaluator     │  generate grounded answer, score it,
          │  self-correction loop      │  correct up to N times, keep best.
          └──────────────┬─────────────┘
                         ▼
                    final answer + evaluation + citations
```

Each stage is a small, independently testable component wired together by a
composition root (`scrag/container.py`), so backends and models are swappable
without touching orchestration logic.

## Features

- **Multi-agent pipeline** — retriever, guardrail, generator, evaluator.
- **Real self-correction** — bounded generate→evaluate loop that never regresses
  to a worse answer than one it already produced.
- **Pluggable vector store** — dependency-free NumPy backend by default (exact
  cosine, `argpartition` top-k), optional **FAISS** backend for large corpora.
- **Provider-agnostic LLM** — any OpenAI-compatible endpoint (Groq by default,
  or OpenAI / vLLM / Ollama) via config, with timeouts and retries.
- **Proper ingestion** — overlapping, boundary-aware chunking that scales past
  toy documents; batched embedding.
- **REST API** (FastAPI) **and CLI** (Typer) over the same core.
- **Typed configuration** (pydantic-settings), **structured logging** (optional
  JSON), robust output parsing, and clean error → HTTP mapping.
- **Fully tested** — 31 tests run with fakes (no network, no torch) and in CI
  across Python 3.10–3.12. **Dockerfile** + **docker-compose** included.

## Install

```bash
# 1. Clone and create a virtual environment
python -m venv .venv
source .venv/bin/activate            # Windows (PowerShell): .venv\Scripts\Activate.ps1

# 2. Install with the local embedding stack (sentence-transformers + torch)
pip install -e ".[local]"            # add ,dev for tests + lint: ".[local,dev]"

# 3. Configure your key
cp .env.example .env                 # Windows: copy .env.example .env
#     then edit .env and set GROQ_API_KEY
```

Optional extras: `.[faiss]` (FAISS backend), `.[dev]` (tests + lint), `.[all]`.
VS Code users get a pre-wired interpreter/test config in `.vscode/settings.json`.

> Get a free Groq API key at <https://console.groq.com>. Any OpenAI-compatible
> endpoint works — set `SCRAG_LLM_BASE_URL` and the model names accordingly.

## Quickstart — CLI

```bash
scrag ingest                                   # build the index from data/docs
scrag query "What triggers self-correction?"   # ask a question
scrag info                                      # show config + index status
```

## Quickstart — REST API

```bash
scrag serve                     # http://localhost:8000  (docs at /docs)
```

```bash
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"reset": true}'

curl -X POST localhost:8000/query -H 'content-type: application/json' \
     -d '{"query": "What does the guardrail agent do?"}'
```

| Method | Path      | Description                                       |
| ------ | --------- | ------------------------------------------------- |
| GET    | `/health` | Liveness, index size, model + backend info        |
| POST   | `/ingest` | (Re)build the index from the configured docs dir  |
| POST   | `/query`  | Run the full self-correcting pipeline for a query |
| GET    | `/docs`   | Interactive OpenAPI (Swagger) UI                  |

## Quickstart — Docker

```bash
export GROQ_API_KEY=your_key_here
docker compose up --build
# → API on http://localhost:8000
```

## Configuration

All settings are environment-driven (prefix `SCRAG_`), with a `.env` fallback.
The API key also accepts the conventional `GROQ_API_KEY` / `OPENAI_API_KEY`.
See [`.env.example`](.env.example) for the full list. Highlights:

| Variable                       | Default                          | Meaning                                |
| ------------------------------ | -------------------------------- | -------------------------------------- |
| `GROQ_API_KEY`                 | —                                | LLM API key (required for `/query`)    |
| `SCRAG_LLM_BASE_URL`           | `https://api.groq.com/openai/v1` | OpenAI-compatible endpoint             |
| `SCRAG_GENERATOR_MODEL`        | `llama-3.1-8b-instant`           | Model for generation/guardrail/eval    |
| `SCRAG_VECTOR_BACKEND`         | `numpy`                          | `numpy` or `faiss`                     |
| `SCRAG_TOP_K`                  | `5`                              | Chunks retrieved per query             |
| `SCRAG_GUARDRAIL_THRESHOLD`    | `0.3`                            | Min relevance to keep a chunk          |
| `SCRAG_EVAL_THRESHOLD`         | `0.7`                            | Min grounding score to accept an answer|
| `SCRAG_MAX_SELF_CORRECT_STEPS` | `2`                              | Max generate→evaluate attempts         |
| `SCRAG_API_AUTH_TOKEN`         | _(unset)_                        | If set, `/query` + `/ingest` require it |

## Security

- **Secrets** are read from the environment / `.env` only — never hard-coded and
  never logged. `.env` is git-ignored; commit `.env.example` instead.
- **Optional API auth:** set `SCRAG_API_AUTH_TOKEN` to require a shared secret on
  `/query` and `/ingest` (`Authorization: Bearer …` or `X-API-Key: …`), compared
  in constant time. `/health` stays open for probes.
- **Restrict CORS** in production via `SCRAG_CORS_ALLOW_ORIGINS` (defaults to all
  origins for easy local use).
- **No arbitrary file access:** ingestion only reads the server-configured
  `docs_dir`; request bodies never carry file paths.
- The container image runs as a **non-root** user.

## Project structure

```text
src/scrag/
  config.py            typed settings (pydantic-settings)
  models.py            pydantic domain + API schemas
  chunking.py          boundary-aware overlapping chunker
  embeddings.py        Embedder protocol + lazy sentence-transformers backend
  vectorstore/         VectorStore ABC · numpy (default) · faiss (optional)
  llm/                 provider-agnostic OpenAI-compatible ChatClient
  agents/              guardrail · generator · evaluator (+ robust parsing)
  retriever.py         query → embed → search
  ingest.py            load → chunk → embed → index
  pipeline.py          self-correcting orchestrator (keeps best attempt)
  container.py         composition root / dependency wiring
  api/                 FastAPI app factory + routes
  cli.py               Typer CLI (ingest / query / serve / info)
tests/                 pytest suite (fakes; no network or torch)
```

## Testing

```bash
pip install -e ".[dev]"
ruff check .
pytest
```

The suite injects a fake embedder and a scriptable fake LLM client, so it runs
in well under a second with no API key, no network, and no torch — the same path
CI uses on Python 3.10, 3.11 and 3.12.

## Scaling notes

- The NumPy backend does exact brute-force cosine search with `O(n)` top-k
  selection — comfortable into the hundreds-of-thousands-of-chunks range. Beyond
  that, switch `SCRAG_VECTOR_BACKEND=faiss` (swap `IndexFlatIP` for IVF/HNSW for
  approximate search at very large scale — the persistence contract is unchanged).
- Guardrail scoring is fanned out across a thread pool, so per-query latency
  stays roughly flat as `top_k` grows.
- The LLM layer is stateless and provider-agnostic, so the API scales
  horizontally behind a load balancer; the vector index is memory-mapped per
  worker or externalized via FAISS.

## License

[MIT](LICENSE) © Syed Waleed Ahmed
