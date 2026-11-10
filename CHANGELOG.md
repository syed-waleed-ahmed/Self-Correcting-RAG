# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Phase 1 (core reliability) of the roadmap in `docs/planning/`.

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
