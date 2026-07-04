# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
