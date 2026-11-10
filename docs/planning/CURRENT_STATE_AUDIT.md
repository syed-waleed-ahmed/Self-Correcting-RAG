# Current State Audit

Phase 0 of the roadmap. Audit date: 2026-11-10. Baseline: commit `b02d357` on
`main` (package version 0.1.0). The Phase 1 fixes described at the end are on
branch `feat/phase-0-1-audit-reliability`.

## 1. What the project is

SCRAG is a Python 3.10+ package with a FastAPI service and a Typer CLI over one
core. It answers questions over a local document corpus. A query is embedded and
matched against an in-process vector index. An LLM "guardrail" scores each
retrieved chunk and drops irrelevant ones. A generator writes an answer from
the kept chunks, and an evaluator scores how well that answer is grounded. If
the score is below a threshold, the answer is regenerated, up to a bounded
number of times, and the best-scoring attempt is returned.

## 2. Inventory (baseline)

| Path | Responsibility |
| --- | --- |
| `src/scrag/config.py` | `Settings` (pydantic-settings, `SCRAG_` env prefix, `.env` fallback) |
| `src/scrag/models.py` | Pydantic domain models and API schemas |
| `src/scrag/chunking.py` | Paragraph/sentence-aware overlapping character chunker |
| `src/scrag/embeddings.py` | `Embedder` protocol; lazy `SentenceTransformerEmbedder` (torch) |
| `src/scrag/vectorstore/` | `VectorStore` ABC; NumPy (default) and FAISS backends |
| `src/scrag/llm/client.py` | `ChatClient` protocol; `OpenAICompatibleClient` (Groq default) |
| `src/scrag/agents/` | Guardrail, generator, evaluator; defensive output parsing |
| `src/scrag/retriever.py` | Embed query, then search the store |
| `src/scrag/ingest.py` | Load `.txt`/`.md`, chunk, embed in batches, save |
| `src/scrag/pipeline.py` | Self-correcting orchestrator |
| `src/scrag/container.py` | Composition root (`build_container`) |
| `src/scrag/api/` | `create_app` factory and routes `/health`, `/query`, `/ingest` |
| `src/scrag/cli.py` | `scrag ingest | query | serve | info` |
| `tests/` | 31 tests using a fake embedder and a scriptable fake LLM |
| `Dockerfile`, `docker-compose.yml` | CPU-only torch image, non-root, health check |
| `.github/workflows/ci.yml` | ruff check + pytest on Python 3.10, 3.11, 3.12 |
| `data/docs/` | 3 sample documents (about 1.2 KB in total) |
| `graphify-out/` | Generated knowledge-graph artifacts (not runtime code) |

## 3. Public interfaces recorded at baseline

These are the extension points other code depends on. Phase 1 kept every one of
them working; additions are noted in section 7.

```python
class Embedder(Protocol):
    dim: int                                   # property
    def embed(self, texts: list[str]) -> np.ndarray: ...       # (n, dim) float32

class VectorStore(ABC):
    dim: int
    def add(self, vectors, chunks) -> None
    def search(self, query_vector, top_k) -> list[RetrievedChunk]
    def clear(self) -> None
    def save(self) -> None
    def load(self) -> bool
    def __len__(self) -> int

class ChatClient(Protocol):
    def complete(self, *, model, system, user, temperature=0.2,
                 max_tokens=None, json_mode=False) -> str: ...

build_container(settings, *, embedder=None, store=None, chat_client=None) -> Container
create_app(settings=None, *, container=None) -> FastAPI
SelfCorrectingRAGPipeline.run(query, top_k=None) -> PipelineResult
```

| HTTP | Request | Response (baseline) |
| --- | --- | --- |
| `GET /health` | none | `status, version, llm_configured, embedding_model, vector_backend, index_size` |
| `POST /ingest` | `{reset: bool = true}` | `documents, chunks, index_size` |
| `POST /query` | `{query: str, top_k?: 1..100}` | `query, answer, evaluation_score, evaluation_explanation, attempts, used_chunks[]` |

## 4. Dependencies

Declared in `pyproject.toml`:

- Core: fastapi, uvicorn[standard], typer, openai, numpy, pydantic, pydantic-settings, python-dotenv.
- Extras:
  - `local`: sentence-transformers, torch
  - `faiss`: faiss-cpu
  - `dev`: pytest, pytest-cov, httpx, ruff, mypy

Installed in the development venv on 2026-11-10: Python 3.10.11, fastapi 0.139.0,
starlette 1.3.1, pydantic 2.13.4, pydantic-settings 2.14.2, openai 2.44.0,
numpy 2.2.6, sentence-transformers 5.6.0, torch 2.12.1, typer 0.26.8,
uvicorn 0.50.0, pytest 9.1.1, ruff 0.15.20, mypy 2.1.0.

## 5. Baseline quality results (commit `b02d357`)

| Check | Result | In CI? |
| --- | --- | --- |
| `pytest` | 31 passed | yes |
| `ruff check .` | clean | yes |
| `ruff format --check .` | 14 files would be reformatted | no |
| `mypy src` | 7 errors: 6 unused `type: ignore`, 1 real overload error in `llm/client.py:88` | no |

Live provider checks (2026-11-10, with the configured Groq key; listing models
uses no tokens, and the smoke test used a few hundred):

| Check | Result |
| --- | --- |
| Models available to the key | 11. `llama-3.1-8b-instant` (the configured default) is **not** among them |
| `openai/gpt-oss-20b`, guardrail call with `max_tokens=8` | **empty content**, `finish_reason=length`: hidden reasoning used up the budget |
| `openai/gpt-oss-20b`, `max_tokens=256` / JSON mode | works (about 105 completion tokens per guardrail call) |
| `qwen/qwen3.8-27b`, all three agent call shapes | works; guardrail used 4 completion tokens |
| End-to-end `scrag query` with qwen on the existing index | answer returned, score 1.00 on attempt 1, legacy index loaded |

## 6. Defects found

The baseline line numbers refer to commit `b02d357`. Every defect below is fixed
on the Phase 1 branch and covered by a regression test.

| # | Defect | Baseline location | Severity |
| --- | --- | --- | --- |
| 1 | Default model `llama-3.1-8b-instant` was retired for free/developer tiers on 2026-08-16, so every `/query` fails on a free key | `config.py:44-46` | Critical |
| 2 | Retries never receive the evaluator's explanation, so self-correction is blind | `generator.py:36-42`, `pipeline.py:75` | High |
| 3 | `attempts` reports the winning attempt's number, not how many attempts ran | `pipeline.py:96` | Medium |
| 4 | An evaluator outage or unparseable reply becomes a score of `0.0`. That triggers a pointless retry and reports a fabricated metric | `evaluator.py:49-55` | High |
| 5 | Score parsing takes the first number and clamps it: `8/10` and `85%` both become 1.0 | `_parsing.py:16-25` | Medium |
| 6 | `/ingest` clears the store and then adds batches, so a concurrent `/query` sees an empty or partial index. Saves are not atomic, and nothing stops two ingests from running at once | `ingest.py:81-89`, `numpy_store.py:75-83` | High |
| 7 | With several workers, each holds its own index; `/ingest` updates only the worker that served it | `container.py`, `api/app.py:45` | High |
| 8 | `reset=false` appends every chunk again (duplicates) | `ingest.py:81-87` | Medium |
| 9 | When the real embedding dimension differs from config, the embedder silently adopts it and the store then rejects the vectors | `embeddings.py:71-78` | Medium |
| 10 | A missing docs directory raises `FileNotFoundError`, which becomes a 500 | `ingest.py:34-35` | Low |
| 11 | `ValueError -> 400` returns the raw exception text. `LLMError -> 502` returns the provider's message | `api/app.py:69-79` | Medium (information leak) |
| 12 | A non-ASCII API token crashes `secrets.compare_digest` with `TypeError`, giving a 500 | `api/routes.py:43` | Low |
| 13 | A generator failure on a retry discards an earlier successful answer | `pipeline.py:75` | Medium |
| 14 | A guardrail failure is reported as `guardrail_score = 1.0` (a fabricated metric) | `guardrail.py:56-60` | Low |
| 15 | One non-UTF-8 file aborts the whole ingestion | `ingest.py:40` | Low |
| 16 | `Settings(llm_api_key=...)` from code is silently ignored (only the env alias names work). The test fixture relied on it | `config.py:23-39` | Low |
| 17 | A reasoning model with the hard-coded guardrail `max_tokens=8` returns empty output, so the guardrail silently stops filtering | `guardrail.py:54` | Medium |

## 7. What Phase 1 changed (summary)

See `CHANGELOG.md` (Unreleased) for the full list.

- **Default model:** `qwen/qwen3.8-27b`.
- **Self-correction:** the evaluator's critique is fed back on retries.
- **Attempt reporting:** `attempts`, `best_attempt` and `attempt_history`.
- **Evaluation status:** an explicit `evaluation_status`, with `null` scores
  instead of fabricated ones.
- **Error handling:**
  - a domain error hierarchy (`scrag/errors.py`);
  - a sanitized error contract with `X-Request-ID`;
  - the request ID in log lines.
- **Persistence:**
  - immutable index generations plus an atomic `CURRENT` pointer (the legacy
    flat layout still loads);
  - lock-free snapshot reads;
  - cross-process refresh;
  - one ingest per process.
- **Ingestion:**
  - incremental upsert by content hash;
  - embedding shape and finiteness checks;
  - clean errors for a missing or empty docs directory.
- **Interface additions:**
  - `VectorStore.replace()`, `snapshot()` and `refresh_if_stale()`;
  - `GeneratorAgent.generate(..., feedback=None)`;
  - `Container.run_ingest()` and `refresh_index()`;
  - setting `SCRAG_GUARDRAIL_MAX_TOKENS`.
- **Quality gates:**
  - `python scripts/verify.py`;
  - CI now also runs format and type checks and a FAISS job;
  - tests grew from 31 to 112.

## 8. Architectural weaknesses still open (input to the roadmap)

| Area | Weakness | Planned phase |
| --- | --- | --- |
| Deployment footprint | torch + sentence-transformers needs far more than the 512 MB of a free Render instance | 2 (ONNX embedder) |
| Shared state | The index lives in process memory, synchronized through a shared directory. Separate hosts without a shared disk do not converge, and cross-process `reset=false` ingests are last-writer-wins | 2/4 (pgvector) |
| Retrieval | Dense-only cosine search over fixed-size character chunks; no BM25, reranking, metadata filters or collections | 2 |
| Evaluation | The same small model generates and judges its own answers, and no golden dataset measures whether self-correction helps | 2 (minimal harness), 6 |
| LLM cost | About 7-9 sequential or fanned-out LLM calls per query, with no token or cost accounting and no response cache | 2, 6 |
| Concurrency model | Synchronous provider calls in a threadpool; no streaming | 3/5 |
| Security | One optional shared token; no users, sessions, rate limits, quotas or upload validation. Retrieved text goes straight into prompts (prompt-injection exposure) | 4, 7 |
| Ingestion formats | `.txt`/`.md` only; nothing is ever deleted when `reset=false` | 2 |
| Observability | Request IDs and logs only; no metrics, traces or latency breakdown | 6 |
| Durability | Index files are not fsync'ed (acceptable for derived data) | n/a |
