# Graph Report - D:\Projects & Stuff\self_correcting_rag  (2026-11-10)

## Corpus Check
- 84 files · ~191,050 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1110 nodes · 3936 edges · 53 communities detected
- Extraction: 42% EXTRACTED · 58% INFERRED · 0% AMBIGUOUS · INFERRED: 2292 edges (avg confidence: 0.62)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]
- [[_COMMUNITY_Community 42|Community 42]]
- [[_COMMUNITY_Community 43|Community 43]]
- [[_COMMUNITY_Community 44|Community 44]]
- [[_COMMUNITY_Community 45|Community 45]]
- [[_COMMUNITY_Community 46|Community 46]]
- [[_COMMUNITY_Community 47|Community 47]]
- [[_COMMUNITY_Community 48|Community 48]]
- [[_COMMUNITY_Community 49|Community 49]]
- [[_COMMUNITY_Community 50|Community 50]]
- [[_COMMUNITY_Community 51|Community 51]]
- [[_COMMUNITY_Community 52|Community 52]]

## God Nodes (most connected - your core abstractions)
1. `Chunk` - 118 edges
2. `RetrievedChunk` - 90 edges
3. `FakeChatClient` - 75 edges
4. `Settings` - 67 edges
5. `NumpyVectorStore` - 63 edges
6. `LLMError` - 60 edges
7. `VectorStore` - 53 edges
8. `ScragError` - 51 edges
9. `Database` - 50 edges
10. `Container` - 48 edges

## Surprising Connections (you probably didn't know these)
- `Record real pipeline runs for the demo's sample questions.  The public demo se` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\scripts\record_samples.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\config.py
- `build_verifier()` --calls--> `test_verifier_is_only_built_when_admins_and_keys_are_configured()`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\auth.py → D:\Projects & Stuff\self_correcting_rag\tests\test_auth.py
- `Bearer-JWT verification for the operator/admin (e.g. Supabase Auth).  Two key` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\auth.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\config.py
- `The token's claims if it is valid, else None (the reason is logged).` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\auth.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\config.py
- `A verifier when JWT admin auth is configured (keys and at least one admin), else` --uses--> `Settings`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\auth.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\config.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.04
Nodes (105): The demo's daily live-run budget, stored in Postgres so it survives restarts., database(), A migrated, emptied test database (skips without TEST_DATABASE_URL)., ConversationStore, make_title(), connection(), Database, _driver() (+97 more)

### Community 1 - "Community 1"
Cohesion: 0.04
Nodes (113): Vector store interface shared by all backends., Abstract nearest-neighbour store over chunk embeddings.      Implementations own, Add ``vectors`` (shape ``(n, dim)``) with their ``chunks``., Return the ``top_k`` most similar chunks to ``query_vector``., Remove all vectors and chunks., Persist the index to disk., Load a persisted index if present. Returns True if anything loaded., Number of stored chunks. (+105 more)

### Community 2 - "Community 2"
Cohesion: 0.05
Nodes (64): ABC, _normalize(), snapshot(), info(), Show effective configuration and index status., IndexIncompatibleError, A persisted index cannot be loaded with the current configuration., FaissVectorStore (+56 more)

### Community 3 - "Community 3"
Cohesion: 0.04
Nodes (77): Chunker, Text chunking.  Splits documents into overlapping, roughly sentence-aligned char, Character-window chunker with overlap.      Args:         chunk_size: Maximum, Split ``text`` into non-empty chunks, none longer than ``chunk_size``., Yield sentence-ish units split on paragraph then sentence boundaries., Seed the next chunk with overlap context, staying within chunk_size., sample_chunks(), DocumentsNotFoundError (+69 more)

### Community 4 - "Community 4"
Cohesion: 0.09
Nodes (76): error_body(), FastAPI application factory.  ``create_app`` builds the app, wires a :class:`Con, Reject oversized uploads before the multipart body is read.      Starlette spo, Assign a request ID, expose it in a header, and contain unhandled errors., RequestContextMiddleware, UploadLimitMiddleware, BaseModel, _bearer() (+68 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (74): Create the FastAPI app.      Args:         settings: Configuration; falls back t, BaseSettings, agent(), _bootstrap(), _db_connect(), db_downgrade(), db_status(), db_upgrade() (+66 more)

### Community 6 - "Community 6"
Cohesion: 0.07
Nodes (60): build_verifier(), JWTVerifier, Bearer-JWT verification for the operator/admin (e.g. Supabase Auth).  Two key, A verifier when JWT admin auth is configured (keys and at least one admin), else, VectorStore, OpenAICompatibleClient, build_container(), build_retriever() (+52 more)

### Community 7 - "Community 7"
Cohesion: 0.04
Nodes (69): create_app(), Create the FastAPI app.      Args:         settings: Configuration; falls bac, The token's claims if it is valid, else None (the reason is logged)., fake_embedder(), FakeEmbedder, _hermetic_env(), _next(), Shared test fixtures and fakes.  The fakes let the entire pipeline (retrieval → (+61 more)

### Community 8 - "Community 8"
Cohesion: 0.06
Nodes (64): FakeChatClient, Scriptable ChatClient. Routes each call by role via the system prompt.      Ea, evaluate_answers(), GeneratorAgent, Attempt, A single generate→evaluate attempt within the self-correction loop., _select_best(), _sort_by_relevance() (+56 more)

### Community 9 - "Community 9"
Cohesion: 0.13
Nodes (39): Conversation history, scoped to a session.  Every query filters on ``session_i, Raise :class:`NotFoundError` unless the conversation belongs to the session., Record a question and its answer. Creates the conversation when         ``conve, get_persistence(), require_session(), DocumentTooLongError, UnreadableDocumentError, UnsupportedFileTypeError (+31 more)

### Community 10 - "Community 10"
Cohesion: 0.15
Nodes (17): PersistentDailyBudget, status(), _client(), test_daily_budget_exhausts_and_resets_next_utc_day(), test_demo_budget_exhaustion_and_refund(), test_demo_ingest_requires_operator_token(), test_demo_query_length_and_top_k_caps(), test_demo_rate_limit_returns_429_with_retry_after() (+9 more)

### Community 11 - "Community 11"
Cohesion: 0.22
Nodes (14): evaluate_retrieval(), hit_at_k(), load_golden(), ndcg_at_k(), recall_at_k(), reciprocal_rank(), RetrievalReport, to_jsonable() (+6 more)

### Community 12 - "Community 12"
Cohesion: 0.29
Nodes (5): ApiError, request(), call(), main(), Smoke-test a running SCRAG API (local container or the live deployment).  Usag

### Community 13 - "Community 13"
Cohesion: 0.5
Nodes (0): 

### Community 14 - "Community 14"
Cohesion: 0.67
Nodes (0): 

### Community 15 - "Community 15"
Cohesion: 1.0
Nodes (1): Backward-compatible entry point.  Prefer installing the package (``pip install -

### Community 16 - "Community 16"
Cohesion: 1.0
Nodes (0): 

### Community 17 - "Community 17"
Cohesion: 1.0
Nodes (1): Whether an LLM key is configured (agents require this).

### Community 18 - "Community 18"
Cohesion: 1.0
Nodes (1): Whether a database is configured (sessions, uploads, history, jobs).

### Community 19 - "Community 19"
Cohesion: 1.0
Nodes (1): Deterministic, collision-resistant id for a chunk.

### Community 20 - "Community 20"
Cohesion: 1.0
Nodes (1): The value retrieval ranked by: fused score in hybrid mode, else cosine.

### Community 21 - "Community 21"
Cohesion: 1.0
Nodes (0): 

### Community 22 - "Community 22"
Cohesion: 1.0
Nodes (0): 

### Community 23 - "Community 23"
Cohesion: 1.0
Nodes (0): 

### Community 24 - "Community 24"
Cohesion: 1.0
Nodes (0): 

### Community 25 - "Community 25"
Cohesion: 1.0
Nodes (1): Character-window chunker with overlap.      Args:         chunk_size: Maximum ch

### Community 26 - "Community 26"
Cohesion: 1.0
Nodes (1): Split ``text`` into non-empty chunks, none longer than ``chunk_size``.

### Community 27 - "Community 27"
Cohesion: 1.0
Nodes (1): Yield sentence-ish units split on paragraph then sentence boundaries.

### Community 28 - "Community 28"
Cohesion: 1.0
Nodes (1): Seed the next chunk with overlap context, staying within chunk_size.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Build the vector index from documents in the configured docs directory.

### Community 30 - "Community 30"
Cohesion: 1.0
Nodes (1): Ask a question against the ingested corpus.

### Community 31 - "Community 31"
Cohesion: 1.0
Nodes (1): Run the FastAPI service with uvicorn.

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): Show effective configuration and index status.

### Community 33 - "Community 33"
Cohesion: 1.0
Nodes (1): Whether an LLM key is configured (agents require this).

### Community 34 - "Community 34"
Cohesion: 1.0
Nodes (1): Return a process-wide cached Settings instance.

### Community 35 - "Community 35"
Cohesion: 1.0
Nodes (1): Anything that can turn text into fixed-size vectors.

### Community 36 - "Community 36"
Cohesion: 1.0
Nodes (1): Dimensionality of the produced vectors.

### Community 37 - "Community 37"
Cohesion: 1.0
Nodes (1): Local embedder backed by ``sentence-transformers`` (lazy-loaded).

### Community 38 - "Community 38"
Cohesion: 1.0
Nodes (1): Render log records as single-line JSON documents.

### Community 39 - "Community 39"
Cohesion: 1.0
Nodes (1): Configure the root logger. Idempotent within a process.

### Community 40 - "Community 40"
Cohesion: 1.0
Nodes (1): Return a module-scoped logger.

### Community 41 - "Community 41"
Cohesion: 1.0
Nodes (1): A unit of retrievable text derived from a source document.

### Community 42 - "Community 42"
Cohesion: 1.0
Nodes (1): Deterministic, collision-resistant id for a chunk.

### Community 43 - "Community 43"
Cohesion: 1.0
Nodes (1): A chunk plus the scores assigned to it during retrieval/guardrailing.

### Community 44 - "Community 44"
Cohesion: 1.0
Nodes (1): A single generate→evaluate attempt within the self-correction loop.

### Community 45 - "Community 45"
Cohesion: 1.0
Nodes (1): The final result returned by the pipeline for a query.

### Community 46 - "Community 46"
Cohesion: 1.0
Nodes (1): Trimmed chunk representation for API responses.

### Community 47 - "Community 47"
Cohesion: 1.0
Nodes (1): Extract the first number from ``text`` and clamp it to [0, 1].

### Community 48 - "Community 48"
Cohesion: 1.0
Nodes (1): Parse a JSON object from ``text``, tolerating surrounding prose/fences.

### Community 49 - "Community 49"
Cohesion: 1.0
Nodes (1): Raised when an LLM request ultimately fails (after retries).

### Community 50 - "Community 50"
Cohesion: 1.0
Nodes (1): Minimal chat interface the agents depend on (easily faked in tests).

### Community 51 - "Community 51"
Cohesion: 1.0
Nodes (1): Return the assistant message content for a single-turn prompt.

### Community 52 - "Community 52"
Cohesion: 1.0
Nodes (1): Concrete :class:`ChatClient` backed by the ``openai`` SDK.

## Knowledge Gaps
- **90 isolated node(s):** `Backward-compatible entry point.  Prefer installing the package (``pip install -`, `Smoke-test a running SCRAG API (local container or the live deployment).  Usag`, `Run every quality gate CI runs, in the same order: one command, one exit code.`, `Text chunking.  Splits documents into overlapping, roughly sentence-aligned char`, `Character-window chunker with overlap.      Args:         chunk_size: Maximum` (+85 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 15`** (2 nodes): `main.py`, `Backward-compatible entry point.  Prefer installing the package (``pip install -`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 16`** (2 nodes): `layout.tsx`, `RootLayout()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 17`** (1 nodes): `Whether an LLM key is configured (agents require this).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 18`** (1 nodes): `Whether a database is configured (sessions, uploads, history, jobs).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 19`** (1 nodes): `Deterministic, collision-resistant id for a chunk.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 20`** (1 nodes): `The value retrieval ranked by: fused score in hybrid mode, else cosine.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 21`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 22`** (1 nodes): `next-env.d.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 23`** (1 nodes): `next.config.ts`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 24`** (1 nodes): `postcss.config.mjs`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 25`** (1 nodes): `Character-window chunker with overlap.      Args:         chunk_size: Maximum ch`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (1 nodes): `Split ``text`` into non-empty chunks, none longer than ``chunk_size``.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (1 nodes): `Yield sentence-ish units split on paragraph then sentence boundaries.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 28`** (1 nodes): `Seed the next chunk with overlap context, staying within chunk_size.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (1 nodes): `Build the vector index from documents in the configured docs directory.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 30`** (1 nodes): `Ask a question against the ingested corpus.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 31`** (1 nodes): `Run the FastAPI service with uvicorn.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `Show effective configuration and index status.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 33`** (1 nodes): `Whether an LLM key is configured (agents require this).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 34`** (1 nodes): `Return a process-wide cached Settings instance.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 35`** (1 nodes): `Anything that can turn text into fixed-size vectors.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 36`** (1 nodes): `Dimensionality of the produced vectors.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 37`** (1 nodes): `Local embedder backed by ``sentence-transformers`` (lazy-loaded).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 38`** (1 nodes): `Render log records as single-line JSON documents.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 39`** (1 nodes): `Configure the root logger. Idempotent within a process.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 40`** (1 nodes): `Return a module-scoped logger.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 41`** (1 nodes): `A unit of retrievable text derived from a source document.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 42`** (1 nodes): `Deterministic, collision-resistant id for a chunk.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 43`** (1 nodes): `A chunk plus the scores assigned to it during retrieval/guardrailing.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 44`** (1 nodes): `A single generate→evaluate attempt within the self-correction loop.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 45`** (1 nodes): `The final result returned by the pipeline for a query.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 46`** (1 nodes): `Trimmed chunk representation for API responses.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 47`** (1 nodes): `Extract the first number from ``text`` and clamp it to [0, 1].`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 48`** (1 nodes): `Parse a JSON object from ``text``, tolerating surrounding prose/fences.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 49`** (1 nodes): `Raised when an LLM request ultimately fails (after retries).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 50`** (1 nodes): `Minimal chat interface the agents depend on (easily faked in tests).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 51`** (1 nodes): `Return the assistant message content for a single-turn prompt.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 52`** (1 nodes): `Concrete :class:`ChatClient` backed by the ``openai`` SDK.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Chunk` connect `Community 1` to `Community 0`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 6`, `Community 7`, `Community 8`, `Community 11`?**
  _High betweenness centrality (0.109) - this node is a cross-community bridge._
- **Why does `Pluggable vector store backends.` connect `Community 0` to `Community 1`, `Community 2`, `Community 4`, `Community 5`, `Community 6`, `Community 8`, `Community 9`, `Community 10`?**
  _High betweenness centrality (0.100) - this node is a cross-community bridge._
- **Why does `NumpyVectorStore` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 8`, `Community 10`?**
  _High betweenness centrality (0.068) - this node is a cross-community bridge._
- **Are the 115 inferred relationships involving `Chunk` (e.g. with `RateLimiter` and `Budget`) actually correct?**
  _`Chunk` has 115 INFERRED edges - model-reasoned connections that need verification._
- **Are the 87 inferred relationships involving `RetrievedChunk` (e.g. with `SelfCorrectingRAGPipeline` and `The self-correcting RAG orchestrator.  Stages:     1. Retrieval          — embed`) actually correct?**
  _`RetrievedChunk` has 87 INFERRED edges - model-reasoned connections that need verification._
- **Are the 70 inferred relationships involving `FakeChatClient` (e.g. with `Settings` and `Chunk`) actually correct?**
  _`FakeChatClient` has 70 INFERRED edges - model-reasoned connections that need verification._
- **Are the 63 inferred relationships involving `Settings` (e.g. with `Record real pipeline runs for the demo's sample questions.  The public demo se` and `JWTVerifier`) actually correct?**
  _`Settings` has 63 INFERRED edges - model-reasoned connections that need verification._