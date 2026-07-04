# Graph Report - D:\Projects & Stuff\self_correcting_rag  (2026-07-05)

## Corpus Check
- 33 files · ~35,124 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 264 nodes · 694 edges · 15 communities detected
- Extraction: 51% EXTRACTED · 49% INFERRED · 0% AMBIGUOUS · INFERRED: 341 edges (avg confidence: 0.63)
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

## God Nodes (most connected - your core abstractions)
1. `RetrievedChunk` - 35 edges
2. `Chunk` - 33 edges
3. `NumpyVectorStore` - 26 edges
4. `Chunker` - 23 edges
5. `VectorStore` - 23 edges
6. `Pluggable vector store backends.` - 21 edges
7. `Settings` - 18 edges
8. `build_container()` - 18 edges
9. `ChatClient` - 18 edges
10. `FakeChatClient` - 18 edges

## Surprising Connections (you probably didn't know these)
- `Chunker` --uses--> `Container`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\chunking.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\container.py
- `Chunker` --uses--> `Composition root: build and wire all components from Settings.  Centralizing con`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\chunking.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\container.py
- `Chunker` --uses--> `Load a persisted index into the store if one exists.`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\chunking.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\container.py
- `Chunker` --uses--> `Construct a fully wired :class:`Container`.      The optional overrides exist so`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\chunking.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\container.py
- `Chunker` --uses--> `IngestStats`  [INFERRED]
  D:\Projects & Stuff\self_correcting_rag\src\scrag\chunking.py → D:\Projects & Stuff\self_correcting_rag\src\scrag\ingest.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.09
Nodes (20): FastAPI application factory.  ``create_app`` builds the app, wires a :class:`Con, ChatClient, LLMError, Provider-agnostic chat client over any OpenAI-compatible endpoint.  Works with G, Raised when an LLM request ultimately fails (after retries)., Minimal chat interface the agents depend on (easily faked in tests)., Return the assistant message content for a single-turn prompt., Typed, environment-driven application configuration.  Settings are read (in prio (+12 more)

### Community 1 - "Community 1"
Cohesion: 0.11
Nodes (27): BaseSettings, Central configuration object for the whole application., Settings, fake_embedder(), FakeChatClient, FakeEmbedder, _next(), Shared test fixtures and fakes.  The fakes let the entire pipeline (retrieval → (+19 more)

### Community 2 - "Community 2"
Cohesion: 0.12
Nodes (32): Vector store interface shared by all backends., Abstract nearest-neighbour store over chunk embeddings.      Implementations own, Add ``vectors`` (shape ``(n, dim)``) with their ``chunks``., Return the ``top_k`` most similar chunks to ``query_vector``., Remove all vectors and chunks., Persist the index to disk., Load a persisted index if present. Returns True if anything loaded., Number of stored chunks. (+24 more)

### Community 3 - "Community 3"
Cohesion: 0.2
Nodes (21): BaseModel, Attempt, ChunkView, HealthResponse, IngestRequest, IngestResponse, PipelineResult, QueryRequest (+13 more)

### Community 4 - "Community 4"
Cohesion: 0.17
Nodes (13): OpenAICompatibleClient, Concrete :class:`ChatClient` backed by the ``openai`` SDK., build_container(), Container, Composition root: build and wire all components from Settings.  Centralizing con, Load a persisted index into the store if one exists., Construct a fully wired :class:`Container`.      The optional overrides exist so, Local embedder backed by ``sentence-transformers`` (lazy-loaded). (+5 more)

### Community 5 - "Community 5"
Cohesion: 0.11
Nodes (18): create_app(), Create the FastAPI app.      Args:         settings: Configuration; falls back t, _bootstrap(), info(), ingest(), query(), Command-line interface (Typer): ingest, query, serve, info., Show effective configuration and index status. (+10 more)

### Community 6 - "Community 6"
Cohesion: 0.12
Nodes (7): ABC, _normalize(), FaissVectorStore, _import_faiss(), create_vector_store(), Instantiate the vector store selected by configuration.      ``numpy`` is the de, VectorStore

### Community 7 - "Community 7"
Cohesion: 0.16
Nodes (14): build_context(), Render retrieved chunks into a numbered, source-attributed context block., GuardrailAgent, parse_json_object(), parse_score(), Extract the first number from ``text`` and clamp it to [0, 1]., Parse a JSON object from ``text``, tolerating surrounding prose/fences., _rc() (+6 more)

### Community 8 - "Community 8"
Cohesion: 0.18
Nodes (12): Chunker, Text chunking.  Splits documents into overlapping, roughly sentence-aligned char, Character-window chunker with overlap.      Args:         chunk_size: Maximum ch, Split ``text`` into non-empty chunks, none longer than ``chunk_size``., Yield sentence-ish units split on paragraph then sentence boundaries., Seed the next chunk with overlap context, staying within chunk_size., test_empty_text_yields_no_chunks(), test_hard_split_of_a_single_long_token() (+4 more)

### Community 9 - "Community 9"
Cohesion: 0.28
Nodes (10): NumpyVectorStore, test_query_without_llm_returns_503(), _chunk(), test_add_search_ordering(), test_clear(), test_length_mismatch_rejected(), test_load_dim_mismatch_raises(), test_load_missing_index_returns_false() (+2 more)

### Community 10 - "Community 10"
Cohesion: 1.0
Nodes (1): Backward-compatible entry point.  Prefer installing the package (``pip install -

### Community 11 - "Community 11"
Cohesion: 1.0
Nodes (1): Whether an LLM key is configured (agents require this).

### Community 12 - "Community 12"
Cohesion: 1.0
Nodes (1): Dimensionality of the produced vectors.

### Community 13 - "Community 13"
Cohesion: 1.0
Nodes (1): Deterministic, collision-resistant id for a chunk.

### Community 14 - "Community 14"
Cohesion: 1.0
Nodes (0): 

## Knowledge Gaps
- **39 isolated node(s):** `Backward-compatible entry point.  Prefer installing the package (``pip install -`, `Text chunking.  Splits documents into overlapping, roughly sentence-aligned char`, `Character-window chunker with overlap.      Args:         chunk_size: Maximum ch`, `Split ``text`` into non-empty chunks, none longer than ``chunk_size``.`, `Yield sentence-ish units split on paragraph then sentence boundaries.` (+34 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 10`** (2 nodes): `main.py`, `Backward-compatible entry point.  Prefer installing the package (``pip install -`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 11`** (1 nodes): `Whether an LLM key is configured (agents require this).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 12`** (1 nodes): `Dimensionality of the produced vectors.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 13`** (1 nodes): `Deterministic, collision-resistant id for a chunk.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 14`** (1 nodes): `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `Chunk` connect `Community 2` to `Community 0`, `Community 1`, `Community 3`, `Community 6`, `Community 7`, `Community 9`?**
  _High betweenness centrality (0.134) - this node is a cross-community bridge._
- **Why does `Chunker` connect `Community 8` to `Community 2`, `Community 4`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Why does `RetrievedChunk` connect `Community 2` to `Community 0`, `Community 3`, `Community 4`, `Community 6`, `Community 7`, `Community 9`?**
  _High betweenness centrality (0.104) - this node is a cross-community bridge._
- **Are the 32 inferred relationships involving `RetrievedChunk` (e.g. with `SelfCorrectingRAGPipeline` and `The self-correcting RAG orchestrator.  Stages:     1. Retrieval          — embed`) actually correct?**
  _`RetrievedChunk` has 32 INFERRED edges - model-reasoned connections that need verification._
- **Are the 30 inferred relationships involving `Chunk` (e.g. with `IngestStats` and `Document ingestion: load → chunk → embed → index.  Reads ``.txt``/``.md`` files`) actually correct?**
  _`Chunk` has 30 INFERRED edges - model-reasoned connections that need verification._
- **Are the 17 inferred relationships involving `NumpyVectorStore` (e.g. with `Chunk` and `RetrievedChunk`) actually correct?**
  _`NumpyVectorStore` has 17 INFERRED edges - model-reasoned connections that need verification._
- **Are the 16 inferred relationships involving `Chunker` (e.g. with `Container` and `Composition root: build and wire all components from Settings.  Centralizing con`) actually correct?**
  _`Chunker` has 16 INFERRED edges - model-reasoned connections that need verification._