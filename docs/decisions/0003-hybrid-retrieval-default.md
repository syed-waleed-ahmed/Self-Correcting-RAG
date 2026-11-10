# 0003: Hybrid retrieval is the default

**Status:** Accepted (2026-11-10).

**Decision.**
- `SCRAG_RETRIEVAL_MODE=hybrid` is the default.
- Dense and BM25 candidates (a pool of 20 from each) are fused with Reciprocal
  Rank Fusion (k = 60).
- Results keep their true cosine `score` and add `fusion_score` and
  `keyword_score`.

**Evidence.** On the 40-question golden set at k = 5:
- MRR rose from 0.915 to 0.961.
- Recall rose from 0.950 to 0.988.
- Lexical-question MRR rose from 0.838 to 0.914.
- Latency was unchanged (about 4 ms).

See `docs/evaluation.md`.

**Consequences.**
- BM25 is in-process and rebuilt when the store's revision changes. That is
  fine for small and medium corpora.
- With pgvector (Phase 4), the keyword side moves to Postgres full-text search.
- Dense-only mode remains available.
