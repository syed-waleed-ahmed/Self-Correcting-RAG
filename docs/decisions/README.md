# Architecture decision records

Short records of decisions that shape the system, with the evidence behind
them. They were proposed in `docs/planning/TARGET_ARCHITECTURE.md` and are
recorded here once implemented.

| ADR | Decision | Status |
| --- | --- | --- |
| [0001](0001-fastembed-deployment-embedder.md) | fastembed (ONNX) is the deployment embedder | Accepted, implemented in 0.2.0 |
| [0002](0002-bundled-index-and-recorded-samples.md) | Bundle the sample index in the image; serve recorded sample answers | Accepted, implemented in 0.2.0 |
| [0003](0003-hybrid-retrieval-default.md) | Hybrid retrieval (BM25 + dense, RRF) is the default | Accepted, implemented in 0.2.0 |
| [0004](0004-groq-only-public-demo.md) | The public demo uses Groq only; no Gemini free tier | Accepted |
| [0005](0005-in-process-demo-limits.md) | Demo limits live in process; no Redis | Accepted, implemented in 0.2.0 |
| [0006](0006-defer-pgvector-to-phase-4.md) | pgvector arrives with the database in Phase 4 | Accepted |
