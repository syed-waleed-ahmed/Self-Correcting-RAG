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
| [0006](0006-defer-pgvector-to-phase-4.md) | pgvector arrives with the database in Phase 4 | Accepted, implemented in Phase 4 |
| [0007](0007-plain-sql-migrations.md) | Plain SQL migrations (up + down) instead of Alembic | Accepted, implemented in Phase 4 |
| [0008](0008-postgres-persistence-design.md) | Persistence design: Neon Free, anonymous sessions, durable jobs, no idle traffic | Accepted, implemented in Phase 4 |
| [0009](0009-sse-stage-streaming.md) | Stream progress per stage over SSE, not answer tokens | Accepted, implemented in Phase 5 |
| [0010](0010-observability.md) | Observability: Prometheus metrics, optional OpenTelemetry, real cost per answer | Accepted, implemented in Phase 6 |
