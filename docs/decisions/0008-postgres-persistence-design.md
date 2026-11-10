# 0008: Persistence design: one Postgres, anonymous sessions, no idle traffic

**Status:** Accepted (2026-11-10). Implemented in Phase 4.

**Context.** Phase 4 adds:
- anonymous sessions;
- uploads with search over them;
- conversation history;
- a durable job queue;
- quotas that survive a restart.

All of it has to run on a free database tier while the API keeps its
free-tier behaviour: it sleeps after 15 idle minutes and has 0.1 CPU.

## Provider: Neon Free (recommended), any Postgres with pgvector supported

The code needs only a connection string (`SCRAG_DATABASE_URL`) and the
`vector` extension. The terms below were checked on 2026-11-10 against the
providers' own pages.

| | Neon Free | Supabase Free |
| --- | --- | --- |
| When idle | Compute scales to zero after 5 minutes and resumes on the next connection | Project pauses after 7 days without activity; someone must restore it in the dashboard |
| Storage | 1 GB per project | 500 MB per database |
| Over quota | Never billed: writes that would grow storage fail | Read-only past the cap |
| pgvector | Supported | Supported |

**Neon is the recommended host.** A portfolio demo sits idle for weeks, and on
Supabase the first visitor after that would find uploads and history down
until a manual restore. Neon resumes on its own. Supabase remains a supported
alternative: use its transaction pooler URL on port 6543.

## Design decisions

- **The database is optional and never on the startup path (extends D2).**
  - Building the container connects nowhere.
  - The pool keeps no idle connections (`min_size=0`).
  - The first use connects and migrates.
  - When the database is down, a circuit breaker fails calls fast for 30
    seconds.
  - Sample-corpus questions keep working throughout. A visitor with a session
    gets an answer with `history_saved: false`.
- **No keep-alive traffic.**
  - The job worker waits on an in-process event while the queue is empty.
  - It polls only while jobs are pending.
  - `/health` never queries the database.
  - With no visitors, the database receives no queries and can scale to zero,
    as the free-tier rules require.
- **Isolation by construction.**
  - Every row a visitor owns references `scrag.sessions` with
    `ON DELETE CASCADE`.
  - Chunks and messages carry `session_id` denormalized, so every search and
    read filters on it directly.
  - Another session's resources answer 404, never 403.
  - Tests prove that session B can't list, read, delete, search or append to
    session A's data.
- **The schema is not reachable from outside.**
  - Tables live in a dedicated `scrag` schema. Supabase's Data API exposes
    only `public`.
  - Row-level security is enabled with no policies, as a second barrier.
- **Session tokens.**
  - A token is `scs_` plus 256 random bits. Only its SHA-256 is stored.
  - The TTL is fixed from creation, 24 hours by default, with no sliding
    renewal.
  - Expired sessions are purged with all of their data.
- **Uploads.**
  - Text is extracted at upload time, giving immediate 415/413/422 errors.
  - Only the extracted text is stored, never the original bytes. This is less
    to protect, less storage used, and there is no Storage bucket to secure.
  - Chunking and embedding run as a job.
  - Limits apply per session (documents, bytes, chunks) and globally (total
    documents), to stay inside the free storage quota.
  - Oversized bodies are rejected before multipart parsing. Starlette would
    otherwise spool them to disk.
- **Jobs.**
  - Jobs are claimed with `FOR UPDATE SKIP LOCKED` and a lease, and the lease
    is checked when a job completes.
  - Retries use exponential backoff, and `PermanentJobError` skips them.
  - A job whose worker died is reclaimed once its lease expires: interrupted
    work resumes after a restart.
  - Delivery is at-least-once, so handlers are idempotent: ingestion replaces
    a document's chunks.
- **Vector search is an exact scan per session, not HNSW (revises D3).**
  - A session holds at most a few hundred chunks.
  - An HNSW index would post-filter by `session_id` and could silently return
    fewer than k results.
  - Rows record their embedding model, and searches filter on it, so a model
    change never compares vectors from different spaces.
- **Keyword search uses Postgres full text.** Terms are OR-ed and ranked with
  `ts_rank_cd`. Terms come from the BM25 tokenizer, reduced to `[a-z0-9]`, so
  no `tsquery` operator can be injected. The dense and keyword rankings are
  fused with RRF. `scope=all` fuses the session's ranking with the sample
  corpus, so each counts as one list.
- **The daily budget lives in Postgres.**
  - The in-process budget reset on every free-tier restart, so "30 per day"
    meant "30 per wake-up".
  - A conditional upsert enforces the limit atomically.
  - If the database is down, the budget falls back to the in-process one.

## Consequences

- Sessions are anonymous only, and the web UI for uploads and history arrives
  in Phase 5.
- Agent runs still search the sample corpus only. Session-aware tools are a
  later increment.
- A job is lost only if every attempt fails. Its document is then marked
  `failed` with a stable error code.
- Database load is bounded by the document cap. Measured numbers belong in the
  evidence log once deployed.
