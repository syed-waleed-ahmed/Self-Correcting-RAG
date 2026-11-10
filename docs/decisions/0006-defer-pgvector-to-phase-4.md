# 0006: pgvector arrives with the database in Phase 4

**Status:** Accepted (2026-11-10). This changes the original roadmap, which
placed pgvector in Phase 2.

**Rationale.**
- The walking-skeleton demo (Phase 2b) deliberately needs no database
  (ADR 0002).
- Building `PgVectorStore` before there is a schema, migrations and connection
  management would mean writing that plumbing twice.
- The file-backed stores already handle the single-instance demo correctly,
  including atomic swaps and cross-process refresh.

**Consequences.**
- Shared vector storage across hosts (risk R23) stays open until Phase 4.
