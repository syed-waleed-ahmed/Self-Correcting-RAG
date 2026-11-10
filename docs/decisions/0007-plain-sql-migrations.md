# 0007: Plain SQL migrations instead of Alembic

**Status:** Accepted (2026-11-10). Implemented in Phase 4. This departs from
the roadmap, which named Alembic.

**Context.**
- SCRAG has no ORM. Every query is explicit SQL through psycopg 3.
- Alembic needs SQLAlchemy at runtime, a large dependency for a container
  measured at 210 MB peak under a 512 MB limit.
- The deployment plan requires that every migration ships with a downgrade.

**Decision.**
- Migrations are numbered SQL file pairs in the package,
  `src/scrag/persistence/sql/NNNN_name.up.sql` and `.down.sql`. Loading fails
  if either half is missing.
- `apply_migrations` and `rollback_migrations` run in a single transaction
  under `pg_advisory_xact_lock`. Several API processes starting together
  serialize on the lock. Transaction-scoped locks also work through
  transaction-mode poolers (PgBouncer, Supavisor), where session locks don't.
- The applied versions are recorded in `scrag.schema_migrations`.
- The API migrates on first database use (`SCRAG_DB_MIGRATE_ON_START=true`,
  the default) because Render Free has no pre-deploy hook. Operators can use
  `scrag db status|upgrade|downgrade` instead.

**Evidence.** The test suite checks that up, down and up again restore the
schema. CI repeats that round trip through the CLI against Postgres 17 with
pgvector.

**Consequences.**
- There is no autogeneration, so schema changes are written by hand. This is
  fine at the current scale of seven tables.
- If a migration fails, the whole transaction rolls back. The API keeps serving
  sample-corpus questions and reports `database_unavailable` for the
  persistence endpoints.
