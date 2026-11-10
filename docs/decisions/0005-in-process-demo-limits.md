# 0005: Demo limits live in process; no Redis

**Status:** Accepted (2026-11-10).

**Decision.**
- Demo mode enforces a per-client token bucket, a global daily budget of live
  pipeline runs, a query-length cap, a `top_k` cap, and operator-only ingestion.
  All of it is held in process memory.
- The client is identified by the address a trusted proxy hop appended to
  `X-Forwarded-For`. The left-most value is client-controlled, so it is never
  used.

**Rationale.**
- Render Free runs exactly one instance, so in-process state is correct there.
- Upstash switches to pay-as-you-go once a card is added. Redis would add a
  cost risk and an extra dependency for no benefit.

**Consequences.**
- A restart resets the counters. A restart can't be triggered by visitors, and
  the provider's own quota still bounds total use.
- With more than one instance, the counters must move to Postgres (Phase 4),
  as `docs/planning/TARGET_ARCHITECTURE.md` (D5) already plans.
