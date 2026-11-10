# Risk Register

Last reviewed: 2026-11-10.
- Likelihood and impact are each rated L (low), M (medium) or H (high).
- Status: Open, Mitigated (controls in place) or Accepted (risk taken knowingly).

## Platform and cost

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R1 | The API (torch + sentence-transformers) doesn't fit Render Free's 512 MB, so the container is OOM-killed | L | H | ADR 0001: fastembed. The real image peaked at 210 MB under `--memory=512m --cpus=0.1` in CI (run 42735463057); the CI container job enforces peak < 450 MB on every PR | Mitigated (verified in CI) |
| R2 | The free database is unreachable (Supabase pauses after 7 idle days; any provider can have outages), so uploads and history go down | M | M | ADR 0008: Neon Free recommended (scales to zero, resumes on connect, no manual restore); the database is never on the startup path; a circuit breaker fails fast; sample Q&A keeps working and answers say `history_saved: false`; health reports the last DB status without I/O; **no keep-alive traffic** (the worker sleeps while the queue is empty). Tested with an unreachable database | Mitigated (Phase 4) |
| R3 | Groq quotas (per model, per org) are exhausted by demo traffic or abuse | H | H | Implemented: per-IP token bucket, global daily budget (30), recorded sample answers, TTL answer cache, batched guardrail, `top_k=5` (raised from 3 after measurement: about 2.5k tokens per answer), quota-reached UX. Still open: automatic fallback to a second Groq model | Partly mitigated |
| R4 | Groq retires a configured model with little notice (happened 2026-08-16 and 2026-09-14) | H | H | Model IDs in config; startup `GET /models` probe that warns; fallback model list; re-check before each deploy | Mitigated for the default (Phase 1); probe Open |
| R5 | A card on file lets Render bill bandwidth/build overage automatically ($0.15/GB beyond 5 GB) | M | M | Avoid adding a card if possible; keep API responses small; static assets on Vercel; monthly usage check (`COST_ZERO_VERIFICATION.md`) | Open |
| R6 | Adding a card to Upstash switches it to pay-as-you-go | L | M | Redis isn't used (D5); if ever used, no card | Mitigated |
| R7 | Free tiers change terms (e.g. HF Spaces Docker and Koyeb became paid) | M | H | Re-verify the matrix before every deploy; keep the provider-agnostic seams so a host swap is configuration, not code | Open |
| R8 | Render's 750 h/month or 5 GB bandwidth runs out and free services are suspended until month end | L | M | One service only; static assets elsewhere; monitor usage | Open |
| R9 | Vercel Hobby forbids commercial use | L | M | Keep the demo non-commercial (no ads, no paid work through it) | Accepted |
| R10 | Cold start (~1 minute on Render) makes the demo look broken | H | M | Implemented: the UI polls `/health` and shows a "waking the server" state; the model and index are baked into the image so startup needs no download; no keep-alive | Mitigated |

## Product and quality

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R11 | An LLM-as-judge score is mistaken for verification | M | H | `evaluation_status` wording; docs state it is heuristic; citation checks (Phase 3); offline golden-set evaluation (Phase 2/6) | Partly mitigated (Phase 1 wording) |
| R12 | "Advanced RAG" gives no measurable improvement | M | M | Eval harness before retrieval changes (Phase 2); publish only measured deltas | Open |
| R13 | Scope creep (11 agents, full RBAC, 11 pages) leaves many half-finished features | H | H | Reduced roster (D7); walking skeleton first; phase gates; "done" requires evidence | Open |
| R14 | A reasoning model silently breaks the guardrail via empty output under small token budgets | M | M | `SCRAG_GUARDRAIL_MAX_TOKENS`; `<think>` stripping; fail-open with `guardrail_score=null` and a warning log | Mitigated (Phase 1) |
| R15 | Reranking or embedding on 0.1 CPU is too slow for a usable demo | M | M | Reranker optional and off by default; measure p50/p95 latency on the real instance in Phase 2b | Open |

## Security and privacy

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R16 | Prompt injection through uploaded documents or tool output | H | H | Treat retrieved text and tool output as data; delimit them; tools never chosen by retrieved text; permission levels and approval for sensitive tools (Phase 3/7) | Open |
| R17 | Cross-session data leakage in the public demo | M | H | Implemented: every row hangs off `scrag.sessions` and every query filters on `session_id` (chunks and messages carry it); other sessions' resources answer 404; tests prove B can't list, read, delete, search or append to A's data; 24 h TTL with purge; tokens stored hashed; RLS enabled on a non-public schema | Mitigated (Phase 4, tested) |
| R18 | Malicious uploads (zip bombs, huge PDFs, parser exploits) | M | H | Implemented: Content-Length required and checked before the multipart body is parsed (1 MB default); suffix allow-list; PDF page cap (50); chunk cap per document; extraction in a private temp dir under a fixed name; safe base names. Still open: MIME sniffing and a parse time limit (Phase 7) | Partly mitigated (Phase 4) |
| R19 | The upload feature collects personal data from EU visitors (GDPR) | M | M | Implemented: 24 h retention with automatic deletion, `DELETE /sessions/me` deletes everything at once, only extracted text is stored, no analytics cookies. Still open: the UI notice (Phase 5) | Partly mitigated (Phase 4) |
| R20 | Secrets leak through errors or logs | L | H | Sanitized error contract with request IDs (Phase 1); keys only in env; never sent to the browser | Mitigated for the API (Phase 1) |
| R21 | Groq / LLM providers see visitor prompts and uploads | H | M | Disclose in the demo UI; don't send uploads to Gemini (paid data terms apply in the EEA); minimal context sent | Accepted with disclosure |
| R22 | Dependency vulnerabilities | M | M | `pip-audit` and `npm audit` in CI (Phase 7); Dependabot | Open |

## Engineering

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R23 | Separate hosts without a shared disk serve different index versions | M | M | Uploaded documents live in Postgres (shared by every host). The bundled sample index is baked into each image, so every host serves the same version | Mitigated (Phase 4) |
| R24 | Cross-process `reset=false` ingests lose updates (last writer wins) | L | L | Single-ingest lock per process; pgvector transactions in Phase 2 | Accepted until Phase 2 |
| R25 | Breaking API changes surprise existing clients | L | M | `CHANGELOG.md` "Changed (breaking)"; `/api/v1` versioning in Phase 4 | Mitigated (documented) |
