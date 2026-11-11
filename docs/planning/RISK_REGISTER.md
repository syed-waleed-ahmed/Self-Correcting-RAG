# Risk Register

Last reviewed: 2026-11-10.
- Likelihood and impact are each rated L (low), M (medium) or H (high).
- Status: Open, Mitigated (controls in place) or Accepted (risk taken knowingly).

## Platform and cost

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R1 | The API (torch + sentence-transformers) doesn't fit Render Free's 512 MB, so the container is OOM-killed | L | H | ADR 0001: fastembed. The real image peaks at about 215 MB under `--memory=512m --cpus=0.1` in CI's container job, which enforces peak < 450 MB on every PR. Phase 7 load tests found that a runtime ingest (how the demo embeds uploads) at the old embedding batch size of 32 added 229 MB for good; the default is now 8 (+65 MB), and the load-test workflow measures a runtime ingest at both | Mitigated (measured in CI, Phase 7) |
| R2 | The free database is unreachable (Supabase pauses after 7 idle days; any provider can have outages), so uploads and history go down | M | M | ADR 0008: Neon Free recommended (scales to zero, resumes on connect, no manual restore); the database is never on the startup path; a circuit breaker fails fast; sample Q&A keeps working and answers say `history_saved: false`; health reports the last DB status without I/O; **no keep-alive traffic** (the worker sleeps while the queue is empty). Tested with an unreachable database | Mitigated (Phase 4) |
| R3 | Groq quotas (per model, per org) are exhausted by demo traffic or abuse | H | H | Implemented: per-IP token bucket, each client's daily share (10 live runs, Phase 7), global daily budget (30), recorded sample answers, TTL answer cache, batched guardrail, `top_k=5` (raised from 3 after measurement: about 2.5k tokens per answer), quota-reached UX. Still open: automatic fallback to a second Groq model | Partly mitigated |
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
| R15 | Reranking or embedding on 0.1 CPU is too slow for a usable demo | M | M | Reranker optional and off by default. Measured in Phase 7 load tests at `--cpus 0.1`: with ONNX Runtime's default thread per core, 10 busy users saw search at 990 / 2,300 ms (median / p95). The image now runs one thread: 7 / 120 ms, live answers 1.0 / 1.2 s, no failures (docs/load-testing.md) | Mitigated (Phase 7, measured in CI) |

## Security and privacy

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R16 | Prompt injection through uploaded documents or tool output | H | H | Implemented: the planner sees only the task, so retrieved text and tool output can't choose tools; evidence is delimited as untrusted with tag-escape protection; the writer has no tools; sensitive tools need approval (operator-only on the demo); new tools are read-only by construction (threat model T3-T5). Residual: injected text can still bias an answer | Mitigated (Phase 3/7); residual accepted |
| R17 | Cross-session data leakage in the public demo | M | H | Implemented: every row hangs off `scrag.sessions` and every query filters on `session_id` (chunks and messages carry it); other sessions' resources answer 404; tests prove B can't list, read, delete, search or append to A's data; 24 h TTL with purge; tokens stored hashed; RLS enabled on a non-public schema | Mitigated (Phase 4, tested) |
| R18 | Malicious uploads (zip bombs, huge PDFs, parser exploits) | M | H | Implemented: Content-Length required and checked before the multipart body is parsed (1 MB default); suffix allow-list; content sniffing (bytes must match the extension; NUL bytes refused in text); DOCX ZIP directory checked before inflating (30 MB, 2,000 parts, 200:1); deep JSON and oversized CSV fields fail cleanly; PDF page cap (50) and a 30 s extraction budget; chunk cap per document; extraction in a private temp dir under a fixed name; safe base names. Residual: one pathological PDF page can run long | Mitigated (Phase 7, tested) |
| R19 | The upload feature collects personal data from EU visitors (GDPR) | M | M | Implemented: 24 h retention with automatic deletion, `DELETE /sessions/me` deletes everything at once, only extracted text is stored, no analytics cookies. Still open: the UI notice (Phase 5) | Partly mitigated (Phase 4) |
| R20 | Secrets leak through errors or logs | L | H | Sanitized error contract with request IDs (Phase 1); keys only in env; never sent to the browser; secrets excluded from `repr(settings)`; `Cache-Control: no-store` on API responses (Phase 7); secret scanning with push protection | Mitigated (Phase 7, tested) |
| R21 | Groq / LLM providers see visitor prompts and uploads | H | M | Disclose in the demo UI; don't send uploads to Gemini (paid data terms apply in the EEA); minimal context sent | Accepted with disclosure |
| R22 | Dependency vulnerabilities | M | M | Implemented: `pip-audit` of the deployed dependency set and `npm audit` in CI; Dependabot for pip, npm, GitHub Actions and the Docker base image; CodeQL; read-only CI token | Mitigated (Phase 7) |
| R26 | The agent's web tool is used for SSRF (cloud metadata, internal network) | M | H | Off unless hosts are allow-listed; https:443 only; resolved addresses must be public (IPv6 forms that embed IPv4 included); connection pinned to the checked address; redirects re-checked; text-only, size and time caps (threat model T5) | Mitigated (Phase 7, tested) |
| R27 | The agent's SQL tool reads or writes data it shouldn't, or pins a CPU | M | H | A private in-memory catalog of public data; authorizer allows only SELECT over two tables and pure functions; `query_only`; 1 s budget; row and value caps (threat model T4) | Mitigated (Phase 7, tested) |
| R28 | A request flood exhausts the single free instance's CPU or the free database | M | M | Per-endpoint rate limits; a flood guard on all requests before routing; body-size cap. Measured with Locust (docs/load-testing.md): a one-client flood is contained (at most 0.15% of its requests served; all of them without the guard on unlimited endpoints), and no visitor request failed. Residual: a contained flood still roughly triples other visitors' median retrieval latency; volumetric attacks need edge protection | Partly mitigated (Phase 7, measured) |

## Engineering

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R23 | Separate hosts without a shared disk serve different index versions | M | M | Uploaded documents live in Postgres (shared by every host). The bundled sample index is baked into each image, so every host serves the same version | Mitigated (Phase 4) |
| R24 | Cross-process `reset=false` ingests lose updates (last writer wins) | L | L | Single-ingest lock per process; pgvector transactions in Phase 2 | Accepted until Phase 2 |
| R25 | Breaking API changes surprise existing clients | L | M | `CHANGELOG.md` "Changed (breaking)"; `/api/v1` versioning in Phase 4 | Mitigated (documented) |
