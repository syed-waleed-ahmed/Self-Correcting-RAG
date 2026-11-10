# Risk Register

Last reviewed: 2026-11-10.
- Likelihood and impact are each rated L (low), M (medium) or H (high).
- Status: Open, Mitigated (controls in place) or Accepted (risk taken knowingly).

## Platform and cost

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R1 | The API (torch + sentence-transformers) doesn't fit Render Free's 512 MB, so the container is OOM-killed | H | H | D1: fastembed ONNX embedder for deployment; measure RSS under `--memory=512m` before deploying | Open |
| R2 | Supabase Free pauses after 7 idle days and resuming is manual, so uploads and history go down | H | M | D2: bundled sample index keeps core Q&A working; health endpoint reports DB status; documented manual resume; **no keep-alive pings** (forbidden) | Open |
| R3 | Groq quotas (per model, per org) are exhausted by demo traffic or abuse | H | H | Demo budgets (per session, per IP, global daily); cached answers for sample questions; batched guardrail; `top_k=3`; clear quota-reached UX; fallback to a second Groq model | Open |
| R4 | Groq retires a configured model with little notice (happened 2026-08-16 and 2026-09-14) | H | H | Model IDs in config; startup `GET /models` probe that warns; fallback model list; re-check before each deploy | Mitigated for the default (Phase 1); probe Open |
| R5 | A card on file lets Render bill bandwidth/build overage automatically ($0.15/GB beyond 5 GB) | M | M | Avoid adding a card if possible; keep API responses small; static assets on Vercel; monthly usage check (`COST_ZERO_VERIFICATION.md`) | Open |
| R6 | Adding a card to Upstash switches it to pay-as-you-go | L | M | Redis isn't used (D5); if ever used, no card | Mitigated |
| R7 | Free tiers change terms (e.g. HF Spaces Docker and Koyeb became paid) | M | H | Re-verify the matrix before every deploy; keep the provider-agnostic seams so a host swap is configuration, not code | Open |
| R8 | Render's 750 h/month or 5 GB bandwidth runs out and free services are suspended until month end | L | M | One service only; static assets elsewhere; monitor usage | Open |
| R9 | Vercel Hobby forbids commercial use | L | M | Keep the demo non-commercial (no ads, no paid work through it) | Accepted |
| R10 | Cold start (~1 minute on Render) makes the demo look broken | H | M | Frontend shows a "waking the server" state with retry; the health check is cheap; no keep-alive | Open |

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
| R17 | Cross-session data leakage in the public demo | M | H | Every row scoped by session/org; isolation tests (Phase 4); short TTL and automatic cleanup | Open |
| R18 | Malicious uploads (zip bombs, huge PDFs, parser exploits) | M | H | Size, page and type limits; MIME sniffing; parse in a time-limited background job; no OCR in the demo | Open |
| R19 | The upload feature collects personal data from EU visitors (GDPR) | M | M | Demo notice: don't upload personal data; short retention with automatic deletion; per-session delete; no analytics cookies | Open |
| R20 | Secrets leak through errors or logs | L | H | Sanitized error contract with request IDs (Phase 1); keys only in env; never sent to the browser | Mitigated for the API (Phase 1) |
| R21 | Groq / LLM providers see visitor prompts and uploads | H | M | Disclose in the demo UI; don't send uploads to Gemini (paid data terms apply in the EEA); minimal context sent | Accepted with disclosure |
| R22 | Dependency vulnerabilities | M | M | `pip-audit` and `npm audit` in CI (Phase 7); Dependabot | Open |

## Engineering

| ID | Risk | L | I | Mitigation | Status |
| --- | --- | --- | --- | --- | --- |
| R23 | Separate hosts without a shared disk serve different index versions | M | M | pgvector as the shared store (Phase 2); file stores documented as single-host | Accepted until Phase 2 |
| R24 | Cross-process `reset=false` ingests lose updates (last writer wins) | L | L | Single-ingest lock per process; pgvector transactions in Phase 2 | Accepted until Phase 2 |
| R25 | Breaking API changes surprise existing clients | L | M | `CHANGELOG.md` "Changed (breaking)"; `/api/v1` versioning in Phase 4 | Mitigated (documented) |
