# Deployment Plan

Last updated: 2026-11-10. **Nothing is deployed publicly yet.** This document
will only claim a live deployment once one is verified, with evidence recorded
in section 6.

## 1. What works today: local Docker

```bash
export GROQ_API_KEY=...            # never commit it
docker compose up --build          # API on http://localhost:8000
curl -X POST localhost:8000/ingest -H 'content-type: application/json' -d '{"reset": true}'
curl -X POST localhost:8000/query  -H 'content-type: application/json' -d '{"query": "What triggers self-correction?"}'
```

The current image installs torch (CPU) and sentence-transformers. That works
locally but is not expected to fit Render Free's 512 MB. Phase 2 adds the
fastembed image variant (decision D1 in `TARGET_ARCHITECTURE.md`).

## 2. Target topology (free tier)

| Component | Service | Notes |
| --- | --- | --- |
| Frontend | Vercel Hobby, `*.vercel.app` | Static/edge only; no secrets in client code |
| API | Render Free Web Service (Docker), Frankfurt | One instance; sleeps after 15 minutes; ~1 minute cold start |
| Sample knowledge | NumPy index baked into the image | Works without the database (D2) |
| Database, vectors, files | Supabase Free | Connect through the pooler `:6543` (IPv4); pauses after 7 idle days |
| LLM | Groq free tier | Server-side key only |

## 3. Environment variables

**API (Render)**, set as Render environment secrets:

| Variable | Value / note |
| --- | --- |
| `GROQ_API_KEY` | Secret. Owner's Groq key |
| `SCRAG_GENERATOR_MODEL` / `_EVALUATOR_` / `_GUARDRAIL_MODEL` | `qwen/qwen3.8-27b` (re-verify on deploy day) |
| `SCRAG_CORS_ALLOW_ORIGINS` | `["https://<project>.vercel.app"]`, never `*` in production |
| `SCRAG_API_AUTH_TOKEN` | Set for admin endpoints (Phase 4 replaces it with real auth) |
| `SCRAG_LOG_JSON` | `true` |
| `SCRAG_TOP_K` | `3` in demo mode |
| `DATABASE_URL` | Phase 4. Supabase pooler URI, transaction mode, `sslmode=require` |
| `SCRAG_DEMO_MODE` | Phase 2b. Enables quotas and caching |

**Frontend (Vercel):** `NEXT_PUBLIC_API_BASE_URL=https://<service>.onrender.com`,
the only public variable. No keys.

## 4. Manual steps that need the owner's accounts

These can't be automated from the repository. Each needs a person to sign in.

1. **Groq:** confirm the free-plan limits for the chosen models on the console's
   Limits page and record them in `FREE_TIER_SERVICE_MATRIX.md`.
2. **Render:**
   - Create a Web Service from the GitHub repo (Docker runtime, Free instance
     type, Frankfurt region).
   - Set the variables from section 3.
   - Note whether a card was required (risk R5).
3. **Supabase** (Phase 4):
   - Create a Free project in an EU region.
   - Enable the `vector` extension.
   - Copy the **transaction pooler** connection string.
   - Do not enable add-ons.
4. **Vercel:**
   - Import the repo with the `web/` root directory (Phase 2b).
   - Set `NEXT_PUBLIC_API_BASE_URL`.
5. **GitHub:** keep the repository public, so CI minutes are free. Add a repo
   secret only if a deploy workflow is introduced.

## 5. Smoke tests (to automate in Phase 2b)

Run these against the live API with `scripts/smoke.py --base-url ...`:

1. `GET /health` returns 200 and `index_size > 0` (bundled index).
2. `POST /query` with a curated sample question returns 200, a non-empty
   answer, at least one `used_chunks` entry, and an `evaluation_status` other
   than `unavailable`.
3. `POST /query` with an empty query returns 422; one with whitespace only
   returns 400 with `error=invalid_query`.
4. Every response carries an `X-Request-ID` header.
5. CORS: a preflight from the Vercel origin is allowed, and one from another
   origin is rejected.
6. Rate limit (Phase 2b): a burst beyond the per-IP limit returns 429 with
   `error=rate_limited`.

## 6. Deployment evidence log

| Date | Commit | Target | Smoke test result | Notes |
| --- | --- | --- | --- | --- |
| none yet | | | | |

## 7. Rollback

- **Render:** redeploy the previous successful deploy from the dashboard. The
  image is immutable per commit.
- **Vercel:** promote the previous deployment.
- **Database migrations** (Phase 4): every migration ships with a downgrade.
  Take a manual backup (`pg_dump` through the pooler) before applying a
  migration to the demo project.

## 8. Things never to do

- Run scheduled keep-alive traffic to avoid sleep or pause policies.
- Put a provider key in frontend code or `NEXT_PUBLIC_*` variables.
- Store authoritative state on Render's filesystem (it is wiped on restart).
- Add a payment method to Upstash, or enable Supabase or Render add-ons.
