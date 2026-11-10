# Cost-Zero Verification

Goal: EUR 0 per month, with no billing mechanism able to charge silently.

## Current state (2026-11-10)

**Nothing is deployed, and the project incurs no infrastructure cost.**
- Local development uses the owner's machine.
- CI runs on free public-repository GitHub Actions runners.
- LLM calls use the owner's Groq free-tier key. During this audit a few hundred
  tokens were used for verification calls (a model listing and small smoke tests).

## Billing exposure by service

| Service | Can it charge money? | How it could happen | Control |
| --- | --- | --- | --- |
| Vercel Hobby | No (V) | Over-limit features pause instead of billing | Stay on Hobby; non-commercial use only |
| Render Free | **Yes, if a card is on file** (V) | Bandwidth over 5 GB/month ($0.15/GB) and build minutes over 500 ($5 per 1,000) are billed automatically | Prefer no card. If one is required, check usage monthly, keep responses small, serve static assets from Vercel, and never pick a paid instance type |
| Neon Free | No (V): overages aren't billed; writes fail at the storage cap | Upgrading to Launch (pay-as-you-go) | Stay on Free; no card; the app caps stored documents (`SCRAG_PERSISTENCE_MAX_DOCUMENTS`) |
| Supabase Free (alternative) | Not on the Free plan (U: the docs only say paid plans need a card) | Upgrading the plan or enabling compute add-ons | Never upgrade; never enable add-ons; spend cap stays on if prompted |
| Groq | Not without a card (V) | Upgrading to Developer requires a payment method | Don't add a payment method |
| Upstash | **Yes, once a card is added** (V) | Adding a card converts the database to pay-as-you-go | Not used. If adopted later, never add a card |
| GitHub Actions | Not for public repos (V) | A private repo beyond $12 of compute with a payment method | Keep the repo public; no payment method required |
| Gemini API | Not used | Paid Services would be required for EU-facing use | Excluded from the design |

## ZERO_COST_MODE (planned for Phase 2b)

This is a design only; it is not implemented yet. `SCRAG_ZERO_COST_MODE=true`
(the default) will make the application refuse configurations that can incur
cost:

- An LLM base URL outside an allow-list of providers verified to be free
  (Groq, local Ollama/vLLM).
- Any integration marked `paid=True` in its adapter metadata. Paid adapters
  ship disabled.
- Unbounded budgets. Demo mode always has finite per-session, per-IP and global
  daily LLM budgets.

At startup the server logs which mode it is in, and `/health` reports it.

## Re-verification checklist

Run this before every deployment and on the first of every month:

- [ ] Re-read the free-tier pages for Vercel, Render, Supabase and Groq, and
  update `FREE_TIER_SERVICE_MATRIX.md` (date, numbers, V/U).
- [ ] Groq: `GET /openai/v1/models` still lists every configured model.
- [ ] Render: usage page under 750 h, 5 GB bandwidth and 500 build minutes; no
  paid instance types; the card status is noted.
- [ ] Neon (or Supabase, if used instead): plan still "Free", no card or
  add-ons, storage well under the cap, compute hours under 100 per month.
- [ ] Vercel: plan still "Hobby"; usage within limits.
- [ ] GitHub: repo still public; Actions usage shows no billable minutes.
- [ ] Billing pages of every service show EUR 0 / $0 for the period. Record the
  date and the result below.

| Date | Checked by | All services at 0? | Notes |
| --- | --- | --- | --- |
| 2026-11-10 | Phase 0 audit | n/a (nothing deployed) | Terms verified from official docs; see matrix |
