# Free-Tier Service Matrix

Everything below was checked against official vendor pages on **2026-11-10**.
V means the number was confirmed on an official page. U means it could not be
confirmed (for example, it is shown only inside a logged-in console). Free
tiers change without notice: re-verify before each deployment (see
`COST_ZERO_VERIFICATION.md`).

## Decision summary

| Need | Choice | Status |
| --- | --- | --- |
| Frontend hosting | Vercel Hobby | Use |
| API hosting | Render Free Web Service | Use, with constraints (512 MB RAM, sleeps) |
| Database + vectors | **Neon Free** (Postgres + pgvector), re-verified 2026-11-10; Supabase Free supported as an alternative | Use (ADR 0008). No file storage needed: only extracted text is kept |
| LLM inference | Groq free tier | Use; the only free LLM usable for an EU-reachable public demo |
| Second free LLM provider | Google Gemini API free tier | **Do not use** for the public demo (EU terms, below) |
| Local LLM | Ollama | Local development only |
| Embeddings | fastembed (ONNX) running in-process | Use in deployment; torch is too large for Render Free |
| Cache / queue | Postgres and in-process memory | Use; Redis not needed |
| Redis | Upstash Redis Free | Optional only; never add a card |
| CI | GitHub Actions (public repository) | Use |
| Alternatives rejected | Hugging Face Spaces (Docker now needs a paid plan), Koyeb (no free plan for new users), Qdrant Cloud free (deleted after 4 idle weeks) | Do not use |

## Details

### Vercel Hobby (frontend)

- **Monthly allowance (V):**
  - 100 GB Fast Data Transfer and 10 GB Fast Origin Transfer
  - 1,000,000 function invocations and 1,000,000 CDN requests
  - 4 CPU-hours Active CPU
  - 300 s maximum function duration
  - 100 deployments per day
- **Over the limit (V):** no charge. The feature stops, in most cases for 30 days.
  There is no spend management on Hobby.
- **Restriction (V):** non-commercial personal use only. A portfolio demo
  qualifies; ads, affiliate links or paid work would not.
- **Sources:** https://vercel.com/docs/plans/hobby,
  https://vercel.com/docs/limits/fair-use-guidelines

### Render Free Web Service (API)

- **Instance (V):** 0.1 CPU and 512 MB RAM.
- **Sleep (V):** spins down after 15 minutes without inbound traffic, and waking
  takes about 1 minute. Render may restart a free service at any time.
- **Hours (V):** 750 instance hours per workspace per month, with no rollover.
  When they run out, all free services are suspended until the next month.
- **Workspace allowance (V):**
  - 5 GB bandwidth per month; beyond that, $0.15/GB
  - 500 build minutes; beyond that, $5 per 1,000
- **Disk (V):** no persistent disk. The local filesystem is lost on redeploy,
  restart and spin-down.
- **Docker (V):** fully supported.
- **Card (U):** the tutorial says no payment is required, but staff forum posts
  say a card is often requested for verification.
- **Billing (V):** with a card on file, bandwidth and build overage is billed
  automatically. Without a card, free services are suspended instead.
- **Region (V):** Frankfurt is available.
- **Sources:** https://render.com/docs/free, https://render.com/docs/docker,
  https://render.com/pricing

### Neon Free (Postgres, pgvector): chosen for Phase 4 (checked 2026-11-10)

- **Quotas (V, Neon FAQ updated 2026-10-01):**
  - 100 CU-hours of compute per project per month
  - 1 GB of storage per project (some older pages say 0.5 GB; the app's
    global document cap keeps usage far below either)
  - 5 GB of public network transfer
- **Idle behaviour (V):** compute scales to zero after 5 minutes of inactivity
  and resumes on the next connection. There is no manual-restore pause.
- **Overage (V):** never billed on Free. Writes that would grow storage past
  the cap fail until space is freed.
- **pgvector (V):** supported (`CREATE EXTENSION vector`).
- **Card (U):** not expected for Free. Don't add one: the Launch plan is
  pay-as-you-go.
- **Sources:** https://neon.com/faqs/free-plan-limits-and-quotas,
  https://neon.com/docs/extensions/pg-extensions

### Supabase Free (Postgres, pgvector, Auth, Storage)

- **Quotas (V):**
  - 500 MB database per project
  - 1 GB file storage
  - 5 GB egress plus 5 GB cached egress
  - 50,000 Auth MAU
  - 2 active projects
- **Compute (V):** Nano (shared CPU, up to 0.5 GB RAM), 60 direct connections,
  200 pooler clients.
- **Pause (V):** a project is paused after 1 week of low activity. **Resuming is
  manual**, from the dashboard. A paused project can be restored for up to 1 year.
- **pgvector (V):** available as the `vector` extension.
- **Connectivity (V):** direct connections on Free are IPv6 only. IPv4 hosts must
  use the shared pooler: port 6543 in transaction mode, port 5432 in session mode.
- **Card (U for Free):** the docs say paid plans require a card.
- **Sources:** https://supabase.com/pricing,
  https://supabase.com/docs/guides/platform/compute-and-disk,
  https://supabase.com/docs/guides/database/connecting-to-postgres

### Groq (LLM)

- **Retired models (V):** `llama-3.1-8b-instant` and `llama-3.3-70b-versatile`
  were shut down for free and developer tiers on 2026-08-16. Groq's suggested
  replacement `qwen3.6-27b` was itself shut down on 2026-09-14.
- **Free models confirmed for our key (2026-11-10):** `openai/gpt-oss-20b`,
  `openai/gpt-oss-120b` and `qwen/qwen3.8-27b`, among 11 models available.
- **Rate limits:** the only published table is labelled "base limits for the
  Developer plan". It gives these models 30 RPM, 1K RPD, 8K TPM and 200K TPD
  each. The free plan's own numbers are shown only in the logged-in console (U).
  Limits apply per organization, per model.
- **Card (V):** a payment method is needed only to upgrade to Developer.
- **Sources:** https://console.groq.com/docs/deprecations,
  https://console.groq.com/docs/models, https://console.groq.com/docs/rate-limits

### Google Gemini API free tier (rejected for the public demo)

- **Terms (V, effective 2026-03-23):** making API clients available to users in
  the EEA, Switzerland or the UK requires Paid Services. A public demo URL is
  reachable from the EU, and the owner appears to be EU-based. In the EEA, UK
  and Switzerland, the paid data terms apply even to free use.
- **Free models (V):**
  - the Gemini Flash and Flash-Lite families
  - 2.5 Pro
  - Gemma 4
  - Gemini Embedding 2 and Gemini Embedding
- **Rate limits (U):** shown per project in AI Studio, not in the docs.
- **OpenAI-compatible endpoint (V, beta):**
  `https://generativelanguage.googleapis.com/v1beta/openai/`
- **Sources:** https://aistudio.google.com/docs/terms,
  https://aistudio.google.com/docs/available-regions,
  https://aistudio.google.com/docs/pricing,
  https://aistudio.google.com/docs/openai

### Upstash Redis Free (optional, not planned)

- **Quotas (V):** 500K commands per month, 10 GB bandwidth, 1 database, about 256 MB data.
- **Billing (V):** adding a card switches the database to pay-as-you-go.
- **Source:** https://upstash.com/pricing/redis

### GitHub Actions (CI)

- **Public repositories (V):** standard GitHub-hosted runners are free.
- **Private repositories on GitHub Free (V):** $12 of compute per month
  (about 2,000 Linux minutes); usage is blocked when it runs out if there is
  no payment method.
- **Source:** https://docs.github.com/en/billing/concepts/product-billing/github-actions

### fastembed (embedding library, runs inside the API process)

- **Model support (V):** supports `sentence-transformers/all-MiniLM-L6-v2`
  (384 dimensions, Apache-2.0, 0.090 GB model) on ONNX Runtime, with no PyTorch.
- **Memory (U):** no published RAM figure. Measure it in Phase 2 under a
  512 MB container limit.
- **Embedding compatibility:** this is the same model as the current default,
  so the 384-dim index format is unchanged. Re-ingest after switching anyway,
  because numerical output may differ slightly between runtimes.

### Rejected alternatives

- **Hugging Face Spaces (V):** CPU Basic (2 vCPU, 16 GB RAM) is free, but Docker
  and Gradio Spaces now require a paid plan to create.
- **Koyeb:** the docs still describe a free instance (512 MB, 0.1 vCPU), but a
  2026-02-17 announcement says new users must subscribe to a paid plan (U,
  probably unavailable).
- **Qdrant Cloud free (V):** 1 GB RAM, 0.5 vCPU, 4 GB disk. It is suspended
  after 1 idle week and **deleted after 4 weeks**.
