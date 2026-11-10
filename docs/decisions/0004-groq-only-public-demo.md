# 0004: The public demo uses Groq only

**Status:** Accepted (2026-11-10).

**Context.** The Gemini API terms (effective 2026-03-23) require Paid Services
when an API client is made available to users in the EEA, Switzerland or the
UK. A public demo is reachable from all of them. Groq's free tier has no such
restriction, but it retired the Llama 3.x models for free tiers on 2026-08-16.

**Decision.**
- The demo uses Groq with `qwen/qwen3.8-27b` for all three agents. All three
  call shapes were verified live on 2026-11-10.
- The guardrail scores chunks in one batched call, saving requests.
- The fallback model is another Groq model (limits apply per model), for
  example `openai/gpt-oss-20b` with `SCRAG_GUARDRAIL_MAX_TOKENS=256`.
- Ollama is for local development only.

**Consequences.**
- One provider is a single point of failure.
- The demo degrades to recorded sample answers and retrieval-only search
  rather than failing completely.
