# 0002: Bundle the sample index; serve recorded sample answers

**Status:** Accepted (2026-11-10).

**Context.**
- Free hosts wipe their local disk on restart and sleep when idle.
- Free databases pause after a week of inactivity.
- The free LLM quota allows only a few dozen full pipeline runs per day.

A public demo has to make a good first impression under all of these.

**Decision.**
- The Docker build runs `scrag ingest`, baking the embedding model and the
  sample index into the image. The core demo needs no database.
- `scripts/record_samples.py` runs the real pipeline for 8 curated questions and
  stores the results in `data/demo/samples.json`, together with a corpus
  fingerprint.
- `/query` serves a recorded result for a matching sample question only while
  the fingerprint matches the live index. Every cached response is labelled
  `cached: true`, with `cached_at` and `cache_source` set. The UI shows
  "Recorded run" with the date and offers "Run live".
- Repeated live questions are served from a TTL cache (`cache_source:
  "recent"`), also labelled.

**Consequences.**
- The demo answers its sample questions with no LLM quota, and with no database.
- Cached answers are real past executions, never synthesized.
- Changing the corpus without re-recording simply disables the samples instead
  of serving stale answers.
