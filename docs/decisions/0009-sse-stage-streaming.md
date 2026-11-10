# 0009: Stream progress per stage over SSE, not answer tokens

**Status:** Accepted (2026-11-10). Implemented in Phase 5.

**Context.**
- Answering a question takes several seconds: retrieval, a guardrail call,
  then up to two generate-and-grade attempts. An agent run takes longer.
- A blank spinner hides the most interesting part of the system: the
  self-correction loop.
- D8 chose Server-Sent Events.

**Decision.**
- `POST /query/stream` and `POST /agent/run/stream` return
  `text/event-stream`:
  - `progress`: one per pipeline stage (retrieve, guardrail, generate,
    evaluate per attempt) or per agent trace event;
  - `result`: exactly the body of the non-streaming endpoint;
  - `error`: the usual sanitized error body plus its HTTP status.
- Validation, rate limits and session checks run before the stream starts, so
  they keep their HTTP status codes.
- The work runs on a thread that inherits the request's context, so logs keep
  the request ID. Events pass through a queue. A failing listener is logged
  and ignored.
- A comment line goes out every 15 s while the work is quiet, so proxies keep
  long runs open. It is sent only inside an active request; it is not
  keep-alive.
- Responses send `Cache-Control: no-cache, no-transform` and
  `X-Accel-Buffering: no` so intermediaries don't buffer or compress them.
- The browser reads the stream with `fetch`. `EventSource` only supports GET,
  and these endpoints are POSTs with JSON bodies. The incremental parser is
  unit-tested.

**Why not stream answer tokens?**
- The answer that matters is the graded one. Streaming the first attempt's
  tokens would show text that the evaluator may then reject and replace,
  which is misleading for a self-correcting system.
- Token streaming would also need streaming support in every LLM call path.

**Consequences.**
- Progress is honest and fine-grained, and every stage carries its timing.
- The final answer appears at once when grading finishes.
- If a client disconnects, the work finishes anyway. Its budget unit and saved
  history turn are kept, which is the same outcome as a completed request.
