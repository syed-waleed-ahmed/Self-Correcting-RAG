# SCRAG HTTP API

SCRAG serves a versioned REST API under /api/v1, and keeps the original
unversioned paths for backward compatibility. GET /health reports liveness,
the index size and whether a language model is configured. POST /query runs
the full self-correcting pipeline and returns the answer, the evaluation
score, the evaluation status, the number of attempts executed, the attempt
that was returned, the history of every attempt and the chunks that were used.
POST /search runs retrieval only, without calling a language model, and
returns the ranked chunks. POST /ingest rebuilds or updates the index from the
server's configured documents directory.

Errors share one JSON shape with three fields: error, a stable machine
readable code such as llm_error or index_empty; detail, a message that is safe
to show to clients; and request_id. Every response carries an X-Request-ID
header with the same identifier, which also appears in the server logs, so an
operator can find the full internal details of a failure without those
details ever being sent to the client.

When the server runs in demo mode it enforces a per-client rate limit, a
global daily budget of language model queries, and limits on query length,
and it disables ingestion for anonymous visitors. A rate-limited request
receives HTTP status 429 with the error code rate_limited, and a request made
after the daily budget is spent receives 429 with the code
demo_budget_exhausted.
