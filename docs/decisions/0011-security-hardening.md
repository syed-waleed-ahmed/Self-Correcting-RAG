# 0011: Security hardening: narrow read-only tools, layered quotas, measured limits

**Status:** Accepted (2026-11-10). Implemented in Phase 7. The analysis behind
it is the [threat model](../threat-model.md).

**Context.**
- The demo is public and anonymous, runs on free tiers (0.1 CPU, 512 MB, a
  daily LLM quota), and its agent is steered by visitors: whatever the planner
  sends a tool is effectively visitor input.
- The roadmap asked for an HTTP tool and a SQL tool. Both are classic ways for
  an LLM agent to become a proxy into the network or the database.
- A review found concrete gaps:
  - deeply nested JSON and oversized CSV fields crashed extraction (500s);
  - nothing checked that an upload's bytes matched its extension;
  - a DOCX was inflated without bounds;
  - `round(5, -10**9)` made the calculator compute 10**(10**9) in a thread
    its timeout can't stop;
  - request bodies outside uploads were unbounded;
  - `repr(settings)` printed the provider key;
  - one visitor could spend the whole day's LLM budget.

**Decision.**
- **Tools are narrow and read-only by construction, not by prompt.**
  - `sql_query` runs on a private in-memory SQLite catalog built from the
    public index (documents and chunks). An authorizer allows only SELECT,
    reads of the two tables and an allow-list of pure functions, under
    `PRAGMA query_only`, with a progress-handler time budget and row and value
    caps. There is no access to the application database at all: no
    credentials or connection strings are ever in reach.
  - `web_fetch` is off unless the operator allow-lists hosts. When on, it:
    - fetches `https` on port 443 only;
    - refuses non-public addresses (including IPv6 forms that embed IPv4);
    - pins the connection to the checked address, with certificate checks
      for the host name;
    - re-checks every redirect;
    - reads only uncompressed text, within size and time caps.
  - The planner never sees tool output or retrieved text (unchanged), so
    neither can choose the next tool.
- **Uploads are sniffed and bounded before any parser runs:**
  - magic bytes must match the extension;
  - a DOCX's ZIP directory is checked before inflating (total size, part
    count, ratio);
  - text with NUL bytes is refused;
  - every parser failure becomes a 422, never a 500.
- **Quotas are layered, each for a different resource:**
  - a flood guard per client on all requests, before routing (CPU and the
    database, on endpoints that have no limit of their own);
  - the existing per-client token bucket on expensive endpoints (the
    provider's requests per minute);
  - a new per-client daily share of live runs (fairness between visitors);
  - the global daily budget (the provider's daily quota);
  - a global cap on live sessions next to the existing document caps
    (database storage);
  - a body-size cap on every request (memory).
- **Defaults are hardened:**
  - security headers and `Cache-Control: no-store` on API responses;
  - secrets excluded from the settings `repr`;
  - a read-only CI token;
  - Dependabot also for the Docker base image.
- **Limits are measured, not assumed.** A Locust suite with an LLM stand-in
  (`loadtests/`) runs locally and in CI against the real image at the free
  plan's CPU and memory ([load-testing.md](../load-testing.md)).
  - The measurements decided what the flood guard is for. Against `GET /demo`
    it cut a flooding client from every request served (900 a second) to 417
    in three minutes, and other visitors' p95 latency went back to about
    baseline.
  - Where the endpoint already had a rate limit, it made no measurable
    difference.
- **Everything stays in process (ADR 0005).** No Redis, no WAF account.

**Consequences.**
- Every control has a test, and the threat model names its residual risks.
- In-process limits reset on restart and are per instance. That is correct for
  the single free instance, and needs a shared store if the API is scaled out.
- A contained flood still slows other visitors: it roughly tripled their
  median retrieval latency in the measurement. One Python process can't fully
  isolate a busy connection; volumetric protection belongs at the edge.
- The SQL tool answers questions about the corpus, not about a visitor's
  uploads: the session-scoped agent keeps only search and the calculator.
