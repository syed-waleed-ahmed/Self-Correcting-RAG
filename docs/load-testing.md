# Load testing

How the API behaves under load, measured with [Locust](https://locust.io)
(`loadtests/`). Every number here comes from a recorded run; the raw CSVs of
the CI runs are kept as workflow artifacts.

## Method

- **No real LLM.** The server's `SCRAG_LLM_BASE_URL` points at
  `loadtests/mock_llm.py`, an OpenAI-compatible stand-in that answers each
  agent in the shape it expects after a fixed **300 ms** per call (about a
  hosted model's response time). Load tests therefore never spend a provider's
  quota, and they measure this server, not the provider's rate limits.
- **The real code path.** Retrieval embeds every query with the deployment
  embedder (fastembed, ONNX) and runs hybrid BM25 + dense search over the
  bundled sample index; a live answer makes the same three LLM calls as
  production (batched guardrail, generator, evaluator).
- **Profiles** (`loadtests/locustfile.py`):
  - `Visitor`, a busy visitor with 1-3 s between actions. The mix: health 1,
    demo info 2, retrieval-only search 4, recorded sample answer 4, live
    question 1 (a unique question, so it misses every cache).
  - `DemoVisitor`, the same mix at a reading pace (10-20 s), each user from
    its own client address.
  - `Flooder`, one abusive client (one address) sending `/search` as fast as
    it can over `FLOOD_CONNECTIONS` concurrent connections.
- **Summaries** come from `loadtests/summarize.py`. Latencies are client-side,
  including the 300 ms LLM wait where there is one.

## Capacity (local, demo mode off)

One `scrag serve` process (uvicorn, one worker, Python 3.10) on a 12-thread
AMD laptop CPU, with no rate limits. Two minutes per step.

| Users | Requests | Failures | Req/s | Search median / p95 | Sample answer median / p95 | Live answer median / p95 | Health p95 |
| ---: | ---: | ---: | ---: | --- | --- | --- | ---: |
| 20 | 1,140 | 0 | 9.6 | 10 / 18 ms | 6 / 10 ms | 950 / 970 ms | 8 ms |
| 100 | 5,602 | 0 | 47.1 | 11 / 25 ms | 7 / 20 ms | 950 / 980 ms | 12 ms |
| 300 | 16,275 | 0 | 135.6 | 29 / 410 ms | 20 / 390 ms | 1,000 / 1,400 ms | 270 ms |

- **No request failed** at any step.
- **A live answer costs about 35 ms of server time** on top of its three LLM
  calls. Warm breakdown: retrieval 4-5 ms, then about 310 ms for each LLM
  stage.
- **The knee is between 100 and 300 users** (50-135 requests/s). Medians stay
  low but p95 latency grows tenfold as requests queue in the single process.

## Abuse: one flooding client (local, demo mode on)

The same machine, with a demo-mode server running the production protections:
- 6 requests/min per client on the expensive endpoints, burst 3;
- a flood guard of 120/min per client, burst 60;
- a large daily budget, only so that the budget can't end a run.

Each run is 3 minutes: 30 `DemoVisitor`s, each from its own address, plus one
abusive client flooding over 8 concurrent connections from a single address
(`FLOOD_CONNECTIONS=8`). "No guard" disables only the flood guard, by setting
its limit impossibly high.

**Flooding `/search`** (an endpoint with its own rate limit):

| Run | Flood req/s | Flood requests served | Visitor search median / p95 | Sample answer median / p95 | Live answer median / p99 | Visitor failures |
| --- | ---: | ---: | --- | --- | --- | ---: |
| No flooder | n/a | n/a | 11 / 26 ms | 7 / 18 ms | 950 / 980 ms | 0 |
| No guard | 697 | 20 of 124,360 | 35 / 72 ms | 21 / 60 ms | 990 / 3,300 ms | 0 |
| With guard | 709 | 20 of 126,136 | 46 / 100 ms | 32 / 100 ms | 980 / 1,100 ms | 0 |

**Flooding `GET /demo`** (no rate limit of its own, like every GET endpoint):

| Run | Flood req/s | Flood requests served | Visitor search median / p95 | Sample answer median / p95 | Live answer median / p99 | Visitor failures |
| --- | ---: | ---: | --- | --- | --- | ---: |
| No guard | 901 | all 160,900 | 28 / 69 ms | 21 / 42 ms | 970 / 1,100 ms | 0 |
| With guard | 1,551 | 417 of 277,140 | 19 / 27 ms | 15 / 22 ms | 950 / 980 ms | 0 |

What this shows:
- **No visitor request failed in any run, and none was rate limited.** Each
  visitor has its own address and its own buckets.
- **The per-endpoint rate limit contains a flood on the endpoints it covers.**
  Of about 125,000 flood requests to `/search`, 20 got through: the burst of 3
  plus 6 a minute.
  - The flood guard adds nothing measurable there. A refusal costs about the
    same from either layer, because most of a request's cost is HTTP handling,
    not routing.
- **The guard matters on endpoints without their own limit.**
  - Without it, the flooding client got all 160,900 `/demo` responses (900 a
    second).
  - With it, the client got 417 (its burst of 60 plus 120 a minute), the rest
    were refused in 3 ms, and visitors' p95 latencies went back to about the
    no-flood baseline.
  - On the deployed demo, those endpoints include the database-backed reads of
    sessions, documents, jobs and history, so the guard also protects the free
    database's compute.
- **A contained flood still slows everyone else.** It roughly tripled
  visitors' median retrieval latency in the `/search` runs, and the medians
  stay above baseline with the guard too. One Python process can't fully
  isolate a busy connection; volumetric protection belongs at the edge.

## Free-tier container (CI)

The `Load test` workflow (`.github/workflows/load-test.yml`) builds the
deployment image and runs three profiles against it with `--cpus 0.1
--memory 512m`, Render Free's allowance:
- capacity at 1 CPU;
- capacity at 0.1 CPU;
- abuse at 0.1 CPU (10 `DemoVisitor`s and one flood connection).

It records end and peak memory and whether the container was OOM-killed, and
writes every table to the job summary. It runs manually and on pull requests
that change the load tests or the image.

## Running it

```bash
pip install -e ".[load]"
python loadtests/mock_llm.py --port 8090 --latency-ms 300 &
SCRAG_LLM_API_KEY=mock SCRAG_LLM_BASE_URL=http://127.0.0.1:8090/v1 scrag serve --port 8000 &
locust -f loadtests/locustfile.py --headless -u 20 -r 5 -t 2m \
    --host http://127.0.0.1:8000 --csv out/capacity Visitor
python loadtests/summarize.py out/capacity_stats.csv out/capacity_failures.csv
```

For the abuse profile, start the server with `SCRAG_DEMO_MODE=true`,
`SCRAG_FORWARDED_IP_HEADER=x-forwarded-for` and a large
`SCRAG_DEMO_DAILY_QUERY_BUDGET` (so the budget doesn't end the run), then run
`DemoVisitor Flooder` with `FLOOD_CONNECTIONS` set.

Never point a load test at the public demo: it runs on free tiers, and
[SECURITY.md](../SECURITY.md) puts load testing it out of scope.
