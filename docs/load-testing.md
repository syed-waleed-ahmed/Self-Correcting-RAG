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

It records memory before and after each run (process memory and page cache
separately), the cgroup's peak, and whether the container was OOM-killed, and
writes every table to the job summary. It runs manually and on pull requests
that change the load tests or the image.

First run ([42829377689](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/runs/42829377689),
a 4-CPU GitHub runner):

| Profile | Req/s | Failures | Search median / p95 | Sample answer median / p95 | Live answer median / p95 | Peak memory | OOM |
| --- | ---: | ---: | --- | --- | --- | ---: | --- |
| 1 CPU, 20 `Visitor`s | 9.8 | 0 | 3 / 5 ms | 1 / 2 ms | 990 / 1,000 ms | 457 MiB | no |
| 0.1 CPU, 10 `Visitor`s | 4.8 | 0 | 21 / 270 ms | 1 / 100 ms | 1,100 / 1,500 ms | 459 MiB | no |
| 0.1 CPU, 10 `DemoVisitor`s + 1 flood connection | 354 | 0 (visitors) | 83 / 390 ms | 49 / 290 ms | 1,111 / 2,200 ms | 236 MiB | no |

- **At Render Free's 0.1 CPU,** the real image served a busy 10-user load
  without a failure. Retrieval p95 rose to 270 ms, and live answers stayed
  LLM-bound.
- **The single flood connection got 14 of 42,068 requests served** (its burst
  of 3 plus 6 a minute), with no visitor errors.
- **Memory rose from about 210 MiB at startup to a 457-459 MiB peak** in the
  two capacity profiles, close to the 512 MiB limit.

## Memory: what the 460 MiB was, and the fix

The next run split process memory from page cache. It showed **439 MiB of
process memory before any load**, with no cache. The demo-mode container job
in CI measures about 215 MiB for the same image, so the load test was doing
something production doesn't:
- **The smoke script's ingest call.** Outside demo mode, `scripts/smoke.py`
  sent `POST /ingest {"reset": true}` (meant only to check that the demo
  refuses it), so the server re-embedded the whole corpus at runtime.
- **ONNX Runtime keeps the memory its largest batch needed.** One batch of 32
  long chunks sets the process size for good.

Measured locally (process memory, one full ingest of the 29-chunk corpus;
each run took about 0.5 s):

| Embedding batch size | 4 | 8 | 16 | 32 (old default) |
| --- | ---: | ---: | ---: | ---: |
| Memory added by the ingest, never released | +36 MB | +65 MB | +137 MB | **+229 MB** |

This mattered in production too. On the demo, the job worker embeds every
visitor upload at runtime at that batch size: one upload would have taken the
API from about 215 MiB to about 440 MiB of a 512 MiB instance. Two fixes:
- **The default `SCRAG_EMBEDDING_BATCH_SIZE` is now 8.** It keeps 165 MB of
  headroom, and a large upload still needs only 25 embedding calls.
- **`scripts/smoke.py` only calls `/ingest` against the demo**, where the call
  must be refused. It no longer re-ingests a server's corpus as a side effect.

The workflow now measures a runtime ingest directly, at 0.1 CPU and 512 MB,
with batch 32 and with the default:

| Run ([42833040513](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/runs/42833040513)) | Process memory before | After one runtime ingest | Peak (cgroup) | OOM-killed |
| --- | ---: | ---: | ---: | --- |
| Batch 32 (old default) | 199 MiB | 431 MiB (+232) | 438 MiB | no |
| **Batch 8 (new default)** | 205 MiB | **275 MiB (+70)** | 287 MiB | no |

In the same run, the serving profiles stayed at 214-222 MiB of process memory,
with peaks of 237-239 MiB. That was 457-462 MiB while the smoke script was
re-ingesting at batch 32.

## Latency at 0.1 CPU depends on the runner

The free-tier capacity profile, the same code apart from the memory fix, in
two runs:

| Run | Req/s | Failures | Search median / p95 | Live answer median / p95 |
| --- | ---: | ---: | --- | --- |
| [42829377689](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/runs/42829377689) | 4.8 | 0 | 21 / 270 ms | 1,100 / 1,500 ms |
| [42833040513](https://github.com/syed-waleed-ahmed/Self-Correcting-RAG/actions/runs/42833040513) | 3.2 | 0 | 1,200 / 3,000 ms | 3,500 / 6,100 ms |

- **The 1-CPU profile was stable across runs** (search 9 / 63-76 ms).
- **`--cpus 0.1` gives the container 10 ms of CPU per 100 ms.** Once ten busy
  users' requests need more than that in a period, they queue in 100 ms steps.
  A few milliseconds' difference in per-request CPU between runner machines
  then becomes a large difference in latency.
- **The workflow now records the runner's CPU model.** The robust conclusions
  from 0.1 CPU are the ones that held in both runs: no request failed, memory
  stayed bounded, and a flooding client was contained.
- **Ten users acting every 1-3 s is far busier than the demo's real traffic,**
  whose rate limit allows each visitor 6 expensive requests a minute.

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
