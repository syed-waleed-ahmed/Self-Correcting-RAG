# Evaluation

SCRAG ships an offline evaluation harness (`scrag eval`) and a golden dataset
(`data/eval/golden.jsonl`). Every number on this page was measured, and the
date and setup are recorded so it can be reproduced.

## How to run it

```bash
scrag ingest                              # index data/docs
scrag eval --k 5                          # dense vs hybrid retrieval, no LLM calls
scrag eval --k 5 --answers --limit 8      # also run the full pipeline (spends LLM quota)
scrag eval --output report.json           # full per-question report
```

## Methodology

- **Golden set:** 40 questions over the 16 sample documents in `data/docs`.
  Each question is labelled with the document(s) containing its answer. 18 are
  tagged `lexical` (exact identifiers, numbers, names) and 22 `semantic`
  (paraphrased concepts). A test (`tests/test_evaluation.py`) checks that every
  label points to a real document.
- **Retrieval metrics** are computed at the source-document level: the top-k
  chunks are mapped to their files, keeping the first occurrence of each.
  - recall@k: share of a question's relevant documents that appear in the
    top k.
  - hit@k: share of questions with at least one relevant document in the top k.
  - MRR: mean of 1 / rank of the first relevant document.
  - nDCG@k: log-discounted gain, normalized so a perfect ranking scores 1.
- **Latency** is the mean wall time of one `retrieve()` call. It is measured
  after a warm-up call, so one-time model loading is excluded.

## Results: dense vs hybrid retrieval (2026-11-10)

Setup:
- 16 documents and 29 chunks (1000 characters, 150 overlap).
- Embeddings from `all-MiniLM-L6-v2` via fastembed (ONNX).
- Hybrid mode: BM25 (k1 = 1.5, b = 0.75) fused with dense results by Reciprocal
  Rank Fusion (k = 60, candidate pool 20).
- Measured on a Windows development machine.

| k | Mode | Recall@k | Hit@k | MRR | nDCG@k | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | dense | 0.925 | 0.950 | 0.908 | 0.898 | |
| 3 | hybrid | 0.925 | 0.950 | **0.950** | **0.929** | |
| 5 | dense | 0.950 | 0.975 | 0.915 | 0.908 | 4.2 ms |
| 5 | hybrid | **0.988** | **1.000** | **0.961** | **0.957** | 4.0 ms |

By question type at k = 5:

| Mode | Lexical MRR | Semantic MRR |
| --- | --- | --- |
| dense | 0.838 | 0.977 |
| hybrid | **0.914** | **1.000** |

Hybrid retrieval helps most on lexical questions such as "What constant is
commonly used in RRF?" and "Which NumPy function selects the best matches?".
These are the questions where dense embeddings blur exact terms. Its latency is
the same as dense. On this evidence `hybrid` is the default
`SCRAG_RETRIEVAL_MODE`.

Two questions are still answered poorly by both modes: q35 ("What relevance
threshold does the guardrail use by default?") and q36 ("Which language model
does SCRAG use by default?"). Their answers are single values inside a long
architecture document. They are left in the golden set as known misses rather
than tuned away.

## Embedding backends

Measured on 2026-11-10, embedding 32 texts and then running 20 single-query
embeddings, each backend in a fresh process. Memory is the peak working set on
Windows.

| Backend | Peak memory | Per-query time | Vector agreement |
| --- | --- | --- | --- |
| fastembed (ONNX) | 211 MB | 3.0 ms | identical (cosine 1.00000 on every text) |
| sentence-transformers (torch) | 428 MB | 12.3 ms | reference |

This is why the deployment image uses fastembed (decision D1).

The CI `container` job runs the real deployment image on Linux under the free
tier's limits (`--memory=512m --cpus=0.1`). It smoke-tests the image, sends 15
retrieval requests, and reads the cgroup counters. First recorded run
(2026-11-10, GitHub Actions run 42735463057):

| Metric | Value |
| --- | --- |
| Peak memory | 210 MB of 512 MB |
| Memory after load | 196 MB |
| OOM killed | false |
| Image size | 296 MB |

## Recorded demo answers

`scripts/record_samples.py` ran the full pipeline live for the 8 demo sample
questions on 2026-11-10:
- model: `qwen/qwen3.8-27b`;
- retrieval: hybrid, top_k = 3;
- guardrail: batched.

All 8 runs passed evaluation on the first attempt (judge score 1.0), so these
recordings don't exercise self-correction. Live queries show the attempt
timeline whenever a retry happens.

## Limitations

- The corpus and golden set are small and were written by the same author,
  which makes the absolute numbers optimistic. The comparison between modes
  on identical data is what matters.
- The evaluator is an LLM judge. Its scores are heuristic signals, and the same
  model family also generates the answers (self-preference bias).
- Answer-level evaluation (`--answers`) was not run on the full golden set,
  because about 40 full pipeline runs would exceed the free-tier daily token
  quota in one go. Run it in slices with `--limit`.
