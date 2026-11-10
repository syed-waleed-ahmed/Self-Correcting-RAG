# Evaluation

SCRAG ships an offline evaluation harness (`scrag eval`) and a golden dataset
(`data/eval/golden.jsonl`). Every number on this page was measured, and the
date and setup are recorded so it can be reproduced.

## How to run it

```bash
scrag ingest                              # index data/docs
scrag eval --k 5                          # dense vs hybrid retrieval, no LLM calls
scrag eval --k 5 --answers --limit 8      # also run the full pipeline (spends LLM quota)
scrag eval --compare 8                    # compare four answer configurations (spends LLM quota)
scrag eval --output report.json           # full per-question report
```

## Methodology

- **Golden set:** 60 questions over the 16 sample documents in `data/docs`.
  - Each question is labelled with the document(s) containing its answer and
    with keywords a correct answer should contain (`answer_contains`).
  - 25 are tagged `lexical` (exact identifiers, numbers, names) and 35
    `semantic` (paraphrased concepts).
  - Questions q41 to q60 were added on 2026-11-10 to cover the weakest
    documents (chunking, free-tier deployment, hallucination, prompt
    injection, evaluation metrics, API).
  - Each new question's keywords were checked to occur in its labelled source
    text, and a test (`tests/test_evaluation.py`) checks that every label
    points to a real document.
- **Retrieval metrics** are computed at the source-document level: the top-k
  chunks are mapped to their files, keeping the first occurrence of each.
  - recall@k: share of a question's relevant documents that appear in the
    top k.
  - hit@k: share of questions with at least one relevant document in the top k.
  - MRR: mean of 1 / rank of the first relevant document.
  - nDCG@k: log-discounted gain, normalized so a perfect ranking scores 1.
- **Latency** is the mean wall time of one `retrieve()` call. It is measured
  after a warm-up call, so one-time model loading is excluded.

## Results: dense vs hybrid retrieval (2026-11-10, 60 questions)

Setup:
- 16 documents and 29 chunks (1000 characters, 150 overlap).
- Embeddings from `all-MiniLM-L6-v2` via fastembed (ONNX).
- Hybrid mode: BM25 (k1 = 1.5, b = 0.75) fused with dense results by Reciprocal
  Rank Fusion (k = 60, candidate pool 20).
- Measured on a Windows development machine.

| k | Mode | Recall@k | Hit@k | MRR | nDCG@k | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| 3 | dense | 0.950 | 0.967 | 0.939 | 0.932 | 3.1 ms |
| 3 | hybrid | 0.950 | 0.967 | **0.958** | **0.946** | 2.7 ms |
| 5 | dense | 0.967 | 0.983 | 0.943 | 0.939 | 2.5 ms |
| 5 | hybrid | **0.992** | **1.000** | **0.966** | **0.965** | 2.7 ms |

By question type at k = 5:

| Mode | Lexical recall | Lexical MRR | Semantic recall | Semantic MRR |
| --- | --- | --- | --- | --- |
| dense | 0.960 | 0.883 | 0.971 | 0.986 |
| hybrid | **1.000** | **0.938** | **0.986** | 0.986 |

Hybrid retrieval helps most on lexical questions such as "What constant is
commonly used in RRF?" and "Which response header carries the request
identifier?". These are the questions where dense embeddings blur exact terms.
Its latency is the same as dense. On this evidence `hybrid` is the default
`SCRAG_RETRIEVAL_MODE`.

Known weak spots, kept in the set rather than tuned away:
- q35 ("What relevance threshold does the guardrail use by default?") and q36
  ("Which language model does SCRAG use by default?") are both missed at k = 3
  by both modes.
- At k = 5, hybrid finds them at ranks 4 and 5.
- Their answers are single values inside a long architecture document.

The first version of this page (40 questions) measured hybrid MRR 0.961
against dense 0.915 at k = 5. Adding the 20 questions narrowed the gap but
kept the ranking.

## Answer quality: baseline vs hybrid vs self-correcting vs agent (2026-11-10)

```bash
scrag eval --no-retrieval --compare 8 --answer-top-k 3        # the four configurations
scrag eval --golden data/eval/hard.jsonl --no-retrieval \
  --compare 6 --configs baseline,self_correcting               # hard and unanswerable questions
```

**The four configurations.** All use `qwen/qwen3.8-27b` on Groq's free tier
for every role:

| Configuration | What it does |
| --- | --- |
| `baseline` | dense retrieval, one generation |
| `hybrid` | hybrid retrieval, one generation |
| `self_correcting` | the deployed pipeline: hybrid retrieval, guardrail, then evaluate and correct |
| `agent` | plan, tools, write, verify |

**How each answer is measured:**
- **Judge score:** graded once by the same judge call (the evaluator model,
  against the context that configuration used).
- **Pass rate:** the share of answers with a judge score of at least 0.7.
- **Keyword coverage:** the share of the golden `answer_contains` keywords
  present in the answer. It is the model-independent signal.
- **Tokens and calls:** the configuration's own LLM usage. The judge's tokens
  are measured separately.
- **Latency:** mean wall time per answer.
- **Pacing:** runs stayed under 6,000 tokens per minute (Groq's limit for this
  model is 8,000).

### Answerable questions (8 questions, top_k = 3)

A stratified sample of the golden set: q01, q05, q11, q16, q28, q34, q41 and
q47, four lexical and four semantic. The run used 63.7k tokens in total.

| Configuration | Judge score | Pass rate | Keyword coverage | Tokens per answer | LLM calls | Latency |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 1.000 | 8/8 | 1.000 | 688 | 1.0 | 441 ms |
| hybrid | 1.000 | 8/8 | 1.000 | 696 | 1.0 | 431 ms |
| self-correcting | 1.000 | 8/8 | 1.000 | 1,797 | 3.0 | 1,145 ms |
| agent | 1.000 | 8/8 | 1.000 | 1,924 | 3.0 | 869 ms |

### Hard and unanswerable questions (`data/eval/hard.jsonl`)

- **h01 and h02:** the two hardest golden questions (the q35 and q36 weak
  spots).
- **h03 to h06:** questions the corpus cannot answer, including traps the
  model could answer from its training data: the year of the RRF paper and
  the capital of Australia. The right answer is to decline.

| Question set | top_k | baseline (dense) | hybrid | self-correcting |
| --- | --- | --- | --- | --- |
| Unanswerable (4): declined | 3 | 4/4 | not run | 4/4 |
| Hard (2): answered correctly | 3 | 0/2 (declined) | not run | 0/2 (declined) |
| Hard (2): answered correctly | 5 | 0/2 | **2/2** | **2/2** |

### What this shows

1. **On this corpus, self-correction never changed an outcome.**
   - In every measured case its first attempt passed.
   - It costs about 2.6 times the tokens and latency of single-shot hybrid
     answering.
   - Its value here is a checked grounding status, not better answers.
   - The sample corpus is small and clean. Noisy, contradictory or adversarial
     corpora are where correction should matter, and they are untested here.
2. **Retrieval is the bottleneck.** Every wrong answer measured was a missing
   chunk, not a bad generation.
   - Hybrid retrieval at k = 5 fixed both hard questions; neither
     self-correction nor dense retrieval did.
   - This is why the demo's `top_k` was raised from 3 to 5. It costs about
     40% more tokens per answer, and on the retrieval benchmark it took hit@k
     from 0.967 to 1.000.
3. **The model declines instead of hallucinating.** Under the "say you are
   not sure" instruction, it declined all four unanswerable questions in both
   configurations, including the traps.
4. **The judge alone is not a quality measure.**
   - It scored every answer 1.00, including correct refusals to answer
     questions that were answerable.
   - It measures faithfulness to the context, not completeness.
   - It is the same model family as the generator.
   - Keyword coverage and abstention decided every comparison above.

These samples are small (8, 6, 2 and 2 questions), so the numbers are
indicative. The harness (`scrag eval --compare N`) is there to run larger or
harder sets as quota allows.

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
