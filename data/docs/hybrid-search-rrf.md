# Hybrid search with Reciprocal Rank Fusion

Hybrid search runs a dense vector search and a keyword search for the same
query and merges the two ranked lists. Merging is difficult to do with raw
scores, because cosine similarities and BM25 scores live on unrelated scales.

Reciprocal Rank Fusion (RRF) avoids the problem by using only ranks. Each
result receives, from every list it appears in, a contribution of 1 / (k +
rank), where rank starts at 1 and k is a smoothing constant. The constant is
commonly set to 60, the value used in the original RRF paper by Cormack,
Clarke and Buettcher. The contributions are summed and the results are sorted
by the fused score. A chunk ranked highly by both retrievers rises to the top;
a chunk found by only one retriever can still appear if it ranked well there.

RRF needs no training and no score calibration, which makes it a robust
default. Its main tuning knobs are the constant k and the number of candidates
taken from each retriever before fusion.

In SCRAG's hybrid mode, each retriever contributes a candidate pool larger
than the final top-k, the pools are fused with RRF using k = 60, and the best
fused chunks are passed on to the guardrail. Each returned chunk still reports
its cosine similarity, and additionally a fusion score so that the ordering
can be inspected.
