# Reranking retrieved passages

A reranker re-orders a short list of retrieved candidates using a more
expensive model than the one used for the first retrieval pass. The most
common choice is a cross-encoder: instead of embedding the query and each
passage separately, it reads the query and the passage together and outputs a
single relevance score. Because it sees both texts at once, a cross-encoder is
usually more accurate than comparing two independent embeddings.

The cost is speed. A cross-encoder must run once for every candidate passage,
so it is applied only to the top few dozen results of a cheaper first stage
such as dense, keyword or hybrid retrieval. On a machine with a small fraction
of a CPU core, reranking even twenty passages can add noticeable latency.

An LLM can also act as a reranker or relevance filter by scoring each passage.
SCRAG's guardrail agent works this way: it asks a language model to rate how
much each retrieved chunk helps answer the question on a scale from 0.0 to
1.0, and drops chunks below a threshold. To save requests against provider
rate limits, the guardrail can score all chunks in a single batched call and
falls back to one call per chunk if the batched answer cannot be parsed.
