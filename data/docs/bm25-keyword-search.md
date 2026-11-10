# BM25 keyword search

BM25 is a classic ranking function for keyword search. It scores a document
for a query by summing, over the query terms, a weight that grows with how
often the term appears in the document and with how rare the term is across
the whole collection. Rare terms, measured by inverse document frequency
(IDF), count for more than common ones.

Two parameters shape the formula. The parameter k1 controls term-frequency
saturation: after a term has appeared a few times, additional occurrences add
less and less to the score. A typical value of k1 is between 1.2 and 2.0. The
parameter b controls length normalization: with b set to 0.75, a long document
needs more matches than a short one to earn the same score, and with b set to
0 document length is ignored.

BM25 has no notion of meaning, so it cannot match synonyms or paraphrases. Its
strength is exact matching: identifiers, error codes, configuration variable
names, numbers and rare jargon are found reliably even when an embedding model
would blur them. Because dense and keyword retrieval fail in different ways,
combining them usually beats either one alone.

SCRAG implements BM25 in-process with k1 = 1.5 and b = 0.75 over the same
chunks that are stored in the vector index, and rebuilds the keyword index
automatically whenever the vector index changes.
