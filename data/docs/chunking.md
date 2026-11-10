# Chunking documents for retrieval

Retrieval systems rarely index whole documents. Long documents are split into
smaller passages called chunks, and each chunk is embedded and searched
separately. Chunking matters because an embedding of a very long text averages
many topics together, which makes it hard to match a specific question, and
because the language model's context window can only hold a limited amount of
retrieved text.

SCRAG uses a character-based chunker with a default chunk size of 1000
characters and an overlap of 150 characters. It prefers to split on paragraph
boundaries first, then on sentence boundaries, and only falls back to hard
character cuts when a single sentence is longer than the chunk size. The
overlap carries the tail of one chunk into the start of the next, so a fact
that straddles a boundary still appears intact in at least one chunk.

Chunk size is a trade-off. Small chunks give precise matches but may lose the
surrounding context needed to answer a question. Large chunks keep context but
dilute the embedding and consume more of the prompt budget. Semantic chunking
is an alternative that groups sentences by topic similarity instead of by a
fixed length.

Every SCRAG chunk records its source file, its position within that file, and
a hash of the source document's content, which is used to detect when a
document has changed and needs to be re-indexed.
