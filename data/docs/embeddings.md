# Text embeddings

An embedding model maps a piece of text to a fixed-length vector of numbers so
that texts with similar meaning end up close together in vector space. Dense
retrieval embeds the user's question with the same model used for the
documents and returns the chunks whose vectors are nearest to the question
vector.

SCRAG's default embedding model is all-MiniLM-L6-v2, a small sentence
transformer that produces 384-dimensional vectors. It is fast enough to run on
a CPU, which keeps the system free of GPU requirements. The model can run
through two interchangeable backends: sentence-transformers, which uses
PyTorch, and fastembed, which runs the same model with ONNX Runtime. Both
produce the same vectors for the same text, but the ONNX backend needs far
less memory because it does not load PyTorch, which matters on small hosting
instances with only 512 MB of RAM.

The dimensionality of the vectors must match the index. If the configured
embedding dimension does not match what the model actually produces, SCRAG
fails fast with an error that tells the operator which value to set, instead
of silently writing incompatible vectors. Changing the embedding model always
requires re-ingesting the documents, because vectors produced by different
models are not comparable.

Embeddings capture meaning well, so they handle paraphrases and synonyms, but
they can miss exact identifiers such as error codes, version numbers or rare
product names. Keyword search complements them in those cases.
