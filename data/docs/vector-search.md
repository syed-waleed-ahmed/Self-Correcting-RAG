# Vector search and vector stores

A vector store holds chunk embeddings and answers nearest-neighbour queries.
SCRAG normalizes every vector to unit length, so the inner product of two
vectors equals their cosine similarity, a score between -1 and 1 where higher
means more similar.

The default backend is a NumPy store that performs exact brute-force search:
it multiplies the query vector by the matrix of all chunk vectors and selects
the best matches with argpartition, which keeps a query linear in the number
of chunks. Exact search is simple and accurate, and it remains practical well
into hundreds of thousands of chunks. An optional FAISS backend uses a flat
inner-product index and can be swapped for approximate structures such as
HNSW or IVF when the corpus grows larger.

Approximate nearest-neighbour indexes such as HNSW trade a small amount of
recall for much faster search on very large collections. PostgreSQL can also
act as a vector store through the pgvector extension, which keeps vectors next
to relational data and supports HNSW indexes.

SCRAG persists its index as immutable generations. Each save writes a new
generation directory and then atomically repoints a small file named CURRENT
at it. Readers never see a half-written index, and other server processes
notice the new generation and reload it before answering their next query.
