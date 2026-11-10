# SCRAG architecture

SCRAG, short for Self-Correcting Retrieval-Augmented Generation, is an open
source Python service that answers questions over a document collection and
checks its own answers. It exposes the same core through a FastAPI REST API
and a Typer command-line interface.

A query passes through four stages. The retriever embeds the question and
fetches the most relevant chunks using dense, keyword or hybrid search. The
guardrail agent scores each chunk's relevance with a language model and drops
chunks below a threshold of 0.3; if it drops everything, the pipeline falls
back to the raw retrieval results. The generator agent writes an answer using
only the kept chunks. The evaluator agent scores how well the answer is
grounded in those chunks.

If the evaluator's score is below the threshold of 0.7, the pipeline runs a
self-correction step: it sends the previous answer and the evaluator's
explanation back to the generator and asks for an improved answer. The loop
runs at most two attempts by default and always returns the highest-scoring
attempt, so self-correction can never make the final answer worse. If the
evaluator itself fails, the loop stops and the answer is reported with an
evaluation status of unavailable.

Components are wired together in a single composition root, so the embedding
model, vector store and language model provider can be swapped through
configuration. The default language model is qwen/qwen3.8-27b served by Groq,
and any OpenAI-compatible endpoint can be used instead.
