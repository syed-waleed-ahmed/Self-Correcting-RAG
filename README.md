# Self-Correcting RAG Pipeline (Multi-Agent)

This project implements a **self-correcting Retrieval-Augmented Generation (RAG) pipeline** with 4 LLM roles:

1. **Retriever** – uses local sentence-transformer embeddings to retrieve relevant documents.
2. **Guardrail Agent** – scores each retrieved chunk for relevance and filters out off-topic context.
3. **Generator Agent** – generates answers strictly grounded in the filtered context.
4. **Evaluator Agent** – evaluates the factual consistency of the answer with the context and can trigger self-correction.

## Tech Stack

- Python 3
- `sentence-transformers` (`all-MiniLM-L6-v2`) for local embeddings
- Groq API (OpenAI-compatible) for:
  - Guardrail relevance scoring
  - Answer generation
  - Answer evaluation

## Project Structure

```text
self_correcting_rag/
  data/
    docs/           # knowledge base as .txt files
  src/
    config.py
    llm_client.py
    embeddings.py
    retriever.py
    guardrail_agent.py
    generator_agent.py
    evaluator_agent.py
    pipeline.py
  main.py
  requirements.txt
  .env              # contains GROQ_API_KEY (not committed)
```

  ## Author
  
  SYED WALEED AHMED
