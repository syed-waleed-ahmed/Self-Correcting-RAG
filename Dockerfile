# syntax=docker/dockerfile:1
#
# Deployment image, sized for free hosting tiers (512 MB RAM):
# - embeddings run on fastembed (ONNX Runtime), not PyTorch: same vectors, about
#   half the memory, and a much smaller image;
# - the embedding model and the sample knowledge-base index are baked in at
#   build time, because free hosts wipe their local disk on every restart.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    SCRAG_EMBEDDING_BACKEND=fastembed \
    SCRAG_EMBEDDING_CACHE_DIR=/app/models

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install ".[onnx,docs]"

COPY data/docs ./data/docs
COPY data/demo ./data/demo
# Downloads the embedding model into /app/models and builds data/index.
RUN scrag ingest

RUN useradd --create-home appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import os, sys, urllib.request; port = os.environ.get('PORT', '8000'); sys.exit(0 if urllib.request.urlopen(f'http://localhost:{port}/health').status == 200 else 1)"

# Hosts such as Render inject PORT; default to 8000 locally.
CMD ["sh", "-c", "exec scrag serve --host 0.0.0.0 --port ${PORT:-8000}"]
