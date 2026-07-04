"""Document ingestion: load → chunk → embed → index.

Reads ``.txt``/``.md`` files from a directory, splits them into overlapping
chunks, embeds them in batches and writes them to a vector store. Embedding is
batched so ingestion scales to large corpora without exhausting memory.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunker
from .embeddings import Embedder
from .logging_config import get_logger
from .models import Chunk
from .vectorstore.base import VectorStore

logger = get_logger(__name__)

_SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass
class IngestStats:
    documents: int
    chunks: int
    index_size: int


def load_documents(docs_dir: Path | str) -> list[tuple[str, str]]:
    """Return ``(source_name, text)`` pairs for supported files under ``docs_dir``."""
    docs_dir = Path(docs_dir)
    if not docs_dir.exists():
        raise FileNotFoundError(f"Documents directory not found: {docs_dir}")
    documents: list[tuple[str, str]] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in _SUPPORTED_SUFFIXES or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8").strip()
        if text:
            documents.append((path.relative_to(docs_dir).as_posix(), text))
    return documents


def chunk_documents(
    documents: list[tuple[str, str]], chunker: Chunker
) -> list[Chunk]:
    """Split ``(source, text)`` pairs into a flat list of chunks."""
    chunks: list[Chunk] = []
    for source, text in documents:
        for position, piece in enumerate(chunker.split(text)):
            chunks.append(
                Chunk(
                    id=Chunk.make_id(source, position, piece),
                    source=source,
                    text=piece,
                    position=position,
                )
            )
    return chunks


def ingest(
    *,
    embedder: Embedder,
    store: VectorStore,
    chunker: Chunker,
    docs_dir: Path | str,
    batch_size: int = 32,
    reset: bool = True,
) -> IngestStats:
    """Run the full ingestion pipeline and persist the resulting index."""
    documents = load_documents(docs_dir)
    if not documents:
        raise ValueError(f"No ingestable documents found in {docs_dir}")

    chunks = chunk_documents(documents, chunker)
    logger.info("Chunked %d documents into %d chunks", len(documents), len(chunks))

    if reset:
        store.clear()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        vectors = embedder.embed([c.text for c in batch])
        store.add(vectors, batch)

    store.save()
    return IngestStats(
        documents=len(documents), chunks=len(chunks), index_size=len(store)
    )
