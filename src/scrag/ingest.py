"""Document ingestion: load → chunk → embed → index.

Reads supported files (text, Markdown, HTML, CSV, JSON, and PDF/DOCX with the
``docs`` extra; see :mod:`scrag.loaders`) from a directory, splits them into
overlapping chunks, embeds them in batches and writes them to a vector store.
Files that can't be read, or exceed the size limits, are skipped with a warning.

Modes:
    * ``reset=True``  — rebuild: the index mirrors the documents directory.
    * ``reset=False`` — upsert by source: unchanged documents (same content hash)
      are skipped, changed documents have their old chunks replaced, new
      documents are added. Nothing is ever deleted in this mode.

The new index is computed in full and then published with one atomic
``store.replace``, so concurrent queries see either the old index or the new
one, never a partially ingested state.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .chunking import Chunker
from .embeddings import Embedder, embed_checked
from .errors import DocumentsNotFoundError, NoDocumentsError
from .loaders import SUPPORTED_SUFFIXES, LoaderError, extract_text
from .logging_config import get_logger
from .models import Chunk
from .vectorstore.base import VectorStore

logger = get_logger(__name__)

DOC_HASH_KEY = "doc_hash"
DEFAULT_MAX_DOCUMENT_BYTES = 10_000_000


@dataclass
class IngestStats:
    documents: int
    chunks: int
    index_size: int
    added_documents: int = 0
    updated_documents: int = 0
    skipped_documents: int = 0


def content_hash(text: str) -> str:
    """Stable fingerprint of a document's text, used to detect changes."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def load_documents(
    docs_dir: Path | str,
    *,
    max_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
    max_pdf_pages: int = 300,
) -> list[tuple[str, str]]:
    """Return ``(source_name, text)`` pairs for supported files under ``docs_dir``.

    Unreadable or oversized files are skipped with a warning rather than
    aborting the whole ingestion.
    """
    docs_dir = Path(docs_dir)
    if not docs_dir.is_dir():
        raise DocumentsNotFoundError(f"Documents directory not found: {docs_dir}")
    documents: list[tuple[str, str]] = []
    for path in sorted(docs_dir.rglob("*")):
        if path.suffix.lower() not in SUPPORTED_SUFFIXES or not path.is_file():
            continue
        source = path.relative_to(docs_dir).as_posix()
        try:
            text = extract_text(path, max_bytes=max_bytes, max_pdf_pages=max_pdf_pages)
        except LoaderError as exc:
            logger.warning("Skipping %s: %s", source, exc)
            continue
        if text:
            documents.append((source, text))
    return documents


def chunk_documents(documents: list[tuple[str, str]], chunker: Chunker) -> list[Chunk]:
    """Split ``(source, text)`` pairs into a flat list of chunks tagged with the doc hash."""
    chunks: list[Chunk] = []
    for source, text in documents:
        doc_hash = content_hash(text)
        for position, piece in enumerate(chunker.split(text)):
            chunks.append(
                Chunk(
                    id=Chunk.make_id(source, position, piece),
                    source=source,
                    text=piece,
                    position=position,
                    metadata={DOC_HASH_KEY: doc_hash},
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
    max_document_bytes: int = DEFAULT_MAX_DOCUMENT_BYTES,
    max_pdf_pages: int = 300,
) -> IngestStats:
    """Run the full ingestion pipeline and persist the resulting index."""
    documents = load_documents(docs_dir, max_bytes=max_document_bytes, max_pdf_pages=max_pdf_pages)
    if not documents:
        raise NoDocumentsError(f"No ingestable documents found in {docs_dir}")

    kept_vectors: np.ndarray
    if reset:
        kept_vectors = np.zeros((0, store.dim), dtype="float32")
        kept_chunks: list[Chunk] = []
        to_embed = documents
        added, updated, skipped = len(documents), 0, 0
    else:
        existing_vectors, existing_chunks = store.snapshot()
        known_hashes: dict[str, str | None] = {}
        for chunk in existing_chunks:
            known_hashes.setdefault(chunk.source, chunk.metadata.get(DOC_HASH_KEY))
        to_embed = []
        added = updated = skipped = 0
        for source, text in documents:
            if source not in known_hashes:
                added += 1
                to_embed.append((source, text))
            elif known_hashes[source] == content_hash(text):
                skipped += 1
            else:  # changed, or indexed before hashes were recorded
                updated += 1
                to_embed.append((source, text))
        replaced = {source for source, _ in to_embed}
        keep = [i for i, chunk in enumerate(existing_chunks) if chunk.source not in replaced]
        kept_vectors = existing_vectors[keep]
        kept_chunks = [existing_chunks[i] for i in keep]

    new_chunks = chunk_documents(to_embed, chunker)
    logger.info(
        "Ingest: %d documents (%d added, %d updated, %d unchanged) -> %d new chunks",
        len(documents),
        added,
        updated,
        skipped,
        len(new_chunks),
    )

    if reset or new_chunks:
        new_vectors = _embed_in_batches(embedder, new_chunks, store.dim, batch_size)
        store.replace(np.vstack([kept_vectors, new_vectors]), kept_chunks + new_chunks)
        store.save()

    return IngestStats(
        documents=len(documents),
        chunks=len(new_chunks),
        index_size=len(store),
        added_documents=added,
        updated_documents=updated,
        skipped_documents=skipped,
    )


def _embed_in_batches(
    embedder: Embedder, chunks: list[Chunk], dim: int, batch_size: int
) -> np.ndarray:
    batches = [
        embed_checked(embedder, [c.text for c in chunks[start : start + batch_size]], dim)
        for start in range(0, len(chunks), batch_size)
    ]
    return np.vstack(batches) if batches else np.zeros((0, dim), dtype="float32")
