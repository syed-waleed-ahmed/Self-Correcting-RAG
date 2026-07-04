"""Text chunking.

Splits documents into overlapping, roughly sentence-aligned character windows.
Whole-file "chunks" don't scale (retrieval precision collapses on long docs and
context windows overflow), so ingestion always chunks. The splitter prefers to
break on paragraph/sentence boundaries and falls back to hard character cuts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Boundaries preferred for splitting, strongest first.
_PARAGRAPH = re.compile(r"\n\s*\n")
_SENTENCE = re.compile(r"(?<=[.!?])\s+")


@dataclass(frozen=True)
class Chunker:
    """Character-window chunker with overlap.

    Args:
        chunk_size: Maximum chunk length in characters.
        overlap: Number of trailing characters carried into the next chunk.
    """

    chunk_size: int = 1000
    overlap: int = 150

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.overlap < 0:
            raise ValueError("overlap must be non-negative")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")

    def split(self, text: str) -> list[str]:
        """Split ``text`` into non-empty chunks, none longer than ``chunk_size``."""
        text = text.strip()
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        chunks: list[str] = []
        current = ""
        for sent in self._sentences(text):
            if len(sent) > self.chunk_size:
                # A single over-long unit (e.g. a boundary-less blob): flush and
                # hard-split it. Those pieces already carry their own overlap.
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(self._hard_split(sent))
                continue
            if not current:
                current = sent
            elif len(current) + 1 + len(sent) <= self.chunk_size:
                current = f"{current} {sent}"
            else:
                chunks.append(current)
                current = self._with_overlap(current, sent)
        if current:
            chunks.append(current)
        return [c.strip() for c in chunks if c.strip()]

    def _sentences(self, text: str):  # type: ignore[no-untyped-def]
        """Yield sentence-ish units split on paragraph then sentence boundaries."""
        for para in _PARAGRAPH.split(text):
            para = para.strip()
            if not para:
                continue
            for sent in _SENTENCE.split(para):
                sent = sent.strip()
                if sent:
                    yield sent

    def _hard_split(self, text: str) -> list[str]:
        step = self.chunk_size - self.overlap  # > 0: overlap < chunk_size (enforced)
        return [text[i : i + self.chunk_size] for i in range(0, len(text), step)]

    def _with_overlap(self, previous: str, seg: str) -> str:
        """Seed the next chunk with overlap context, staying within chunk_size."""
        if self.overlap <= 0:
            return seg
        budget = self.chunk_size - len(seg) - 1
        if budget <= 0:
            return seg
        take = min(self.overlap, budget, len(previous))
        tail = previous[-take:].strip()
        return f"{tail} {seg}" if tail else seg
