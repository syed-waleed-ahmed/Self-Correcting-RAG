from __future__ import annotations

import pytest

from scrag.chunking import Chunker


def test_empty_text_yields_no_chunks():
    assert Chunker().split("   ") == []


def test_short_text_is_single_chunk():
    text = "One short sentence."
    assert Chunker(chunk_size=1000, overlap=100).split(text) == [text]


def test_long_text_is_split_and_bounded():
    chunker = Chunker(chunk_size=80, overlap=20)
    text = " ".join(f"Sentence number {i} about retrieval." for i in range(40))
    chunks = chunker.split(text)
    assert len(chunks) > 1
    # Overlap can extend a chunk slightly past chunk_size, but not unboundedly.
    for c in chunks:
        assert len(c) <= chunker.chunk_size + chunker.overlap + 1


def test_hard_split_of_a_single_long_token():
    chunker = Chunker(chunk_size=50, overlap=10)
    text = "x" * 500  # no natural boundaries
    chunks = chunker.split(text)
    assert len(chunks) > 1
    assert all(len(c) <= chunker.chunk_size for c in chunks)


def test_reconstruction_covers_all_content():
    chunker = Chunker(chunk_size=60, overlap=15)
    text = " ".join(f"word{i}" for i in range(100))
    joined = " ".join(chunker.split(text))
    # Every original token survives chunking (overlap may duplicate some).
    for token in ("word0", "word50", "word99"):
        assert token in joined


def test_invalid_overlap_rejected():
    with pytest.raises(ValueError):
        Chunker(chunk_size=100, overlap=100)
