from __future__ import annotations

import numpy as np
import pytest

from scrag.models import Chunk
from scrag.vectorstore.numpy_store import NumpyVectorStore


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", source="s.txt", text=f"chunk {i}", position=i)


def test_add_search_ordering(tmp_path):
    store = NumpyVectorStore(dim=3, path=tmp_path)
    vectors = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype="float32")
    store.add(vectors, [_chunk(0), _chunk(1), _chunk(2)])
    assert len(store) == 3

    results = store.search(np.array([0.9, 0.1, 0.0], dtype="float32"), top_k=2)
    assert [r.chunk.id for r in results] == ["c0", "c1"]
    assert results[0].score > results[1].score
    assert results[0].score == pytest.approx(1.0, abs=1e-2)


def test_search_on_empty_store(tmp_path):
    assert NumpyVectorStore(dim=3, path=tmp_path).search(np.ones(3, dtype="float32"), 5) == []


def test_length_mismatch_rejected(tmp_path):
    store = NumpyVectorStore(dim=3, path=tmp_path)
    with pytest.raises(ValueError):
        store.add(np.zeros((2, 3), dtype="float32"), [_chunk(0)])


def test_persistence_round_trip(tmp_path):
    store = NumpyVectorStore(dim=3, path=tmp_path)
    store.add(np.eye(3, dtype="float32"), [_chunk(0), _chunk(1), _chunk(2)])
    store.save()

    reloaded = NumpyVectorStore(dim=3, path=tmp_path)
    assert reloaded.load() is True
    assert len(reloaded) == 3
    top = reloaded.search(np.array([1, 0, 0], dtype="float32"), top_k=1)
    assert top[0].chunk.id == "c0"


def test_load_missing_index_returns_false(tmp_path):
    assert NumpyVectorStore(dim=3, path=tmp_path).load() is False


def test_load_dim_mismatch_raises(tmp_path):
    store = NumpyVectorStore(dim=3, path=tmp_path)
    store.add(np.eye(3, dtype="float32"), [_chunk(0), _chunk(1), _chunk(2)])
    store.save()
    with pytest.raises(ValueError):
        NumpyVectorStore(dim=4, path=tmp_path).load()


def test_clear(tmp_path):
    store = NumpyVectorStore(dim=3, path=tmp_path)
    store.add(np.eye(3, dtype="float32"), [_chunk(0), _chunk(1), _chunk(2)])
    store.clear()
    assert len(store) == 0
