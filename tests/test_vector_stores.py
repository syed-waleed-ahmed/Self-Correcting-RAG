"""Backend-agnostic vector store tests: persistence, refresh and concurrency.

Every test runs against the NumPy store and, when ``faiss-cpu`` is installed,
the FAISS store.
"""

from __future__ import annotations

import json
import threading

import numpy as np
import pytest

from scrag.errors import IndexIncompatibleError
from scrag.models import Chunk
from scrag.vectorstore._persistence import GENERATIONS_DIR, POINTER_FILE, read_pointer
from scrag.vectorstore.numpy_store import NumpyVectorStore

DIM = 4


def _faiss_store(path):
    pytest.importorskip("faiss")
    from scrag.vectorstore.faiss_store import FaissVectorStore

    return FaissVectorStore(dim=DIM, path=path)


@pytest.fixture(params=["numpy", "faiss"])
def make_store(request):
    factory = (
        (lambda path: NumpyVectorStore(dim=DIM, path=path))
        if request.param == "numpy"
        else _faiss_store
    )
    return factory


def _chunks(prefix: str, n: int) -> list[Chunk]:
    return [
        Chunk(id=f"{prefix}{i}", source=f"{prefix}.txt", text=f"{prefix} {i}") for i in range(n)
    ]


def _vectors(n: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((n, DIM), dtype=np.float32) + 0.01


def test_save_writes_generation_and_pointer(make_store, tmp_path):
    store = make_store(tmp_path)
    store.replace(_vectors(3), _chunks("a", 3))
    store.save()

    generation = read_pointer(tmp_path)
    assert generation == store.generation
    meta = json.loads((tmp_path / GENERATIONS_DIR / generation / "meta.json").read_text())
    assert meta["count"] == 3
    assert meta["dim"] == DIM

    reloaded = make_store(tmp_path)
    assert reloaded.load() is True
    assert len(reloaded) == 3
    assert reloaded.generation == generation


def test_only_recent_generations_are_kept(make_store, tmp_path):
    store = make_store(tmp_path)
    for i in range(5):
        store.replace(_vectors(2, seed=i), _chunks(f"g{i}", 2))
        store.save()
    remaining = list((tmp_path / GENERATIONS_DIR).iterdir())
    assert len(remaining) == 2
    assert store.generation in {p.name for p in remaining}


def test_refresh_picks_up_index_saved_by_another_process(make_store, tmp_path):
    """Simulates two API workers sharing an index directory."""
    worker_a, worker_b = make_store(tmp_path), make_store(tmp_path)
    worker_a.replace(_vectors(2), _chunks("old", 2))
    worker_a.save()
    worker_b.load()

    worker_a.replace(_vectors(5, seed=1), _chunks("new", 5))
    worker_a.save()

    assert worker_b.refresh_if_stale() is True
    assert len(worker_b) == 5
    assert worker_b.refresh_if_stale() is False  # already current
    assert worker_a.refresh_if_stale() is False  # its own save is not "stale"


def test_refresh_without_any_persisted_index_is_noop(make_store, tmp_path):
    assert make_store(tmp_path).refresh_if_stale() is False


def test_dangling_pointer_raises_incompatible(make_store, tmp_path):
    (tmp_path / POINTER_FILE).write_text("00000000000000000001-deadbeef")
    with pytest.raises(IndexIncompatibleError):
        make_store(tmp_path).load()


def test_corrupt_pointer_raises_incompatible(make_store, tmp_path):
    (tmp_path / POINTER_FILE).write_text("../../etc")
    with pytest.raises(IndexIncompatibleError):
        make_store(tmp_path).load()


def test_backend_mismatch_is_rejected(tmp_path):
    store = NumpyVectorStore(dim=DIM, path=tmp_path)
    store.replace(_vectors(1), _chunks("a", 1))
    store.save()
    meta_path = tmp_path / GENERATIONS_DIR / store.generation / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["backend"] = "faiss"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(IndexIncompatibleError):
        NumpyVectorStore(dim=DIM, path=tmp_path).load()


def test_legacy_flat_layout_still_loads(tmp_path):
    """Indexes written before 0.2 (files directly in index_dir) remain readable."""
    chunks = _chunks("legacy", 2)
    np.save(tmp_path / "vectors.npy", np.eye(2, DIM, dtype="float32"))
    (tmp_path / "chunks.jsonl").write_text(
        "".join(c.model_dump_json() + "\n" for c in chunks), encoding="utf-8"
    )
    store = NumpyVectorStore(dim=DIM, path=tmp_path)
    assert store.load() is True
    assert len(store) == 2
    assert store.search(np.array([1, 0, 0, 0], dtype="float32"), 1)[0].chunk.id == "legacy0"


def test_snapshot_round_trips_through_replace(make_store, tmp_path):
    store = make_store(tmp_path)
    store.replace(_vectors(3), _chunks("a", 3))
    vectors, chunks = store.snapshot()
    other = make_store(tmp_path / "other")
    other.replace(vectors, chunks)
    query = _vectors(1, seed=9)
    assert [r.chunk.id for r in other.search(query, 3)] == [
        r.chunk.id for r in store.search(query, 3)
    ]


def test_replace_validates_shapes(make_store, tmp_path):
    store = make_store(tmp_path)
    with pytest.raises(ValueError):
        store.replace(_vectors(2), _chunks("a", 3))


def test_searches_never_observe_a_mixed_index(make_store, tmp_path):
    """Concurrent replace() must be atomic from the point of view of search()."""
    store = make_store(tmp_path)
    generations = [(_vectors(50, seed=s), _chunks(p, 50)) for s, p in ((1, "a"), (2, "b"))]
    store.replace(*generations[0])

    errors: list[str] = []
    stop = threading.Event()

    def writer() -> None:
        for i in range(200):
            store.replace(*generations[i % 2])
        stop.set()

    def reader() -> None:
        query = _vectors(1, seed=7)
        while not stop.is_set():
            prefixes = {r.chunk.id[0] for r in store.search(query, 10)}
            if len(prefixes) != 1:
                errors.append(f"mixed result: {prefixes}")

    threads = [threading.Thread(target=writer)] + [
        threading.Thread(target=reader) for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not errors, errors[:3]
