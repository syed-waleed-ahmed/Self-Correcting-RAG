"""Atomic, multi-process-friendly index persistence shared by the file backends.

Layout under the index directory::

    CURRENT              name of the live generation (replaced atomically)
    generations/<gen>/   immutable files written by a single save()
        meta.json        {"format_version", "backend", "dim", "count", "generation"}
        ...              backend payload (e.g. vectors.npy + chunks.jsonl)

``save()`` writes a brand-new generation directory and only then repoints
``CURRENT`` at it with an atomic rename. Readers therefore never observe a
half-written index, and other processes (extra API workers, the CLI) detect a
newer generation by reading ``CURRENT`` and reload it. The newest generations
are kept so a reader that is mid-load when another save commits still finds
its files.

Durability on power loss is not guaranteed (files are not fsync'ed); the index
is derived data that ``scrag ingest`` can rebuild.

The pre-0.2 flat layout (payload files directly in the index directory) is still
loaded when no ``CURRENT`` pointer exists.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import threading
import time
import uuid
from abc import abstractmethod
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, ClassVar, TypeVar

from ..errors import IndexIncompatibleError
from ..logging_config import get_logger
from ..models import Chunk
from .base import VectorStore

logger = get_logger(__name__)

POINTER_FILE = "CURRENT"
GENERATIONS_DIR = "generations"
META_FILE = "meta.json"
FORMAT_VERSION = 2
LEGACY_GENERATION = "legacy"
KEEP_GENERATIONS = 2

_GENERATION_RE = re.compile(r"\d{20}-[0-9a-f]{8}")
_T = TypeVar("_T")


def write_chunks(path: Path, chunks: Sequence[Chunk]) -> None:
    """Write chunks as JSON Lines, one chunk per line, row-aligned with vectors."""
    with path.open("w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")


def read_chunks(path: Path) -> list[Chunk]:
    return [
        Chunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def check_count(directory: Path, meta: dict[str, Any] | None, vectors: int, chunks: int) -> None:
    """Raise if the vector, chunk and recorded counts disagree."""
    expected = meta.get("count") if meta else None
    if vectors != chunks or (expected is not None and expected != chunks):
        raise IndexIncompatibleError(
            f"Corrupt index in {directory}: {vectors} vectors, {chunks} chunks, "
            f"meta count {expected}."
        )


def new_generation_id() -> str:
    """Sortable, unique generation name: zero-padded nanosecond time + random suffix."""
    return f"{time.time_ns():020d}-{uuid.uuid4().hex[:8]}"


def _retry_os(fn: Callable[[], _T], attempts: int = 10, delay: float = 0.02) -> _T:
    """Retry transient ``PermissionError``s.

    On Windows, renaming over (or opening) a file that another process holds
    open fails briefly instead of succeeding as it would on POSIX.
    """
    for attempt in range(attempts):
        try:
            return fn()
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)
    raise AssertionError("unreachable")  # pragma: no cover


def read_pointer(root: Path) -> str | None:
    """Return the live generation name, or ``None`` if no pointer exists."""
    pointer = root / POINTER_FILE
    try:
        name = _retry_os(lambda: pointer.read_text(encoding="utf-8")).strip()
    except FileNotFoundError:
        return None
    if not _GENERATION_RE.fullmatch(name):
        raise IndexIncompatibleError(f"Corrupt index pointer in {pointer}: {name!r}")
    return name


def write_pointer(root: Path, generation: str) -> None:
    """Atomically point ``CURRENT`` at ``generation``."""
    tmp = root / f".{POINTER_FILE}.{uuid.uuid4().hex}.tmp"
    tmp.write_text(generation, encoding="utf-8")
    try:
        _retry_os(lambda: os.replace(tmp, root / POINTER_FILE))
    finally:
        tmp.unlink(missing_ok=True)


def prune_generations(root: Path, current: str, keep: int = KEEP_GENERATIONS) -> None:
    """Delete all but the newest ``keep`` generations (always keeping ``current``)."""
    gens_dir = root / GENERATIONS_DIR
    if not gens_dir.is_dir():
        return
    names = sorted(p.name for p in gens_dir.iterdir() if _GENERATION_RE.fullmatch(p.name))
    survivors = set(names[-keep:]) | {current}
    for name in names:
        if name not in survivors:
            shutil.rmtree(gens_dir / name, ignore_errors=True)


class PersistentVectorStore(VectorStore):
    """Base for file-backed stores: generation-based save/load/refresh."""

    backend: ClassVar[str]
    legacy_files: ClassVar[tuple[str, ...]]

    def __init__(self, dim: int, path: Path | str) -> None:
        self.dim = dim
        self.path = Path(path)
        # Serializes writers (add/replace/clear/load/save). Searches never take
        # it: they read one immutable state snapshot instead.
        self._lock = threading.RLock()
        self._generation: str | None = None

    @property
    def generation(self) -> str | None:
        """Name of the persisted generation currently loaded (or last saved)."""
        return self._generation

    # --- backend hooks ---------------------------------------------------------
    @abstractmethod
    def _write_payload(self, directory: Path) -> int:
        """Write the current state into ``directory``; return the chunk count."""

    @abstractmethod
    def _read_payload(self, directory: Path, meta: dict[str, Any] | None) -> None:
        """Load and validate files in ``directory``, then publish them as the state."""

    # --- persistence -----------------------------------------------------------
    def save(self) -> None:
        with self._lock:
            generation = new_generation_id()
            directory = self.path / GENERATIONS_DIR / generation
            directory.mkdir(parents=True)
            count = self._write_payload(directory)
            meta = {
                "format_version": FORMAT_VERSION,
                "backend": self.backend,
                "dim": self.dim,
                "count": count,
                "generation": generation,
            }
            (directory / META_FILE).write_text(json.dumps(meta), encoding="utf-8")
            write_pointer(self.path, generation)
            self._generation = generation
            prune_generations(self.path, current=generation)
        logger.info("Saved %s index generation %s (%d chunks)", self.backend, generation, count)

    def load(self) -> bool:
        with self._lock:
            generation = read_pointer(self.path)
            if generation is None:
                return self._load_legacy()
            self._load_generation(generation)
            return True

    def refresh_if_stale(self) -> bool:
        with self._lock:
            generation = read_pointer(self.path)
            if generation is None or generation == self._generation:
                return False
            logger.info("Index generation changed on disk (%s); reloading", generation)
            self._load_generation(generation)
            return True

    def _load_generation(self, generation: str) -> None:
        directory = self.path / GENERATIONS_DIR / generation
        try:
            meta = json.loads((directory / META_FILE).read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise IndexIncompatibleError(
                f"Index generation {generation} referenced by {POINTER_FILE} is missing."
            ) from exc
        if meta.get("backend") != self.backend:
            raise IndexIncompatibleError(
                f"Index in {self.path} was built by the {meta.get('backend')!r} backend, "
                f"not {self.backend!r}. Re-ingest or switch SCRAG_VECTOR_BACKEND."
            )
        if meta.get("dim") != self.dim:
            raise IndexIncompatibleError(
                f"Persisted index dim {meta.get('dim')} != configured dim {self.dim}. "
                "Re-ingest after changing the embedding model."
            )
        self._read_payload(directory, meta)
        self._generation = generation
        logger.info(
            "Loaded %s index generation %s (%d chunks)", self.backend, generation, len(self)
        )

    def _load_legacy(self) -> bool:
        if not all((self.path / name).exists() for name in self.legacy_files):
            return False
        self._read_payload(self.path, None)
        self._generation = LEGACY_GENERATION
        logger.info("Loaded legacy flat-layout index (%d chunks) from %s", len(self), self.path)
        return True
