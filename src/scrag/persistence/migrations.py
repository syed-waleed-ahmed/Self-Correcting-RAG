"""Versioned SQL migrations with downgrades.

Migrations are plain SQL files shipped in the package
(``sql/NNNN_name.up.sql`` and ``.down.sql``). They are applied in one
transaction under a transaction-scoped advisory lock, so several API processes
starting at once never race. Transaction-scoped locks also work through
transaction-mode poolers, where session-level advisory locks do not.

Plain SQL instead of Alembic (ADR 0007): the code base has no ORM, Alembic would
add SQLAlchemy to a 512 MB container, and these migrations need nothing beyond
ordered DDL with a downgrade.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from importlib import resources
from typing import Any

SCHEMA = "scrag"
_LOCK_KEY = 0x5C5A_0004  # arbitrary constant shared by every SCRAG process
_FILE_RE = re.compile(r"^(\d{4})_([a-z0-9_]+)\.(up|down)\.sql$")


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    up: str
    down: str


def load_migrations() -> list[Migration]:
    """All packaged migrations, ordered by version (every one needs up and down)."""
    found: dict[int, dict[str, str]] = {}
    root = resources.files(__package__ or "scrag.persistence").joinpath("sql")
    for entry in root.iterdir():
        match = _FILE_RE.match(entry.name)
        if not match:
            continue
        version, name, direction = int(match[1]), match[2], match[3]
        slot = found.setdefault(version, {"name": name})
        slot[direction] = entry.read_text(encoding="utf-8")
    migrations = []
    for version in sorted(found):
        slot = found[version]
        if "up" not in slot or "down" not in slot:
            raise RuntimeError(f"migration {version:04d} needs both .up.sql and .down.sql")
        migrations.append(Migration(version, slot["name"], slot["up"], slot["down"]))
    return migrations


def _prepare(conn: Any) -> set[int]:
    conn.execute("SELECT pg_advisory_xact_lock(%s)", (_LOCK_KEY,))
    conn.execute(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA}")
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS {SCHEMA}.schema_migrations ("
        " version integer PRIMARY KEY,"
        " name text NOT NULL,"
        " applied_at timestamptz NOT NULL DEFAULT now())"
    )
    rows = conn.execute(f"SELECT version FROM {SCHEMA}.schema_migrations").fetchall()
    return {_first(row) for row in rows}


def _first(row: Any) -> Any:
    return next(iter(row.values())) if isinstance(row, dict) else row[0]


def current_version(conn: Any) -> int:
    with conn.transaction():
        applied = _prepare(conn)
    return max(applied, default=0)


def apply_migrations(conn: Any, target: int | None = None) -> list[int]:
    """Apply pending migrations up to ``target`` (default: latest). Returns the versions."""
    done: list[int] = []
    with conn.transaction():
        applied = _prepare(conn)
        for migration in load_migrations():
            if migration.version in applied:
                continue
            if target is not None and migration.version > target:
                break
            conn.execute(migration.up)
            conn.execute(
                f"INSERT INTO {SCHEMA}.schema_migrations (version, name) VALUES (%s, %s)",
                (migration.version, migration.name),
            )
            done.append(migration.version)
    return done


def rollback_migrations(conn: Any, target: int = 0) -> list[int]:
    """Undo applied migrations newer than ``target``, newest first. Returns the versions."""
    done: list[int] = []
    with conn.transaction():
        applied = _prepare(conn)
        for migration in reversed(load_migrations()):
            if migration.version <= target or migration.version not in applied:
                continue
            conn.execute(migration.down)
            conn.execute(
                f"DELETE FROM {SCHEMA}.schema_migrations WHERE version = %s", (migration.version,)
            )
            done.append(migration.version)
    return done
