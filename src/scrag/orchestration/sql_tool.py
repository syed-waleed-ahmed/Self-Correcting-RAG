"""Read-only SQL over an allow-listed catalog of the knowledge base (``sql_query``).

The agent can answer questions *about* the corpus ("which document is longest?",
"how many chunks mention BM25?") with one SQLite query over two tables built from
the index::

    documents(source TEXT, chunks INTEGER, characters INTEGER)
    chunks(id TEXT, source TEXT, position INTEGER, characters INTEGER, text TEXT)

The SQL is written by the planner, which a visitor steers, so it is treated as
hostile. Defense in depth:

1. The database is a private in-memory copy of public index data: there is no
   other data to reach, and nothing written to it would outlive the process.
2. ``PRAGMA query_only`` plus an authorizer that allows only SELECT, reads of
   the two catalog tables and a short list of pure functions. Everything else is
   refused when the statement is prepared, before it runs: writes, ATTACH,
   PRAGMA, ``sqlite_master``, recursive CTEs, ``load_extension``, and the
   functions that can allocate unbounded memory (``randomblob``, ``zeroblob``,
   ``printf``, ``group_concat``, ``replace``, ...).
3. Exactly one SELECT (or ``WITH ... SELECT``) statement.
4. A wall-clock budget enforced by a progress handler, so a cross join can't
   pin a CPU; row and value-length caps on the result; and SQLite's own length
   limits on Python 3.11+.
"""

from __future__ import annotations

import re
import sqlite3
import threading
import time
from collections.abc import Callable
from typing import Any

from pydantic import BaseModel, Field

from ..models import Chunk
from .tools import Tool

SCHEMA: dict[str, str] = {
    "documents": "source TEXT PRIMARY KEY, chunks INTEGER NOT NULL, characters INTEGER NOT NULL",
    "chunks": (
        "id TEXT PRIMARY KEY, source TEXT NOT NULL, position INTEGER NOT NULL,"
        " characters INTEGER NOT NULL, text TEXT NOT NULL"
    ),
}
SCHEMA_SUMMARY = (
    "documents(source, chunks, characters) and chunks(id, source, position, characters, text)"
)

# Pure functions with output bounded by their input. Anything not listed is denied.
ALLOWED_FUNCTIONS = frozenset(
    {
        "abs",
        "avg",
        "coalesce",
        "count",
        "glob",
        "ifnull",
        "iif",
        "instr",
        "length",
        "like",
        "lower",
        "ltrim",
        "max",
        "min",
        "nullif",
        "round",
        "rtrim",
        "substr",
        "substring",
        "sum",
        "total",
        "trim",
        "typeof",
        "upper",
    }
)

_STATEMENT_START = re.compile(r"\s*(select|with)\b", re.IGNORECASE)
# (limit id, value) pairs applied where Python supports Connection.setlimit (3.11+).
_LIMITS = (
    ("SQLITE_LIMIT_LENGTH", 1_000_000),
    ("SQLITE_LIMIT_SQL_LENGTH", 4_000),
    ("SQLITE_LIMIT_COMPOUND_SELECT", 4),
    ("SQLITE_LIMIT_EXPR_DEPTH", 100),
    ("SQLITE_LIMIT_ATTACHED", 0),
)


class SqlRejected(ValueError):
    """The query was refused or failed; the message is safe to show."""


def _authorize(action: int, arg1: str | None, arg2: str | None, _db: Any, _src: Any) -> int:
    if action == sqlite3.SQLITE_SELECT:
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        return sqlite3.SQLITE_OK if arg1 in SCHEMA else sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        return (
            sqlite3.SQLITE_OK if (arg2 or "").lower() in ALLOWED_FUNCTIONS else sqlite3.SQLITE_DENY
        )
    return sqlite3.SQLITE_DENY


def build_catalog_db(chunks: list[Chunk]) -> sqlite3.Connection:
    """A locked-down, in-memory catalog of ``chunks`` (read-only once returned)."""
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    for table, columns in SCHEMA.items():
        conn.execute(f"CREATE TABLE {table} ({columns})")
    per_document: dict[str, tuple[int, int]] = {}
    rows = []
    for chunk in chunks:
        rows.append((chunk.id, chunk.source, chunk.position, len(chunk.text), chunk.text))
        count, characters = per_document.get(chunk.source, (0, 0))
        per_document[chunk.source] = (count + 1, characters + len(chunk.text))
    conn.executemany("INSERT OR IGNORE INTO chunks VALUES (?, ?, ?, ?, ?)", rows)
    conn.executemany(
        "INSERT INTO documents VALUES (?, ?, ?)",
        [(source, n, chars) for source, (n, chars) in sorted(per_document.items())],
    )
    conn.commit()
    if hasattr(conn, "setlimit"):
        for name, value in _LIMITS:
            conn.setlimit(getattr(sqlite3, name), value)
    conn.execute("PRAGMA query_only = ON")
    conn.set_authorizer(_authorize)  # last: from here on only SELECTs are prepared
    return conn


class CorpusCatalog:
    """Runs read-only queries against a catalog rebuilt whenever the index changes.

    ``snapshot`` returns ``(revision, chunks)``; a revision of ``-1`` means
    "unknown", and the catalog is then rebuilt for every query.
    """

    def __init__(
        self,
        snapshot: Callable[[], tuple[int, list[Chunk]]],
        *,
        max_rows: int = 50,
        max_value_chars: int = 300,
        time_budget_s: float = 1.0,
    ) -> None:
        self._snapshot = snapshot
        self.max_rows = max_rows
        self.max_value_chars = max_value_chars
        self.time_budget_s = time_budget_s
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._revision: int | None = None

    def _connection(self) -> sqlite3.Connection:
        revision, chunks = self._snapshot()
        if self._conn is None or revision == -1 or revision != self._revision:
            if self._conn is not None:
                self._conn.close()
            self._conn = build_catalog_db(chunks)
            self._revision = revision
        return self._conn

    def query(self, sql: str) -> dict[str, Any]:
        """Run one SELECT. Returns ``columns``, ``rows``, ``row_count``, ``truncated``."""
        statement = sql.strip().rstrip(";").strip()
        if not _STATEMENT_START.match(statement):
            raise SqlRejected("only a single SELECT (or WITH ... SELECT) statement is allowed")
        with self._lock:  # one query at a time on the shared connection
            conn = self._connection()
            deadline = time.monotonic() + self.time_budget_s
            conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 1_000)
            try:
                cursor = conn.execute(statement)
                rows = cursor.fetchmany(self.max_rows + 1)
                columns = [column[0] for column in cursor.description or ()]
            except (sqlite3.Error, sqlite3.Warning) as exc:
                if "interrupted" in str(exc):
                    raise SqlRejected(f"query exceeded {self.time_budget_s:g}s") from exc
                raise SqlRejected(str(exc)) from exc
            finally:
                conn.set_progress_handler(None, 0)
        return {
            "columns": columns,
            "rows": [[self._cell(value) for value in row] for row in rows[: self.max_rows]],
            "row_count": min(len(rows), self.max_rows),
            "truncated": len(rows) > self.max_rows,
        }

    def _cell(self, value: Any) -> Any:
        if isinstance(value, bytes):
            return f"<{len(value)} bytes>"
        if isinstance(value, str) and len(value) > self.max_value_chars:
            return value[: self.max_value_chars] + "..."
        return value


class SqlQueryInput(BaseModel):
    sql: str = Field(
        min_length=1,
        max_length=1000,
        description=f"One read-only SQLite SELECT over {SCHEMA_SUMMARY}.",
    )


def sql_query_tool(catalog: CorpusCatalog, *, enabled: bool = True) -> Tool:
    def run(args: BaseModel) -> dict[str, Any]:
        assert isinstance(args, SqlQueryInput)
        try:
            return {"sql": args.sql, **catalog.query(args.sql)}
        except SqlRejected as exc:
            return {"sql": args.sql, "error": str(exc)}

    return Tool(
        name="sql_query",
        description=(
            "Read-only SQL (one SQLite SELECT) over a catalog of the sample knowledge "
            f"base: {SCHEMA_SUMMARY}. Use it to count, list, rank or compare "
            "documents and chunks, not to answer questions about their content."
        ),
        input_model=SqlQueryInput,
        run=run,
        timeout_s=5.0,
        enabled=enabled,
    )
