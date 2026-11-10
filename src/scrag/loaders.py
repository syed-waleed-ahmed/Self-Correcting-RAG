"""Text extraction for supported document formats.

Every loader turns a file into plain text, or raises :class:`LoaderError` with a
reason that ingestion logs before skipping the file (one bad file never aborts
an ingestion). PDF and DOCX need the optional ``docs`` extra
(``pip install "self-correcting-rag[docs]"``); everything else is stdlib.
"""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any


class LoaderError(Exception):
    """A file could not be turned into text (reason is in the message)."""


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LoaderError("not valid UTF-8 text") from exc


class _TextExtractor(HTMLParser):
    _SKIP = {"script", "style", "noscript", "template", "svg"}
    _BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self._SKIP:
            self._skip_depth += 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip_depth:
            self._skip_depth -= 1
        elif tag in self._BLOCK:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._skip_depth:
            self.parts.append(data)


def load_html(path: Path, **_: Any) -> str:
    parser = _TextExtractor()
    parser.feed(_read_text(path))
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def load_csv(path: Path, **_: Any) -> str:
    reader = csv.reader(io.StringIO(_read_text(path)))
    rows = list(reader)
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    if not body:
        return ", ".join(header)
    return "\n".join(
        "; ".join(f"{h}: {v}" for h, v in zip(header, row, strict=False) if v) for row in body
    )


def _flatten_json(value: Any, prefix: str = "", depth: int = 0) -> list[str]:
    if depth > 20:
        return []
    if isinstance(value, dict):
        out: list[str] = []
        for key, item in value.items():
            out += _flatten_json(item, f"{prefix}.{key}" if prefix else str(key), depth + 1)
        return out
    if isinstance(value, list):
        out = []
        for i, item in enumerate(value):
            out += _flatten_json(item, f"{prefix}[{i}]", depth + 1)
        return out
    return [f"{prefix}: {value}" if prefix else str(value)]


def load_json(path: Path, **_: Any) -> str:
    try:
        data = json.loads(_read_text(path))
    except json.JSONDecodeError as exc:
        raise LoaderError(f"invalid JSON ({exc.msg})") from exc
    return "\n".join(_flatten_json(data))


def load_pdf(path: Path, *, max_pdf_pages: int = 300, **_: Any) -> str:
    try:
        from pypdf import PdfReader
        from pypdf.errors import PdfReadError
    except ImportError as exc:
        raise LoaderError('PDF support needs: pip install "self-correcting-rag[docs]"') from exc
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            raise LoaderError("encrypted PDF")
        if len(reader.pages) > max_pdf_pages:
            raise LoaderError(f"{len(reader.pages)} pages exceeds the {max_pdf_pages}-page limit")
        return "\n\n".join((page.extract_text() or "").strip() for page in reader.pages)
    except PdfReadError as exc:
        raise LoaderError(f"unreadable PDF ({exc})") from exc


def load_docx(path: Path, **_: Any) -> str:
    try:
        import docx
    except ImportError as exc:
        raise LoaderError('DOCX support needs: pip install "self-correcting-rag[docs]"') from exc
    try:
        document = docx.Document(str(path))
    except Exception as exc:  # noqa: BLE001 - python-docx raises assorted errors
        raise LoaderError(f"unreadable DOCX ({type(exc).__name__})") from exc
    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))
    return "\n\n".join(parts)


def _load_plain(path: Path, **_: Any) -> str:
    return _read_text(path)


LOADERS: dict[str, Callable[..., str]] = {
    ".txt": _load_plain,
    ".md": _load_plain,
    ".markdown": _load_plain,
    ".html": load_html,
    ".htm": load_html,
    ".csv": load_csv,
    ".json": load_json,
    ".pdf": load_pdf,
    ".docx": load_docx,
}

SUPPORTED_SUFFIXES = frozenset(LOADERS)


def extract_text(path: Path, *, max_bytes: int, max_pdf_pages: int = 300) -> str:
    """Extract text from ``path`` or raise :class:`LoaderError`."""
    loader = LOADERS.get(path.suffix.lower())
    if loader is None:
        raise LoaderError(f"unsupported file type {path.suffix!r}")
    size = path.stat().st_size
    if size > max_bytes:
        raise LoaderError(f"{size} bytes exceeds the {max_bytes}-byte limit")
    text = loader(path, max_pdf_pages=max_pdf_pages)
    # Normalize line endings so a document hashes (and chunks) identically on
    # Windows checkouts (CRLF) and Linux builds (LF).
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()
