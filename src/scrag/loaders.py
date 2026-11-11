"""Text extraction for supported document formats.

Every loader turns a file into plain text, or raises :class:`LoaderError` with a
reason that ingestion logs before skipping the file (one bad file never aborts
an ingestion). PDF and DOCX need the optional ``docs`` extra
(``pip install "self-correcting-rag[docs]"``); everything else is stdlib.

Untrusted input (uploads) gets extra checks before any parser sees it:
:func:`sniff_upload` verifies the bytes really are the format the extension
claims, DOCX containers are inspected for zip bombs before anything is
inflated, and every parser failure (including deep nesting and oversized CSV
fields) becomes a :class:`LoaderError` instead of an unhandled exception.
"""

from __future__ import annotations

import csv
import io
import json
import re
import tempfile
import time
import zipfile
from collections.abc import Callable
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

# A DOCX is a ZIP of XML parts. These bound what one may expand to, checked from
# the ZIP directory before anything is inflated (zipfile never inflates an entry
# past its declared size, so the declared sizes are a real bound).
DOCX_MAX_ENTRIES = 2_000
DOCX_MAX_UNCOMPRESSED_BYTES = 30_000_000
DOCX_MAX_RATIO = 200  # inflated/compressed size, for entries over 1 MB
# Checked between pages, so one pathological page can still run long.
PDF_EXTRACTION_BUDGET_S = 30.0

_PDF_MAGIC = b"%PDF-"
_ZIP_MAGIC = b"PK\x03\x04"


class LoaderError(Exception):
    """A file could not be turned into text (reason is in the message)."""


def _read_text(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise LoaderError("not valid UTF-8 text") from exc
    if "\x00" in text:  # valid UTF-8, but a binary file renamed to a text type
        raise LoaderError("binary content (NUL bytes) in a text file")
    return text


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


def html_to_text(markup: str) -> str:
    """Visible text of an HTML document (scripts, styles and markup dropped)."""
    parser = _TextExtractor()
    parser.feed(markup)
    lines = (" ".join(line.split()) for line in "".join(parser.parts).splitlines())
    return "\n".join(line for line in lines if line)


def load_html(path: Path, **_: Any) -> str:
    return html_to_text(_read_text(path))


def load_csv(path: Path, **_: Any) -> str:
    try:
        rows = list(csv.reader(io.StringIO(_read_text(path))))
    except csv.Error as exc:  # e.g. a field over csv.field_size_limit()
        raise LoaderError(f"invalid CSV ({exc})") from exc
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
    except RecursionError as exc:  # e.g. [[[[...]]]] nested thousands deep
        raise LoaderError("JSON is nested too deeply") from exc
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
        deadline = time.monotonic() + PDF_EXTRACTION_BUDGET_S
        pages = []
        for page in reader.pages:
            if time.monotonic() > deadline:
                raise LoaderError(f"text extraction took over {PDF_EXTRACTION_BUDGET_S:g}s")
            pages.append((page.extract_text() or "").strip())
        return "\n\n".join(pages)
    except PdfReadError as exc:
        raise LoaderError(f"unreadable PDF ({exc})") from exc
    except RecursionError as exc:  # deeply nested PDF objects
        raise LoaderError("PDF structure is nested too deeply") from exc


def check_docx_container(data: bytes) -> None:
    """Refuse a DOCX whose ZIP container is malformed or would inflate too far."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            entries = archive.infolist()
    except (zipfile.BadZipFile, ValueError) as exc:
        raise LoaderError("not a DOCX (corrupt ZIP container)") from exc
    if len(entries) > DOCX_MAX_ENTRIES:
        raise LoaderError(f"DOCX has {len(entries)} parts (limit {DOCX_MAX_ENTRIES})")
    names = {entry.filename for entry in entries}
    if not {"[Content_Types].xml", "word/document.xml"} <= names:
        raise LoaderError("not a DOCX (no word/document.xml)")
    total = sum(entry.file_size for entry in entries)
    if total > DOCX_MAX_UNCOMPRESSED_BYTES:
        raise LoaderError(f"DOCX inflates to {total} bytes (limit {DOCX_MAX_UNCOMPRESSED_BYTES})")
    for entry in entries:
        if entry.file_size > 1_000_000 and entry.file_size > DOCX_MAX_RATIO * max(
            entry.compress_size, 1
        ):
            raise LoaderError("DOCX part has a zip-bomb compression ratio")


def load_docx(path: Path, **_: Any) -> str:
    try:
        import docx
    except ImportError as exc:
        raise LoaderError('DOCX support needs: pip install "self-correcting-rag[docs]"') from exc
    data = path.read_bytes()
    check_docx_container(data)  # before python-docx inflates any part
    try:
        document = docx.Document(io.BytesIO(data))
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


_UNSAFE_NAME_CHARS = re.compile(r"[\x00-\x1f\x7f/\\:*?\"<>|]+")


def safe_filename(name: str | None, *, max_length: int = 120) -> str:
    """A display-safe base name for an uploaded file (no paths or control characters)."""
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1]
    base = _UNSAFE_NAME_CHARS.sub("_", base).strip(" .")
    if not base:
        return "upload.txt"
    stem, dot, suffix = base.rpartition(".")
    if dot and len(suffix) <= 10 and len(base) > max_length:
        return stem[: max_length - len(suffix) - 1] + "." + suffix
    return base[:max_length]


def sniff_upload(data: bytes, suffix: str) -> None:
    """Check that ``data`` really is the format ``suffix`` claims (content sniffing).

    The client's file name and Content-Type are never trusted: a PDF must carry
    the ``%PDF-`` header (allowed within the first 1024 bytes, as readers do), a
    DOCX must be a ZIP container that passes :func:`check_docx_container`, and
    the text formats are checked for binary content when they are decoded.
    """
    if suffix == ".pdf":
        if _PDF_MAGIC not in data[:1024]:
            raise LoaderError("not a PDF (no %PDF- header)")
    elif suffix == ".docx":
        if not data.startswith(_ZIP_MAGIC):
            raise LoaderError("not a DOCX (not a ZIP container)")
        check_docx_container(data)
    elif data.startswith((_PDF_MAGIC, _ZIP_MAGIC)):
        raise LoaderError(f"a PDF or ZIP file named {suffix}")


def extract_upload(data: bytes, filename: str, *, max_bytes: int, max_pdf_pages: int = 300) -> str:
    """Extract text from uploaded bytes with the same loaders as files on disk.

    The bytes are sniffed first (:func:`sniff_upload`), then written to a private
    temporary directory under a fixed name (never the client's filename), so an
    upload can't choose where it is written.
    """
    suffix = Path(filename).suffix.lower()
    if suffix not in LOADERS:
        raise LoaderError(f"unsupported file type {suffix!r}")
    if len(data) > max_bytes:
        raise LoaderError(f"{len(data)} bytes exceeds the {max_bytes}-byte limit")
    sniff_upload(data, suffix)
    with tempfile.TemporaryDirectory(prefix="scrag-upload-") as tmp:
        path = Path(tmp) / f"upload{suffix}"
        path.write_bytes(data)
        return extract_text(path, max_bytes=max_bytes, max_pdf_pages=max_pdf_pages)


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
