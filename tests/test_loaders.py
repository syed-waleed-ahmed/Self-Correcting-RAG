"""Document format loaders and ingestion limits (Phase 2)."""

from __future__ import annotations

import json

import pytest

from scrag.ingest import load_documents
from scrag.loaders import LoaderError, extract_text

LIMIT = 1_000_000


def _minimal_pdf(text: str) -> bytes:
    """Build a valid one-page PDF whose page draws ``text`` (computed xref offsets)."""
    stream = f"BT /F1 12 Tf 72 720 Td ({text}) Tj ET".encode()
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
        + b"/Resources << /Font << /F1 5 0 R >> >> /Contents 4 0 R >>",
        b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (
        len(objects) + 1,
        xref,
    )
    return bytes(out)


def test_html_drops_scripts_and_keeps_text(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><head><style>p{}</style><script>alert('x')</script></head>"
        "<body><h1>Guardrails</h1><p>Drop off-topic &amp; harmful text.</p></body></html>",
        encoding="utf-8",
    )
    text = extract_text(path, max_bytes=LIMIT)
    assert "Guardrails" in text
    assert "Drop off-topic & harmful text." in text
    assert "alert" not in text and "p{}" not in text


def test_csv_rows_become_labelled_lines(tmp_path):
    path = tmp_path / "models.csv"
    path.write_text("model,dim\nall-MiniLM-L6-v2,384\n", encoding="utf-8")
    assert extract_text(path, max_bytes=LIMIT) == "model: all-MiniLM-L6-v2; dim: 384"


def test_json_is_flattened_to_paths(tmp_path):
    path = tmp_path / "cfg.json"
    path.write_text(json.dumps({"retrieval": {"mode": "hybrid", "k": [3, 5]}}), encoding="utf-8")
    text = extract_text(path, max_bytes=LIMIT)
    assert "retrieval.mode: hybrid" in text
    assert "retrieval.k[1]: 5" in text


def test_invalid_json_is_a_loader_error(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(LoaderError):
        extract_text(path, max_bytes=LIMIT)


def test_pdf_text_extraction(tmp_path):
    pytest.importorskip("pypdf")
    path = tmp_path / "doc.pdf"
    path.write_bytes(_minimal_pdf("Hybrid retrieval fuses rankings"))
    assert "Hybrid retrieval fuses rankings" in extract_text(path, max_bytes=LIMIT)


def test_pdf_page_limit(tmp_path):
    pytest.importorskip("pypdf")
    path = tmp_path / "doc.pdf"
    path.write_bytes(_minimal_pdf("one page"))
    with pytest.raises(LoaderError, match="page limit"):
        extract_text(path, max_bytes=LIMIT, max_pdf_pages=0)


def test_docx_paragraphs_and_tables(tmp_path):
    docx = pytest.importorskip("docx")
    document = docx.Document()
    document.add_paragraph("The evaluator scores grounding.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "threshold"
    table.rows[0].cells[1].text = "0.7"
    path = tmp_path / "doc.docx"
    document.save(str(path))
    text = extract_text(path, max_bytes=LIMIT)
    assert "The evaluator scores grounding." in text
    assert "threshold | 0.7" in text


def test_oversized_and_unsupported_files_are_rejected(tmp_path):
    big = tmp_path / "big.txt"
    big.write_text("x" * 100, encoding="utf-8")
    with pytest.raises(LoaderError, match="limit"):
        extract_text(big, max_bytes=10)
    with pytest.raises(LoaderError, match="unsupported"):
        extract_text(tmp_path / "image.png", max_bytes=LIMIT)


def test_load_documents_mixes_formats_and_skips_bad_files(tmp_path):
    (tmp_path / "a.md").write_text("# Title\nbody", encoding="utf-8")
    (tmp_path / "b.html").write_text("<p>html body</p>", encoding="utf-8")
    (tmp_path / "c.json").write_text("{broken", encoding="utf-8")
    (tmp_path / "d.png").write_bytes(b"\x89PNG")  # not a supported suffix: ignored
    (tmp_path / "e.txt").write_text("y" * 50, encoding="utf-8")
    docs = dict(load_documents(tmp_path, max_bytes=40))
    assert set(docs) == {"a.md", "b.html"}  # c.json invalid, e.txt too big


def test_line_endings_do_not_change_document_hashes(tmp_path):
    """A Windows (CRLF) checkout must fingerprint the corpus like a Linux build."""
    from scrag.ingest import content_hash

    (tmp_path / "lf").mkdir()
    (tmp_path / "crlf").mkdir()
    (tmp_path / "lf" / "a.txt").write_bytes(b"line one\nline two\n")
    (tmp_path / "crlf" / "a.txt").write_bytes(b"line one\r\nline two\r\n")
    lf = dict(load_documents(tmp_path / "lf"))["a.txt"]
    crlf = dict(load_documents(tmp_path / "crlf"))["a.txt"]
    assert lf == crlf
    assert content_hash(lf) == content_hash(crlf)
