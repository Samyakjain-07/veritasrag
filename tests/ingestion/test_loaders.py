import pytest
from fpdf import FPDF
from src.ingestion.loaders import (
    chunk_documents,
    load_and_chunk,
    load_document,
    load_html,
    load_pdf,
    load_text,
)


def test_load_text(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("hello from a plain text file", encoding="utf-8")

    docs = load_text(path)

    assert len(docs) == 1
    assert "hello from a plain text file" in docs[0].page_content


def test_load_html(tmp_path):
    path = tmp_path / "page.html"
    path.write_text(
        "<html><body><h1>Title</h1><p>hello from html</p><script>ignored()</script></body></html>",
        encoding="utf-8",
    )

    docs = load_html(path)

    assert len(docs) == 1
    assert "hello from html" in docs[0].page_content
    assert "ignored()" not in docs[0].page_content


def test_load_pdf(tmp_path):
    path = tmp_path / "doc.pdf"
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    pdf.cell(0, 10, text="hello from a pdf")
    pdf.output(str(path))

    docs = load_pdf(path)

    assert len(docs) >= 1
    assert "hello from a pdf" in docs[0].page_content


def test_load_document_dispatches_by_extension(tmp_path):
    path = tmp_path / "note.md"
    path.write_text("markdown content", encoding="utf-8")

    docs = load_document(path)

    assert "markdown content" in docs[0].page_content


def test_load_document_unsupported_extension(tmp_path):
    path = tmp_path / "note.xyz"
    path.write_text("nope", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file extension"):
        load_document(path)


def test_chunk_documents_splits_long_text():
    from langchain_core.documents import Document

    long_doc = Document(page_content="word " * 400, metadata={"source": "synthetic"})

    chunks = chunk_documents([long_doc], chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(chunk.page_content) <= 100 for chunk in chunks)
    assert all(chunk.metadata["source"] == "synthetic" for chunk in chunks)


def test_load_and_chunk(tmp_path):
    path = tmp_path / "note.txt"
    path.write_text("word " * 400, encoding="utf-8")

    chunks = load_and_chunk(path, chunk_size=100, chunk_overlap=20)

    assert len(chunks) > 1
