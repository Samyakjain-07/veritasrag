"""Document loaders for text, PDF, and HTML sources, plus a shared chunking helper.

Wraps LangChain's document loaders so the rest of the ingestion pipeline works
against a single, format-agnostic list of `Document`s regardless of source
file type.
"""

from __future__ import annotations

from pathlib import Path

from langchain_community.document_loaders import BSHTMLLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.observability.otel_setup import get_tracer

DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 50

_TEXT_SUFFIXES = {".txt", ".md"}
_PDF_SUFFIXES = {".pdf"}
_HTML_SUFFIXES = {".html", ".htm"}


def load_text(path: str | Path) -> list[Document]:
    """Load a plain-text or Markdown file into LangChain `Document`s."""
    return TextLoader(str(path), encoding="utf-8").load()


def load_pdf(path: str | Path) -> list[Document]:
    """Load a PDF file into one `Document` per page."""
    return PyPDFLoader(str(path)).load()


def load_html(path: str | Path) -> list[Document]:
    """Load an HTML file, stripping markup down to visible text."""
    return BSHTMLLoader(str(path)).load()


_LOADERS_BY_SUFFIX = {
    **dict.fromkeys(_TEXT_SUFFIXES, load_text),
    **dict.fromkeys(_PDF_SUFFIXES, load_pdf),
    **dict.fromkeys(_HTML_SUFFIXES, load_html),
}


def load_document(path: str | Path) -> list[Document]:
    """Load a document, dispatching to the right loader by file extension.

    Supports `.txt`/`.md` (text), `.pdf` (PDF), and `.html`/`.htm` (HTML).
    Raises `ValueError` for any other extension.
    """
    with get_tracer(__name__).start_as_current_span("veritasrag.ingestion.load_document") as span:
        path = Path(path)
        span.set_attribute("veritasrag.ingestion.source_path", str(path))
        loader = _LOADERS_BY_SUFFIX.get(path.suffix.lower())
        if loader is None:
            supported = ", ".join(sorted(_LOADERS_BY_SUFFIX))
            raise ValueError(
                f"Unsupported file extension '{path.suffix}' for {path}. Supported: {supported}"
            )
        documents = loader(path)
        span.set_attribute("veritasrag.ingestion.document_count", len(documents))
        return documents


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks using a recursive character splitter."""
    with get_tracer(__name__).start_as_current_span("veritasrag.ingestion.chunk_documents") as span:
        span.set_attribute("veritasrag.ingestion.input_document_count", len(documents))
        span.set_attribute("veritasrag.ingestion.chunk_size", chunk_size)
        span.set_attribute("veritasrag.ingestion.chunk_overlap", chunk_overlap)
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_documents(documents)
        span.set_attribute("veritasrag.ingestion.chunk_count", len(chunks))
        return chunks


def load_and_chunk(
    path: str | Path,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Load a document by path and immediately chunk it."""
    return chunk_documents(load_document(path), chunk_size=chunk_size, chunk_overlap=chunk_overlap)
