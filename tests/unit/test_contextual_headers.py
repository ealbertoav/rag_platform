"""T-120 — contextual chunk headers tests."""

from __future__ import annotations

from src.core.constants import CHUNK_RAW_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking import get_chunker
from src.rag.chunking.contextual_headers import (
    ContextualHeadersChunker,
    build_header_line,
    chunk_context_text,
    prepend_headers,
)
from src.rag.chunking.recursive_chunker import RecursiveChunker


def _doc(
    content: str = "Revenue grew 12% year over year.",
    *,
    source: str = "/data/raw/annual_report_2023.pdf",
    metadata: dict | None = None,
) -> Document:
    base_meta = {"filename": "annual_report_2023.pdf", "section": "Revenue", "page": 42}
    if metadata:
        base_meta.update(metadata)
    return Document(source=source, content=content, metadata=base_meta)


def _chunk(text: str = "Revenue grew 12% year over year.", metadata: dict | None = None) -> Chunk:
    meta = {"filename": "annual_report_2023.pdf", "section": "Revenue", "page": 42}
    if metadata:
        meta.update(metadata)
    return Chunk(document_id="doc-1", text=text, metadata=meta)


class TestBuildHeaderLine:
    def test_uses_loader_metadata(self):
        doc = _doc()
        chunk = _chunk()
        header = build_header_line(doc, chunk)
        assert "annual_report_2023.pdf" in header
        assert "Revenue" in header
        assert "42" in header

    def test_falls_back_to_source_basename(self):
        doc = Document(source="/tmp/report.md", content="text", metadata={})
        chunk = Chunk(document_id=doc.id, text="text", metadata={})
        header = build_header_line(doc, chunk)
        assert "report.md" in header


class TestPrependHeaders:
    def test_prefixes_chunk_text(self):
        doc = _doc()
        chunk = _chunk()
        result = prepend_headers(doc, chunk)
        assert result.startswith("[Document:")
        assert result.endswith(chunk.text)

    def test_example_format(self):
        doc = _doc()
        chunk = _chunk()
        result = prepend_headers(doc, chunk)
        assert "[Document: annual_report_2023.pdf | Section: Revenue | Page: 42]" in result


class TestContextualHeadersChunker:
    def test_embedded_text_includes_header(self):
        doc = _doc("Body text here.")
        inner = RecursiveChunker(chunk_size=500)
        chunker = ContextualHeadersChunker(inner)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text.startswith("[Document:")
        assert "Body text here." in chunks[0].text

    def test_raw_text_preserved_in_metadata(self):
        doc = _doc("Body text here.")
        chunker = ContextualHeadersChunker(RecursiveChunker(chunk_size=500))
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata[CHUNK_RAW_TEXT_KEY] == "Body text here."

    def test_disabled_via_factory_leaves_text_unchanged(self):
        doc = _doc("Plain chunk.")
        chunker = get_chunker("recursive", use_contextual_headers=False, chunk_size=500)
        chunks = chunker.chunk(doc)
        assert chunks[0].text == "Plain chunk."
        assert CHUNK_RAW_TEXT_KEY not in chunks[0].metadata

    def test_enabled_via_factory_applies_headers(self):
        doc = _doc("Plain chunk.")
        chunker = get_chunker("recursive", use_contextual_headers=True, chunk_size=500)
        chunks = chunker.chunk(doc)
        assert chunks[0].text.startswith("[Document:")
        assert chunks[0].metadata[CHUNK_RAW_TEXT_KEY] == "Plain chunk."


class TestChunkContextText:
    def test_prefers_raw_text_when_present(self):
        chunk = Chunk(
            document_id="d1",
            text="[Document: x]\nActual content.",
            metadata={CHUNK_RAW_TEXT_KEY: "Actual content."},
        )
        assert chunk_context_text(chunk) == "Actual content."

    def test_falls_back_to_chunk_text(self):
        chunk = Chunk(document_id="d1", text="No header applied.")
        assert chunk_context_text(chunk) == "No header applied."

    def test_exclude_false_returns_prefixed_text(self):
        chunk = Chunk(
            document_id="d1",
            text="[Document: x]\nActual content.",
            metadata={CHUNK_RAW_TEXT_KEY: "Actual content."},
        )
        assert (
            chunk_context_text(chunk, exclude_from_llm_context=False)
            == "[Document: x]\nActual content."
        )

    def test_exclude_true_strips_header(self):
        chunk = Chunk(
            document_id="d1",
            text="[Document: x]\nActual content.",
            metadata={CHUNK_RAW_TEXT_KEY: "Actual content."},
        )
        assert chunk_context_text(chunk, exclude_from_llm_context=True) == "Actual content."
