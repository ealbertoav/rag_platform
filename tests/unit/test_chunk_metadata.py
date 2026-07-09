"""Unit tests for chunk metadata filtering."""

from __future__ import annotations

from src.core.constants import CHUNK_SECTION_KEY, LAYOUT_DOCUMENT_METADATA_KEYS
from src.domain.entities.document import Document
from src.rag.chunking.metadata import chunk_metadata
from src.rag.chunking.recursive_chunker import RecursiveChunker


class TestChunkMetadata:
    def test_layout_keys_constant(self) -> None:
        assert frozenset({"tables", "figures", "sections"}) == LAYOUT_DOCUMENT_METADATA_KEYS

    def test_chunk_metadata_excludes_layout_keys(self) -> None:
        metadata = {
            "loader": "docling",
            "filename": "report.pdf",
            CHUNK_SECTION_KEY: "Intro",
            "tables": [{"table_id": "table-1"}],
            "figures": [{"figure_id": "figure-1"}],
            "sections": ["Intro"],
        }
        filtered = chunk_metadata(metadata)
        assert filtered == {
            "loader": "docling",
            "filename": "report.pdf",
            CHUNK_SECTION_KEY: "Intro",
        }

    def test_recursive_chunker_uses_filtered_metadata(self) -> None:
        document = Document(
            source="/tmp/report.pdf",
            content="Paragraph one.\n\nParagraph two.",
            metadata={
                "loader": "docling",
                "tables": [{"table_id": "table-1"}],
                "figures": [{"figure_id": "figure-1"}],
                "sections": ["Intro"],
            },
        )
        chunks = RecursiveChunker(chunk_size=100, overlap=10).chunk(document)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert "tables" not in chunk.metadata
            assert "figures" not in chunk.metadata
            assert "sections" not in chunk.metadata
