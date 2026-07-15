"""T-241 — page-boundary chunker tests."""

from __future__ import annotations

import pytest

from src.core.constants import CHUNK_INDEX_KEY, CHUNK_PAGE_KEY, CHUNK_SOURCE_KEY
from src.domain.entities.document import Document
from src.rag.chunking import get_chunker
from src.rag.chunking.page_chunker import PageAwareChunker

_PARA = "word " * 120  # ~120 tokens


def _doc(
    content: str,
    *,
    source: str = "test.pdf",
    metadata: dict[str, object] | None = None,
) -> Document:
    return Document(source=source, content=content, metadata=metadata or {})


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class TestPageAwareChunker:
    def test_no_pages_metadata_behaves_like_recursive(self):
        content = "Short plain document."
        chunks = PageAwareChunker().chunk(_doc(content, source="test.md"))
        assert len(chunks) == 1
        assert chunks[0].text == content
        assert CHUNK_PAGE_KEY not in chunks[0].metadata

    def test_single_page_gets_page_one(self):
        content = "Only page text."
        chunks = PageAwareChunker().chunk(_doc(content, metadata={"pages": [content]}))
        assert len(chunks) == 1
        assert chunks[0].metadata[CHUNK_PAGE_KEY] == 1
        assert chunks[0].text == content

    def test_multi_page_document_maps_chunks_to_pages(self):
        pages = ["First page body.", "Second page body.", "Third page body."]
        doc = _doc("\n\n".join(pages), metadata={"pages": pages})
        chunks = PageAwareChunker().chunk(doc)
        assert len(chunks) == 3
        assert [c.metadata[CHUNK_PAGE_KEY] for c in chunks] == [1, 2, 3]
        assert chunks[0].text == pages[0]
        assert chunks[1].text == pages[1]
        assert chunks[2].text == pages[2]
        for chunk in chunks:
            other_pages = [p for p in pages if p != chunk.text]
            assert not any(other in chunk.text for other in other_pages)

    def test_oversized_page_is_recursively_split_and_shares_page_number(self):
        huge_page = (_PARA + "\n\n") * 8
        doc = _doc(huge_page, metadata={"pages": ["Short page one.", huge_page]})
        chunks = PageAwareChunker(chunk_size=200, overlap=20).chunk(doc)
        page_two_chunks = [c for c in chunks if c.metadata[CHUNK_PAGE_KEY] == 2]
        assert len(page_two_chunks) > 1
        assert all(_approx_tokens(c.text) <= 200 for c in page_two_chunks)

    def test_chunk_index_is_globally_sequential_across_pages(self):
        huge_page = (_PARA + "\n\n") * 8
        pages = [huge_page, "Short second page."]
        doc = _doc(huge_page, metadata={"pages": pages})
        chunks = PageAwareChunker(chunk_size=200, overlap=20).chunk(doc)
        assert len(chunks) > 2
        assert [c.metadata[CHUNK_INDEX_KEY] for c in chunks] == list(range(len(chunks)))
        assert chunks[-1].metadata[CHUNK_PAGE_KEY] == 2

    def test_document_id_and_source_propagate(self):
        pages = ["Page one.", "Page two."]
        doc = _doc("\n\n".join(pages), source="docs/report.pdf", metadata={"pages": pages})
        chunks = PageAwareChunker().chunk(doc)
        assert all(c.document_id == doc.id for c in chunks)
        assert all(c.metadata[CHUNK_SOURCE_KEY] == "docs/report.pdf" for c in chunks)

    def test_empty_document_returns_empty(self):
        assert PageAwareChunker().chunk(_doc("")) == []

    def test_empty_pages_list_behaves_like_recursive(self):
        content = "Short plain document."
        chunks = PageAwareChunker().chunk(_doc(content, metadata={"pages": []}))
        assert len(chunks) == 1
        assert chunks[0].text == content
        assert CHUNK_PAGE_KEY not in chunks[0].metadata

    def test_overlap_validation_delegates_to_recursive(self):
        with pytest.raises(ValueError, match="overlap"):
            PageAwareChunker(chunk_size=100, overlap=100)


class TestPageChunkerFactory:
    def test_get_chunker_returns_page_aware(self):
        chunker = get_chunker("page", chunk_size=300, overlap=20)
        assert isinstance(chunker, PageAwareChunker)
