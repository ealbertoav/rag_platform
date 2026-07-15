"""Page-boundary chunker — split on page boundaries and label "metadata.page" (T-241)."""

from __future__ import annotations

from typing import Any

from src.core.constants import (
    CHUNK_INDEX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SOURCE_KEY,
    LAYOUT_DOCUMENT_METADATA_KEYS,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.recursive_chunker import RecursiveChunker


def _page_document_metadata(document_metadata: dict[str, Any]) -> dict[str, Any]:
    """Metadata for a single page's segment — drop document-level outline lists."""
    return {
        key: value
        for key, value in document_metadata.items()
        if key not in LAYOUT_DOCUMENT_METADATA_KEYS
    }


class PageAwareChunker:
    """Split documents on page boundaries, then recursively size each page.

    Page boundaries come from "document.metadata['pages']" (a list of one
    text string per page, set by PdfLoader). Each resulting chunk gets
    "metadata.page" set to its 1-indexed page number; sources without a
    "pages" list (DOCX, HTML, Markdown, PPTX) chunk as a single segment with
    "metadata.page" omitted.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size: int = chunk_size
        self.overlap: int = overlap
        self._splitter: RecursiveChunker = RecursiveChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        pages = document.metadata.get("pages")
        if not pages:
            return self._splitter.chunk(document)

        chunks: list[Chunk] = []
        for page_number, page_text in enumerate(pages, start=1):
            page_doc = document.model_copy(
                update={
                    "content": page_text,
                    "metadata": _page_document_metadata(document.metadata),
                }
            )
            for sub in self._splitter.chunk(page_doc):
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        text=sub.text,
                        metadata={
                            **sub.metadata,
                            CHUNK_SOURCE_KEY: document.source,
                            CHUNK_INDEX_KEY: len(chunks),
                            CHUNK_PAGE_KEY: page_number,
                        },
                    )
                )
        return chunks
