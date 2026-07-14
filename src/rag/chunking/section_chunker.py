"""Section-boundary chunker — split on headings and label "metadata.section" (T-240)."""

from __future__ import annotations

from typing import Any

from src.core.constants import (
    CHUNK_INDEX_KEY,
    CHUNK_SECTION_KEY,
    CHUNK_SOURCE_KEY,
    LAYOUT_DOCUMENT_METADATA_KEYS,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.headings import iter_section_segments
from src.rag.chunking.recursive_chunker import RecursiveChunker


def _segment_document_metadata(
    document_metadata: dict[str, Any],
    section_title: str | None,
) -> dict[str, Any]:
    """Build per-segment metadata so "chunk_metadata" keeps the right section.

    Drops document-level outline lists and any default first-title "section" so
    preamble / titled segments are labeled independently.
    """
    meta = {
        key: value
        for key, value in document_metadata.items()
        if key not in LAYOUT_DOCUMENT_METADATA_KEYS and key != CHUNK_SECTION_KEY
    }
    if section_title:
        meta[CHUNK_SECTION_KEY] = section_title
    return meta


class SectionChunker:
    """Split documents on section boundaries, then recursively size each section.

    Boundaries (in priority order) come from PptxLoader "slides" records,
    Markdown ATX headings (outside fenced code), PPTX "---" separators
    (loader-gated fallback), or outline titles as whole lines. Each resulting
    chunk gets "metadata.section" set to its section title (preamble chunks omit
    it).
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size: int = chunk_size
        self.overlap: int = overlap
        self._splitter: RecursiveChunker = RecursiveChunker(
            chunk_size=chunk_size,
            overlap=overlap,
        )

    def chunk(self, document: Document) -> list[Chunk]:
        segments = iter_section_segments(document.content, document.metadata)
        if not segments:
            return []

        chunks: list[Chunk] = []
        for segment in segments:
            section_doc = document.model_copy(
                update={
                    "content": segment.body,
                    "metadata": _segment_document_metadata(
                        document.metadata,
                        segment.title,
                    ),
                }
            )
            for sub in self._splitter.chunk(section_doc):
                metadata = {
                    **sub.metadata,
                    CHUNK_SOURCE_KEY: document.source,
                    CHUNK_INDEX_KEY: len(chunks),
                }
                if segment.title:
                    metadata[CHUNK_SECTION_KEY] = segment.title
                else:
                    metadata.pop(CHUNK_SECTION_KEY, None)
                chunks.append(
                    Chunk(
                        document_id=document.id,
                        text=sub.text,
                        metadata=metadata,
                    )
                )
        return chunks
