"""Typed smoke checks for contextual header APIs — validated by mypy at lint time (T-171)."""

from __future__ import annotations

from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.contextual_headers import (
    ContextualHeadersChunker,
    build_header_line,
    chunk_context_text,
    group_chunks_by_passage,
    join_chunk_context,
    passage_context_key,
    prepend_headers,
)
from src.rag.chunking.recursive_chunker import RecursiveChunker


def check_contextual_headers_api_types(
    doc: Document,
    chunk: Chunk,
) -> tuple[str, str, str, str, list[tuple[Chunk, list[Chunk]]], str]:
    """Exercise contextual header helper return types."""
    header: str = build_header_line(doc, chunk)
    prefixed: str = prepend_headers(doc, chunk)
    context: str = chunk_context_text(chunk)
    key: str = passage_context_key(chunk)
    groups: list[tuple[Chunk, list[Chunk]]] = group_chunks_by_passage([chunk])
    joined: str = join_chunk_context([chunk])
    return header, prefixed, context, key, groups, joined


def check_contextual_headers_chunker_returns_chunks(doc: Document) -> list[Chunk]:
    """Exercise ContextualHeadersChunker.chunk return type."""
    chunker: ContextualHeadersChunker = ContextualHeadersChunker(RecursiveChunker(chunk_size=500))
    chunks: list[Chunk] = chunker.chunk(doc)
    return chunks
