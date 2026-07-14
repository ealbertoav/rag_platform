"""Shared skip-path sync helpers for structured layout chunks (tables, captions)."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from src.core.constants import BBOX_KEY, CHUNK_PAGE_KEY, CHUNK_SOURCE_KEY, CHUNK_TYPE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking.metadata import chunk_metadata


def built_layout_ids(built_chunks: Iterable[Chunk], id_key: str) -> set[str]:
    """Return layout entity IDs represented by *built_chunks* under *id_key*."""
    return {str(chunk.metadata.get(id_key)) for chunk in built_chunks if chunk.metadata.get(id_key)}


def embedding_succeeded(
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
) -> bool:
    """Return True when every built structured chunk was embedded successfully."""
    desired = {chunk.id for chunk in built_chunks}
    if not desired:
        return True
    successful = {chunk.id for chunk in embedded_chunks}
    return successful == desired


def chunks_needing_upsert(
    chunks: Iterable[Chunk],
    existing_chunk_ids: Iterable[str],
    *,
    bm25: object | None = None,
) -> list[Chunk]:
    """Return embedded chunks that are new or have updated text in the index."""
    existing_id_set = set(existing_chunk_ids)
    get_by_id = getattr(bm25, "get_by_id", None) if bm25 is not None else None
    needing: list[Chunk] = []
    for chunk in chunks:
        if chunk.id not in existing_id_set:
            needing.append(chunk)
            continue
        if get_by_id is None:
            continue
        indexed = get_by_id(chunk.id)
        if indexed is None or indexed.text != chunk.text:
            needing.append(chunk)
    return needing


def structured_chunk_metadata(
    document: Document,
    raw_entry: dict[str, Any],
    *,
    chunk_type: str,
    layout_id_key: str,
    layout_id: str,
) -> dict[str, Any]:
    """Build chunk metadata for a structured layout entry (table, caption, …)."""
    metadata = chunk_metadata(document.metadata)
    metadata[CHUNK_TYPE_KEY] = chunk_type
    metadata[layout_id_key] = layout_id
    metadata[CHUNK_SOURCE_KEY] = document.source
    if CHUNK_PAGE_KEY in raw_entry:
        metadata[CHUNK_PAGE_KEY] = raw_entry[CHUNK_PAGE_KEY]
    if BBOX_KEY in raw_entry:
        metadata[BBOX_KEY] = raw_entry[BBOX_KEY]
    return metadata
