from __future__ import annotations

import logging
import re
import uuid
from collections.abc import Iterable
from typing import Any

from src.core.constants import (
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_TABLE,
    TABLE_ID_KEY,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.rag.chunking.metadata import chunk_metadata

logger = logging.getLogger(__name__)

_TABLE_TEXT_KEY = "text"

# GFM pipe table: header row, separator row, one or more body rows.
_MARKDOWN_TABLE_RE = re.compile(
    r"(\|[^\n]+\|\n\|[-: |]+\|\n(?:\|[^\n]+\|\n?)+)",
    re.MULTILINE,
)
_TABLE_ID_NUMERIC_RE = re.compile(r"^table-(\d+)$")
_TABLE_CHUNK_NAMESPACE = uuid.UUID("a3f8c2e1-7b4d-4e9f-8c1a-2d6e5f0b9a37")


def table_chunk_id(source: str, table_id: str) -> str:
    """Stable chunk ID for idempotent table backfill on unchanged documents.

    Keys off *source* (resolved file path), not ephemeral "Document.id" values
    assigned at load time, so unchanged re-ingests produce the same chunk IDs.
    """
    return str(uuid.uuid5(_TABLE_CHUNK_NAMESPACE, f"{source}:{table_id}"))


def is_table_chunk(chunk: Chunk) -> bool:
    """Return True when *chunk* is a structured table index point."""
    return chunk.metadata.get(CHUNK_TYPE_KEY) == CHUNK_TYPE_TABLE


def known_table_chunk_ids(source: str, table_ids: Iterable[str]) -> set[str]:
    """Return stable table chunk IDs for *table_ids* at *source*."""
    return {table_chunk_id(source, table_id) for table_id in table_ids}


def collect_table_ids(document: Document, existing_chunk_ids: Iterable[str]) -> set[str]:
    """Union table IDs from layout metadata and indexed stable chunk IDs."""
    table_ids: set[str] = set()
    tables = document.metadata.get("tables")
    if isinstance(tables, list):
        for entry in tables:
            if isinstance(entry, dict) and entry.get(TABLE_ID_KEY):
                table_ids.add(str(entry[TABLE_ID_KEY]))
    table_ids.update(_discover_table_ids_from_chunk_ids(document.source, existing_chunk_ids))
    return table_ids


def existing_table_chunk_ids(
    source: str,
    existing_chunk_ids: Iterable[str],
    *,
    document: Document,
    bm25: object | None = None,
) -> set[str]:
    """Identify indexed table chunk IDs for a *source* from metadata and BM25 payloads."""
    existing = list(existing_chunk_ids)
    table_ids = collect_table_ids(document, existing)
    indexed = known_table_chunk_ids(source, table_ids)
    if bm25 is None:
        return indexed

    get_by_id = getattr(bm25, "get_by_id", None)
    if get_by_id is None:
        return indexed

    for chunk_id in existing:
        chunk = get_by_id(chunk_id)
        if chunk is not None and is_table_chunk(chunk):
            indexed.add(chunk_id)
    return indexed


def extract_markdown_tables(content: str) -> list[str]:
    """Return Markdown table blocks in document order."""
    return [block.strip() for block in _MARKDOWN_TABLE_RE.findall(content) if block.strip()]


def _discover_table_ids_from_chunk_ids(source: str, chunk_ids: Iterable[str]) -> set[str]:
    """Infer Docling-style ``table-N`` ids from stable indexed chunk IDs."""
    chunk_id_set = set(chunk_ids)
    if not chunk_id_set:
        return set()
    discovered: set[str] = set()
    for index in range(1, len(chunk_id_set) + 50):
        table_id = f"table-{index}"
        if table_chunk_id(source, table_id) in chunk_id_set:
            discovered.add(table_id)
    return discovered


def _content_table_for_id(table_id: str, content_tables: list[str]) -> str | None:
    """Map Docling "table-N" ids to the Nth Markdown table in document content."""
    match = _TABLE_ID_NUMERIC_RE.match(table_id)
    if match is None:
        return None
    index = int(match.group(1)) - 1
    if 0 <= index < len(content_tables):
        return content_tables[index]
    return None


def _resolve_table_text(entry: dict[str, Any], fallback: str | None) -> str | None:
    for key in (_TABLE_TEXT_KEY, "markdown"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if fallback and fallback.strip():
        return fallback.strip()
    return None


def build_table_chunks(document: Document) -> list[Chunk]:
    """Build unembedded table chunks from document layout metadata."""
    tables = document.metadata.get("tables")
    if not isinstance(tables, list) or not tables:
        return []

    content_tables = extract_markdown_tables(document.content)
    chunks: list[Chunk] = []

    for index, raw_entry in enumerate(tables):
        if not isinstance(raw_entry, dict):
            logger.debug("Skipping non-dict table entry at index %d", index)
            continue
        table_id = raw_entry.get(TABLE_ID_KEY)
        if not table_id:
            logger.debug("Skipping table entry without %s at index %d", TABLE_ID_KEY, index)
            continue

        fallback = _content_table_for_id(str(table_id), content_tables)
        text = _resolve_table_text(raw_entry, fallback)
        if not text:
            logger.warning(
                "No text for table %s in %s — skipping structured table chunk",
                table_id,
                document.source,
            )
            continue

        metadata = chunk_metadata(document.metadata)
        metadata[CHUNK_TYPE_KEY] = CHUNK_TYPE_TABLE
        metadata[TABLE_ID_KEY] = str(table_id)
        metadata[CHUNK_SOURCE_KEY] = document.source
        if CHUNK_PAGE_KEY in raw_entry:
            metadata[CHUNK_PAGE_KEY] = raw_entry[CHUNK_PAGE_KEY]
        if BBOX_KEY in raw_entry:
            metadata[BBOX_KEY] = raw_entry[BBOX_KEY]

        chunks.append(
            Chunk(
                id=table_chunk_id(document.source, str(table_id)),
                document_id=document.id,
                text=text,
                metadata=metadata,
            )
        )

    return chunks


class TableChunker:
    """Emits embedded structured table chunks at ingested time."""

    def __init__(self, embedder: EmbeddingRepository) -> None:
        self._embedder: EmbeddingRepository = embedder

    def index(self, document: Document) -> list[Chunk]:
        """Return embedded table chunks for a *document*."""
        chunks = build_table_chunks(document)
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        try:
            dense_vecs, sparse_vecs = self._embedder.embed_both(texts)
        except Exception as exc:
            logger.warning("Embedding table chunks failed for %s: %s", document.source, exc)
            return []

        return [
            chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
            for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs, strict=True)
        ]
