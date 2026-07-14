"""T-232 — Index "type=caption" chunks linked to "figure_id".

Mirrors the T-202 table-chunk pattern: build unembedded caption chunks from
"figures[].caption", embed via "CaptionChunker", and support skip-path
upsert / stale purge helpers with stable UUIDv5 IDs.
"""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterable
from typing import Any

from src.core.constants import (
    ASSET_PATH_KEY,
    CHUNK_TYPE_CAPTION,
    CHUNK_TYPE_KEY,
    FIGURE_CAPTION_KEY,
    FIGURE_ID_KEY,
    MODALITY_CAPTION,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.rag.ingestion.structured_chunk_sync import (
    built_layout_ids,
    chunks_needing_upsert,
    embedding_succeeded,
    structured_chunk_metadata,
)

logger = logging.getLogger(__name__)

_CAPTION_CHUNK_NAMESPACE = uuid.UUID("b5d0e8f2-3c7a-4b19-9d4e-6a1f8c0b2e54")


def caption_chunk_id(source: str, figure_id: str) -> str:
    """Stable chunk ID for idempotent caption backfill on unchanged documents.

    Keys off *source* (resolved file path), not ephemeral "Document.id" values
    assigned at load time, so unchanged re-ingests produce the same chunk IDs.
    """
    return str(uuid.uuid5(_CAPTION_CHUNK_NAMESPACE, f"{source}:{figure_id}"))


def is_caption_chunk(chunk: Chunk) -> bool:
    """Return True when *chunk* is a structured caption index point."""
    return chunk.metadata.get(CHUNK_TYPE_KEY) == CHUNK_TYPE_CAPTION


def known_caption_chunk_ids(source: str, figure_ids: Iterable[str]) -> set[str]:
    """Return stable caption chunk IDs for *figure_ids* at *source*."""
    return {caption_chunk_id(source, figure_id) for figure_id in figure_ids}


def collect_caption_figure_ids(document: Document, existing_chunk_ids: Iterable[str]) -> set[str]:
    """Union captionable figure IDs from layout metadata and indexed chunk IDs."""
    figure_ids: set[str] = set()
    figure_ids.update(metadata_caption_figure_ids(document))
    figure_ids.update(_discover_figure_ids_from_chunk_ids(document.source, existing_chunk_ids))
    return figure_ids


def metadata_caption_figure_ids(document: Document) -> set[str]:
    """Return figure IDs that have a non-empty caption in layout metadata."""
    figures = document.metadata.get("figures")
    if not isinstance(figures, list):
        return set()
    figure_ids: set[str] = set()
    for entry in figures:
        if not isinstance(entry, dict) or not entry.get(FIGURE_ID_KEY):
            continue
        caption = _resolve_caption_text(entry)
        if caption is not None:
            figure_ids.add(str(entry[FIGURE_ID_KEY]))
    return figure_ids


def built_caption_figure_ids(built_chunks: Iterable[Chunk]) -> set[str]:
    """Return figure IDs represented by *built_chunks*."""
    return built_layout_ids(built_chunks, FIGURE_ID_KEY)


def caption_embedding_succeeded(
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
) -> bool:
    """Return True when every built caption chunk was embedded successfully."""
    return embedding_succeeded(built_chunks, embedded_chunks)


def caption_build_succeeded(document: Document, built_chunks: Iterable[Chunk]) -> bool:
    """Return True when every captionable figure produced a structured chunk."""
    expected = metadata_caption_figure_ids(document)
    if not expected:
        return True
    return built_caption_figure_ids(built_chunks) == expected


def caption_sync_succeeded(
    document: Document,
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
) -> bool:
    """Return True when caption build and embed both completed successfully."""
    return caption_build_succeeded(document, built_chunks) and caption_embedding_succeeded(
        built_chunks,
        embedded_chunks,
    )


def stale_caption_ids_safe_to_purge(
    document: Document,
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
    stale_caption_ids: Iterable[str],
) -> list[str]:
    """Return stale caption IDs safe to remove after a caption sync attempt."""
    if caption_sync_succeeded(document, built_chunks, embedded_chunks):
        return list(stale_caption_ids)
    return []


def retained_caption_chunk_ids_on_embed_failure(
    source: str,
    document: Document,
    old_chunk_ids: Iterable[str],
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
    *,
    bm25: object | None = None,
) -> set[str]:
    """Return previously indexed caption chunk IDs to keep when caption sync failed."""
    if caption_sync_succeeded(document, built_chunks, embedded_chunks):
        return set()
    desired = {chunk.id for chunk in built_chunks}
    known_old = existing_caption_chunk_ids(
        source,
        old_chunk_ids,
        document=document,
        bm25=bm25,
    )
    unbuildable = metadata_caption_figure_ids(document) - built_caption_figure_ids(built_chunks)
    retain_unbuildable = known_caption_chunk_ids(source, unbuildable)
    return {
        chunk_id
        for chunk_id in old_chunk_ids
        if chunk_id in known_old and (chunk_id in desired or chunk_id in retain_unbuildable)
    }


def merged_caption_chunk_ids(
    existing_ids: Iterable[str],
    known_caption_ids: set[str],
    document: Document,
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
) -> list[str]:
    """Compute caption chunk IDs to store after a skip-path caption sync."""
    embedded = [chunk.id for chunk in embedded_chunks]
    if caption_sync_succeeded(document, built_chunks, embedded_chunks):
        return list(dict.fromkeys(embedded))
    return [chunk_id for chunk_id in existing_ids if chunk_id in known_caption_ids]


def caption_chunks_needing_upsert(
    caption_chunks: Iterable[Chunk],
    existing_chunk_ids: Iterable[str],
    *,
    bm25: object | None = None,
) -> list[Chunk]:
    """Return embedded caption chunks that are new or have updated text in the index."""
    return chunks_needing_upsert(caption_chunks, existing_chunk_ids, bm25=bm25)


def existing_caption_chunk_ids(
    source: str,
    existing_chunk_ids: Iterable[str],
    *,
    document: Document,
    bm25: object | None = None,
) -> set[str]:
    """Identify indexed caption chunk IDs for a *source* from metadata and BM25 payloads."""
    existing = list(existing_chunk_ids)
    figure_ids = collect_caption_figure_ids(document, existing)
    indexed = known_caption_chunk_ids(source, figure_ids)
    if bm25 is None:
        return indexed

    get_by_id = getattr(bm25, "get_by_id", None)
    if get_by_id is None:
        return indexed

    for chunk_id in existing:
        chunk = get_by_id(chunk_id)
        if chunk is not None and is_caption_chunk(chunk):
            indexed.add(chunk_id)
    return indexed


def _discover_figure_ids_from_chunk_ids(source: str, chunk_ids: Iterable[str]) -> set[str]:
    """Infer Docling-style "figure-N" ids from stable indexed caption chunk IDs."""
    chunk_id_set = set(chunk_ids)
    if not chunk_id_set:
        return set()
    discovered: set[str] = set()
    for index in range(1, len(chunk_id_set) + 50):
        figure_id = f"figure-{index}"
        if caption_chunk_id(source, figure_id) in chunk_id_set:
            discovered.add(figure_id)
    return discovered


def _resolve_caption_text(entry: dict[str, Any]) -> str | None:
    """Return stripped caption text from a figure metadata entry, if present."""
    for key in (FIGURE_CAPTION_KEY, "caption"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def build_caption_chunks(document: Document) -> list[Chunk]:
    """Build unembedded caption chunks from document figure metadata with captions."""
    figures = document.metadata.get("figures")
    if not isinstance(figures, list) or not figures:
        return []

    chunks: list[Chunk] = []

    for index, raw_entry in enumerate(figures):
        if not isinstance(raw_entry, dict):
            logger.debug("Skipping non-dict figure entry at index %d", index)
            continue
        figure_id = raw_entry.get(FIGURE_ID_KEY)
        if not figure_id:
            logger.debug("Skipping figure entry without %s at index %d", FIGURE_ID_KEY, index)
            continue

        text = _resolve_caption_text(raw_entry)
        if not text:
            logger.debug(
                "No caption for figure %s in %s — skipping caption chunk",
                figure_id,
                document.source,
            )
            continue

        metadata = structured_chunk_metadata(
            document,
            raw_entry,
            chunk_type=CHUNK_TYPE_CAPTION,
            layout_id_key=FIGURE_ID_KEY,
            layout_id=str(figure_id),
        )

        asset_path = raw_entry.get(ASSET_PATH_KEY) or raw_entry.get("asset_path")
        asset_path_str = str(asset_path) if asset_path else None
        if asset_path_str:
            metadata[ASSET_PATH_KEY] = asset_path_str

        chunks.append(
            Chunk(
                id=caption_chunk_id(document.source, str(figure_id)),
                document_id=document.id,
                text=text,
                metadata=metadata,
                modality=MODALITY_CAPTION,
                asset_path=asset_path_str,
            )
        )

    return chunks


class CaptionChunker:
    """Emits embedded structured caption chunks at ingested time."""

    def __init__(self, embedder: EmbeddingRepository) -> None:
        self._embedder: EmbeddingRepository = embedder

    def index(self, document: Document) -> list[Chunk]:
        """Return embedded caption chunks for a *document*."""
        chunks = build_caption_chunks(document)
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        try:
            dense_vecs, sparse_vecs = self._embedder.embed_both(texts)
        except Exception as exc:
            logger.warning("Embedding caption chunks failed for %s: %s", document.source, exc)
            return []

        return [
            chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
            for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs, strict=True)
        ]
