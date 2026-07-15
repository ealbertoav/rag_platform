"""T-253 — Index "type=figure" chunks with an image vector when supported.

Mirrors the T-202/T-232 table/caption-chunk pattern: build unembedded figure
chunks via :func:`~src.rag.ingestion.figure_extractor.build_figure_chunks`
(T-230), text-embed them via :class:`FigureChunker`, and additionally attach
``Chunk.image_embedding`` from ``asset_path`` through
:meth:`EmbeddingRepository.embed_image` (T-250) whenever the active provider
supports it (T-251). Providers without image support soft-fail the image
step only — the text-embedded chunk still indexes, unchanged from the
pre-T-253 behavior.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from pathlib import Path

from src.core.constants import ASSET_PATH_KEY, FIGURE_ID_KEY
from src.core.exceptions import EmbeddingError
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.rag.ingestion.figure_extractor import build_figure_chunks, figure_chunk_id, is_figure_chunk
from src.rag.ingestion.structured_chunk_sync import (
    built_layout_ids,
    chunks_needing_upsert,
    embedding_succeeded,
)

logger = logging.getLogger(__name__)


def known_figure_chunk_ids(source: str, figure_ids: Iterable[str]) -> set[str]:
    """Return stable figure chunk IDs for *figure_ids* at *source*."""
    return {figure_chunk_id(source, figure_id) for figure_id in figure_ids}


def metadata_figure_ids(document: Document) -> set[str]:
    """Return figure IDs that have a persisted asset_path in layout metadata."""
    figures = document.metadata.get("figures")
    if not isinstance(figures, list):
        return set()
    figure_ids: set[str] = set()
    for entry in figures:
        if not isinstance(entry, dict) or not entry.get(FIGURE_ID_KEY):
            continue
        asset_path = entry.get(ASSET_PATH_KEY) or entry.get("asset_path")
        if asset_path:
            figure_ids.add(str(entry[FIGURE_ID_KEY]))
    return figure_ids


def collect_figure_ids(document: Document, existing_chunk_ids: Iterable[str]) -> set[str]:
    """Union figure IDs from layout metadata and indexed stable chunk IDs."""
    figure_ids: set[str] = set()
    figure_ids.update(metadata_figure_ids(document))
    figure_ids.update(_discover_figure_ids_from_chunk_ids(document.source, existing_chunk_ids))
    return figure_ids


def built_figure_ids(built_chunks: Iterable[Chunk]) -> set[str]:
    """Return layout figure IDs represented by *built_chunks*."""
    return built_layout_ids(built_chunks, FIGURE_ID_KEY)


def figure_embedding_succeeded(
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
) -> bool:
    """Return True when every built figure chunk was text-embedded successfully."""
    return embedding_succeeded(built_chunks, embedded_chunks)


def figure_build_succeeded(document: Document, built_chunks: Iterable[Chunk]) -> bool:
    """Return True when every asset-bearing figure produced a structured chunk."""
    expected = metadata_figure_ids(document)
    if not expected:
        return True
    return built_figure_ids(built_chunks) == expected


def figure_sync_succeeded(
    document: Document,
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
) -> bool:
    """Return True when figure build and text-embed both completed successfully."""
    return figure_build_succeeded(document, built_chunks) and figure_embedding_succeeded(
        built_chunks,
        embedded_chunks,
    )


def stale_figure_ids_safe_to_purge(
    document: Document,
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
    stale_figure_ids: Iterable[str],
) -> list[str]:
    """Return stale figure IDs safe to remove after a figure sync attempt."""
    if figure_sync_succeeded(document, built_chunks, embedded_chunks):
        return list(stale_figure_ids)
    return []


def retained_figure_chunk_ids_on_embed_failure(
    source: str,
    document: Document,
    old_chunk_ids: Iterable[str],
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
    *,
    bm25: object | None = None,
) -> set[str]:
    """Return previously indexed figure chunk IDs to keep when figure sync failed."""
    if figure_sync_succeeded(document, built_chunks, embedded_chunks):
        return set()
    desired = {chunk.id for chunk in built_chunks}
    known_old = existing_figure_chunk_ids(
        source,
        old_chunk_ids,
        document=document,
        bm25=bm25,
    )
    unbuildable = metadata_figure_ids(document) - built_figure_ids(built_chunks)
    retain_unbuildable = known_figure_chunk_ids(source, unbuildable)
    return {
        chunk_id
        for chunk_id in old_chunk_ids
        if chunk_id in known_old and (chunk_id in desired or chunk_id in retain_unbuildable)
    }


def merged_figure_chunk_ids(
    existing_ids: Iterable[str],
    known_figure_ids: set[str],
    document: Document,
    built_chunks: Iterable[Chunk],
    embedded_chunks: Iterable[Chunk],
) -> list[str]:
    """Compute figure chunk IDs to store after a skip-path figure sync."""
    embedded = [chunk.id for chunk in embedded_chunks]
    if figure_sync_succeeded(document, built_chunks, embedded_chunks):
        return list(dict.fromkeys(embedded))
    return [chunk_id for chunk_id in existing_ids if chunk_id in known_figure_ids]


def figure_chunks_needing_upsert(
    figure_chunks: Iterable[Chunk],
    existing_chunk_ids: Iterable[str],
    *,
    bm25: object | None = None,
) -> list[Chunk]:
    """Return embedded figure chunks that are new or have updated text in the index."""
    return chunks_needing_upsert(figure_chunks, existing_chunk_ids, bm25=bm25)


def existing_figure_chunk_ids(
    source: str,
    existing_chunk_ids: Iterable[str],
    *,
    document: Document,
    bm25: object | None = None,
) -> set[str]:
    """Identify indexed figure chunk IDs for a *source* from metadata and BM25 payloads."""
    existing = list(existing_chunk_ids)
    figure_ids = collect_figure_ids(document, existing)
    indexed = known_figure_chunk_ids(source, figure_ids)
    if bm25 is None:
        return indexed

    get_by_id = getattr(bm25, "get_by_id", None)
    if get_by_id is None:
        return indexed

    for chunk_id in existing:
        chunk = get_by_id(chunk_id)
        if chunk is not None and is_figure_chunk(chunk):
            indexed.add(chunk_id)
    return indexed


def _discover_figure_ids_from_chunk_ids(source: str, chunk_ids: Iterable[str]) -> set[str]:
    """Infer Docling-style "figure-N" ids from stable indexed figure chunk IDs."""
    chunk_id_set = set(chunk_ids)
    if not chunk_id_set:
        return set()
    discovered: set[str] = set()
    for index in range(1, len(chunk_id_set) + 50):
        figure_id = f"figure-{index}"
        if figure_chunk_id(source, figure_id) in chunk_id_set:
            discovered.add(figure_id)
    return discovered


class FigureChunker:
    """Emits embedded structured figure chunks at ingested time, with image
    vectors when the active embedding provider supports embed_image() (T-253)."""

    def __init__(self, embedder: EmbeddingRepository) -> None:
        self._embedder: EmbeddingRepository = embedder

    def index(self, document: Document) -> list[Chunk]:
        """Return embedded figure chunks for a *document*, image-embedded when supported."""
        chunks = build_figure_chunks(document)
        if not chunks:
            return []

        texts = [c.text for c in chunks]
        try:
            dense_vecs, sparse_vecs = self._embedder.embed_both(texts)
        except Exception as exc:
            logger.warning("Embedding figure chunks failed for %s: %s", document.source, exc)
            return []

        embedded = [
            chunk.model_copy(update={"embedding": dense, "sparse_vector": sparse})
            for chunk, dense, sparse in zip(chunks, dense_vecs, sparse_vecs, strict=True)
        ]
        return self._attach_image_embeddings(embedded, document.source)

    def _attach_image_embeddings(self, chunks: list[Chunk], source: str) -> list[Chunk]:
        """Populate image_embedding from asset_path; leaves it None on any failure."""
        targets = [(index, chunk) for index, chunk in enumerate(chunks) if chunk.asset_path]
        if not targets:
            return chunks

        paths = [Path(chunk.asset_path) for _, chunk in targets]  # type: ignore[arg-type]
        try:
            image_vecs = self._embedder.embed_image(paths)
        except EmbeddingError as exc:
            logger.debug(
                "Image embeddings unavailable for figure chunks in %s: %s",
                source,
                exc,
            )
            return chunks
        except Exception as exc:
            logger.warning("Embedding figure images failed for %s: %s", source, exc)
            return chunks

        updated = list(chunks)
        for (index, chunk), image_embedding in zip(targets, image_vecs, strict=True):
            updated[index] = chunk.model_copy(update={"image_embedding": image_embedding})
        return updated
