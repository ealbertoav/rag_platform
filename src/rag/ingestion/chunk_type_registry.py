"""T-243 — Centralized BM25/dense index routing per chunk type.

Replaces the scattered, per-type predicate imports that used to live in
`ingestion_pipeline._bm25_indexable` (`is_hype_question`, `is_summary_chunk`, ...)
with a single lookup table. Chunks whose `type` metadata is absent (plain text
from any chunking strategy) or unrecognized fall back to `DEFAULT_ROUTING`
(indexed in both stores), preserving prior behavior.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.core.constants import (
    CHUNK_TYPE_CAPTION,
    CHUNK_TYPE_DETAIL,
    CHUNK_TYPE_FIGURE,
    CHUNK_TYPE_HYPE,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_PAGE,
    CHUNK_TYPE_PROPOSITION,
    CHUNK_TYPE_SUMMARY,
    CHUNK_TYPE_SYNTHETIC,
    CHUNK_TYPE_TABLE,
)
from src.domain.entities.chunk import Chunk


@dataclass(frozen=True)
class ChunkIndexRouting:
    """Which index(es) a chunk type should be written to."""

    index_dense: bool = True
    index_bm25: bool = True


DEFAULT_ROUTING = ChunkIndexRouting(index_dense=True, index_bm25=True)

# Vector-only index points (LLM-generated, not lexically matchable text
# users would type — question paraphrases / document-level summaries).
_VECTOR_ONLY = ChunkIndexRouting(index_dense=True, index_bm25=False)

CHUNK_TYPE_INDEX_ROUTING: dict[str, ChunkIndexRouting] = {
    CHUNK_TYPE_HYPE: _VECTOR_ONLY,
    CHUNK_TYPE_SUMMARY: _VECTOR_ONLY,
    CHUNK_TYPE_SYNTHETIC: DEFAULT_ROUTING,
    CHUNK_TYPE_TABLE: DEFAULT_ROUTING,
    CHUNK_TYPE_CAPTION: DEFAULT_ROUTING,
    CHUNK_TYPE_DETAIL: DEFAULT_ROUTING,
    CHUNK_TYPE_PROPOSITION: DEFAULT_ROUTING,
    CHUNK_TYPE_PAGE: DEFAULT_ROUTING,
    CHUNK_TYPE_FIGURE: DEFAULT_ROUTING,
}


def routing_for_type(chunk_type: str | None) -> ChunkIndexRouting:
    """Look up index routing for a chunk `type` metadata value.

    Missing or unrecognized types default to `DEFAULT_ROUTING`.
    """
    if chunk_type is None:
        return DEFAULT_ROUTING
    return CHUNK_TYPE_INDEX_ROUTING.get(chunk_type, DEFAULT_ROUTING)


def routing_for_chunk(chunk: Chunk) -> ChunkIndexRouting:
    return routing_for_type(chunk.metadata.get(CHUNK_TYPE_KEY))


def is_dense_indexable(chunk: Chunk) -> bool:
    return routing_for_chunk(chunk).index_dense


def is_bm25_indexable(chunk: Chunk) -> bool:
    return routing_for_chunk(chunk).index_bm25


def filter_dense_indexable(chunks: list[Chunk]) -> list[Chunk]:
    return [chunk for chunk in chunks if is_dense_indexable(chunk)]


def filter_bm25_indexable(chunks: list[Chunk]) -> list[Chunk]:
    return [chunk for chunk in chunks if is_bm25_indexable(chunk)]
