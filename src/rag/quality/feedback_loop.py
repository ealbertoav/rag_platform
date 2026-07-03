from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.constants import FEEDBACK_SCORE_KEY
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository

if TYPE_CHECKING:
    from src.infrastructure.vectordb.bm25 import BM25Index

logger = logging.getLogger(__name__)

_POSITIVE_DELTA = 1.0
_NEGATIVE_DELTA = -1.0


def score_from_relevant(relevant: bool) -> float:
    """Map a boolean relevance vote to a signed feedback delta."""
    return _POSITIVE_DELTA if relevant else _NEGATIVE_DELTA


def record_feedback(
    vector_store: VectorStoreRepository,
    query_id: str,
    chunk_id: str,
    score: float,
    *,
    bm25_index: BM25Index | None = None,
) -> None:
    """Accumulate user feedback for *chunk_id* and persist to vector-store metadata."""
    current = vector_store.get_feedback_score(chunk_id)
    updated = current + score
    vector_store.set_feedback_score(chunk_id, updated)
    if bm25_index is not None:
        bm25_index.update_chunk_metadata(chunk_id, {FEEDBACK_SCORE_KEY: updated})
    logger.info(
        "Recorded retrieval feedback query_id=%r chunk_id=%r delta=%.2f total=%.2f",
        query_id,
        chunk_id,
        score,
        updated,
    )


def apply_feedback_boost(
    results: list[SearchResult],
    *,
    boost_multiplier: float,
) -> list[SearchResult]:
    """Boost fused RRF scores for chunks with positive accumulated feedback."""
    if boost_multiplier <= 0 or not results:
        return results

    boosted: list[SearchResult] = []
    for chunk, fused_score in results:
        raw = chunk.metadata.get(FEEDBACK_SCORE_KEY)
        if isinstance(raw, int | float) and not isinstance(raw, bool) and raw > 0:
            fused_score += boost_multiplier * float(raw)
        boosted.append((chunk, fused_score))

    boosted.sort(key=lambda item: item[1], reverse=True)
    return boosted
