from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

from src.core.constants import FEEDBACK_SCORE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository

logger = logging.getLogger(__name__)

_POSITIVE_DELTA = 1.0
_NEGATIVE_DELTA = -1.0


def score_from_relevant(relevant: bool) -> float:
    """Map a boolean relevance vote to a signed feedback delta."""
    return _POSITIVE_DELTA if relevant else _NEGATIVE_DELTA


def feedback_score_from_metadata(metadata: Mapping[str, Any]) -> float:
    """Return a numeric feedback score from chunk metadata, or 0.0 when unset."""
    return _feedback_score_from_raw(metadata.get(FEEDBACK_SCORE_KEY))


def _feedback_score_from_raw(raw: object) -> float:
    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, int | float):
        return float(raw)
    return 0.0


def merge_chunk_views(left: Chunk, right: Chunk) -> Chunk:
    """Merge two views of the same chunk, keeping the highest feedback score."""
    merged_meta: dict[str, Any] = dict(left.metadata)
    for key, value in right.metadata.items():
        if key == FEEDBACK_SCORE_KEY:
            best = max(
                feedback_score_from_metadata(merged_meta),
                _feedback_score_from_raw(value),
            )
            if best > 0:
                merged_meta[FEEDBACK_SCORE_KEY] = best
            elif FEEDBACK_SCORE_KEY in right.metadata and FEEDBACK_SCORE_KEY not in merged_meta:
                merged_meta[FEEDBACK_SCORE_KEY] = value
        elif key not in merged_meta:
            merged_meta[key] = value
    return left.model_copy(update={"metadata": merged_meta})


def resolve_feedback_score(
    chunk: Chunk,
    *,
    vector_scores: dict[str, float] | None = None,
) -> float:
    """Resolve the feedback score for *chunk*.

    Qdrant is the source of truth when *vector_scores* is supplied (written by
    ``record_feedback``). Otherwise, fall back to chunk metadata from search payloads.
    """
    if vector_scores is not None:
        return vector_scores.get(chunk.id, 0.0)
    return feedback_score_from_metadata(chunk.metadata)


def record_feedback(
    vector_store: VectorStoreRepository,
    query_id: str,
    chunk_id: str,
    score: float,
) -> None:
    """Accumulate user feedback for *chunk_id* in the vector store (source of truth)."""
    updated = vector_store.accumulate_feedback_score(chunk_id, score)
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
    vector_store: VectorStoreRepository | None = None,
) -> list[SearchResult]:
    """Boost fused RRF scores for chunks with positive accumulated feedback."""
    if boost_multiplier <= 0 or not results:
        return results

    vector_scores: dict[str, float] | None = None
    if vector_store is not None:
        chunk_ids = [chunk.id for chunk, _ in results]
        vector_scores = vector_store.get_feedback_scores(chunk_ids)

    boosted: list[SearchResult] = []
    for chunk, fused_score in results:
        feedback = resolve_feedback_score(chunk, vector_scores=vector_scores)
        if feedback > 0:
            fused_score += boost_multiplier * feedback
        boosted.append((chunk, fused_score))

    boosted.sort(key=lambda item: item[1], reverse=True)
    return boosted
