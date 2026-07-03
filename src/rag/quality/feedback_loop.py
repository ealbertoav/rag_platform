from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.constants import FEEDBACK_SCORE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository

if TYPE_CHECKING:
    from src.infrastructure.vectordb.bm25 import BM25Index

logger = logging.getLogger(__name__)

_POSITIVE_DELTA = 1.0
_NEGATIVE_DELTA = -1.0


def score_from_relevant(relevant: bool) -> float:
    """Map a boolean relevance vote to a signed feedback delta."""
    return _POSITIVE_DELTA if relevant else _NEGATIVE_DELTA


def feedback_score_from_metadata(metadata: dict[str, object]) -> float:
    """Return a numeric feedback score from chunk metadata, or 0.0 when unset."""
    raw = metadata.get(FEEDBACK_SCORE_KEY)
    if isinstance(raw, bool):
        return 0.0
    if isinstance(raw, int | float):
        return float(raw)
    return 0.0


def merge_chunk_views(left: Chunk, right: Chunk) -> Chunk:
    """Merge two views of the same chunk, keeping the highest feedback score."""
    merged_meta = dict(left.metadata)
    for key, value in right.metadata.items():
        if key == FEEDBACK_SCORE_KEY:
            best = max(
                feedback_score_from_metadata(merged_meta),
                feedback_score_from_metadata({FEEDBACK_SCORE_KEY: value}),
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
    bm25_index: BM25Index | None = None,
    vector_scores: dict[str, float] | None = None,
) -> float:
    """Resolve the best-known feedback score for *chunk* across retrieval stores."""
    score = feedback_score_from_metadata(chunk.metadata)
    if bm25_index is not None:
        stored = bm25_index.get_by_id(chunk.id)
        if stored is not None:
            score = max(score, feedback_score_from_metadata(stored.metadata))
    if vector_scores is not None:
        score = max(score, vector_scores.get(chunk.id, 0.0))
    return score


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
        if bm25_index.update_chunk_metadata(chunk_id, {FEEDBACK_SCORE_KEY: updated}):
            bm25_index.save()
        else:
            logger.warning(
                "Chunk %r not found in BM25 index; feedback persisted to vector store only",
                chunk_id,
            )
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
    bm25_index: BM25Index | None = None,
    vector_store: VectorStoreRepository | None = None,
) -> list[SearchResult]:
    """Boost fused RRF scores for chunks with positive accumulated feedback."""
    if boost_multiplier <= 0 or not results:
        return results

    vector_scores: dict[str, float] = {}
    if vector_store is not None:
        chunk_ids = [chunk.id for chunk, _ in results]
        vector_scores = vector_store.get_feedback_scores(chunk_ids)

    boosted: list[SearchResult] = []
    for chunk, fused_score in results:
        feedback = resolve_feedback_score(
            chunk,
            bm25_index=bm25_index,
            vector_scores=vector_scores,
        )
        if feedback > 0:
            fused_score += boost_multiplier * feedback
        boosted.append((chunk, fused_score))

    boosted.sort(key=lambda item: item[1], reverse=True)
    return boosted
