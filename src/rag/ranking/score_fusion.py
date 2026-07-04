from __future__ import annotations

from src.core.constants import RRF_K
from src.domain.entities.chunk import Chunk
from src.domain.repositories.vector_store_repository import SearchResult
from src.rag.quality.feedback_loop import merge_chunk_views


def _register_chunk(chunks: dict[str, Chunk], chunk: Chunk) -> None:
    existing = chunks.get(chunk.id)
    if existing is None:
        chunks[chunk.id] = chunk
        return
    chunks[chunk.id] = merge_chunk_views(existing, chunk)


def rrf_fuse(
    *ranked_lists: list[SearchResult],
    top_k: int,
    k: int = RRF_K,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion over any number of ranked result lists.

    For each chunk that appears in at least one list, its fused score is:

        score = Σ 1 / (k + rank_i) for each list i that contains the chunk

    The constant k (default 60) controls how many early ranks are penalized.
    Higher k → less penalty for lower ranks; lower k → stronger top-rank boost.

    Chunks are deduplicated by "Chunk.id" — non-feedback metadata is merged
    across retriever views of the same chunk (feedback scores stay on the first view).
    """
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    for ranked in ranked_lists:
        for rank, (chunk, _) in enumerate(ranked):
            scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
            _register_chunk(chunks, chunk)

    return [
        (chunks[cid], scores[cid])
        for cid in sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    ]


def weighted_linear_fuse(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    alpha: float,
    top_k: int,
) -> list[SearchResult]:
    """Weighted linear combination of normalized dense and sparse scores.

    "alpha=1.0" → dense only; "alpha=0.0" → sparse only.

    Both score lists are min-max normalized to [0, 1] before combining so that
    the raw score scales (cosine similarity vs. BM25) don't dominate.
    When the same chunk appears in both lists, non-feedback metadata is merged
    rather than letting one retriever overwrite the other.
    """

    def _normalise(results: list[SearchResult]) -> dict[str, float]:
        if not results:
            return {}
        scores = [s for _, s in results]
        lo, hi = min(scores), max(scores)
        denom = hi - lo if hi != lo else 1.0
        return {result.id: (s - lo) / denom for result, s in results}

    dense_norm = _normalise(dense)
    sparse_norm = _normalise(sparse)

    all_ids = set(dense_norm) | set(sparse_norm)
    chunks: dict[str, Chunk] = {}
    for result, _ in dense + sparse:
        _register_chunk(chunks, result)
    fused: dict[str, float] = {
        cid: alpha * dense_norm.get(cid, 0.0) + (1.0 - alpha) * sparse_norm.get(cid, 0.0)
        for cid in all_ids
    }

    return [
        (chunks[cid], fused[cid])
        for cid in sorted(fused, key=fused.__getitem__, reverse=True)[:top_k]
    ]
