from __future__ import annotations

import numpy as np

from src.domain.entities.chunk import Chunk


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(np.dot(a, b) / denom) if denom else 0.0


def _normalize_relevance_scores(n: int) -> list[float]:
    """Derive normalized relevance from reranker order (index 0 = most relevant)."""
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    return [(n - i) / n for i in range(n)]


def mmr_select(
    chunks: list[Chunk],
    embeddings: list[list[float]],
    lambda_: float,
    top_k: int,
) -> list[Chunk]:
    """Select up to *top_k* chunks via Maximal Marginal Relevance (MMR).

    Greedy selection maximizes::

        lambda_ * relevance - (1 - lambda_) * max_sim_to_selected

    where *relevance* comes from input order (reranker ranking) and
    pairwise similarity uses cosine distance between *embeddings*.

    When ``lambda_ == 1.0``, returns the first *top_k* chunks unchanged
    (pure relevance, no diversity penalty).
    """
    if not chunks:
        return []
    if len(chunks) != len(embeddings):
        msg = f"chunks ({len(chunks)}) and embeddings ({len(embeddings)}) length mismatch"
        raise ValueError(msg)

    k = min(top_k, len(chunks))
    if k == 0:
        return []

    if lambda_ >= 1.0:
        return list(chunks[:k])

    relevance = _normalize_relevance_scores(len(chunks))
    vectors = [np.asarray(v, dtype=np.float64) for v in embeddings]

    selected_indices: list[int] = []
    remaining = set(range(len(chunks)))

    while len(selected_indices) < k and remaining:
        best_idx = -1
        best_score = float("-inf")

        for idx in remaining:
            rel = relevance[idx]
            if not selected_indices:
                mmr_score = lambda_ * rel
            else:
                max_sim = max(_cosine(vectors[idx], vectors[s]) for s in selected_indices)
                mmr_score = lambda_ * rel - (1.0 - lambda_) * max_sim

            if mmr_score > best_score:
                best_score = mmr_score
                best_idx = idx

        selected_indices.append(best_idx)
        remaining.remove(best_idx)

    return [chunks[i] for i in selected_indices]
