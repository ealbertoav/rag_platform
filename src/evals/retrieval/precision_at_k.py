from __future__ import annotations


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of the top-K retrieved results that are relevant.

    Args:
        retrieved_ids: Ranked list of retrieved chunk IDs (the best first).
        relevant_ids:  Ground-truth relevant chunk IDs.
        k:             Cut-off rank.

    Returns:
        Value in [0, 1].  Returns 0.0 when *k* ≤ 0.
    """
    if k <= 0:
        return 0.0
    top_k = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for r in top_k if r in relevant) / k
