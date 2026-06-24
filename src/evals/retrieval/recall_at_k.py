from __future__ import annotations


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Fraction of relevant documents found in the top-K retrieved results.

    Args:
        retrieved_ids: Ranked list of retrieved chunk IDs (the best first).
        relevant_ids:  Ground-truth relevant chunk IDs.
        k:             Cut-off rank.

    Returns:
        Value in [0, 1].  Returns 0.0 when *relevant_ids* is empty.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    top_k = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top_k & relevant) / len(relevant)
