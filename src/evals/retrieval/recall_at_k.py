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


def oracle_recall_at_k(relevant_ids: list[str], k: int) -> float:
    """Recall@K when all ground-truth relevant chunks are ranked before any other result.

    Oracle evaluation only considers the first *k* relevant documents as the
    recall target, since only that many can appear in the top-*k* retrieved
    slots. This avoids penalising multi-chunk golden rows where len(relevant) > k.
    """
    if not relevant_ids or k <= 0:
        return 0.0
    return recall_at_k(relevant_ids, relevant_ids[:k], k)
