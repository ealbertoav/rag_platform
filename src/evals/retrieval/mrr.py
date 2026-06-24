from __future__ import annotations


def mrr(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """Mean Reciprocal Rank for a single query.

    Returns 1/rank of the first relevant document found in *retrieved_ids*,
    or 0.0 if no relevant document appears in the list.
    """
    if not relevant_ids or not retrieved_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    for rank, id_ in enumerate(retrieved_ids, 1):
        if id_ in relevant_set:
            return 1.0 / rank
    return 0.0
