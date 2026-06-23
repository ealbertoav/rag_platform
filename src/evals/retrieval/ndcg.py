from __future__ import annotations

import math


def ndcg_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Normalized Discounted Cumulative Gain at rank K.

    Assumes binary relevance: a document is either relevant (gain = 1) or not.

    DCG = Σ gain_i / log2(i + 2), for i = 0... k-1
    IDCG = DCG of the ideal ordering (all relevant docs first)
    NDCG = DCG / IDCG

    Args:
        retrieved_ids: Ranked list of retrieved chunk IDs (the best first).
        relevant_ids:  Ground-truth relevant chunk IDs.
        k:             Cut-off rank.

    Returns:
        Value in [0, 1].  Returns 0.0 when there are no relevant documents or
        *k* ≤ 0.
    """
    if not relevant_ids or k <= 0:
        return 0.0

    relevant = set(relevant_ids)

    dcg = sum(
        1.0 / math.log2(i + 2) for i, doc_id in enumerate(retrieved_ids[:k]) if doc_id in relevant
    )

    # Ideal DCG: place as many relevant docs as possible in the top-K slots.
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))

    return dcg / idcg if idcg > 0 else 0.0
