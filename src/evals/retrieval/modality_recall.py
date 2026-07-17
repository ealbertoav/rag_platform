from __future__ import annotations

import dataclasses
from collections.abc import Iterable

from src.core.constants import MODALITY_FIGURE, MODALITY_TABLE
from src.evals.retrieval.recall_at_k import recall_at_k

MODALITIES: tuple[str, str] = (MODALITY_TABLE, MODALITY_FIGURE)


@dataclasses.dataclass
class ModalityRetrievalSample:
    query_id: str
    modality: str  # "table" or "figure" (T-210 modality labels)
    retrieved_ids: list[str]  # system output ranked best-first
    relevant_ids: list[str]  # ground-truth


def modality_recall_at_k(
    samples: Iterable[ModalityRetrievalSample], modality: str, k: int
) -> float:
    """Mean Recall@K over *samples* whose modality equals *modality*.

    Returns 0.0 when no sample matches *modality*, mirroring recall_at_k's
    empty-input convention.
    """
    matching = [s for s in samples if s.modality == modality]
    if not matching:
        return 0.0
    return sum(recall_at_k(s.retrieved_ids, s.relevant_ids, k) for s in matching) / len(matching)


def table_recall_at_k(samples: Iterable[ModalityRetrievalSample], k: int) -> float:
    """Mean Recall@K over table-modality samples."""
    return modality_recall_at_k(samples, MODALITY_TABLE, k)


def figure_recall_at_k(samples: Iterable[ModalityRetrievalSample], k: int) -> float:
    """Mean Recall@K over figure-modality samples."""
    return modality_recall_at_k(samples, MODALITY_FIGURE, k)


def load_modality_samples(pairs: list[dict[str, object]]) -> list[ModalityRetrievalSample]:
    """Build samples (with empty "retrieved_ids") from multimodal QA pair dicts.

    Expects rows shaped like "MultimodalQAPair.to_dict()" (T-280) —
    "relevant_chunks" and "modality" keys. Callers fill "retrieved_ids" by
    running the retrieval pipeline before scoring.
    """
    samples: list[ModalityRetrievalSample] = []
    for i, pair in enumerate(pairs):
        raw_relevant = pair.get("relevant_chunks", [])
        modality = pair.get("modality", "")
        samples.append(
            ModalityRetrievalSample(
                query_id=str(i),
                modality=modality if isinstance(modality, str) else "",
                retrieved_ids=[],
                relevant_ids=[
                    r
                    for r in (raw_relevant if isinstance(raw_relevant, list) else [])
                    if isinstance(r, str)
                ],
            )
        )
    return samples


__all__ = [
    "MODALITIES",
    "ModalityRetrievalSample",
    "figure_recall_at_k",
    "load_modality_samples",
    "modality_recall_at_k",
    "table_recall_at_k",
]
