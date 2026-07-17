from __future__ import annotations

import dataclasses

from rich.console import Console
from rich.table import Table

from src.evals.retrieval.modality_recall import (
    MODALITIES,
    ModalityRetrievalSample,
    modality_recall_at_k,
)


@dataclasses.dataclass
class ModalityMetricsAtK:
    modality: str
    k: int
    recall: float


class ModalityRetrievalEvaluator:
    """Compute Recall@K separately for table- and figure-modality samples."""

    def __init__(self, k_values: list[int] | None = None) -> None:
        self.k_values: list[int] = k_values or [1, 3, 5, 10]

    def evaluate(self, samples: list[ModalityRetrievalSample]) -> list[ModalityMetricsAtK]:
        """Return per-modality, per-K recall averaged over *samples*."""
        if not samples:
            return []

        results: list[ModalityMetricsAtK] = []
        for modality in MODALITIES:
            for k in self.k_values:
                recall = modality_recall_at_k(samples, modality, k)
                results.append(ModalityMetricsAtK(modality=modality, k=k, recall=recall))
        return results

    @staticmethod
    def print_table(
        metrics: list[ModalityMetricsAtK], title: str = "Modality Retrieval Metrics"
    ) -> None:
        """Print a Rich table summarizing recall across modalities and K values."""
        table = Table(title=title, show_header=True)
        table.add_column("Modality", style="cyan")
        table.add_column("K", justify="right")
        table.add_column("Recall@K", justify="right")

        for m in metrics:
            table.add_row(m.modality, str(m.k), f"{m.recall:.4f}")

        Console().print(table)


__all__ = ["ModalityMetricsAtK", "ModalityRetrievalEvaluator"]
