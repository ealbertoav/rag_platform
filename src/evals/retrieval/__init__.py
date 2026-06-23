from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable
from pathlib import Path

from rich.console import Console
from rich.table import Table

from src.evals.retrieval.ndcg import ndcg_at_k
from src.evals.retrieval.precision_at_k import precision_at_k
from src.evals.retrieval.recall_at_k import recall_at_k


@dataclasses.dataclass
class RetrievalSample:
    query_id: str
    retrieved_ids: list[str]   # system output ranked best-first
    relevant_ids: list[str]    # ground-truth


@dataclasses.dataclass
class MetricsAtK:
    k: int
    recall: float
    precision: float
    ndcg: float


class RetrievalEvaluator:
    """Compute Recall@K, Precision@K, NDCG@K across a list of samples."""

    def __init__(self, k_values: list[int] | None = None) -> None:
        self.k_values: list[int] = k_values or [1, 3, 5, 10]

    def evaluate(self, samples: list[RetrievalSample]) -> list[MetricsAtK]:
        """Return averaged metrics for each K value in *k_values*."""
        if not samples:
            return []

        results: list[MetricsAtK] = []
        for k in self.k_values:
            recall = _mean(recall_at_k(s.retrieved_ids, s.relevant_ids, k) for s in samples)
            precision = _mean(
                precision_at_k(s.retrieved_ids, s.relevant_ids, k) for s in samples
            )
            ndcg_score = _mean(ndcg_at_k(s.retrieved_ids, s.relevant_ids, k) for s in samples)
            results.append(MetricsAtK(k=k, recall=recall, precision=precision, ndcg=ndcg_score))

        return results

    @staticmethod
    def print_table(metrics: list[MetricsAtK], title: str = "Retrieval Metrics") -> None:
        """Print a Rich table summarizing metrics across K values."""
        table = Table(title=title, show_header=True)
        table.add_column("K", justify="right", style="cyan")
        table.add_column("Recall@K", justify="right")
        table.add_column("Precision@K", justify="right")
        table.add_column("NDCG@K", justify="right")

        for m in metrics:
            table.add_row(
                str(m.k),
                f"{m.recall:.4f}",
                f"{m.precision:.4f}",
                f"{m.ndcg:.4f}",
            )

        Console().print(table)


def load_retrieval_dataset(path: Path) -> list[RetrievalSample]:
    """Load a golden retrieval dataset from a *path*.

    Expected format:
        [{"id": "...", "query": "...", "relevant_chunk_ids": ["c1", "c2"]}, ...]

    Returns samples with empty "retrieved_ids" — callers must fill these by
    running the retrieval pipeline before passing to "RetrievalEvaluator".
    """
    raw: list[object] = json.loads(path.read_text(encoding="utf-8"))
    samples: list[RetrievalSample] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        query_id = entry.get("id", "")
        raw_ids = entry.get("relevant_chunk_ids", [])
        samples.append(RetrievalSample(
            query_id=query_id if isinstance(query_id, str) else "",
            retrieved_ids=[],
            relevant_ids=[r for r in (raw_ids if isinstance(raw_ids, list) else [])
                          if isinstance(r, str)],
        ))
    return samples


# ── internal ───────────────────────────────────────────────────────────────────


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items) if items else 0.0
