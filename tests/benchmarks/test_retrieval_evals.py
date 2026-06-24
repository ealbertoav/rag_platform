"""T-041 benchmark tests — Retrieval Evals against the golden dataset.

Run with:
    uv run pytest tests/benchmarks/test_retrieval_evals.py -v -s

The tests are skipped when the golden retrieval dataset is empty (default state
before documents are ingested and the dataset is populated).
"""

from __future__ import annotations

import pytest

from src.core.constants import DATASETS_DIR
from src.evals.retrieval import (
    RetrievalEvaluator,
    RetrievalSample,
    load_retrieval_dataset,
)

_GOLDEN_PATH = DATASETS_DIR / "goldens" / "retrieval_dataset.json"


def _load_samples() -> list[RetrievalSample]:
    """Load golden dataset; returns an empty list if the file has only the placeholder."""
    try:
        samples = load_retrieval_dataset(_GOLDEN_PATH)
        # Skip if only the example placeholder row is present
        # Skip placeholder rows (relevant_ids start with "chunk_id_")
        return [s for s in samples if not all(r.startswith("chunk_id_") for r in s.relevant_ids)]
    except (OSError, ValueError, KeyError):
        return []


_SAMPLES = _load_samples()

pytestmark = pytest.mark.skipif(
    len(_SAMPLES) == 0,
    reason="Golden retrieval dataset is empty — populate via T-040 first.",
)


@pytest.fixture(scope="module")
def evaluator() -> RetrievalEvaluator:
    return RetrievalEvaluator(k_values=[1, 3, 5, 10])


class TestRetrievalBenchmark:
    def test_dataset_loaded(self):
        assert len(_SAMPLES) > 0

    def test_evaluate_returns_metrics_for_each_k(self, evaluator):
        results = evaluator.evaluate(_SAMPLES)
        assert len(results) == len(evaluator.k_values)

    def test_metrics_in_valid_range(self, evaluator):
        results = evaluator.evaluate(_SAMPLES)
        for m in results:
            assert 0.0 <= m.recall <= 1.0
            assert 0.0 <= m.precision <= 1.0
            assert 0.0 <= m.ndcg <= 1.0

    def test_recall_non_decreasing_with_k(self, evaluator):
        results = evaluator.evaluate(_SAMPLES)
        recalls = [m.recall for m in results]
        for a, b in zip(recalls[:-1], recalls[1:], strict=False):
            assert a <= b + 1e-9  # non-decreasing

    def test_print_summary_table(self, evaluator, capsys):
        results = evaluator.evaluate(_SAMPLES)
        evaluator.print_table(results, title="Retrieval Benchmark")
