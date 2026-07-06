"""T-041 / T-152 benchmark tests — Retrieval Evals against the golden dataset.

Run with:
    uv run pytest tests/benchmarks/test_retrieval_evals.py -v -s

The tests are skipped when the golden retrieval dataset is empty or contains only
placeholder rows (default state before documents are ingested and evals are run).
"""

from __future__ import annotations

import pytest

from src.core.constants import DATASETS_DIR
from src.evals.golden_dataset import (
    MIN_QA_PAIRS,
    count_real_qa_pairs,
    is_placeholder_retrieval_row,
    load_qa_dicts,
    retrieval_rows_match_qa,
)
from src.evals.regression_gate import load_real_retrieval_rows, load_regression_baseline
from src.evals.retrieval import (
    RetrievalEvaluator,
    RetrievalSample,
    load_retrieval_dataset,
    recall_at_k,
)

_GOLDEN_PATH = DATASETS_DIR / "goldens" / "retrieval_dataset.json"
_BASELINE_PATH = DATASETS_DIR / "goldens" / "retrieval_baseline.json"
_QA_PATH = DATASETS_DIR / "goldens" / "qa_dataset.json"


def _load_baseline() -> dict[str, object]:
    return load_regression_baseline(_BASELINE_PATH)


def _load_real_samples() -> list[RetrievalSample]:
    """Load golden dataset; returns an empty list if only placeholder rows are present."""
    if not load_real_retrieval_rows(_GOLDEN_PATH):
        return []
    try:
        samples = load_retrieval_dataset(_GOLDEN_PATH)
        return [
            s
            for s in samples
            if s.relevant_ids
            and not is_placeholder_retrieval_row({"relevant_chunk_ids": s.relevant_ids})
        ]
    except (OSError, ValueError, KeyError):
        return []


_SAMPLES = _load_real_samples()
_BASELINE = _load_baseline()

pytestmark = pytest.mark.skipif(
    len(_SAMPLES) == 0,
    reason="Golden retrieval dataset is empty — populate via `make evals` first.",
)


@pytest.fixture(scope="module")
def evaluator() -> RetrievalEvaluator:
    return RetrievalEvaluator(k_values=[1, 3, 5, 10])


class TestRetrievalBenchmark:
    def test_dataset_loaded(self):
        assert len(_SAMPLES) > 0

    def test_minimum_sample_count(self):
        min_samples = _BASELINE.get("min_samples", MIN_QA_PAIRS)
        assert isinstance(min_samples, int)
        assert len(_SAMPLES) >= min_samples
        assert count_real_qa_pairs(_QA_PATH) >= min_samples

    def test_retrieval_matches_qa_golden(self):
        qa_pairs = load_qa_dicts(_QA_PATH)
        real_rows = load_real_retrieval_rows(_GOLDEN_PATH)
        assert retrieval_rows_match_qa(qa_pairs, real_rows)

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

    def test_oracle_recall_meets_baseline(self, evaluator):
        """Ground-truth retrieval (retrieved=relevant) must meet the committed baseline."""
        oracle_samples = [
            RetrievalSample(
                query_id=s.query_id,
                retrieved_ids=list(s.relevant_ids),
                relevant_ids=list(s.relevant_ids),
            )
            for s in _SAMPLES
        ]
        results = evaluator.evaluate(oracle_samples)
        recall_at_5 = next(m.recall for m in results if m.k == 5)
        expected = _BASELINE.get("oracle_recall_at_5", 1.0)
        assert isinstance(expected, (int, float))
        assert recall_at_5 >= float(expected) - 1e-9

    def test_recall_at_5_above_regression_threshold(self, evaluator):
        """Per-sample oracle Recall@5 must clear the CI regression floor."""
        min_recall = _BASELINE.get("min_recall_at_5", 0.5)
        assert isinstance(min_recall, (int, float))
        for sample in _SAMPLES:
            score = recall_at_k(sample.relevant_ids, sample.relevant_ids, k=5)
            assert score >= float(min_recall)

    def test_print_summary_table(self, evaluator, capsys):
        results = evaluator.evaluate(_SAMPLES)
        evaluator.print_table(results, title="Retrieval Benchmark")
