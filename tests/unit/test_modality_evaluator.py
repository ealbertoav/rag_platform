"""T-281 — ModalityRetrievalEvaluator unit tests (reporting layer)."""

from __future__ import annotations

import pytest

from src.core.constants import MODALITY_FIGURE, MODALITY_TABLE
from src.evals.retrieval.modality_evaluator import ModalityMetricsAtK, ModalityRetrievalEvaluator
from src.evals.retrieval.modality_recall import MODALITIES, ModalityRetrievalSample


def _sample(modality: str, retrieved: list[str], relevant: list[str]) -> ModalityRetrievalSample:
    return ModalityRetrievalSample(
        query_id="q", modality=modality, retrieved_ids=retrieved, relevant_ids=relevant
    )


class TestModalityRetrievalEvaluator:
    def test_evaluate_covers_all_modalities_and_k_values(self):
        samples = [
            _sample(MODALITY_TABLE, ["a"], ["a"]),
            _sample(MODALITY_FIGURE, [], ["b"]),
        ]
        evaluator = ModalityRetrievalEvaluator(k_values=[1, 5])
        metrics = evaluator.evaluate(samples)
        assert len(metrics) == len(MODALITIES) * 2
        by_modality_k = {(m.modality, m.k): m.recall for m in metrics}
        assert by_modality_k[(MODALITY_TABLE, 1)] == pytest.approx(1.0)
        assert by_modality_k[(MODALITY_FIGURE, 1)] == pytest.approx(0.0)

    def test_default_k_values(self):
        evaluator = ModalityRetrievalEvaluator()
        assert evaluator.k_values == [1, 3, 5, 10]

    def test_evaluate_empty_samples_returns_empty_list(self):
        assert ModalityRetrievalEvaluator().evaluate([]) == []

    def test_print_table_smoke(self, capsys: pytest.CaptureFixture[str]):
        metrics = [ModalityMetricsAtK(modality=MODALITY_TABLE, k=5, recall=0.5)]
        ModalityRetrievalEvaluator.print_table(metrics)
        captured = capsys.readouterr()
        assert "table" in captured.out
