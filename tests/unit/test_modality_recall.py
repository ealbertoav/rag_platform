"""T-281 — Modality retrieval metric unit tests (pure functions, no I/O)."""

from __future__ import annotations

import pytest

from src.core.constants import MODALITY_FIGURE, MODALITY_TABLE
from src.evals.retrieval.modality_recall import (
    ModalityRetrievalSample,
    figure_recall_at_k,
    load_modality_samples,
    modality_recall_at_k,
    table_recall_at_k,
)


def _sample(modality: str, retrieved: list[str], relevant: list[str]) -> ModalityRetrievalSample:
    return ModalityRetrievalSample(
        query_id="q", modality=modality, retrieved_ids=retrieved, relevant_ids=relevant
    )


# ── modality_recall_at_k / table_recall_at_k / figure_recall_at_k ──────────────


class TestModalityRecallAtK:
    def test_averages_only_matching_modality(self):
        samples = [
            _sample(MODALITY_TABLE, ["a", "b"], ["a"]),  # recall 1.0
            _sample(MODALITY_TABLE, ["x"], ["a"]),  # recall 0.0
            _sample(MODALITY_FIGURE, ["a"], ["a"]),  # ignored
        ]
        assert modality_recall_at_k(samples, MODALITY_TABLE, k=2) == pytest.approx(0.5)

    def test_no_matching_modality_returns_zero(self):
        samples = [_sample(MODALITY_FIGURE, ["a"], ["a"])]
        assert modality_recall_at_k(samples, MODALITY_TABLE, k=5) == pytest.approx(0.0)

    def test_empty_samples_returns_zero(self):
        assert modality_recall_at_k([], MODALITY_TABLE, k=5) == pytest.approx(0.0)

    def test_result_in_zero_one(self):
        samples = [_sample(MODALITY_TABLE, ["a", "b"], ["a", "c"])]
        score = modality_recall_at_k(samples, MODALITY_TABLE, k=2)
        assert 0.0 <= score <= 1.0


class TestTableAndFigureRecallAtK:
    def test_table_recall_uses_table_samples_only(self):
        samples = [
            _sample(MODALITY_TABLE, ["a"], ["a"]),
            _sample(MODALITY_FIGURE, [], ["b"]),
        ]
        assert table_recall_at_k(samples, k=3) == pytest.approx(1.0)

    def test_figure_recall_uses_figure_samples_only(self):
        samples = [
            _sample(MODALITY_TABLE, ["a"], ["a"]),
            _sample(MODALITY_FIGURE, [], ["b"]),
        ]
        assert figure_recall_at_k(samples, k=3) == pytest.approx(0.0)


# ── load_modality_samples ───────────────────────────────────────────────────────


class TestLoadModalitySamples:
    def test_builds_samples_with_empty_retrieved_ids(self):
        pairs: list[dict[str, object]] = [
            {"question": "Q1", "relevant_chunks": ["table0"], "modality": MODALITY_TABLE},
            {"question": "Q2", "relevant_chunks": ["figure0"], "modality": MODALITY_FIGURE},
        ]
        samples = load_modality_samples(pairs)
        assert [s.modality for s in samples] == [MODALITY_TABLE, MODALITY_FIGURE]
        assert all(s.retrieved_ids == [] for s in samples)
        assert samples[0].relevant_ids == ["table0"]

    def test_ignores_non_string_relevant_chunk_entries(self):
        pairs: list[dict[str, object]] = [
            {"relevant_chunks": ["a", 1, None, "b"], "modality": MODALITY_TABLE}
        ]
        samples = load_modality_samples(pairs)
        assert samples[0].relevant_ids == ["a", "b"]

    def test_missing_fields_default_gracefully(self):
        samples = load_modality_samples([{}])
        assert samples[0].modality == ""
        assert samples[0].relevant_ids == []

    def test_empty_input_returns_empty_list(self):
        assert load_modality_samples([]) == []
