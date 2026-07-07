"""T-041 — Retrieval metric unit tests (pure functions, no I/O)."""

from __future__ import annotations

import json
import math

import pytest

from src.evals.retrieval import MetricsAtK, RetrievalEvaluator, RetrievalSample
from src.evals.retrieval.mrr import mrr
from src.evals.retrieval.ndcg import ndcg_at_k
from src.evals.retrieval.precision_at_k import precision_at_k
from src.evals.retrieval.recall_at_k import oracle_recall_at_k, recall_at_k

# ── recall_at_k ────────────────────────────────────────────────────────────────


class TestRecallAtK:
    def test_perfect_recall(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_zero_recall_no_overlap(self):
        assert recall_at_k(["x", "y"], ["a", "b"], k=2) == pytest.approx(0.0)

    def test_partial_recall(self):
        assert recall_at_k(["a", "x", "y"], ["a", "b"], k=3) == pytest.approx(0.5)

    def test_k_limits_retrieved(self):
        # "b" is rank 3 — excluded when k=2
        assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_empty_relevant_returns_zero(self):
        assert recall_at_k(["a", "b"], [], k=3) == pytest.approx(0.0)

    def test_empty_retrieved_returns_zero(self):
        assert recall_at_k([], ["a", "b"], k=5) == pytest.approx(0.0)

    def test_k_zero_returns_zero(self):
        assert recall_at_k(["a", "b"], ["a", "b"], k=0) == pytest.approx(0.0)

    def test_k_larger_than_retrieved(self):
        assert recall_at_k(["a"], ["a", "b"], k=10) == pytest.approx(0.5)

    def test_result_in_zero_one(self):
        score = recall_at_k(["a", "b", "c"], ["a", "b", "d"], k=5)
        assert 0.0 <= score <= 1.0


class TestOracleRecallAtK:
    def test_perfect_oracle_single_chunk(self):
        assert oracle_recall_at_k(["a"], k=5) == pytest.approx(1.0)

    def test_perfect_oracle_multi_chunk_within_k(self):
        assert oracle_recall_at_k(["a", "b", "c"], k=5) == pytest.approx(1.0)

    def test_perfect_oracle_multi_chunk_exceeds_k(self):
        ids = [f"c{i}" for i in range(12)]
        assert oracle_recall_at_k(ids, k=5) == pytest.approx(1.0)

    def test_oracle_differs_from_standard_recall_when_many_relevant(self):
        ids = [f"c{i}" for i in range(12)]
        assert recall_at_k(ids, ids, k=5) == pytest.approx(5 / 12)
        assert oracle_recall_at_k(ids, k=5) == pytest.approx(1.0)

    def test_empty_relevant_returns_zero(self):
        assert oracle_recall_at_k([], k=5) == pytest.approx(0.0)

    def test_k_zero_returns_zero(self):
        assert oracle_recall_at_k(["a", "b"], k=0) == pytest.approx(0.0)


# ── precision_at_k ─────────────────────────────────────────────────────────────


class TestPrecisionAtK:
    def test_perfect_precision(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], k=2) == pytest.approx(1.0)

    def test_zero_precision(self):
        assert precision_at_k(["x", "y"], ["a", "b"], k=2) == pytest.approx(0.0)

    def test_partial_precision(self):
        # 1 out of 3 retrieved are relevant
        assert precision_at_k(["a", "x", "y"], ["a", "b"], k=3) == pytest.approx(1 / 3)

    def test_k_limits_to_top_k(self):
        # Only first 2 evaluated — "a" is relevant (1/2)
        assert precision_at_k(["a", "x", "b"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_k_zero_returns_zero(self):
        assert precision_at_k(["a"], ["a"], k=0) == pytest.approx(0.0)

    def test_empty_retrieved(self):
        assert precision_at_k([], ["a"], k=3) == pytest.approx(0.0)

    def test_result_in_zero_one(self):
        score = precision_at_k(["a", "b", "c"], ["a", "b"], k=5)
        assert 0.0 <= score <= 1.0


# ── ndcg_at_k ──────────────────────────────────────────────────────────────────


class TestNdcgAtK:
    def test_perfect_ndcg(self):
        # All relevant docs first
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_zero_ndcg_no_overlap(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == pytest.approx(0.0)

    def test_relevant_at_rank1_vs_rank2(self):
        # Rank 1 hit beats rank 2 hit (higher DCG)
        score_first = ndcg_at_k(["a", "x"], ["a"], k=2)
        score_second = ndcg_at_k(["x", "a"], ["a"], k=2)
        assert score_first > score_second

    def test_empty_relevant_returns_zero(self):
        assert ndcg_at_k(["a", "b"], [], k=5) == pytest.approx(0.0)

    def test_k_zero_returns_zero(self):
        assert ndcg_at_k(["a"], ["a"], k=0) == pytest.approx(0.0)

    def test_single_relevant_at_rank1(self):
        # DCG = 1/log2(2) = 1; IDCG = 1 → NDCG = 1
        assert ndcg_at_k(["a", "x", "y"], ["a"], k=3) == pytest.approx(1.0)

    def test_single_relevant_at_rank2(self):
        # DCG = 1/log2(3); IDCG = 1/log2(2)
        expected = (1.0 / math.log2(3)) / (1.0 / math.log2(2))
        assert ndcg_at_k(["x", "a", "y"], ["a"], k=3) == pytest.approx(expected)

    def test_result_in_zero_one(self):
        score = ndcg_at_k(["a", "b", "c", "d"], ["a", "c", "e"], k=4)
        assert 0.0 <= score <= 1.0


# ── RetrievalEvaluator ────────────────────────────────────────────────────────


class TestRetrievalEvaluator:
    @staticmethod
    def _perfect_sample() -> RetrievalSample:
        return RetrievalSample(
            query_id="q1",
            retrieved_ids=["a", "b", "c"],
            relevant_ids=["a", "b", "c"],
        )

    def test_returns_metric_per_k(self):
        ev = RetrievalEvaluator(k_values=[1, 3])
        results = ev.evaluate([self._perfect_sample()])
        assert len(results) == 2
        assert all(isinstance(m, MetricsAtK) for m in results)

    def test_k_values_in_results(self):
        ev = RetrievalEvaluator(k_values=[1, 5, 10])
        results = ev.evaluate([self._perfect_sample()])
        assert [m.k for m in results] == [1, 5, 10]

    def test_perfect_retrieval_score_1(self):
        ev = RetrievalEvaluator(k_values=[3])
        m = ev.evaluate([self._perfect_sample()])[0]
        assert m.recall == pytest.approx(1.0)
        assert m.precision == pytest.approx(1.0)
        assert m.ndcg == pytest.approx(1.0)

    def test_empty_samples_returns_empty(self):
        assert RetrievalEvaluator().evaluate([]) == []

    def test_averages_across_samples(self):
        samples = [
            RetrievalSample("q1", ["a", "x"], ["a"]),  # perfect
            RetrievalSample("q2", ["x", "y"], ["a"]),  # zero
        ]
        ev = RetrievalEvaluator(k_values=[2])
        m = ev.evaluate(samples)[0]
        assert m.recall == pytest.approx(0.5)
        assert m.precision == pytest.approx(0.25)

    def test_print_table_no_error(self, capsys):
        ev = RetrievalEvaluator(k_values=[1, 3])
        metrics = ev.evaluate([self._perfect_sample()])
        ev.print_table(metrics)  # must not raise


# ── mrr ────────────────────────────────────────────────────────────────────────


class TestMRR:
    def test_first_hit_at_rank1(self):
        assert mrr(["a", "b", "c"], ["a"]) == pytest.approx(1.0)

    def test_first_hit_at_rank2(self):
        assert mrr(["x", "a", "c"], ["a"]) == pytest.approx(0.5)

    def test_first_hit_at_rank3(self):
        assert mrr(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_no_hit_returns_zero(self):
        assert mrr(["x", "y"], ["a", "b"]) == pytest.approx(0.0)

    def test_empty_retrieved_returns_zero(self):
        assert mrr([], ["a"]) == pytest.approx(0.0)

    def test_empty_relevant_returns_zero(self):
        assert mrr(["a", "b"], []) == pytest.approx(0.0)

    def test_multiple_relevant_uses_first_found(self):
        # "b" at rank 2 is hit before "c" at rank 3 — MRR = 1/2
        assert mrr(["x", "b", "c"], ["b", "c"]) == pytest.approx(0.5)

    def test_result_in_zero_one(self):
        score = mrr(["a", "b", "c"], ["b"])
        assert 0.0 <= score <= 1.0


# ── load_retrieval_dataset ─────────────────────────────────────────────────────


class TestLoadRetrievalDataset:
    def test_loads_valid_entries(self, tmp_path):
        from src.evals.retrieval import load_retrieval_dataset

        path = tmp_path / "retrieval.json"
        path.write_text(
            json.dumps(
                [
                    {"id": "q1", "query": "What is EKS?", "relevant_chunk_ids": ["c1", "c2"]},
                    {"id": "q2", "query": "IAM?", "relevant_chunk_ids": ["c3"]},
                ]
            ),
            encoding="utf-8",
        )
        samples = load_retrieval_dataset(path)
        assert len(samples) == 2
        assert samples[0].query_id == "q1"
        assert samples[0].relevant_ids == ["c1", "c2"]
        assert samples[0].retrieved_ids == []

    def test_skips_non_dict_entries(self, tmp_path):
        from src.evals.retrieval import load_retrieval_dataset

        path = tmp_path / "retrieval.json"
        path.write_text(json.dumps(["not-a-dict", {"id": "q1", "relevant_chunk_ids": []}]))
        samples = load_retrieval_dataset(path)
        assert len(samples) == 1

    def test_coerces_bad_relevant_ids(self, tmp_path):
        from src.evals.retrieval import load_retrieval_dataset

        path = tmp_path / "retrieval.json"
        path.write_text(json.dumps([{"id": 123, "relevant_chunk_ids": [1, "c2", None]}]))
        samples = load_retrieval_dataset(path)
        assert samples[0].query_id == ""
        assert samples[0].relevant_ids == ["c2"]
