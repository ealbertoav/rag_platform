"""Unit tests for benchmark_samples helpers."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.answer import Answer
from src.domain.entities.evaluation import BenchmarkRun
from src.evals.e2e.benchmark_samples import (
    GenerationMetricAccumulator,
    GenerationMetricScores,
    failure_generation_scores,
    pair_str,
    pair_str_list,
    pipeline_error_logger,
    score_generation_sample,
    score_pipeline_question,
)


def _metric(score: float) -> MagicMock:
    mock = MagicMock()
    mock.score.return_value = MagicMock(score=score)
    return mock


class TestPairHelpers:
    def test_pair_str(self):
        assert pair_str("hello") == "hello"
        assert pair_str(1) == ""

    def test_pair_str_list(self):
        assert pair_str_list(["a", 1, "b"]) == ["a", "b"]
        assert pair_str_list("nope") == []


class TestFailureGenerationScores:
    def test_returns_zeros_with_elapsed_latency(self):
        scores = failure_generation_scores(started_at=0.0)
        assert scores.recall_at_k == 0.0
        assert scores.faithfulness == 0.0
        assert scores.relevance == 0.0
        assert scores.latency_ms >= 0.0


class TestPipelineErrorLogger:
    def test_logs_bound_context_and_exception(self):
        logged: list[tuple[str, tuple[object, ...]]] = []

        def capture(message: str, *args: object) -> None:
            logged.append((message, args))

        handler = pipeline_error_logger(capture, "failed %s %r %s", "tech", "question")
        error = RuntimeError("boom")
        handler(error)
        assert logged == [("failed %s %r %s", ("tech", "question", error))]


class TestScoreGenerationSample:
    def test_scores_with_answer_latency(self):
        answer = Answer(query_id="q", text="Generated", sources=["c0"], latency_ms=25.0)
        run = BenchmarkRun(answer=answer, context_texts=["ctx"])
        scores = score_generation_sample(
            run=run,
            question="Q?",
            expected_answer="A",
            relevant_ids=["c0"],
            recall_k=5,
            faithfulness=_metric(0.9),
            relevance=_metric(0.8),
            started_at=0.0,
        )
        assert scores.recall_at_k == 1.0
        assert scores.faithfulness == 0.9
        assert scores.relevance == 0.8
        assert scores.latency_ms == 25.0

    def test_scores_with_elapsed_when_latency_missing(self):
        answer = MagicMock()
        answer.text = "Generated"
        answer.sources = ["c0"]
        answer.latency_ms = None
        run = BenchmarkRun(answer=answer, context_texts=["ctx"])
        scores = score_generation_sample(
            run=run,
            question="Q?",
            expected_answer="A",
            relevant_ids=["c0"],
            recall_k=5,
            faithfulness=_metric(0.7),
            relevance=_metric(0.6),
            started_at=0.0,
        )
        assert scores.latency_ms >= 0.0


class TestGenerationMetricAccumulator:
    def test_append_and_means(self):
        accumulator = GenerationMetricAccumulator()
        accumulator.append(
            GenerationMetricScores(
                recall_at_k=0.0,
                faithfulness=0.0,
                relevance=0.0,
                latency_ms=20.0,
            )
        )
        accumulator.append(
            GenerationMetricScores(
                recall_at_k=1.0,
                faithfulness=0.8,
                relevance=0.7,
                latency_ms=10.0,
            )
        )
        assert accumulator.total_samples == 2
        means = accumulator.means()
        assert means.mean_recall_at_5 == pytest.approx(0.5)
        assert means.mean_faithfulness == pytest.approx(0.4)
        assert means.mean_relevance == pytest.approx(0.35)
        assert means.mean_latency_ms == pytest.approx(15.0)


class TestScorePipelineQuestion:
    @pytest.mark.asyncio
    async def test_success_path(self):
        pipeline = MagicMock()
        pipeline.benchmark = AsyncMock(
            return_value=BenchmarkRun(
                answer=Answer(query_id="q", text="A", sources=["c0"], latency_ms=12.0),
                context_texts=["ctx"],
            )
        )
        scores = await score_pipeline_question(
            pipeline=pipeline,
            question="Q?",
            expected_answer="A",
            relevant_ids=["c0"],
            recall_k=5,
            faithfulness=_metric(0.9),
            relevance=_metric(0.8),
        )
        assert scores.recall_at_k == 1.0
        assert scores.faithfulness == 0.9

    @pytest.mark.asyncio
    async def test_failure_path(self):
        pipeline = MagicMock()
        pipeline.benchmark = AsyncMock(side_effect=RuntimeError("down"))
        errors: list[Exception] = []
        scores = await score_pipeline_question(
            pipeline=pipeline,
            question="Q?",
            expected_answer="A",
            relevant_ids=["c0"],
            recall_k=5,
            faithfulness=_metric(0.9),
            relevance=_metric(0.8),
            on_pipeline_error=errors.append,
        )
        assert isinstance(errors[0], RuntimeError)
        assert scores.recall_at_k == 0.0
