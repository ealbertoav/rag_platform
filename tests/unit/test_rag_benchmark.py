"""T-043 — RAGBenchmark and BenchmarkReport unit tests."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.answer import Answer
from src.evals.e2e.rag_benchmark import BenchmarkReport, RAGBenchmark

# ── helpers ────────────────────────────────────────────────────────────────────


def _pipeline_mock(
    answer_text: str = "EKS is Amazon Kubernetes.",
    sources: list[str] | None = None,
    context: list[str] | None = None,
) -> MagicMock:
    m = MagicMock()
    answer = Answer(
        query_id="q1",
        text=answer_text,
        sources=sources if sources is not None else ["c0", "c1"],
    )
    m.benchmark = AsyncMock(return_value=(answer, context or ["Context chunk A."]))
    return m


def _qa(
    question: str = "What is EKS?",
    answer: str = "It is Kubernetes.",
    relevant: list[str] | None = None,
) -> dict[str, object]:
    return {
        "question": question,
        "answer": answer,
        "relevant_chunks": relevant or ["c0", "c1"],
    }


def _metric_mock(score: float) -> MagicMock:
    m = MagicMock()
    result = MagicMock()
    result.score = score
    m.score.return_value = result
    return m


def _faith_mock(score: float = 0.9) -> MagicMock:
    return _metric_mock(score)


def _relev_mock(score: float = 0.85) -> MagicMock:
    return _metric_mock(score)


def _ctx_mock(score: float = 0.8) -> MagicMock:
    return _metric_mock(score)


def _halluc_mock(score: float = 0.05) -> MagicMock:
    return _metric_mock(score)


def _benchmark(
    faith: float = 0.9,
    relev: float = 0.85,
    ctx: float = 0.8,
    halluc: float = 0.05,
    recall_thresh: float = 0.5,
    faith_thresh: float = 0.8,
    relev_thresh: float = 0.75,
    ctx_thresh: float = 0.7,
    halluc_thresh: float = 0.1,
) -> RAGBenchmark:
    return RAGBenchmark(
        faithfulness=_faith_mock(faith),
        relevance=_relev_mock(relev),
        context_precision=_ctx_mock(ctx),
        hallucination=_halluc_mock(halluc),
        recall_threshold=recall_thresh,
        faithfulness_threshold=faith_thresh,
        relevance_threshold=relev_thresh,
        context_precision_threshold=ctx_thresh,
        hallucination_threshold=halluc_thresh,
    )


# ── BenchmarkReport ────────────────────────────────────────────────────────────


class TestBenchmarkReport:
    @staticmethod
    def _report(passed: bool = True) -> BenchmarkReport:
        return BenchmarkReport(
            timestamp="20250101T000000",
            total_samples=2,
            mean_recall_at_5=0.8,
            mean_faithfulness=0.9,
            mean_relevance=0.85,
            mean_context_precision=0.8,
            mean_hallucination=0.05,
            recall_threshold=0.5,
            faithfulness_threshold=0.8,
            relevance_threshold=0.75,
            context_precision_threshold=0.7,
            hallucination_threshold=0.1,
            passed=passed,
        )

    def test_save_creates_file(self, tmp_path: Path):
        r = self._report()
        out = tmp_path / "report.json"
        r.save(out)
        assert out.exists()

    def test_save_valid_json(self, tmp_path: Path):
        r = self._report()
        out = tmp_path / "report.json"
        r.save(out)
        data = json.loads(out.read_text())
        assert data["total_samples"] == 2

    def test_summary_contains_pass(self):
        assert "PASSED" in self._report(passed=True).summary()

    def test_summary_contains_fail(self):
        assert "FAILED" in self._report(passed=False).summary()

    def test_summary_shows_metrics(self):
        s = self._report().summary()
        assert "Recall@5" in s
        assert "Faithfulness" in s
        assert "Relevance" in s
        assert "Context Precision" in s
        assert "Hallucination" in s


# ── RAGBenchmark.run ───────────────────────────────────────────────────────────


class TestRAGBenchmarkRun:
    @pytest.mark.asyncio
    async def test_returns_report(self):
        pipeline = _pipeline_mock()
        report = await _benchmark().run(pipeline, [_qa()], timestamp="T")
        assert isinstance(report, BenchmarkReport)

    @pytest.mark.asyncio
    async def test_total_samples(self):
        pipeline = _pipeline_mock()
        report = await _benchmark().run(pipeline, [_qa(), _qa()], timestamp="T")
        assert report.total_samples == 2

    @pytest.mark.asyncio
    async def test_empty_qa_returns_empty(self):
        pipeline = _pipeline_mock()
        report = await _benchmark().run(pipeline, [], timestamp="T")
        assert report.total_samples == 0

    @pytest.mark.asyncio
    async def test_perfect_recall_when_sources_match(self):
        pipeline = _pipeline_mock(sources=["c0", "c1"])
        report = await _benchmark().run(pipeline, [_qa(relevant=["c0", "c1"])], timestamp="T")
        assert report.mean_recall_at_5 == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_zero_recall_no_overlap(self):
        pipeline = _pipeline_mock(sources=["x", "y"])
        report = await _benchmark().run(pipeline, [_qa(relevant=["c0", "c1"])], timestamp="T")
        assert report.mean_recall_at_5 == pytest.approx(0.0)

    @pytest.mark.asyncio
    async def test_faithfulness_from_metric(self):
        pipeline = _pipeline_mock()
        report = await _benchmark(faith=0.95).run(pipeline, [_qa()], timestamp="T")
        assert report.mean_faithfulness == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_relevance_from_metric(self):
        pipeline = _pipeline_mock()
        report = await _benchmark(relev=0.88).run(pipeline, [_qa()], timestamp="T")
        assert report.mean_relevance == pytest.approx(0.88)

    @pytest.mark.asyncio
    async def test_hallucination_from_metric(self):
        pipeline = _pipeline_mock()
        report = await _benchmark(halluc=0.03).run(pipeline, [_qa()], timestamp="T")
        assert report.mean_hallucination == pytest.approx(0.03)

    @pytest.mark.asyncio
    async def test_passed_when_all_above_threshold(self):
        pipeline = _pipeline_mock(sources=["c0", "c1"])
        report = await _benchmark(faith=0.9, relev=0.85).run(
            pipeline, [_qa(relevant=["c0"])], timestamp="T"
        )
        assert report.passed is True

    @pytest.mark.asyncio
    async def test_failed_when_any_below_threshold(self):
        pipeline = _pipeline_mock(sources=[])  # zero recall
        report = await _benchmark(recall_thresh=0.5).run(
            pipeline, [_qa(relevant=["c0"])], timestamp="T"
        )
        assert report.passed is False

    @pytest.mark.asyncio
    async def test_failed_when_hallucination_above_threshold(self):
        pipeline = _pipeline_mock(sources=["c0", "c1"])
        report = await _benchmark(halluc=0.2, halluc_thresh=0.1).run(
            pipeline, [_qa(relevant=["c0"])], timestamp="T"
        )
        assert report.passed is False

    @pytest.mark.asyncio
    async def test_pipeline_failure_recorded_as_zero(self):
        pipeline = MagicMock()
        pipeline.benchmark = AsyncMock(side_effect=RuntimeError("LLM down"))
        report = await _benchmark().run(pipeline, [_qa()], timestamp="T")
        assert report.per_sample[0].generated_answer == ""

    @pytest.mark.asyncio
    async def test_per_sample_populated(self):
        pipeline = _pipeline_mock()
        report = await _benchmark().run(pipeline, [_qa("q?", "a.")], timestamp="T")
        assert len(report.per_sample) == 1
        assert report.per_sample[0].question == "q?"

    @pytest.mark.asyncio
    async def test_skips_empty_question(self):
        pipeline = _pipeline_mock()
        pairs = [_qa(), {"question": "", "answer": "a", "relevant_chunks": []}]
        report = await _benchmark().run(pipeline, pairs, timestamp="T")
        assert report.total_samples == 1
