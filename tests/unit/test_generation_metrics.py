"""T-042 — Generation metric unit tests (Ragas/DeepEval mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult, GenerationMetric
from src.evals.generation.context_precision import ContextPrecisionMetric
from src.evals.generation.faithfulness import FaithfulnessMetric
from src.evals.generation.hallucination import HallucinationMetric
from src.evals.generation.relevance import RelevanceMetric

# Patch paths — extracted to avoid long lines in each test.
_FAITH = "src.evals.generation.faithfulness.FaithfulnessMetric._ragas_score"
_RELEV = "src.evals.generation.relevance.RelevanceMetric._ragas_score"
_HALLU = "src.evals.generation.hallucination.HallucinationMetric._deepeval_score"

# ── helpers ────────────────────────────────────────────────────────────────────


def _sample(
    question: str = "What is EKS?",
    expected: str = "Amazon EKS is a managed Kubernetes service.",
    generated: str = "EKS is a managed Kubernetes service from AWS.",
    chunks: list[str] | None = None,
) -> EvalSample:
    # `chunks is None` (not `not chunks`) so callers can pass chunks=[] to test
    # the empty-context guard paths without it falling back to the default.
    return EvalSample(
        question=question,
        expected_answer=expected,
        retrieved_chunks=["EKS is Amazon Elastic Kubernetes Service."]
        if chunks is None
        else chunks,
        generated_answer=generated,
    )


# ── EvalResult ─────────────────────────────────────────────────────────────────


class TestEvalResult:
    def test_higher_is_better_passes_above_threshold(self):
        r = EvalResult.make("f", 0.9, threshold=0.8)
        assert r.passed is True

    def test_higher_is_better_fails_at_threshold(self):
        r = EvalResult.make("f", 0.8, threshold=0.8)
        assert r.passed is False

    def test_lower_is_better_passes_below_threshold(self):
        r = EvalResult.make("h", 0.05, threshold=0.1, higher_is_better=False)
        assert r.passed is True

    def test_lower_is_better_fails_above_threshold(self):
        r = EvalResult.make("h", 0.2, threshold=0.1, higher_is_better=False)
        assert r.passed is False

    def test_fields_set(self):
        r = EvalResult.make("faith", 0.85, threshold=0.8, details="all good")
        assert r.metric == "faith"
        assert r.score == pytest.approx(0.85)
        assert r.details == "all good"


# ── GenerationMetric protocol ──────────────────────────────────────────────────


class TestProtocol:
    def test_faithfulness_satisfies_protocol(self):
        m: GenerationMetric = FaithfulnessMetric()
        assert callable(m.score)

    def test_relevance_satisfies_protocol(self):
        m: GenerationMetric = RelevanceMetric()
        assert callable(m.score)

    def test_hallucination_satisfies_protocol(self):
        m: GenerationMetric = HallucinationMetric()
        assert callable(m.score)


# ── FaithfulnessMetric ─────────────────────────────────────────────────────────


class TestFaithfulnessMetric:
    def test_returns_eval_result(self):
        with patch(_FAITH, return_value=0.9):
            result = FaithfulnessMetric().score(_sample())
        assert isinstance(result, EvalResult)

    def test_passes_above_threshold(self):
        with patch(_FAITH, return_value=0.95):
            result = FaithfulnessMetric(threshold=0.8).score(_sample())
        assert result.passed is True

    def test_fails_below_threshold(self):
        with patch(_FAITH, return_value=0.6):
            result = FaithfulnessMetric(threshold=0.8).score(_sample())
        assert result.passed is False

    def test_empty_answer_returns_zero(self):
        result = FaithfulnessMetric().score(_sample(generated=""))
        assert result.score == pytest.approx(0.0)
        assert result.passed is False

    def test_no_context_returns_zero(self):
        result = FaithfulnessMetric().score(_sample(chunks=[]))
        assert result.score == pytest.approx(0.0)

    def test_parametric_answer_skips_context_guard(self):
        sample = _sample(chunks=[], generated="Hello!")
        sample = EvalSample(
            question=sample.question,
            expected_answer=sample.expected_answer,
            retrieved_chunks=[],
            generated_answer=sample.generated_answer,
            parametric_answer=True,
        )
        result = FaithfulnessMetric().score(sample)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert "Parametric answer" in result.details

    def test_ragas_failure_returns_zero(self):
        with patch(_FAITH, side_effect=RuntimeError("API error")):
            result = FaithfulnessMetric().score(_sample())
        assert result.score == pytest.approx(0.0)
        assert result.passed is False
        assert result.details != ""

    def test_metric_name(self):
        with patch(_FAITH, return_value=0.9):
            result = FaithfulnessMetric().score(_sample())
        assert result.metric == "faithfulness"


# ── RelevanceMetric ────────────────────────────────────────────────────────────


class TestRelevanceMetric:
    def test_passes_above_threshold(self):
        with patch(_RELEV, return_value=0.85):
            result = RelevanceMetric(threshold=0.75).score(_sample())
        assert result.passed is True

    def test_fails_below_threshold(self):
        with patch(_RELEV, return_value=0.5):
            result = RelevanceMetric(threshold=0.75).score(_sample())
        assert result.passed is False

    def test_empty_answer_returns_zero(self):
        result = RelevanceMetric().score(_sample(generated=""))
        assert result.score == pytest.approx(0.0)

    def test_ragas_failure_returns_zero(self):
        with patch(_RELEV, side_effect=ImportError("ragas not installed")):
            result = RelevanceMetric().score(_sample())
        assert result.score == pytest.approx(0.0)

    def test_metric_name(self):
        with patch(_RELEV, return_value=0.8):
            result = RelevanceMetric().score(_sample())
        assert result.metric == "answer_relevancy"


# ── HallucinationMetric ────────────────────────────────────────────────────────


class TestHallucinationMetric:
    def test_passes_below_threshold(self):
        with patch(_HALLU, return_value=0.05):
            result = HallucinationMetric(threshold=0.1).score(_sample())
        assert result.passed is True

    def test_fails_above_threshold(self):
        with patch(_HALLU, return_value=0.4):
            result = HallucinationMetric(threshold=0.1).score(_sample())
        assert result.passed is False

    def test_empty_answer_passes(self):
        result = HallucinationMetric().score(_sample(generated=""))
        assert result.score == pytest.approx(0.0)
        assert result.passed is True

    def test_no_context_is_neutral_not_worst_case(self):
        """#91 — empty retrieved_chunks (e.g. CRAG web-only fallback) is not evaluable,
        not a hallucination; must not score as if maximally hallucinated."""
        result = HallucinationMetric().score(_sample(chunks=[]))
        assert result.score == pytest.approx(0.0)
        assert result.passed is True
        assert result.details == "No context to verify against"

    def test_parametric_answer_skips_context_penalty(self):
        sample = EvalSample(
            question="hello",
            expected_answer="hi",
            retrieved_chunks=[],
            generated_answer="Hello!",
            parametric_answer=True,
        )
        result = HallucinationMetric().score(sample)
        assert result.score == pytest.approx(0.0)
        assert result.passed is True

    def test_deepeval_failure_returns_one(self):
        with patch(_HALLU, side_effect=ImportError("deepeval not installed")):
            result = HallucinationMetric().score(_sample())
        assert result.score == pytest.approx(1.0)
        assert result.passed is False

    def test_metric_name(self):
        with patch(_HALLU, return_value=0.05):
            result = HallucinationMetric().score(_sample())
        assert result.metric == "hallucination"


# ── ContextPrecisionMetric ────────────────────────────────────────────────────

_CTX_PREC = "src.evals.generation.context_precision.ContextPrecisionMetric._ragas_score"


class TestContextPrecisionMetric:
    def test_satisfies_protocol(self):
        m: GenerationMetric = ContextPrecisionMetric()
        assert callable(m.score)

    def test_returns_eval_result(self):
        with patch(_CTX_PREC, return_value=0.8):
            result = ContextPrecisionMetric().score(_sample())
        assert isinstance(result, EvalResult)

    def test_passes_above_threshold(self):
        with patch(_CTX_PREC, return_value=0.85):
            result = ContextPrecisionMetric(threshold=0.7).score(_sample())
        assert result.passed is True

    def test_fails_below_threshold(self):
        with patch(_CTX_PREC, return_value=0.5):
            result = ContextPrecisionMetric(threshold=0.7).score(_sample())
        assert result.passed is False

    def test_no_context_returns_zero(self):
        result = ContextPrecisionMetric().score(_sample(chunks=[]))
        assert result.score == pytest.approx(0.0)
        assert result.passed is False

    def test_parametric_answer_skips_context_guard(self):
        sample = EvalSample(
            question="hello",
            expected_answer="hi",
            retrieved_chunks=[],
            generated_answer="Hello!",
            parametric_answer=True,
        )
        result = ContextPrecisionMetric().score(sample)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True

    def test_empty_question_returns_zero(self):
        result = ContextPrecisionMetric().score(_sample(question=""))
        assert result.score == pytest.approx(0.0)

    def test_ragas_failure_returns_zero(self):
        with patch(_CTX_PREC, side_effect=ImportError("ragas not installed")):
            result = ContextPrecisionMetric().score(_sample())
        assert result.score == pytest.approx(0.0)
        assert result.details != ""

    def test_metric_name(self):
        with patch(_CTX_PREC, return_value=0.75):
            result = ContextPrecisionMetric().score(_sample())
        assert result.metric == "context_precision"


# ── Ragas infrastructure ───────────────────────────────────────────────────────


class TestRagasInfrastructure:
    def test_parametric_eval_result(self):
        from src.evals.generation import parametric_eval_result

        result = parametric_eval_result("faithfulness", threshold=0.8)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert "Parametric answer" in result.details

        low = parametric_eval_result("hallucination", threshold=0.1, higher_is_better=False)
        assert low.score == pytest.approx(0.0)
        assert low.passed is True

    def test_make_ragas_dataset(self):
        from src.evals.generation import _make_ragas_dataset

        mock_from_dict = MagicMock()
        mock_dataset = MagicMock(from_dict=mock_from_dict)
        with patch.dict("sys.modules", {"datasets": MagicMock(Dataset=mock_dataset)}):
            _make_ragas_dataset(_sample())

        mock_from_dict.assert_called_once()
        call_kwargs = mock_from_dict.call_args[0][0]
        assert call_kwargs["question"] == [_sample().question]

    def test_get_ragas_metric_imports_faithfulness(self):
        fake_metrics = MagicMock()
        fake_metrics.faithfulness = MagicMock(name="faithfulness")
        fake_ragas = MagicMock(metrics=fake_metrics)
        with patch.dict("sys.modules", {"ragas": fake_ragas, "ragas.metrics": fake_metrics}):
            metric = FaithfulnessMetric()._get_ragas_metric()
        assert metric is fake_metrics.faithfulness

    def test_get_ragas_metric_imports_relevance(self):
        fake_metrics = MagicMock()
        fake_metrics.answer_relevancy = MagicMock(name="answer_relevancy")
        fake_ragas = MagicMock(metrics=fake_metrics)
        with patch.dict("sys.modules", {"ragas": fake_ragas, "ragas.metrics": fake_metrics}):
            metric = RelevanceMetric()._get_ragas_metric()
        assert metric is fake_metrics.answer_relevancy

    def test_get_ragas_metric_imports_context_precision(self):
        fake_metrics = MagicMock()
        fake_metrics.context_precision = MagicMock(name="context_precision")
        fake_ragas = MagicMock(metrics=fake_metrics)
        with patch.dict("sys.modules", {"ragas": fake_ragas, "ragas.metrics": fake_metrics}):
            metric = ContextPrecisionMetric()._get_ragas_metric()
        assert metric is fake_metrics.context_precision


class TestHallucinationDeepeval:
    def test_deepeval_score_path(self):
        mock_metric = MagicMock()
        mock_metric.score = 0.05
        fake_hm = MagicMock(return_value=mock_metric)
        fake_test_case = MagicMock()
        fake_deepeval = MagicMock(
            metrics=MagicMock(HallucinationMetric=fake_hm),
            test_case=MagicMock(LLMTestCase=fake_test_case),
        )
        with patch.dict(
            "sys.modules",
            {
                "deepeval": fake_deepeval,
                "deepeval.metrics": fake_deepeval.metrics,
                "deepeval.test_case": fake_deepeval.test_case,
            },
        ):
            raw = HallucinationMetric(threshold=0.1)._deepeval_score(_sample())
        assert raw == pytest.approx(0.05)
