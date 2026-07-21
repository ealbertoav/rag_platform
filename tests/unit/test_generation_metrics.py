"""T-042 — Generation metric unit tests (NVIDIA NIM judge / DeepEval mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult, GenerationMetric, extract_json_object
from src.evals.generation.context_precision import ContextPrecisionMetric
from src.evals.generation.faithfulness import FaithfulnessMetric
from src.evals.generation.hallucination import HallucinationMetric
from src.evals.generation.relevance import RelevanceMetric

# Patch paths — extracted to avoid long lines in each test. `_judge_score` is the
# seam shared by every LLMJudgeMetric subclass; mocking it isolates EvalResult/
# threshold wrapping from prompt-building and response-parsing (tested separately
# below, per metric).
_FAITH = "src.evals.generation.faithfulness.FaithfulnessMetric._judge_score"
_RELEV = "src.evals.generation.relevance.RelevanceMetric._judge_score"
_CTX_PREC = "src.evals.generation.context_precision.ContextPrecisionMetric._judge_score"
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


class TestParametricEvalResult:
    def test_higher_is_better(self):
        from src.evals.generation import parametric_eval_result

        result = parametric_eval_result("faithfulness", threshold=0.8)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert "Parametric answer" in result.details

    def test_lower_is_better(self):
        from src.evals.generation import parametric_eval_result

        low = parametric_eval_result("hallucination", threshold=0.1, higher_is_better=False)
        assert low.score == pytest.approx(0.0)
        assert low.passed is True


class TestExtractJsonObject:
    def test_extracts_clean_json(self):
        assert extract_json_object('{"score": 0.5}') == {"score": 0.5}

    def test_extracts_json_wrapped_in_prose(self):
        text = 'Sure, here is my answer:\n{"score": 0.9}\nLet me know if you need more.'
        assert extract_json_object(text) == {"score": 0.9}

    def test_raises_when_no_braces_found(self):
        with pytest.raises(ValueError, match="No JSON object found"):
            extract_json_object("no json here")

    def test_raises_when_top_level_is_not_an_object(self):
        with pytest.raises(ValueError):
            extract_json_object("[1, 2, 3]")


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

    def test_context_precision_satisfies_protocol(self):
        m: GenerationMetric = ContextPrecisionMetric()
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
        sample = EvalSample(
            question="hello",
            expected_answer="hi",
            retrieved_chunks=[],
            generated_answer="Hello!",
            parametric_answer=True,
        )
        result = FaithfulnessMetric().score(sample)
        assert result.score == pytest.approx(1.0)
        assert result.passed is True
        assert "Parametric answer" in result.details

    def test_judge_failure_returns_zero(self):
        with patch(_FAITH, side_effect=RuntimeError("API error")):
            result = FaithfulnessMetric().score(_sample())
        assert result.score == pytest.approx(0.0)
        assert result.passed is False
        assert result.details != ""

    def test_metric_name(self):
        with patch(_FAITH, return_value=0.9):
            result = FaithfulnessMetric().score(_sample())
        assert result.metric == "faithfulness"

    def test_prompt_includes_question_context_and_answer(self):
        sample = _sample()
        prompt = FaithfulnessMetric()._build_prompt(sample)
        assert sample.question in prompt
        assert sample.generated_answer in prompt
        assert sample.retrieved_chunks[0] in prompt

    def test_parse_response_computes_supported_fraction(self):
        response = (
            '{"claims": [{"claim": "a", "supported": true}, {"claim": "b", "supported": false}]}'
        )
        score = FaithfulnessMetric()._parse_response(_sample(), response)
        assert score == pytest.approx(0.5)

    def test_parse_response_no_claims_is_vacuously_faithful(self):
        score = FaithfulnessMetric()._parse_response(_sample(), '{"claims": []}')
        assert score == pytest.approx(1.0)

    def test_parse_response_raises_on_missing_claims_key(self):
        with pytest.raises(ValueError, match="claims"):
            FaithfulnessMetric()._parse_response(_sample(), '{"other": []}')


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

    def test_judge_failure_returns_zero(self):
        with patch(_RELEV, side_effect=RuntimeError("judge unavailable")):
            result = RelevanceMetric().score(_sample())
        assert result.score == pytest.approx(0.0)

    def test_metric_name(self):
        with patch(_RELEV, return_value=0.8):
            result = RelevanceMetric().score(_sample())
        assert result.metric == "answer_relevancy"

    def test_prompt_includes_question_and_answer(self):
        sample = _sample()
        prompt = RelevanceMetric()._build_prompt(sample)
        assert sample.question in prompt
        assert sample.generated_answer in prompt

    def test_parse_response_extracts_score(self):
        score = RelevanceMetric()._parse_response(_sample(), '{"score": 0.65}')
        assert score == pytest.approx(0.65)

    def test_parse_response_clamps_out_of_range_score(self):
        assert RelevanceMetric()._parse_response(_sample(), '{"score": 1.5}') == pytest.approx(1.0)
        assert RelevanceMetric()._parse_response(_sample(), '{"score": -0.5}') == pytest.approx(0.0)

    def test_parse_response_raises_on_non_numeric_score(self):
        with pytest.raises(ValueError, match="score"):
            RelevanceMetric()._parse_response(_sample(), '{"score": "high"}')


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


class TestHallucinationDeepeval:
    def test_deepeval_score_path(self):
        mock_metric = MagicMock()
        mock_metric.measure.return_value = 0.05
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


# ── ContextPrecisionMetric ────────────────────────────────────────────────────


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

    def test_judge_failure_returns_zero(self):
        with patch(_CTX_PREC, side_effect=RuntimeError("judge unavailable")):
            result = ContextPrecisionMetric().score(_sample())
        assert result.score == pytest.approx(0.0)
        assert result.details != ""

    def test_metric_name(self):
        with patch(_CTX_PREC, return_value=0.75):
            result = ContextPrecisionMetric().score(_sample())
        assert result.metric == "context_precision"

    def test_prompt_includes_question_and_indexed_passages(self):
        sample = _sample(chunks=["passage one", "passage two"])
        prompt = ContextPrecisionMetric()._build_prompt(sample)
        assert sample.question in prompt
        assert "[0] passage one" in prompt
        assert "[1] passage two" in prompt

    def test_parse_response_computes_relevant_fraction(self):
        score = ContextPrecisionMetric()._parse_response(
            _sample(chunks=["a", "b"]), '{"relevant": [true, false]}'
        )
        assert score == pytest.approx(0.5)

    def test_parse_response_raises_on_empty_relevant_list(self):
        with pytest.raises(ValueError, match="relevant"):
            ContextPrecisionMetric()._parse_response(_sample(), '{"relevant": []}')

    def test_parse_response_raises_on_verdict_count_mismatch(self):
        """A judge that skips/duplicates a passage must not silently mis-score (#104 review)."""
        with pytest.raises(ValueError, match="2 verdicts for 3 passages"):
            ContextPrecisionMetric()._parse_response(
                _sample(chunks=["a", "b", "c"]), '{"relevant": [true, false]}'
            )
