from __future__ import annotations

import logging

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult, parametric_eval_result

logger = logging.getLogger(__name__)

_METRIC = "hallucination"


class HallucinationMetric:
    """Measures the degree of hallucination in the generated answer.

    Wraps DeepEval "HallucinationMetric" (requires "pip install deepeval").
    Score is in [0, 1]; **lower is better** (fewer hallucinations).
    A result passes when "score < threshold".
    """

    def __init__(self, threshold: float = 0.1) -> None:
        self.threshold: float = threshold

    def score(self, sample: EvalSample) -> EvalResult:
        if sample.parametric_answer:
            return parametric_eval_result(
                _METRIC,
                self.threshold,
                higher_is_better=False,
            )
        if not sample.generated_answer:
            # No answer → cannot hallucinate; score 0 (perfect, passes).
            return EvalResult.make(
                _METRIC,
                0.0,
                self.threshold,
                higher_is_better=False,
                details="Empty generated answer",
            )
        if not sample.retrieved_chunks:
            # No context to check the answer against — e.g. CRAG's web-only fallback
            # (T-142) or a correct "I don't have information about this" refusal.
            # Not evaluable, not a hallucination: treat like a parametric answer
            # rather than assuming the worst case.
            return EvalResult.make(
                _METRIC,
                0.0,
                self.threshold,
                higher_is_better=False,
                details="No context to verify against",
            )
        try:
            raw = self._deepeval_score(sample)
            return EvalResult.make(_METRIC, raw, self.threshold, higher_is_better=False)
        except Exception as exc:
            logger.warning("Hallucination scoring failed: %s", exc)
            return EvalResult.make(
                _METRIC, 1.0, self.threshold, higher_is_better=False, details=str(exc)
            )

    # ── internal ───────────────────────────────────────────────────────────────

    def _deepeval_score(self, sample: EvalSample) -> float:
        from deepeval.metrics import HallucinationMetric as _HM
        from deepeval.test_case import LLMTestCase

        test_case = LLMTestCase(
            input=sample.question,
            actual_output=sample.generated_answer,
            context=list(sample.retrieved_chunks),
        )
        metric = _HM(threshold=self.threshold)
        return metric.measure(test_case)
