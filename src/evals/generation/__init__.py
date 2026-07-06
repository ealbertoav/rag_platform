from __future__ import annotations

import dataclasses
import logging
from abc import ABC, abstractmethod
from typing import Protocol

from src.domain.entities.evaluation import EvalSample

logger = logging.getLogger(__name__)

PARAMETRIC_ANSWER_DETAILS = "Parametric answer (no retrieval context)"


def parametric_eval_result(
    metric: str,
    threshold: float,
    *,
    higher_is_better: bool = True,
) -> EvalResult:
    """Neutral score when faithfulness/context metrics do not apply (no retrieval)."""
    score = 1.0 if higher_is_better else 0.0
    return EvalResult.make(
        metric,
        score,
        threshold,
        higher_is_better=higher_is_better,
        details=PARAMETRIC_ANSWER_DETAILS,
    )


@dataclasses.dataclass
class EvalResult:
    """Outcome of a single generation metric on one EvalSample."""

    metric: str
    score: float
    threshold: float
    passed: bool
    details: str = ""

    @classmethod
    def make(
        cls,
        metric: str,
        score: float,
        threshold: float,
        *,
        higher_is_better: bool = True,
        details: str = "",
    ) -> EvalResult:
        passed = (score > threshold) if higher_is_better else (score < threshold)
        return cls(metric=metric, score=score, threshold=threshold, passed=passed, details=details)


class GenerationMetric(Protocol):
    """Common interface for all generation-quality metrics."""

    def score(self, sample: EvalSample) -> EvalResult: ...


# ── Ragas shared infrastructure ────────────────────────────────────────────────


def _make_ragas_dataset(sample: EvalSample) -> object:
    """Build a single-row Ragas-compatible HuggingFace Dataset from *sample*."""
    from datasets import Dataset  # type: ignore[import-untyped, attr-defined]

    return Dataset.from_dict(
        {
            "question": [sample.question],
            "answer": [sample.generated_answer],
            "contexts": [list(sample.retrieved_chunks)],
            "ground_truth": [sample.expected_answer],
        }
    )


class RagasMetric(ABC):
    """Base for metrics that delegate scoring to a single Ragas metric.

    Subclasses must set "_metric_name" and implement "_get_ragas_metric()".
    Override "_pre_checks()" to add early-exit guards before the Ragas call.
    """

    _metric_name: str

    def __init__(self, threshold: float) -> None:
        self.threshold = threshold

    def score(self, sample: EvalSample) -> EvalResult:
        for early_exit in self._pre_checks(sample):
            return early_exit
        try:
            raw = self._ragas_score(sample)
            return EvalResult.make(self._metric_name, raw, self.threshold)
        except Exception as exc:
            logger.warning("%s scoring failed: %s", type(self).__name__, exc)
            return EvalResult.make(self._metric_name, 0.0, self.threshold, details=str(exc))

    def _pre_checks(self, sample: EvalSample) -> list[EvalResult]:
        return []

    def _guard(self, details: str) -> EvalResult:
        """Return a zero-score EvalResult for use in pre-check guards."""
        return EvalResult.make(self._metric_name, 0.0, self.threshold, details=details)

    def _ragas_score(self, sample: EvalSample) -> float:
        from ragas import evaluate  # type: ignore[import-untyped]

        dataset = _make_ragas_dataset(sample)
        result = evaluate(dataset, metrics=[self._get_ragas_metric()])
        return float(result[self._metric_name])

    @abstractmethod
    def _get_ragas_metric(self) -> object: ...
