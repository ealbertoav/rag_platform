from __future__ import annotations

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult, RagasMetric


class RelevanceMetric(RagasMetric):
    """Measures how relevant the generated answer is to the question.

    Wraps Ragas "answer_relevancy" (requires "pip install ragas datasets").
    Score is in [0, 1]; higher = more relevant.
    """

    _metric_name = "answer_relevancy"

    def __init__(self, threshold: float = 0.75) -> None:
        super().__init__(threshold)

    def _pre_checks(self, sample: EvalSample) -> list[EvalResult]:
        if not sample.generated_answer:
            return [self._guard("Empty generated answer")]
        return []

    def _get_ragas_metric(self) -> object:
        from ragas.metrics import answer_relevancy  # type: ignore[import-untyped]

        return answer_relevancy
