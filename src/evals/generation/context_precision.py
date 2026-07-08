from __future__ import annotations

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult, RagasMetric, parametric_eval_result


class ContextPrecisionMetric(RagasMetric):
    """Measures what fraction of retrieved chunks are relevant to the question.

    Uses Ragas ``context_precision`` (LLM-as-judge).
    Score is in [0, 1]; higher = more precise context (less noise).
    """

    _metric_name = "context_precision"

    def __init__(self, threshold: float = 0.7) -> None:
        super().__init__(threshold)

    def _pre_checks(self, sample: EvalSample) -> list[EvalResult]:
        if sample.parametric_answer:
            return [parametric_eval_result(self._metric_name, self.threshold)]
        if not sample.retrieved_chunks:
            return [self._guard("No context provided")]
        if not sample.question:
            return [self._guard("Empty question")]
        return []

    def _get_ragas_metric(self) -> object:
        from ragas.metrics import context_precision

        return context_precision
