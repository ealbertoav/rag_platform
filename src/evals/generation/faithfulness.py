from __future__ import annotations

from typing import override

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult, RagasMetric, parametric_eval_result


class FaithfulnessMetric(RagasMetric):
    """Measures whether the generated answer is grounded in the retrieved context.

    Wraps Ragas "faithfulness" (requires ``pip install ragas datasets``).
    Score is in [0, 1]; higher = more faithful.
    """

    _metric_name: str = "faithfulness"

    @override
    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__(threshold)

    @override
    def _pre_checks(self, sample: EvalSample) -> list[EvalResult]:
        if sample.parametric_answer:
            return [parametric_eval_result(self._metric_name, self.threshold)]
        if not sample.generated_answer:
            return [self._guard("Empty generated answer")]
        if not sample.retrieved_chunks:
            return [self._guard("No context provided")]
        return []

    @override
    def _get_ragas_metric(self) -> object:
        from ragas.metrics import faithfulness

        return faithfulness
