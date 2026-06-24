from __future__ import annotations

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult, RagasMetric


class FaithfulnessMetric(RagasMetric):
    """Measures whether the generated answer is grounded in the retrieved context.

    Wraps Ragas ``faithfulness`` (requires ``pip install ragas datasets``).
    Score is in [0, 1]; higher = more faithful.
    """

    _metric_name = "faithfulness"

    def __init__(self, threshold: float = 0.8) -> None:
        super().__init__(threshold)

    def _pre_checks(self, sample: EvalSample) -> list[EvalResult]:
        if not sample.generated_answer:
            return [self._guard("Empty generated answer")]
        if not sample.retrieved_chunks:
            return [self._guard("No context provided")]
        return []

    def _get_ragas_metric(self) -> object:
        from ragas.metrics import faithfulness  # type: ignore[import-untyped]

        return faithfulness
