from __future__ import annotations

import logging

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult

logger = logging.getLogger(__name__)

_METRIC = "answer_relevancy"


class RelevanceMetric:
    """Measures how relevant the generated answer is to the question.

    Wraps Ragas ``answer_relevancy`` (requires ``pip install ragas datasets``).
    Score is in [0, 1]; higher = more relevant.
    """

    def __init__(self, threshold: float = 0.75) -> None:
        self.threshold = threshold

    def score(self, sample: EvalSample) -> EvalResult:
        if not sample.generated_answer:
            return EvalResult.make(_METRIC, 0.0, self.threshold, details="Empty generated answer")
        try:
            raw = self._ragas_score(sample)
            return EvalResult.make(_METRIC, raw, self.threshold)
        except Exception as exc:
            logger.warning("Relevance scoring failed: %s", exc)
            return EvalResult.make(_METRIC, 0.0, self.threshold, details=str(exc))

    # ── internal ───────────────────────────────────────────────────────────────

    def _ragas_score(self, sample: EvalSample) -> float:
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import answer_relevancy  # type: ignore[import-untyped]

        from datasets import Dataset  # type: ignore[import-untyped]

        dataset = Dataset.from_dict({
            "question": [sample.question],
            "answer": [sample.generated_answer],
            "contexts": [list(sample.retrieved_chunks)],
            "ground_truth": [sample.expected_answer],
        })
        result = evaluate(dataset, metrics=[answer_relevancy])
        return float(result[_METRIC])
