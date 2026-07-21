from __future__ import annotations

from typing import override

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import (
    EvalResult,
    LLMJudgeMetric,
    extract_json_object,
    parametric_eval_result,
)

_PROMPT_TEMPLATE = """You are evaluating which retrieved passages are actually relevant \
to answering a question.

Question: {question}

Passages (in retrieval order):
{passages}

For each passage, determine whether it is relevant to answering the question \
(true) or not relevant / noise (false).

Respond with ONLY a JSON object in this exact shape, no other text — one \
boolean per passage, in the same order:
{{"relevant": [true|false, ...]}}"""


class ContextPrecisionMetric(LLMJudgeMetric):
    """Measures what fraction of retrieved chunks are relevant to the question.

    Asks the NVIDIA NIM judge (#103/#104) to judge each retrieved passage's
    relevance directly. Score is the fraction judged relevant, in [0, 1];
    higher = more precise context (less noise).
    """

    _metric_name: str = "context_precision"

    @override
    def __init__(self, threshold: float = 0.7) -> None:
        super().__init__(threshold)

    @override
    def _pre_checks(self, sample: EvalSample) -> list[EvalResult]:
        if sample.parametric_answer:
            return [parametric_eval_result(self._metric_name, self.threshold)]
        if not sample.retrieved_chunks:
            return [self._guard("No context provided")]
        if not sample.question:
            return [self._guard("Empty question")]
        return []

    @override
    def _build_prompt(self, sample: EvalSample) -> str:
        passages = "\n".join(f"[{i}] {chunk}" for i, chunk in enumerate(sample.retrieved_chunks))
        return _PROMPT_TEMPLATE.format(question=sample.question, passages=passages)

    @override
    def _parse_response(self, sample: EvalSample, response: str) -> float:
        payload = extract_json_object(response)
        relevant = payload.get("relevant")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"Judge response missing 'relevant' list: {response!r}")
        if len(relevant) != len(sample.retrieved_chunks):
            raise ValueError(
                f"Judge returned {len(relevant)} verdicts for "
                f"{len(sample.retrieved_chunks)} passages: {response!r}"
            )
        return sum(1 for v in relevant if v is True) / len(relevant)
