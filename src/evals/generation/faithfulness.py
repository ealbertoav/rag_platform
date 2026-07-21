from __future__ import annotations

from typing import override

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import (
    EvalResult,
    LLMJudgeMetric,
    extract_json_object,
    parametric_eval_result,
)

_PROMPT_TEMPLATE = """You are evaluating whether an AI-generated answer is faithful to \
(fully grounded in) the provided context.

Question: {question}

Context:
{context}

Answer: {answer}

Break the answer down into its individual factual claims. For each claim, \
determine whether it is directly supported by the context (true) or not \
supported, contradicted, or unverifiable from the context (false).

Respond with ONLY a JSON object in this exact shape, no other text:
{{"claims": [{{"claim": "<claim text>", "supported": true|false}}, ...]}}

If the answer makes no factual claims, respond with {{"claims": []}}."""


class FaithfulnessMetric(LLMJudgeMetric):
    """Measures whether the generated answer is grounded in the retrieved context.

    Asks the NVIDIA NIM judge (#103/#104) to decompose the answer into claims
    and verify each against the retrieved context. Score is the fraction of
    claims judged supported, in [0, 1]; higher = more faithful. An answer
    with no claims scores 1.0 (vacuously faithful).
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
    def _build_prompt(self, sample: EvalSample) -> str:
        context = "\n\n".join(sample.retrieved_chunks)
        return _PROMPT_TEMPLATE.format(
            question=sample.question,
            context=context,
            answer=sample.generated_answer,
        )

    @override
    def _parse_response(self, sample: EvalSample, response: str) -> float:
        payload = extract_json_object(response)
        claims = payload.get("claims")
        if not isinstance(claims, list):
            raise ValueError(f"Judge response missing 'claims' list: {response!r}")
        if not claims:
            return 1.0
        supported = sum(1 for claim in claims if isinstance(claim, dict) and claim.get("supported"))
        return supported / len(claims)
