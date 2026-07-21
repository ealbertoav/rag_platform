from __future__ import annotations

from typing import override

from src.domain.entities.evaluation import EvalSample
from src.evals.generation import EvalResult, LLMJudgeMetric, extract_json_object

_PROMPT_TEMPLATE = """You are evaluating how relevant an AI-generated answer is to the \
question that was asked, independent of whether the answer is factually correct.

Question: {question}

Answer: {answer}

Rate the topical relevance of the answer to the question on a scale from \
0.0 (completely off-topic) to 1.0 (directly and fully addresses the question).

Respond with ONLY a JSON object in this exact shape, no other text:
{{"score": <float between 0.0 and 1.0>}}"""


class RelevanceMetric(LLMJudgeMetric):
    """Measures how relevant the generated answer is to the question.

    Asks the NVIDIA NIM judge (#103/#104) to rate topical relevance directly.
    Score is in [0, 1]; higher = more relevant.
    """

    _metric_name: str = "answer_relevancy"

    @override
    def __init__(self, threshold: float = 0.75) -> None:
        super().__init__(threshold)

    @override
    def _pre_checks(self, sample: EvalSample) -> list[EvalResult]:
        if not sample.generated_answer:
            return [self._guard("Empty generated answer")]
        return []

    @override
    def _build_prompt(self, sample: EvalSample) -> str:
        return _PROMPT_TEMPLATE.format(question=sample.question, answer=sample.generated_answer)

    @override
    def _parse_response(self, sample: EvalSample, response: str) -> float:
        payload = extract_json_object(response)
        score = payload.get("score")
        if not isinstance(score, (int, float)):
            raise ValueError(f"Judge response missing numeric 'score': {response!r}")
        return max(0.0, min(1.0, float(score)))
