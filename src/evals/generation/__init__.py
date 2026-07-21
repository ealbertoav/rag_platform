from __future__ import annotations

import dataclasses
import json
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


# ── NVIDIA NIM judge infrastructure (#103/#104) ─────────────────────────────────


def extract_json_object(text: str) -> dict[str, object]:
    """Extract the first ``{...}`` JSON object from free-form judge LLM output.

    Judge prompts ask for a bare JSON object, but models sometimes wrap it in
    prose or a markdown code fence anyway — take the outermost braces rather
    than assuming the whole response is clean JSON.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"No JSON object found in judge response: {text!r}")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError(f"Judge response JSON is not an object: {text!r}")
    return payload


class LLMJudgeMetric(ABC):
    """Base for metrics that ask the NVIDIA NIM judge LLM to score a sample directly.

    Subclasses must set "_metric_name", and implement "_build_prompt()" (what
    to ask the judge) and "_parse_response()" (how to turn its answer into a
    [0, 1] score). Override "_pre_checks()" to add early-exit guards before
    the judge call.
    """

    _metric_name: str

    def __init__(self, threshold: float) -> None:
        self.threshold: float = threshold

    def score(self, sample: EvalSample) -> EvalResult:
        for early_exit in self._pre_checks(sample):
            return early_exit
        try:
            raw = self._judge_score(sample)
            return EvalResult.make(self._metric_name, raw, self.threshold)
        except Exception as exc:
            logger.warning("%s scoring failed: %s", type(self).__name__, exc)
            return EvalResult.make(self._metric_name, 0.0, self.threshold, details=str(exc))

    def _pre_checks(self, _sample: EvalSample) -> list[EvalResult]:
        return []

    def _guard(self, details: str) -> EvalResult:
        """Return a zero-score EvalResult for use in pre-check guards."""
        return EvalResult.make(self._metric_name, 0.0, self.threshold, details=details)

    def _judge_score(self, sample: EvalSample) -> float:
        from src.evals.generation.nim_judge import build_nim_judge_llm

        judge = build_nim_judge_llm()
        prompt = self._build_prompt(sample)
        response = judge.generate(prompt=prompt, context="")
        return self._parse_response(sample, response)

    @abstractmethod
    def _build_prompt(self, sample: EvalSample) -> str: ...

    @abstractmethod
    def _parse_response(self, sample: EvalSample, response: str) -> float: ...
