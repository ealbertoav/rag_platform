from __future__ import annotations

import dataclasses
from typing import Protocol

from src.domain.entities.evaluation import EvalSample


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
        cls, metric: str, score: float, threshold: float,
        *, higher_is_better: bool = True, details: str = "",
    ) -> EvalResult:
        passed = (score > threshold) if higher_is_better else (score < threshold)
        return cls(metric=metric, score=score, threshold=threshold, passed=passed, details=details)


class GenerationMetric(Protocol):
    """Common interface for all generation-quality metrics."""

    def score(self, sample: EvalSample) -> EvalResult: ...
