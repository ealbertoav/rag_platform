"""Shared helpers for e2e benchmark unit tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

from src.domain.entities.answer import Answer
from src.domain.entities.evaluation import BenchmarkRun


def pipeline_mock(
    *,
    sources: list[str] | None = None,
    text: str = "Answer.",
    context: list[str] | None = None,
    latency_ms: float = 42.0,
    fail: bool = False,
) -> MagicMock:
    pipeline = MagicMock()
    if fail:
        pipeline.benchmark = AsyncMock(side_effect=RuntimeError("pipeline down"))
    else:
        answer = Answer(
            query_id="q1",
            text=text,
            sources=sources if sources is not None else ["c0"],
            latency_ms=latency_ms,
        )
        pipeline.benchmark = AsyncMock(
            return_value=BenchmarkRun(answer=answer, context_texts=context or ["ctx"])
        )
    return pipeline


def metric_mock(score: float) -> MagicMock:
    mock = MagicMock()
    result = MagicMock()
    result.score = score
    mock.score.return_value = result
    return mock
