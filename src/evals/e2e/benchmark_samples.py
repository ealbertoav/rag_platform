"""Shared helpers for end-to-end benchmark sample scoring."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable
from typing import Protocol

from src.domain.entities.evaluation import BenchmarkRun, EvalSample
from src.evals.generation.faithfulness import FaithfulnessMetric
from src.evals.generation.relevance import RelevanceMetric
from src.evals.retrieval.recall_at_k import recall_at_k


class BenchmarkPipeline(Protocol):
    async def benchmark(self, question: str) -> BenchmarkRun: ...


@dataclasses.dataclass(frozen=True)
class GenerationMetricScores:
    recall_at_k: float
    faithfulness: float
    relevance: float
    latency_ms: float


@dataclasses.dataclass(frozen=True)
class GenerationMetricMeans:
    mean_recall_at_5: float
    mean_faithfulness: float
    mean_relevance: float
    mean_latency_ms: float


@dataclasses.dataclass
class GenerationMetricAccumulator:
    recalls: list[float] = dataclasses.field(default_factory=list)
    faith_scores: list[float] = dataclasses.field(default_factory=list)
    relev_scores: list[float] = dataclasses.field(default_factory=list)
    latencies: list[float] = dataclasses.field(default_factory=list)

    def append(self, scores: GenerationMetricScores) -> None:
        self.recalls.append(scores.recall_at_k)
        self.faith_scores.append(scores.faithfulness)
        self.relev_scores.append(scores.relevance)
        self.latencies.append(scores.latency_ms)

    @property
    def total_samples(self) -> int:
        return len(self.recalls)

    def means(self) -> GenerationMetricMeans:
        n = len(self.recalls) or 1
        return GenerationMetricMeans(
            mean_recall_at_5=sum(self.recalls) / n,
            mean_faithfulness=sum(self.faith_scores) / n,
            mean_relevance=sum(self.relev_scores) / n,
            mean_latency_ms=sum(self.latencies) / n,
        )


def pair_str(val: object) -> str:
    return val if isinstance(val, str) else ""


def pair_str_list(val: object) -> list[str]:
    return [v for v in val if isinstance(v, str)] if isinstance(val, list) else []


def failure_generation_scores(*, started_at: float) -> GenerationMetricScores:
    """Zero metric scores when pipeline.benchmark() fails."""
    return GenerationMetricScores(
        recall_at_k=0.0,
        faithfulness=0.0,
        relevance=0.0,
        latency_ms=(time.monotonic() - started_at) * 1000,
    )


def pipeline_error_logger(
    log: Callable[..., None],
    message: str,
    *context: object,
) -> Callable[[Exception], None]:
    """Build a typed pipeline-error callback with bound *context* for *message*."""

    def handler(exc: Exception) -> None:
        log(message, *context, exc)

    return handler


def score_generation_sample(
    *,
    run: BenchmarkRun,
    question: str,
    expected_answer: str,
    relevant_ids: list[str],
    recall_k: int,
    faithfulness: FaithfulnessMetric,
    relevance: RelevanceMetric,
    started_at: float,
) -> GenerationMetricScores:
    """Score recall, faithfulness, relevance, and latency for one benchmark run."""
    answer = run.answer
    context_texts = run.context_texts
    retrieved_ids = list(answer.sources)
    recall = recall_at_k(retrieved_ids, relevant_ids, k=recall_k)

    sample = EvalSample(
        question=question,
        expected_answer=expected_answer,
        retrieved_chunks=context_texts,
        generated_answer=answer.text,
        parametric_answer=run.parametric_answer,
    )
    faith_score = faithfulness.score(sample).score
    relev_score = relevance.score(sample).score
    latency_ms = getattr(answer, "latency_ms", None)
    if isinstance(latency_ms, int | float):
        latency = float(latency_ms)
    else:
        latency = (time.monotonic() - started_at) * 1000

    return GenerationMetricScores(
        recall_at_k=recall,
        faithfulness=faith_score,
        relevance=relev_score,
        latency_ms=latency,
    )


async def score_pipeline_question(
    *,
    pipeline: BenchmarkPipeline,
    question: str,
    expected_answer: str,
    relevant_ids: list[str],
    recall_k: int,
    faithfulness: FaithfulnessMetric,
    relevance: RelevanceMetric,
    on_pipeline_error: Callable[[Exception], None] | None = None,
) -> GenerationMetricScores:
    """Run pipeline.benchmark() for one question and return generation metrics."""
    started_at = time.monotonic()
    try:
        run = await pipeline.benchmark(question)
    except Exception as exc:
        if on_pipeline_error is not None:
            on_pipeline_error(exc)
        return failure_generation_scores(started_at=started_at)
    return score_generation_sample(
        run=run,
        question=question,
        expected_answer=expected_answer,
        relevant_ids=relevant_ids,
        recall_k=recall_k,
        faithfulness=faithfulness,
        relevance=relevance,
        started_at=started_at,
    )
