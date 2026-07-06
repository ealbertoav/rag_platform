from __future__ import annotations

import dataclasses

from pydantic import BaseModel, ConfigDict, Field

from src.domain.entities.answer import Answer


@dataclasses.dataclass(frozen=True)
class BenchmarkRun:
    """Output of "ChatPipeline.benchmark()" / agent benchmark adapters."""

    answer: Answer
    context_texts: list[str]
    parametric_answer: bool = False


class EvalSample(BaseModel):
    """One evaluation data point produced by running the full RAG pipeline."""

    model_config = ConfigDict(frozen=True)

    question: str
    expected_answer: str
    # Chunk IDs returned by the retrieval step for this question.
    retrieved_chunks: list[str] = Field(default_factory=list)
    generated_answer: str = ""
    # True when the pipeline answered without retrieval (e.g. Self-RAG direct path).
    parametric_answer: bool = False
    # Metric name → score, e.g. {"faithfulness": 0.91, "relevance": 0.88}.
    scores: dict[str, float] = Field(default_factory=dict)
