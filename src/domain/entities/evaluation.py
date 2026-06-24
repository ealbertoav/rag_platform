from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class EvalSample(BaseModel):
    """One evaluation data point produced by running the full RAG pipeline."""

    model_config = ConfigDict(frozen=True)

    question: str
    expected_answer: str
    # Chunk IDs returned by the retrieval step for this question.
    retrieved_chunks: list[str] = Field(default_factory=list)
    generated_answer: str = ""
    # Metric name → score, e.g. {"faithfulness": 0.91, "relevance": 0.88}.
    scores: dict[str, float] = Field(default_factory=dict)
