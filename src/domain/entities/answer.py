from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from src.rag.quality.explainable_retrieval import ChunkExplanation


class Answer(BaseModel):
    """The RAG system's response to a Query."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str
    # IDs of the Chunks whose text was included in the LLM context window.
    sources: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0
    explanations: list[ChunkExplanation] | None = None
    highlights: dict[str, list[str]] | None = Field(
        default=None,
        description=(
            "Chunk ID to supporting spans copied verbatim from the LLM-facing passage text "
            "(see chunk_context_text), not necessarily from Chunk.text when parent context "
            "or contextual headers apply."
        ),
    )
