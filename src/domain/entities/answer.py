from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class Answer(BaseModel):
    """The RAG system's response to a Query."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    text: str
    # IDs of the Chunks whose text was included in the LLM context window.
    sources: list[str] = Field(default_factory=list)
    latency_ms: float = 0.0
    token_count: int = 0
