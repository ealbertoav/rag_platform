from __future__ import annotations

from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Query(BaseModel):
    """A user question, optionally expanded into multiple sub-queries."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    # LLM-generated restatements of the original question. Empty when query
    # expansion is disabled — retrieval falls back to `text` only.
    expanded_texts: list[str] = Field(default_factory=list)
    # Dense vector of the original `text`. None until the embedding step.
    embedding: list[float] | None = None
