from __future__ import annotations

from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Chunk(BaseModel):
    """A contiguous slice of a Document, paired with its vector representations.

    `embedding` and `sparse_vector` start as None and are populated after
    the embedding step. Use `model_copy(update={...})` to produce updated
    instances — the model is frozen to prevent accidental mutation.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    # Dense vector (e.g. 1024-dim from BGE-M3). None before embedding.
    embedding: list[float] | None = None
    # Sparse lexical vector {token_id: weight} from BGE-M3 ColBERT head.
    sparse_vector: dict[int, float] | None = None
    # Provenance: source file, page, section, parent_id, content_hash, etc.
    metadata: dict[str, Any] = Field(default_factory=dict)
