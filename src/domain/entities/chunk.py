from __future__ import annotations

from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from src.core.constants import MODALITY_TEXT


class Chunk(BaseModel):
    """A contiguous slice of a Document, paired with its vector representations.

    `embedding` and `sparse_vector` start as None and are populated after
    the embedding step. Use `model_copy(update={...})` to produce updated
    instances — the model is frozen to prevent accidental mutation.

    Multimodal fields (T-210): `modality` defaults to text, so existing
    callers stay unchanged; `image_embedding` / `asset_path` are filled by
    later phases (T-230 figure assets, T-250+ image embeddings).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    # Dense vector (e.g. 1024-dim from BGE-M3). None before embedding.
    embedding: list[float] | None = None
    # Sparse lexical vector {token_id: weight} from BGE-M3 ColBERT head.
    sparse_vector: dict[int, float] | None = None
    # Provenance: source file, page, section, parent_id, content_hash, etc.
    metadata: dict[str, Any] = Field(default_factory=dict)
    # Content modality for retrieval/generation routing (T-210).
    modality: str = MODALITY_TEXT
    # Dense image vector (CLIP / Voyage-Multimodal). None until T-250+.
    image_embedding: list[float] | None = None
    # Path to an extracted figure/page image asset (T-230+). None for text/table.
    asset_path: str | None = None
