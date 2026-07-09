from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class RetrievalFilter(BaseModel):
    """Optional constraints applied at retrieval time."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    document_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, str] = Field(default_factory=dict)
    min_score: float | None = None

    def is_active(self) -> bool:
        return bool(self.document_ids or self.metadata or self.min_score is not None)
