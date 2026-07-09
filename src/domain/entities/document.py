from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, ClassVar
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A raw document loaded from a disk before chunking."""

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid4()))
    source: str  # file path or URL — kept as str for JSON-serialisability
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
