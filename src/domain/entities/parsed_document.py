from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict, Field


class ParsedDocument(BaseModel):
    """Layout-aware parse result before chunking.

    Minimal contract for Phase 19: plain text plus optional metadata.
    Layout parsers add structured blocks (tables, figures, pages)
    in later phases via "metadata".
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    source: str  # file path — kept as str for JSON-serialisability
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
