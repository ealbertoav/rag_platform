from __future__ import annotations

from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from src.core.constants import (
    ASSET_PATH_KEY,
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SECTION_KEY,
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_TO_MODALITY,
    FIGURE_ID_KEY,
    MODALITY_TEXT,
    TABLE_ID_KEY,
)
from src.domain.entities.chunk import Chunk


def resolve_modality(*, modality: str = MODALITY_TEXT, chunk_type: str | None = None) -> str:
    """Resolve effective modality from an explicit field and/or chunk type.

    Explicit non-text "modality" wins. Otherwise, "chunk_type" is mapped via
    "CHUNK_TYPE_TO_MODALITY" so legacy chunks that only set "metadata.type"
    (e.g. T-202 table chunks) still resolve correctly. Falls back to
    "modality" (default "text").
    """
    if modality != MODALITY_TEXT:
        return modality
    if chunk_type is not None:
        mapped = CHUNK_TYPE_TO_MODALITY.get(chunk_type)
        if mapped is not None:
            return mapped
    return modality


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return str(value)


def _optional_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    return None


def _optional_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)):
        return None
    coords: list[float] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            return None
        coords.append(float(item))
    return coords


class SourceReference(BaseModel):
    """Structured citation pointing at a retrieved chunk (T-210).

    "Answer.sources" remains "list[str]" of chunk IDs for backward
    compatibility. "Answer.source_references" carries richer provenance for
    multimodal attribution (enriched further in T-272 API responses).
    """

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str | None = None
    source: str | None = None
    modality: str = MODALITY_TEXT
    page: int | None = None
    section: str | None = None
    table_id: str | None = None
    figure_id: str | None = None
    bbox: list[float] | None = None
    snippet: str | None = None
    score: float | None = None
    asset_path: str | None = None

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        *,
        score: float | None = None,
        snippet: str | None = None,
    ) -> SourceReference:
        """Build a "SourceReference" from a "Chunk" and optional score/snippet."""
        meta = chunk.metadata
        raw_type = meta.get(CHUNK_TYPE_KEY)
        chunk_type = raw_type if isinstance(raw_type, str) else None
        asset = chunk.asset_path if chunk.asset_path is not None else meta.get(ASSET_PATH_KEY)
        return cls(
            chunk_id=chunk.id,
            document_id=chunk.document_id,
            source=_optional_str(meta.get(CHUNK_SOURCE_KEY)),
            modality=resolve_modality(modality=chunk.modality, chunk_type=chunk_type),
            page=_optional_int(meta.get(CHUNK_PAGE_KEY)),
            section=_optional_str(meta.get(CHUNK_SECTION_KEY)),
            table_id=_optional_str(meta.get(TABLE_ID_KEY)),
            figure_id=_optional_str(meta.get(FIGURE_ID_KEY)),
            bbox=_optional_bbox(meta.get(BBOX_KEY)),
            snippet=snippet,
            score=score,
            asset_path=_optional_str(asset),
        )


def source_references_for_chunks(
    chunks: list[Chunk],
    *,
    scores: dict[str, float] | None = None,
) -> list[SourceReference]:
    """Map chunks to the "SourceReference" list, optionally attaching per-id scores."""
    score_map = scores or {}
    return [SourceReference.from_chunk(chunk, score=score_map.get(chunk.id)) for chunk in chunks]
