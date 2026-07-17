from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status

from src.api.dependencies import get_vector_store
from src.api.security import require_api_key
from src.core.exceptions import VectorStoreError
from src.core.settings import settings
from src.domain.entities.source_reference import SourceReference
from src.domain.repositories.vector_store_repository import VectorStoreRepository

router = APIRouter(
    prefix="/chunks",
    tags=["chunks"],
    dependencies=[Depends(require_api_key)],
)


class ChunkResponse(SourceReference):
    """Full chunk lookup response (T-273) — a SourceReference plus its text."""

    text: str


@router.get("/{chunk_id}", response_model=ChunkResponse, response_model_exclude_none=True)
async def get_chunk(
    chunk_id: str,
    vector_store: VectorStoreRepository = Depends(get_vector_store),
) -> ChunkResponse:
    """Look up a single stored chunk by ID.

    Disabled by default via `quality.chunk_lookup.enabled` — behaves as if the
    route did not exist (404) when disabled, same as an unknown chunk ID.
    """
    if not settings.quality.chunk_lookup.enabled:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Not Found")
    try:
        chunk = await asyncio.to_thread(vector_store.get_chunk, chunk_id)
    except VectorStoreError as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
    if chunk is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk {chunk_id!r} not found",
        )
    reference = SourceReference.from_chunk(chunk)
    return ChunkResponse(text=chunk.text, **reference.model_dump())
