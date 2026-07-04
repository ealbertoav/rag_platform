from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from src.api.dependencies import get_vector_store
from src.api.security import require_api_key
from src.core.exceptions import VectorStoreError
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.rag.quality.feedback_loop import record_feedback, score_from_relevant

router = APIRouter(
    prefix="/feedback",
    tags=["feedback"],
    dependencies=[Depends(require_api_key)],
)


class FeedbackRequest(BaseModel):
    query_id: str = Field(min_length=1)
    chunk_id: str = Field(min_length=1)
    relevant: bool


@router.post("", status_code=status.HTTP_204_NO_CONTENT)
async def submit_feedback(
    body: FeedbackRequest,
    vector_store: VectorStoreRepository = Depends(get_vector_store),
) -> None:
    """Record user relevance feedback for a retrieved chunk."""
    score = score_from_relevant(body.relevant)
    try:
        await asyncio.to_thread(
            record_feedback,
            vector_store,
            body.query_id,
            body.chunk_id,
            score,
        )
    except VectorStoreError as exc:
        if "not found" in str(exc).lower():
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        raise HTTPException(status_code=502, detail=str(exc)) from exc
