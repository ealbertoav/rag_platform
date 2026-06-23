from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.api.dependencies import get_chat_pipeline
from src.rag.pipelines.chat_pipeline import ChatPipeline

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str


class ChatFullResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    token_count: int


@router.post("", response_class=StreamingResponse)
async def chat_stream(
    body: ChatRequest,
    pipeline: ChatPipeline = Depends(get_chat_pipeline),
) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Each event is "data: {"token": "..."}".
    The stream ends with "data: [DONE]".
    """

    async def _generate() -> AsyncIterator[str]:
        async for token in await pipeline.chat(body.question):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/full", response_model=ChatFullResponse)
async def chat_full(
    body: ChatRequest,
    pipeline: ChatPipeline = Depends(get_chat_pipeline),
) -> ChatFullResponse:
    """Non-streaming endpoint — returns the complete answer once generated."""
    answer = await pipeline.chat_full(body.question)
    return ChatFullResponse(
        answer=answer.text,
        sources=answer.sources,
        latency_ms=answer.latency_ms,
        token_count=answer.token_count,
    )
