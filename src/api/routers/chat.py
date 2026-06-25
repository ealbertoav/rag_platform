from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_agent_pipeline, get_chat_pipeline
from src.api.schemas.agent import AgentChatResponse
from src.rag.pipelines.agent_pipeline import AgentPipeline
from src.rag.pipelines.chat_pipeline import ChatPipeline

router = APIRouter(prefix="/chat", tags=["chat"])

_MAX_AGENT_ITERATIONS = 5


class ChatRequest(BaseModel):
    question: str


class AgentChatRequest(BaseModel):
    question: str
    max_iterations: int = Field(default=3, ge=1, le=_MAX_AGENT_ITERATIONS)


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


@router.post("/agent", response_class=StreamingResponse)
async def chat_agent_stream(
    body: AgentChatRequest,
    pipeline: AgentPipeline = Depends(get_agent_pipeline),
) -> StreamingResponse:
    """Agentic RAG — multistep retrieval with streaming final answer."""
    max_iter = min(body.max_iterations, _MAX_AGENT_ITERATIONS)

    async def _generate() -> AsyncIterator[str]:
        async for token in await pipeline.chat(body.question, max_iterations=max_iter):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/agent/full", response_model=AgentChatResponse)
async def chat_agent_full(
    body: AgentChatRequest,
    pipeline: AgentPipeline = Depends(get_agent_pipeline),
) -> AgentChatResponse:
    """Agentic RAG — returns a complete answer with iteration metadata."""
    max_iter = min(body.max_iterations, _MAX_AGENT_ITERATIONS)
    result = await pipeline.chat_full(body.question, max_iterations=max_iter)
    return AgentChatResponse.from_run(result)
