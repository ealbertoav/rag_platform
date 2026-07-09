from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import APIRouter, Depends
from fastapi import Query as QueryParam
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.api.dependencies import get_agent_pipeline, get_chat_pipeline
from src.api.schemas.agent import AgentChatResponse
from src.api.security import require_api_key
from src.domain.entities.query import Query
from src.rag.pipelines.agent_pipeline import AgentPipeline
from src.rag.pipelines.chat_pipeline import ChatPipeline
from src.rag.quality.explainable_retrieval import ChunkExplanation
from src.rag.retrieval.filters import filters_from_request

router = APIRouter(
    prefix="/chat",
    tags=["chat"],
    dependencies=[Depends(require_api_key)],
)

_MAX_AGENT_ITERATIONS = 5


class ChatRequest(BaseModel):
    question: str
    document_ids: list[str] = Field(default_factory=list)
    metadata_filters: dict[str, str] = Field(default_factory=dict)
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score for retrieved chunks.",
    )


def _query_from_request(body: ChatRequest) -> Query:
    filters = filters_from_request(
        document_ids=body.document_ids,
        metadata_filters=body.metadata_filters,
        min_score=body.min_score,
    )
    return Query(text=body.question, filters=filters)


class AgentChatRequest(BaseModel):
    question: str
    max_iterations: int = Field(default=3, ge=1, le=_MAX_AGENT_ITERATIONS)


class ChatFullResponse(BaseModel):
    answer: str
    sources: list[str]
    latency_ms: float
    token_count: int
    explanations: list[ChunkExplanation] | None = None
    highlights: dict[str, list[str]] | None = None


@router.post("", response_class=StreamingResponse)
async def chat_stream(
    body: ChatRequest,
    pipeline: ChatPipeline = Depends(get_chat_pipeline),
) -> StreamingResponse:
    """Stream the answer as Server-Sent Events.

    Each event is "data: {"token": "..."}".
    The stream ends with "data: [DONE]".
    """

    query = _query_from_request(body)

    async def _generate() -> AsyncIterator[str]:
        async for token in await pipeline.chat(query):
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(_generate(), media_type="text/event-stream")


@router.post("/full", response_model=ChatFullResponse, response_model_exclude_none=True)
async def chat_full(
    body: ChatRequest,
    explain: bool = QueryParam(False, description="Attach per-source retrieval explanations."),
    highlights: bool = QueryParam(
        False,
        description=(
            "Attach verbatim supporting spans per source chunk. Also enabled when "
            + "quality.source_highlighting.enabled is true in config."
        ),
    ),
    pipeline: ChatPipeline = Depends(get_chat_pipeline),
) -> ChatFullResponse:
    """Non-streaming endpoint — returns the complete answer once generated."""
    query = _query_from_request(body)
    answer = await pipeline.chat_full(query, explain=explain, highlights=highlights)
    return ChatFullResponse(
        answer=answer.text,
        sources=answer.sources,
        latency_ms=answer.latency_ms,
        token_count=answer.token_count,
        explanations=answer.explanations,
        highlights=answer.highlights,
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
