from __future__ import annotations

from fastapi import Request

from src.rag.pipelines.agent_pipeline import AgentPipeline
from src.rag.pipelines.chat_pipeline import ChatPipeline
from src.rag.pipelines.ingestion_pipeline import IngestionPipeline


def get_chat_pipeline(request: Request) -> ChatPipeline:
    pipeline: ChatPipeline = request.app.state.chat_pipeline
    return pipeline


def get_agent_pipeline(request: Request) -> AgentPipeline:
    pipeline: AgentPipeline = request.app.state.agent_pipeline
    return pipeline


def get_ingestion_pipeline(request: Request) -> IngestionPipeline:
    pipeline: IngestionPipeline = request.app.state.ingestion_pipeline
    return pipeline
