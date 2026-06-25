from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routers import chat, evals, health, ingest
from src.api.routers.metrics_router import router as metrics_router
from src.core.logging import configure_logging
from src.core.settings import settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize pipeline singletons on startup; persist indexes on shutdown."""
    configure_logging()
    logger.info("Starting RAG platform (lifespan startup)")

    # Build pipeline objects — actual model loading is lazy (on first request).
    from src.rag.pipelines.agent_pipeline import AgentPipeline
    from src.rag.pipelines.chat_pipeline import ChatPipeline
    from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

    _app.state.chat_pipeline = ChatPipeline.from_settings()
    _app.state.agent_pipeline = AgentPipeline.from_settings()
    _app.state.ingestion_pipeline = IngestionPipeline.from_settings()
    _app.state.models_loaded = True

    logger.info("Pipelines initialised — ready to serve")
    yield

    logger.info("Shutting down — persisting BM25 index")
    try:
        _app.state.ingestion_pipeline.save_indexes()
    except Exception as exc:
        logger.warning("Failed to save BM25 index on shutdown: %s", exc)


def create_app() -> FastAPI:
    _app = FastAPI(
        title="RAG Platform",
        version="0.1.0",
        description="Local enterprise RAG — Hybrid Search, BGE-M3, llama.cpp",
        lifespan=lifespan,
    )

    _app.add_middleware(  # type: ignore[arg-type]
        CORSMiddleware,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.include_router(health.router)
    _app.include_router(ingest.router)
    _app.include_router(chat.router)
    _app.include_router(evals.router)
    _app.include_router(metrics_router)

    return _app


app = create_app()
