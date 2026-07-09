from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, Awaitable, Callable, MutableMapping, Sequence
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.rate_limit import RateLimitHTTPMiddleware, configure_rate_limit
from src.api.routers import chat, evals, feedback, health, ingest
from src.api.routers.metrics_router import router as metrics_router
from src.core.logging import configure_logging
from src.core.settings import settings

logger = logging.getLogger(__name__)

AsgiScope = MutableMapping[str, Any]
AsgiReceive = Callable[[], Awaitable[Any]]
AsgiSend = Callable[[Any], Awaitable[None]]
AsgiApp = Callable[[AsgiScope, AsgiReceive, AsgiSend], Awaitable[None]]


def _cors_middleware_factory(
    asgi_app: AsgiApp,
    /,
    allow_origins: Sequence[str],
    allow_credentials: bool,
    allow_methods: Sequence[str],
    allow_headers: Sequence[str],
) -> AsgiApp:
    return CORSMiddleware(
        asgi_app,
        allow_origins=allow_origins,
        allow_credentials=allow_credentials,
        allow_methods=allow_methods,
        allow_headers=allow_headers,
    )


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize pipeline singletons on startup; persist indexes on shutdown."""
    configure_logging()
    logger.info("Starting RAG platform (lifespan startup)")

    # Build pipeline objects — actual model loading is lazy (on first request).
    from src.domain.repositories.vector_store_repository import VectorStoreRepository
    from src.infrastructure.vectordb.bm25 import BM25Index
    from src.infrastructure.vectordb.feedback_store import build_vector_store_from_settings
    from src.rag.pipelines.agent_pipeline import AgentPipeline
    from src.rag.pipelines.chat_pipeline import ChatPipeline
    from src.rag.pipelines.ingestion_pipeline import IngestionPipeline

    bm25 = BM25Index.load_or_create()
    vector_store: VectorStoreRepository = build_vector_store_from_settings()
    _app.state.bm25_index = bm25
    _app.state.vector_store = vector_store
    _app.state.chat_pipeline = ChatPipeline.from_settings(
        bm25_index=bm25,
        vector_store=vector_store,
    )
    _app.state.agent_pipeline = AgentPipeline.from_settings(
        bm25_index=bm25,
        vector_store=vector_store,
    )
    _app.state.ingestion_pipeline = IngestionPipeline.from_settings(
        bm25=bm25,
        vector_store=vector_store,
    )
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

    _ = configure_rate_limit()
    _app.add_middleware(RateLimitHTTPMiddleware)
    _app.add_middleware(
        _cors_middleware_factory,
        allow_origins=settings.api.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    _app.include_router(health.router)
    _app.include_router(ingest.router)
    _app.include_router(chat.router)
    _app.include_router(feedback.router)
    _app.include_router(evals.router)
    _app.include_router(metrics_router)

    return _app


app = create_app()
