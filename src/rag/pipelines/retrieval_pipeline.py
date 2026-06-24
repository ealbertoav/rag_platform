from __future__ import annotations

import logging

from opentelemetry import trace

from src.domain.entities.query import Query
from src.domain.repositories.llm_repository import LLMRepository
from src.domain.services.retrieval_service import RetrievalResult, RetrievalService
from src.observability.metrics import record_retrieval

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.retrieval")


class RetrievalPipeline:
    """Thin wrapper around "RetrievalService" that adds OTel span context.

    Keeps instrumentation concerns out of the domain service while giving
    every retrieval run a root span that downstream steps can attach to.
    """

    def __init__(self, service: RetrievalService) -> None:
        self._service = service

    @property
    def service(self) -> RetrievalService:
        return self._service

    # ── Public ─────────────────────────────────────────────────────────────────

    async def retrieve(self, query: Query) -> RetrievalResult:
        with _tracer.start_as_current_span("retrieval") as span:
            result = await self._service.retrieve(query)
            span.set_attribute("chunk_count", len(result.chunks))
            span.set_attribute("context_chars", len(result.context))
            span.set_attribute("latency_ms", round(result.latency_ms, 1))
        record_retrieval(len(result.chunks), result.latency_ms / 1000)
        return result

    def retrieve_sync(self, query: Query) -> RetrievalResult:
        """Synchronous wrapper for callers that cannot "await"."""
        import asyncio

        return asyncio.run(self.retrieve(query))

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(
        cls,
        llm: LLMRepository | None = None,
    ) -> RetrievalPipeline:
        """Build the full retrieval pipeline from settings.

        *llm* can be injected by the caller (e.g. "ChatPipeline.from_settings"
        reuses the same model instance).  When omitted, "LlamaCppProvider"
        is created from settings so the pipeline is fully self-contained.
        """
        from src.core.settings import settings
        from src.infrastructure.embeddings import get_embedding_provider
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
        from src.infrastructure.vectordb.bm25 import BM25Index
        from src.infrastructure.vectordb.qdrant import QdrantVectorStore
        from src.rag.compression.contextual_compression import ContextualCompressor
        from src.rag.ranking.cross_encoder import CrossEncoder
        from src.rag.retrieval.bm25_retriever import BM25Retriever
        from src.rag.retrieval.dense_retriever import DenseRetriever
        from src.rag.retrieval.hybrid_retriever import HybridRetriever
        from src.rag.retrieval.query_expansion import QueryExpander

        if llm is None:
            llm = LlamaCppProvider.from_settings()

        cfg = settings.retrieval
        embedder = get_embedding_provider()
        vector_store = QdrantVectorStore.from_settings()
        bm25 = BM25Retriever(BM25Index.load_or_create())

        dense = DenseRetriever(embedder=embedder, vector_store=vector_store)
        hybrid = HybridRetriever(dense=dense, bm25=bm25, alpha=cfg.hybrid_alpha)

        expander = QueryExpander.from_settings(llm) if settings.query_expansion.enabled else None
        reranker = CrossEncoder.from_settings()
        compressor = (
            ContextualCompressor.from_settings(llm) if settings.compression.enabled else None
        )

        service = RetrievalService(
            dense_retriever=dense,
            hybrid_retriever=hybrid,
            query_expander=expander,
            reranker=reranker,
            compressor=compressor,
            top_k_retrieval=cfg.top_k_dense,
            top_k_rerank=settings.reranker.top_k,
        )
        return cls(service=service)
