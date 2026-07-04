from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from opentelemetry import trace

from src.core.settings import settings
from src.domain.entities.query import Query
from src.domain.repositories.llm_repository import LLMRepository
from src.domain.services.retrieval_service import RetrievalResult, RetrievalService
from src.observability.metrics import record_retrieval

if TYPE_CHECKING:
    from src.domain.repositories.vector_store_repository import VectorStoreRepository
    from src.infrastructure.vectordb.bm25 import BM25Index

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.retrieval")


def _build_graph_retriever(
    llm: LLMRepository,
    bm25: object,
) -> object | None:
    """Return a GraphRetriever when Neo4j is enabled, else None."""
    if not settings.neo4j.enabled:
        return None
    try:
        from src.rag.retrieval.graph_retriever import GraphRetriever

        return GraphRetriever.from_settings(llm=llm, bm25=bm25)  # type: ignore[arg-type]
    except Exception as exc:
        logger.warning("Graph retriever unavailable (continuing without it): %s", exc)
        return None


def _build_hype_retriever(
    embedder: object,
    vector_store: VectorStoreRepository,
    bm25: object,
) -> object | None:
    """Return a HyPERetriever when HyPE is enabled, else None."""
    if not settings.retrieval.hype.enabled:
        return None
    try:
        from src.rag.retrieval.hype_retriever import HyPERetriever

        return HyPERetriever(
            embedder=embedder,  # type: ignore[arg-type]
            vector_store=vector_store,
            chunk_lookup=bm25,
        )
    except Exception as exc:
        logger.warning("HyPE retriever unavailable (continuing without it): %s", exc)
        return None


def _build_hyde_retriever(
    llm: LLMRepository,
    embedder: object,
    vector_store: VectorStoreRepository,
    *,
    enabled: bool | None = None,
) -> object | None:
    """Return a HyDERetriever when HyDE is enabled, else None."""
    if enabled is None:
        enabled = settings.retrieval.hyde.enabled
    if not enabled:
        return None
    try:
        from src.rag.retrieval.hyde_retriever import HyDERetriever

        return HyDERetriever(
            llm=llm,
            embedder=embedder,  # type: ignore[arg-type]
            vector_store=vector_store,
        )
    except Exception as exc:
        logger.warning("HyDE retriever unavailable (continuing without it): %s", exc)
        return None


def _build_hierarchical_retriever(
    embedder: object,
    vector_store: VectorStoreRepository,
) -> object | None:
    """Return a HierarchicalRetriever when hierarchical indexing is enabled, else None."""
    if not settings.chunking.hierarchical.enabled:
        return None
    try:
        from src.rag.retrieval.hierarchical_retriever import HierarchicalRetriever

        return HierarchicalRetriever(
            embedder=embedder,  # type: ignore[arg-type]
            vector_store=vector_store,
            summary_top_k=settings.chunking.hierarchical.summary_top_k,
        )
    except Exception as exc:
        logger.warning("Hierarchical retriever unavailable (continuing without it): %s", exc)
        return None


class RetrievalPipeline:
    """Thin wrapper around "RetrievalService" that adds OTel span context.

    Keeps instrumentation concerns out of the domain service while giving
    every retrieval run a root span that downstream steps can attach to.
    """

    def __init__(self, service: RetrievalService) -> None:
        self.service = service

    # ── Public ─────────────────────────────────────────────────────────────────

    async def retrieve(self, query: Query) -> RetrievalResult:
        with _tracer.start_as_current_span("retrieval") as span:
            result = await self.service.retrieve(query)
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
        bm25_index: BM25Index | None = None,
        vector_store: VectorStoreRepository | None = None,
    ) -> RetrievalPipeline:
        """Build the full retrieval pipeline from settings.

        *llm* can be injected by the caller (e.g. "ChatPipeline.from_settings"
        reuses the same model instance).  When omitted, "LlamaCppProvider"
        is created from settings so the pipeline is fully self-contained.

        *bm25_index* can be shared with "IngestionPipeline" so API ingest
        updates are visible to retrieval without reloading from disk.
        """
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
        if vector_store is None:
            vector_store = QdrantVectorStore.from_settings()
        bm25_index = bm25_index or BM25Index.load_or_create()
        bm25 = BM25Retriever(bm25_index)

        dense = DenseRetriever(embedder=embedder, vector_store=vector_store)
        graph = _build_graph_retriever(llm, bm25)
        hype = _build_hype_retriever(embedder, vector_store, bm25)
        # Build HyDE when globally enabled or when adaptive may opt in per query.
        hyde = _build_hyde_retriever(
            llm,
            embedder,
            vector_store,
            enabled=cfg.hyde.enabled or cfg.adaptive.enabled,
        )
        hierarchical = _build_hierarchical_retriever(embedder, vector_store)
        hybrid = HybridRetriever(
            dense=dense,
            bm25=bm25,
            alpha=cfg.hybrid_alpha,
            graph_retriever=graph,  # type: ignore[arg-type]
            hype_retriever=hype,  # type: ignore[arg-type]
            hyde_retriever=hyde,  # type: ignore[arg-type]
            hierarchical_retriever=hierarchical,  # type: ignore[arg-type]
            fusion_mode=cfg.hybrid_fusion,
            feedback_boost_multiplier=(
                settings.quality.feedback_loop.boost_multiplier
                if settings.quality.feedback_loop.enabled
                else 0.0
            ),
        )

        expander = None
        if settings.query_expansion.enabled or settings.query_expansion.step_back.enabled:
            expander = QueryExpander.from_settings(llm)
        classifier = None
        strategy_registry = None
        if settings.retrieval.adaptive.enabled:
            from src.rag.retrieval.adaptive.query_classifier import QueryClassifier
            from src.rag.retrieval.adaptive.strategies import AdaptiveStrategyRegistry

            classifier = QueryClassifier.from_settings(llm)
            strategy_registry = AdaptiveStrategyRegistry.from_settings()
        reranker = CrossEncoder.from_settings()
        compressor = (
            ContextualCompressor.from_settings(llm) if settings.compression.enabled else None
        )

        service = RetrievalService(
            dense_retriever=dense,
            hybrid_retriever=hybrid,
            query_expander=expander,
            query_classifier=classifier,
            strategy_registry=strategy_registry,
            reranker=reranker,
            compressor=compressor,
            top_k_retrieval=cfg.top_k_dense,
            top_k_rerank=settings.reranker.top_k,
            top_k_final=cfg.top_k_final,
            rse_enabled=cfg.rse.enabled,
            rse_max_segment_tokens=cfg.rse.max_segment_tokens,
            parent_context_enabled=cfg.parent_context.enabled,
            parent_child_strategy=settings.chunking.strategy == "parent_child",
            chunk_lookup=bm25_index,
            diversity_enabled=cfg.diversity.enabled,
            diversity_lambda=cfg.diversity.lambda_,
            embedder=embedder if cfg.diversity.enabled else None,
            reliable_rag_enabled=settings.quality.reliable_rag.enabled,
            reliable_rag_min_score=settings.quality.reliable_rag.min_score,
            llm=llm if settings.quality.reliable_rag.enabled else None,
            feedback_boost_multiplier=(
                settings.quality.feedback_loop.boost_multiplier
                if settings.quality.feedback_loop.enabled
                else 0.0
            ),
            vector_store=(vector_store if settings.quality.feedback_loop.enabled else None),
        )
        return cls(service=service)
