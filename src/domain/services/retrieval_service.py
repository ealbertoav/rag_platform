from __future__ import annotations

import asyncio
import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

from opentelemetry import trace

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.chunking.contextual_headers import join_chunk_context
from src.rag.compression.contextual_compression import ContextualCompressor
from src.rag.enrichment.parent_context_resolver import (
    ChunkLookup,
    drop_redundant_parent_hits,
    enrich_with_parent_context,
)
from src.rag.enrichment.relevant_segment_extraction import merge_adjacent
from src.rag.quality.feedback_loop import apply_feedback_boost
from src.rag.ranking.cross_encoder import CrossEncoder
from src.rag.ranking.diversity import mmr_select
from src.rag.ranking.score_fusion import rrf_fuse
from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.query_expansion import QueryExpander

if TYPE_CHECKING:
    from src.domain.repositories.embedding_repository import EmbeddingRepository
    from src.domain.repositories.llm_repository import LLMRepository
    from src.domain.repositories.vector_store_repository import VectorStoreRepository
    from src.rag.retrieval.adaptive.query_classifier import QueryClassifier
    from src.rag.retrieval.adaptive.strategies import (
        AdaptiveStrategyRegistry,
        RetrievalStrategyParams,
    )
    from src.rag.retrieval.hybrid_retriever import HybridRetriever

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.retrieval")


@dataclasses.dataclass
class RetrievalResult:
    """Full output of one retrieval run."""

    query: Query  # with embedding (and expanded_texts) populated
    chunks: list[Chunk]  # final, compressed chunks ready for the LLM
    context: str  # chunks joined for direct injection into the prompt
    latency_ms: float = 0.0
    relevance_scores: list[float] = dataclasses.field(default_factory=list)
    """All Reliable RAG grades (pass + fail) for CRAG quality scoring."""


class RetrievalService:
    """Orchestrates the retrieval pipeline steps.

    All components are optional; the pipeline gracefully degrades:
    - no "query_expander" → single-query retrieval
    - no "reranker"       → hybrid results passed directly to compression
    - no "compressor"     → raw reranked chunks used as context
    - no "reliable_rag"   → all reranked/enriched chunks reach compression
    """

    hybrid: HybridRetriever

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        query_expander: QueryExpander | None = None,
        query_classifier: QueryClassifier | None = None,
        strategy_registry: AdaptiveStrategyRegistry | None = None,
        reranker: CrossEncoder | None = None,
        compressor: ContextualCompressor | None = None,
        top_k_retrieval: int = 50,
        top_k_rerank: int = 10,
        top_k_final: int = 5,
        rse_enabled: bool = False,
        rse_max_segment_tokens: int = 1500,
        parent_context_enabled: bool = False,
        parent_child_strategy: bool = False,
        chunk_lookup: ChunkLookup | None = None,
        diversity_enabled: bool = False,
        diversity_lambda: float = 0.7,
        embedder: EmbeddingRepository | None = None,
        reliable_rag_enabled: bool = False,
        reliable_rag_min_score: float = 0.5,
        llm: LLMRepository | None = None,
        feedback_boost_multiplier: float = 0.0,
        vector_store: VectorStoreRepository | None = None,
    ) -> None:
        self._dense: DenseRetriever = dense_retriever
        self.hybrid = hybrid_retriever
        self._expander: QueryExpander | None = query_expander
        self._classifier: QueryClassifier | None = query_classifier
        self._strategy_registry: AdaptiveStrategyRegistry | None = strategy_registry
        self._reranker: CrossEncoder | None = reranker
        self._compressor: ContextualCompressor | None = compressor
        self._top_k_retrieval: int = top_k_retrieval
        self._top_k_rerank: int = top_k_rerank
        self._top_k_final: int = top_k_final
        self._rse_enabled: bool = rse_enabled
        self._rse_max_segment_tokens: int = rse_max_segment_tokens
        self._parent_context_enabled: Any = parent_context_enabled and parent_child_strategy
        self._chunk_lookup: ChunkLookup | None = chunk_lookup
        self._diversity_enabled: bool = diversity_enabled
        self._diversity_lambda: float = diversity_lambda
        self._embedder: EmbeddingRepository | None = embedder
        self._reliable_rag_enabled: bool = reliable_rag_enabled
        self._reliable_rag_min_score: float = reliable_rag_min_score
        self._llm: LLMRepository | None = llm
        self._feedback_boost_multiplier: float = feedback_boost_multiplier
        self._vector_store: VectorStoreRepository | None = vector_store

    # ── Public ─────────────────────────────────────────────────────────────────

    async def retrieve(self, query: Query) -> RetrievalResult:
        """Execute the full retrieval flow and return a "RetrievalResult"."""
        t0 = time.monotonic()
        relevance_scores: list[float] = []

        # 0. Adaptive query classification (optional)
        query = self._classify(query)
        strategy = self._resolve_strategy(query)

        # 1. Query expansion (optional)
        with _tracer.start_as_current_span("retrieval.expansion"):
            query = self._expand(query, strategy)

        # 2. Multi-query hybrid retrieval (original + expanded variants)
        with _tracer.start_as_current_span("retrieval.multi_query_fusion") as span:
            search_results = await self._retrieve_variants(query, strategy)
            chunks = [c for c, _ in search_results]
            variant_count = len(self._query_variants(query))
            span.set_attribute("variant_count", variant_count)
            span.set_attribute("chunk_count", len(chunks))
            if query.embedding is None:
                query = self._dense.embed_query(query)

        logger.debug("Hybrid retrieval: %d chunks from %d variants", len(chunks), variant_count)

        # 3. Cross-encoder reranking (optional)
        with _tracer.start_as_current_span("retrieval.reranking") as span:
            chunks = self._rerank(query.text, chunks)
            span.set_attribute("chunk_count", len(chunks))

        # 3b. MMR diversity selection (optional, after rerank, before compression)
        if self._diversity_enabled:
            with _tracer.start_as_current_span("retrieval.diversity") as span:
                chunks = self._apply_diversity(chunks)
                span.set_attribute("chunk_count", len(chunks))
                span.set_attribute("diversity.lambda", self._diversity_lambda)

        # 4. Relevant segment extraction (optional)
        if self._rse_enabled:
            with _tracer.start_as_current_span("retrieval.rse") as span:
                chunks, merge_count = merge_adjacent(chunks, self._rse_max_segment_tokens)
                span.set_attribute("merge_count", merge_count)
                span.set_attribute("chunk_count", len(chunks))

        # 5. Parent context expansion (optional, parent_child strategy only)
        if self._parent_context_enabled and self._chunk_lookup is not None:
            with _tracer.start_as_current_span("retrieval.parent_context") as span:
                chunks, resolved_count = enrich_with_parent_context(chunks, self._chunk_lookup)
                chunks = drop_redundant_parent_hits(chunks)
                span.set_attribute("resolved_count", resolved_count)
                span.set_attribute("chunk_count", len(chunks))

        # 5b. Reliable RAG relevance grading (optional, after enrichment, before compression)
        if self._reliable_rag_enabled:
            with _tracer.start_as_current_span("retrieval.relevance_grading") as span:
                chunks, pass_count, fail_count, relevance_scores = self._apply_relevance_grading(
                    query.text, chunks
                )
                span.set_attribute("chunk_count", len(chunks))
                span.set_attribute("relevance.pass_count", pass_count)
                span.set_attribute("relevance.fail_count", fail_count)
                span.set_attribute("relevance.min_score", self._reliable_rag_min_score)

        # 6. Contextual compression (optional)
        with _tracer.start_as_current_span("retrieval.compression") as span:
            chunks = self._compress(query.text, chunks, strategy)
            span.set_attribute("chunk_count", len(chunks))

        # 7. Final top-K cap
        chunks = chunks[: self._top_k_final]

        context = join_chunk_context(chunks)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "Retrieval: %d chunks, %d context chars, %.1fms",
            len(chunks),
            len(context),
            elapsed,
        )
        return RetrievalResult(
            query=query,
            chunks=chunks,
            context=context,
            latency_ms=elapsed,
            relevance_scores=relevance_scores,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _classify(self, query: Query) -> Query:
        if self._classifier is None:
            return query
        return self._classifier.classify(query)

    def _resolve_strategy(self, query: Query) -> RetrievalStrategyParams | None:
        if self._strategy_registry is None:
            return None
        from src.rag.retrieval.adaptive.strategies import record_strategy_span

        category = query.metadata.get("category")
        params = self._strategy_registry.resolve_params(category)
        with _tracer.start_as_current_span("retrieval.adaptive.strategy"):
            record_strategy_span(category, params)
        return params

    def _expand(self, query: Query, strategy: RetrievalStrategyParams | None = None) -> Query:
        if self._expander is None:
            return query
        n_variants = strategy.n_variants if strategy is not None else None
        return self._expander.expand(query, n_variants=n_variants)

    @staticmethod
    def _query_variants(query: Query) -> list[str]:
        """Return deduplicated query strings for multi-query retrieval."""
        seen: set[str] = set()
        variants: list[str] = []
        step_back = query.metadata.get("step_back")
        texts = [query.text, *query.expanded_texts]
        if isinstance(step_back, str):
            texts.append(step_back)
        for text in texts:
            normalized = text.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
        return variants

    async def _retrieve_variants(
        self,
        query: Query,
        strategy: RetrievalStrategyParams | None = None,
    ) -> list[tuple[Chunk, float]]:
        top_k = strategy.top_k if strategy is not None else self._top_k_retrieval
        use_hyde = strategy.hyde if strategy is not None else True
        variants = self._query_variants(query)
        if len(variants) == 1:
            embedded = self._dense.embed_query(query)
            return await self.hybrid.retrieve(embedded, top_k=top_k, use_hyde=use_hyde)

        async def _search_variant(text: str) -> list[tuple[Chunk, float]]:
            variant_query = query.model_copy(update={"text": text, "embedding": None})
            variant_embedded = await asyncio.to_thread(self._dense.embed_query, variant_query)
            return await self.hybrid.retrieve(
                variant_embedded,
                top_k=top_k,
                use_hyde=use_hyde,
            )

        gathered = await asyncio.gather(*[_search_variant(text) for text in variants])
        if len(gathered) == 1:
            return gathered[0]
        fused = rrf_fuse(*gathered, top_k=top_k)
        if self._feedback_boost_multiplier > 0:
            fused = apply_feedback_boost(
                fused,
                boost_multiplier=self._feedback_boost_multiplier,
                vector_store=self._vector_store,
            )
        return fused[:top_k]

    def _rerank(self, query_text: str, chunks: list[Chunk]) -> list[Chunk]:
        if self._reranker is None or not chunks:
            return chunks
        return self._reranker.rerank(
            query_text,
            chunks,
            top_k=self._top_k_rerank,
            boost_multiplier=self._feedback_boost_multiplier,
            vector_store=self._vector_store,
        )

    def _apply_relevance_grading(
        self,
        query_text: str,
        chunks: list[Chunk],
    ) -> tuple[list[Chunk], int, int, list[float]]:
        if not chunks:
            return [], 0, 0, []
        if self._llm is None:
            logger.warning(
                "Reliable RAG enabled but no LLM configured — skipping relevance grading"
            )
            return chunks, len(chunks), 0, []

        from src.rag.quality.reliable_rag import grade_relevance

        return grade_relevance(
            query_text,
            chunks,
            self._llm,
            min_score=self._reliable_rag_min_score,
        )

    def _apply_diversity(self, chunks: list[Chunk]) -> list[Chunk]:
        if not chunks:
            return chunks
        embeddings = self._resolve_chunk_embeddings(chunks)
        if embeddings is None:
            logger.warning(
                "Skipping MMR diversity: could not resolve document embeddings for all chunks"
            )
            return chunks
        return mmr_select(
            chunks,
            embeddings,
            self._diversity_lambda,
            top_k=self._top_k_rerank,
        )

    def _resolve_chunk_embeddings(self, chunks: list[Chunk]) -> list[list[float]] | None:
        """Resolve document-space dense vectors for MMR pairwise similarity."""
        if not chunks:
            return []

        resolved: list[list[float] | None] = [None] * len(chunks)
        missing: list[tuple[int, Chunk]] = []

        for index, chunk in enumerate(chunks):
            if chunk.embedding is not None:
                resolved[index] = chunk.embedding
            else:
                missing.append((index, chunk))

        if missing and self._chunk_lookup is not None:
            still_missing: list[tuple[int, Chunk]] = []
            for index, chunk in missing:
                stored = self._chunk_lookup.get_by_id(chunk.id)
                if stored is not None and stored.embedding is not None:
                    resolved[index] = stored.embedding
                else:
                    still_missing.append((index, chunk))
            missing = still_missing

        if missing:
            if self._embedder is None:
                logger.warning(
                    "Diversity enabled but %d/%d chunks lack embeddings and no embedder configured",
                    len(missing),
                    len(chunks),
                )
                return None
            texts = [chunk.text for _, chunk in missing]
            embedded = self._embedder.embed_passage(texts)
            for (index, _), vector in zip(missing, embedded, strict=True):
                resolved[index] = vector

        if any(vector is None for vector in resolved):
            return None
        return [vector for vector in resolved if vector is not None]

    def _compress(
        self,
        query_text: str,
        chunks: list[Chunk],
        strategy: RetrievalStrategyParams | None = None,
    ) -> list[Chunk]:
        if strategy is not None and not strategy.compression:
            return chunks
        if self._compressor is None or not chunks:
            return chunks
        return self._compressor.compress(query_text, chunks)
