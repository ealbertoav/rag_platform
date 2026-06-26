from __future__ import annotations

import asyncio
import dataclasses
import logging
import time

from opentelemetry import trace

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.chunking.contextual_headers import chunk_context_text
from src.rag.compression.contextual_compression import ContextualCompressor
from src.rag.enrichment.relevant_segment_extraction import merge_adjacent
from src.rag.ranking.cross_encoder import CrossEncoder
from src.rag.ranking.score_fusion import rrf_fuse
from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from src.rag.retrieval.query_expansion import QueryExpander

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.retrieval")


@dataclasses.dataclass
class RetrievalResult:
    """Full output of one retrieval run."""

    query: Query  # with embedding (and expanded_texts) populated
    chunks: list[Chunk]  # final, compressed chunks ready for the LLM
    context: str  # chunks joined for direct injection into the prompt
    latency_ms: float = 0.0


class RetrievalService:
    """Orchestrates the retrieval pipeline steps.

    All components are optional; the pipeline gracefully degrades:
    - no "query_expander" → single-query retrieval
    - no "reranker"       → hybrid results passed directly to compression
    - no "compressor"     → raw reranked chunks used as context
    """

    def __init__(
        self,
        dense_retriever: DenseRetriever,
        hybrid_retriever: HybridRetriever,
        query_expander: QueryExpander | None = None,
        reranker: CrossEncoder | None = None,
        compressor: ContextualCompressor | None = None,
        top_k_retrieval: int = 50,
        top_k_rerank: int = 10,
        top_k_final: int = 5,
        rse_enabled: bool = False,
        rse_max_segment_tokens: int = 1500,
    ) -> None:
        self._dense = dense_retriever
        self._hybrid = hybrid_retriever
        self._expander = query_expander
        self._reranker = reranker
        self._compressor = compressor
        self._top_k_retrieval = top_k_retrieval
        self._top_k_rerank = top_k_rerank
        self._top_k_final = top_k_final
        self._rse_enabled = rse_enabled
        self._rse_max_segment_tokens = rse_max_segment_tokens

    @property
    def hybrid(self) -> HybridRetriever:
        return self._hybrid

    # ── Public ─────────────────────────────────────────────────────────────────

    async def retrieve(self, query: Query) -> RetrievalResult:
        """Execute the full retrieval flow and return a "RetrievalResult"."""
        t0 = time.monotonic()

        # 1. Query expansion (optional)
        with _tracer.start_as_current_span("retrieval.expansion"):
            query = self._expand(query)

        # 2. Multi-query hybrid retrieval (original + expanded variants)
        with _tracer.start_as_current_span("retrieval.multi_query_fusion") as span:
            search_results = await self._retrieve_variants(query)
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

        # 4. Relevant segment extraction (optional)
        if self._rse_enabled:
            with _tracer.start_as_current_span("retrieval.rse") as span:
                chunks, merge_count = merge_adjacent(chunks, self._rse_max_segment_tokens)
                span.set_attribute("merge_count", merge_count)
                span.set_attribute("chunk_count", len(chunks))

        # 5. Contextual compression (optional)
        with _tracer.start_as_current_span("retrieval.compression") as span:
            chunks = self._compress(query.text, chunks)
            span.set_attribute("chunk_count", len(chunks))

        # 6. Final top-K cap
        chunks = chunks[: self._top_k_final]

        context = "\n\n".join(chunk_context_text(c) for c in chunks)
        elapsed = (time.monotonic() - t0) * 1000
        logger.info(
            "Retrieval: %d chunks, %d context chars, %.1fms",
            len(chunks),
            len(context),
            elapsed,
        )
        return RetrievalResult(query=query, chunks=chunks, context=context, latency_ms=elapsed)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _expand(self, query: Query) -> Query:
        if self._expander is None:
            return query
        return self._expander.expand(query)

    @staticmethod
    def _query_variants(query: Query) -> list[str]:
        """Return deduplicated query strings for multi-query retrieval."""
        seen: set[str] = set()
        variants: list[str] = []
        for text in [query.text, *query.expanded_texts]:
            normalized = text.strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                variants.append(normalized)
        return variants

    async def _retrieve_variants(self, query: Query) -> list[tuple[Chunk, float]]:
        variants = self._query_variants(query)
        if len(variants) == 1:
            embedded = self._dense.embed_query(query)
            return await self._hybrid.retrieve(embedded, top_k=self._top_k_retrieval)

        async def _search_variant(text: str) -> list[tuple[Chunk, float]]:
            variant_query = query.model_copy(update={"text": text, "embedding": None})
            variant_embedded = await asyncio.to_thread(self._dense.embed_query, variant_query)
            return await self._hybrid.retrieve(variant_embedded, top_k=self._top_k_retrieval)

        gathered = await asyncio.gather(*[_search_variant(text) for text in variants])
        if len(gathered) == 1:
            return gathered[0]
        return rrf_fuse(*gathered, top_k=self._top_k_retrieval)

    def _rerank(self, query_text: str, chunks: list[Chunk]) -> list[Chunk]:
        if self._reranker is None or not chunks:
            return chunks
        return self._reranker.rerank(query_text, chunks, top_k=self._top_k_rerank)

    def _compress(self, query_text: str, chunks: list[Chunk]) -> list[Chunk]:
        if self._compressor is None or not chunks:
            return chunks
        return self._compressor.compress(query_text, chunks)

