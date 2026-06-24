from __future__ import annotations

import dataclasses
import logging
import time

from opentelemetry import trace

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.compression.contextual_compression import ContextualCompressor
from src.rag.ranking.cross_encoder import CrossEncoder
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
    ) -> None:
        self._dense = dense_retriever
        self._hybrid = hybrid_retriever
        self._expander = query_expander
        self._reranker = reranker
        self._compressor = compressor
        self._top_k_retrieval = top_k_retrieval
        self._top_k_rerank = top_k_rerank

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

        # 2. Embed the query (dense vector)
        with _tracer.start_as_current_span("retrieval.embedding"):
            query = self._dense.embed_query(query)

        # 3. Hybrid retrieval (dense HNSW + BM25)
        with _tracer.start_as_current_span("retrieval.hybrid") as span:
            search_results = await self._hybrid.retrieve(query, top_k=self._top_k_retrieval)
            chunks = [c for c, _ in search_results]
            span.set_attribute("chunk_count", len(chunks))
        logger.debug("Hybrid retrieval: %d chunks", len(chunks))

        # 4. Cross-encoder reranking (optional)
        with _tracer.start_as_current_span("retrieval.reranking") as span:
            chunks = self._rerank(query.text, chunks)
            span.set_attribute("chunk_count", len(chunks))

        # 5. Contextual compression (optional)
        with _tracer.start_as_current_span("retrieval.compression") as span:
            chunks = self._compress(query.text, chunks)
            span.set_attribute("chunk_count", len(chunks))

        context = "\n\n".join(c.text for c in chunks)
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

    def _rerank(self, query_text: str, chunks: list[Chunk]) -> list[Chunk]:
        if self._reranker is None or not chunks:
            return chunks
        return self._reranker.rerank(query_text, chunks, top_k=self._top_k_rerank)

    def _compress(self, query_text: str, chunks: list[Chunk]) -> list[Chunk]:
        if self._compressor is None or not chunks:
            return chunks
        return self._compressor.compress(query_text, chunks)
