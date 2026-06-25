from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.domain.entities.query import Query
from src.domain.repositories.vector_store_repository import SearchResult
from src.rag.ranking.score_fusion import rrf_fuse, weighted_linear_fuse
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.retrieval.dense_retriever import DenseRetriever

if TYPE_CHECKING:
    from src.rag.retrieval.graph_retriever import GraphRetriever

logger = logging.getLogger(__name__)

_EXPANSION = 3  # candidate multiplier fed to each retriever before fusion
_MAX_CANDIDATES = 50


class HybridRetriever:
    """Fuses dense (HNSW) and sparse (BM25) retrieval.

    The default mode is Reciprocal Rank Fusion (RRF), which is alpha-independent.
    Set *fusion_mode* to "weighted_linear" to blend dense/sparse scores using
    *alpha* (1.0 = dense only, 0.0 = BM25 only).

    Both searches run concurrently via "asyncio.gather" + "asyncio.to_thread"
    so that the Qdrant network call and the in-memory BM25 lookup overlap.
    """

    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        alpha: float = 0.7,
        graph_retriever: GraphRetriever | None = None,
        fusion_mode: str = "rrf",
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self.alpha = alpha
        self._graph = graph_retriever
        self._fusion_mode = fusion_mode

    # ── Public ─────────────────────────────────────────────────────────────────

    async def retrieve(self, query: Query, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* (Chunk, score) pairs fused from dense + BM25.

        Uses a 3× candidate pool per source before RRF so that chunks that
        appear in both lists benefit from the rank-boost.
        """
        expansion = min(top_k * _EXPANSION, _MAX_CANDIDATES)

        tasks = [
            asyncio.to_thread(self._dense.retrieve, query, expansion),
            asyncio.to_thread(self._bm25.search, query.text, expansion),
        ]
        if self._graph is not None:
            tasks.append(asyncio.to_thread(self._graph.search, query.text, expansion))

        gathered = await asyncio.gather(*tasks)
        dense_results = gathered[0]
        bm25_results = gathered[1]
        graph_results = gathered[2] if self._graph is not None else []

        if self._fusion_mode == "weighted_linear" and self._graph is None:
            fused = weighted_linear_fuse(dense_results, bm25_results, alpha=self.alpha, top_k=top_k)
        else:
            fused = rrf_fuse(dense_results, bm25_results, graph_results, top_k=top_k)
        logger.debug(
            "Hybrid retrieval: %d dense + %d bm25 + %d graph → %d fused",
            len(dense_results),
            len(bm25_results),
            len(graph_results),
            len(fused),
        )
        return fused

    def retrieve_sync(self, query: Query, top_k: int) -> list[SearchResult]:
        """Synchronous wrapper for contexts that cannot await."""
        return asyncio.run(self.retrieve(query, top_k))

    @property
    def graph(self) -> GraphRetriever | None:
        return self._graph
