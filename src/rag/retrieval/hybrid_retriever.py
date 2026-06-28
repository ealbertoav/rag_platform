from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.domain.entities.query import Query
from src.domain.repositories.vector_store_repository import SearchResult
from src.rag.enrichment.document_augmentation import resolve_synthetic_questions
from src.rag.ranking.score_fusion import rrf_fuse, weighted_linear_fuse
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.retrieval.dense_retriever import DenseRetriever

if TYPE_CHECKING:
    from src.rag.retrieval.graph_retriever import GraphRetriever
    from src.rag.retrieval.hierarchical_retriever import HierarchicalRetriever
    from src.rag.retrieval.hype_retriever import HyPERetriever

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
        hype_retriever: HyPERetriever | None = None,
        hierarchical_retriever: HierarchicalRetriever | None = None,
        fusion_mode: str = "rrf",
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self.alpha = alpha
        self._graph = graph_retriever
        self._hype = hype_retriever
        self._hierarchical = hierarchical_retriever
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
        if self._hype is not None:
            tasks.append(asyncio.to_thread(self._hype.retrieve, query, expansion))
        if self._hierarchical is not None:
            tasks.append(asyncio.to_thread(self._hierarchical.retrieve, query, expansion))

        gathered = await asyncio.gather(*tasks)
        dense_results = resolve_synthetic_questions(
            gathered[0],
            lambda chunk_id: self._bm25.get_by_id(chunk_id),  # type: ignore[arg-type, return-value]
        )
        bm25_results = resolve_synthetic_questions(
            gathered[1],
            lambda chunk_id: self._bm25.get_by_id(chunk_id),  # type: ignore[arg-type, return-value]
        )
        graph_idx = 2
        graph_results: list[SearchResult] = []
        if self._graph is not None:
            graph_results = resolve_synthetic_questions(
                gathered[graph_idx],
                lambda chunk_id: self._bm25.get_by_id(chunk_id),  # type: ignore[arg-type, return-value]
            )
            graph_idx += 1
        hype_results: list[SearchResult] = []
        if self._hype is not None:
            hype_results = gathered[graph_idx]
            graph_idx += 1
        hierarchical_results: list[SearchResult] = []
        if self._hierarchical is not None:
            hierarchical_results = gathered[graph_idx]

        if (
            self._fusion_mode == "weighted_linear"
            and self._graph is None
            and self._hype is None
            and self._hierarchical is None
        ):
            fused = weighted_linear_fuse(dense_results, bm25_results, alpha=self.alpha, top_k=top_k)
        else:
            fused = rrf_fuse(
                dense_results,
                bm25_results,
                graph_results,
                hype_results,
                hierarchical_results,
                top_k=top_k,
            )
        logger.debug(
            (
                "Hybrid retrieval: %d dense + %d bm25 + %d graph + %d hype "
                "+ %d hierarchical → %d fused"
            ),
            len(dense_results),
            len(bm25_results),
            len(graph_results),
            len(hype_results),
            len(hierarchical_results),
            len(fused),
        )
        return fused

    def retrieve_sync(self, query: Query, top_k: int) -> list[SearchResult]:
        """Synchronous wrapper for contexts that cannot await."""
        return asyncio.run(self.retrieve(query, top_k))

    @property
    def graph(self) -> GraphRetriever | None:
        return self._graph

    @property
    def hype(self) -> HyPERetriever | None:
        return self._hype

    @property
    def hierarchical(self) -> HierarchicalRetriever | None:
        return self._hierarchical
