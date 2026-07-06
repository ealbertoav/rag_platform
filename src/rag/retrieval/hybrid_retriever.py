from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.repositories.vector_store_repository import SearchResult
from src.rag.enrichment.document_augmentation import resolve_synthetic_questions
from src.rag.quality.feedback_loop import apply_feedback_boost
from src.rag.ranking.score_fusion import rrf_fuse, weighted_linear_fuse
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.filters import apply_chunk_filters, apply_min_score

if TYPE_CHECKING:
    from src.rag.retrieval.graph_retriever import GraphRetriever
    from src.rag.retrieval.hierarchical_retriever import HierarchicalRetriever
    from src.rag.retrieval.hyde_retriever import HyDERetriever
    from src.rag.retrieval.hype_retriever import HyPERetriever

logger = logging.getLogger(__name__)

_EXPANSION = 3  # candidate multiplier fed to each retriever before fusion
_MAX_CANDIDATES = 50
_MAX_CANDIDATES_FEEDBACK = 150  # allow pool growth when feedback can reorder results


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
        hyde_retriever: HyDERetriever | None = None,
        hierarchical_retriever: HierarchicalRetriever | None = None,
        fusion_mode: str = "rrf",
        feedback_boost_multiplier: float = 0.0,
        feedback_expand_pool: bool = True,
    ) -> None:
        self._dense = dense
        self._bm25 = bm25
        self.alpha = alpha
        self.graph = graph_retriever
        self.hype = hype_retriever
        self.hyde = hyde_retriever
        self.hierarchical = hierarchical_retriever
        self._fusion_mode = fusion_mode
        self._feedback_boost_multiplier = feedback_boost_multiplier
        self._feedback_expand_pool = feedback_expand_pool

    def _uses_expanded_pool(self) -> bool:
        return self._feedback_boost_multiplier > 0 and self._feedback_expand_pool

    def _candidate_limit(self, top_k: int) -> int:
        """Return the per-source candidate cap before fusion."""
        cap = _MAX_CANDIDATES_FEEDBACK if self._uses_expanded_pool() else _MAX_CANDIDATES
        return min(top_k * _EXPANSION, cap)

    # ── Public ─────────────────────────────────────────────────────────────────

    async def retrieve(
        self,
        query: Query,
        top_k: int,
        *,
        use_hyde: bool = True,
    ) -> list[SearchResult]:
        """Return up to *top_k* (Chunk, score) pairs fused from dense + BM25.

        Uses a 3× candidate pool per source before RRF so that chunks that
        appear in both lists benefit from the rank-boost.
        """
        expansion = self._candidate_limit(top_k)

        tasks = [
            asyncio.to_thread(self._dense.retrieve, query, expansion),
            asyncio.to_thread(self._bm25.search, query.text, expansion, filters=query.filters),
        ]
        if self.graph is not None:
            tasks.append(
                asyncio.to_thread(
                    self.graph.search,
                    query.text,
                    expansion,
                    filters=query.filters,
                )
            )
        if self.hype is not None:
            tasks.append(asyncio.to_thread(self.hype.retrieve, query, expansion))
        if self.hyde is not None and use_hyde:
            tasks.append(asyncio.to_thread(self.hyde.retrieve, query, expansion))
        if self.hierarchical is not None:
            tasks.append(asyncio.to_thread(self.hierarchical.retrieve, query, expansion))

        gathered = await asyncio.gather(*tasks)

        def lookup(chunk_id: str) -> Chunk | None:
            chunk = self._bm25.get_by_id(chunk_id)
            return chunk if isinstance(chunk, Chunk) else None

        dense_results = apply_min_score(
            apply_chunk_filters(resolve_synthetic_questions(gathered[0], lookup), query.filters),
            query.filters,
        )
        bm25_results = apply_chunk_filters(
            resolve_synthetic_questions(gathered[1], lookup),
            query.filters,
        )
        graph_idx = 2
        graph_results: list[SearchResult] = []
        if self.graph is not None:
            graph_results = apply_chunk_filters(
                resolve_synthetic_questions(gathered[graph_idx], lookup),
                query.filters,
            )
            graph_idx += 1
        hype_results: list[SearchResult] = []
        if self.hype is not None:
            hype_results = apply_min_score(
                apply_chunk_filters(gathered[graph_idx], query.filters),
                query.filters,
            )
            graph_idx += 1
        hyde_results: list[SearchResult] = []
        if self.hyde is not None and use_hyde:
            hyde_results = apply_min_score(
                apply_chunk_filters(
                    resolve_synthetic_questions(gathered[graph_idx], lookup),
                    query.filters,
                ),
                query.filters,
            )
            graph_idx += 1
        hierarchical_results: list[SearchResult] = []
        if self.hierarchical is not None:
            hierarchical_results = apply_min_score(
                apply_chunk_filters(gathered[graph_idx], query.filters),
                query.filters,
            )

        hyde_active = self.hyde is not None and use_hyde
        fusion_top_k = top_k
        if self._uses_expanded_pool():
            fusion_top_k = self._candidate_limit(top_k)
        if (
            self._fusion_mode == "weighted_linear"
            and self.graph is None
            and self.hype is None
            and not hyde_active
            and self.hierarchical is None
        ):
            fused = weighted_linear_fuse(
                dense_results, bm25_results, alpha=self.alpha, top_k=fusion_top_k
            )
        else:
            fused = rrf_fuse(
                dense_results,
                bm25_results,
                graph_results,
                hype_results,
                hyde_results,
                hierarchical_results,
                top_k=fusion_top_k,
            )
        fused = apply_feedback_boost(
            fused,
            boost_multiplier=self._feedback_boost_multiplier,
            vector_store=self._dense.vector_store if self._feedback_boost_multiplier > 0 else None,
        )
        fused = fused[:top_k]
        logger.debug(
            (
                "Hybrid retrieval: %d dense + %d bm25 + %d graph + %d hype "
                "+ %d hyde + %d hierarchical → %d fused"
            ),
            len(dense_results),
            len(bm25_results),
            len(graph_results),
            len(hype_results),
            len(hyde_results),
            len(hierarchical_results),
            len(fused),
        )
        return fused

    def retrieve_sync(self, query: Query, top_k: int) -> list[SearchResult]:
        """Synchronous wrapper for contexts that cannot await."""
        return asyncio.run(self.retrieve(query, top_k))
