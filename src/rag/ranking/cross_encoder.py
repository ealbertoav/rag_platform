from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository
from src.rag.quality.feedback_loop import apply_feedback_boost

if TYPE_CHECKING:
    from src.domain.repositories.vector_store_repository import VectorStoreRepository

logger = logging.getLogger(__name__)


class CrossEncoder:
    """RAG-layer wrapper around a "RerankerRepository".

    Keeps the retrieval pipeline decoupled from the concrete reranker
    implementation.  "top_k" can be overridden per call for flexibility.
    """

    def __init__(self, reranker: RerankerRepository, top_k: int = 10) -> None:
        self._reranker: RerankerRepository = reranker
        self._top_k: int = top_k

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int | None = None,
        *,
        boost_multiplier: float = 0.0,
        vector_store: VectorStoreRepository | None = None,
    ) -> list[Chunk]:
        """Re-score *chunks* against *query* and return the top-K most relevant.

        Uses the instance "top_k" unless overridden per call. When *boost_multiplier*
        is positive, accumulated user feedback is added to cross-encoder scores before
        the final sort so retrieval feedback survives reranking.
        """
        k = top_k if top_k is not None else self._top_k
        if not chunks:
            return []
        scored = self._reranker.score(query, chunks)
        if boost_multiplier > 0:
            scored = apply_feedback_boost(
                scored,
                boost_multiplier=boost_multiplier,
                vector_store=vector_store,
            )
        else:
            scored.sort(key=lambda item: item[1], reverse=True)
        result = [chunk for chunk, _ in scored[:k]]
        logger.debug("CrossEncoder: %d → %d chunks", len(chunks), len(result))
        return result

    @classmethod
    def from_settings(cls) -> CrossEncoder:
        from src.core.settings import settings
        from src.infrastructure.rerankers.bge_reranker import BGERerankerProvider

        cfg = settings.reranker
        return cls(reranker=BGERerankerProvider.from_settings(), top_k=cfg.top_k)
