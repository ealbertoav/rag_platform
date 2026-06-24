from __future__ import annotations

import logging

from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository

logger = logging.getLogger(__name__)


class CrossEncoder:
    """RAG-layer wrapper around a "RerankerRepository".

    Keeps the retrieval pipeline decoupled from the concrete reranker
    implementation.  "top_k" can be overridden per call for flexibility.
    """

    def __init__(self, reranker: RerankerRepository, top_k: int = 10) -> None:
        self._reranker = reranker
        self._top_k = top_k

    def rerank(self, query: str, chunks: list[Chunk], top_k: int | None = None) -> list[Chunk]:
        """Re-score *chunks* against *query* and return the top-K most relevant.

        Uses the instance "top_k" unless overridden per call.
        """
        k = top_k if top_k is not None else self._top_k
        result = self._reranker.rerank(query, chunks, top_k=k)
        logger.debug("CrossEncoder: %d → %d chunks", len(chunks), len(result))
        return result

    @classmethod
    def from_settings(cls) -> CrossEncoder:
        from src.core.settings import settings
        from src.infrastructure.rerankers.bge_reranker import BGERerankerProvider

        cfg = settings.reranker
        return cls(reranker=BGERerankerProvider.from_settings(), top_k=cfg.top_k)
