from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.core.constants import MODALITY_CAPTION, MODALITY_TABLE
from src.core.exceptions import RetrievalError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository
from src.rag.quality.feedback_loop import apply_feedback_boost

if TYPE_CHECKING:
    from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository

logger = logging.getLogger(__name__)

# Modalities boosted by T-262 — structured content that cross-encoders trained
# on natural-language pairs tend to underscore relative to prose passages.
_BOOSTED_MODALITIES = frozenset({MODALITY_TABLE, MODALITY_CAPTION})


def apply_modality_boost(
    scored: list[SearchResult],
    *,
    boost: float,
    modalities: frozenset[str] = _BOOSTED_MODALITIES,
) -> list[SearchResult]:
    """Add *boost* to the cross-encoder score of table/caption chunks, re-sorted."""
    if boost <= 0 or not scored:
        return scored
    boosted = [
        (chunk, score + boost if chunk.modality in modalities else score) for chunk, score in scored
    ]
    boosted.sort(key=lambda item: item[1], reverse=True)
    return boosted


class CrossEncoder:
    """RAG-layer wrapper around a "RerankerRepository".

    Keeps the retrieval pipeline decoupled from the concrete reranker
    implementation.  "top_k" can be overridden per call for flexibility.
    """

    def __init__(
        self,
        reranker: RerankerRepository,
        top_k: int = 10,
        *,
        modality_boost: float = 0.0,
    ) -> None:
        self._reranker: RerankerRepository = reranker
        self._top_k: int = top_k
        self._modality_boost: float = modality_boost

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

        Uses the instance "top_k" unless overridden per call. The instance's
        "modality_boost" (T-262) is added to table/caption chunk scores first, so
        it composes with the RRF fusion and diversity stages upstream. When
        *boost_multiplier* is positive, accumulated user feedback is then added to
        the (possibly modality-boosted) scores before the final sort so retrieval
        feedback survives reranking.
        """
        from src.observability.metrics import record_reranker_fallback, record_reranker_success

        k = top_k if top_k is not None else self._top_k
        if not chunks:
            return []
        try:
            scored = self._reranker.score(query, chunks)
        except RetrievalError:
            logger.warning(
                "Reranker scoring failed — falling back to raw retrieval order for %d chunks",
                len(chunks),
                exc_info=True,
            )
            record_reranker_fallback()
            return chunks[:k]
        record_reranker_success([score for _, score in scored])
        if self._modality_boost > 0:
            scored = apply_modality_boost(scored, boost=self._modality_boost)
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
        from src.infrastructure.rerankers.nvidia_nim_reranker import NvidiaNimRerankerProvider
        from src.infrastructure.rerankers.qwen_reranker import QwenRerankerProvider

        cfg = settings.reranker
        reranker: RerankerRepository
        if cfg.provider == "qwen_reranker":
            reranker = QwenRerankerProvider.from_settings()
        elif cfg.provider == "nvidia_nim":
            reranker = NvidiaNimRerankerProvider.from_settings()
        else:
            reranker = BGERerankerProvider.from_settings()
        return cls(reranker=reranker, top_k=cfg.top_k, modality_boost=cfg.modality_boost)
