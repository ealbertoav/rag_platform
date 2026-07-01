from __future__ import annotations

import logging

from src.core.constants import CHUNK_TYPE_HYPE
from src.domain.entities.query import Query
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository
from src.rag.enrichment.document_augmentation import resolve_synthetic_questions

logger = logging.getLogger(__name__)


class HyPERetriever:
    """Retrieve source chunks via question-question dense matching (HyPE).

    Embeds the user query and searches only "hype_question" vectors stored
    at index time, then resolves hits back to their source chunks.
    """

    def __init__(
        self,
        embedder: EmbeddingRepository,
        vector_store: VectorStoreRepository,
        chunk_lookup: object,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store
        self._lookup = chunk_lookup

    def retrieve(self, query: Query, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* source chunks ranked by HyPE similarity."""
        embedding = query.embedding or self._embedder.embed_query([query.text])[0]
        hype_hits = self._vector_store.search_dense(
            embedding,
            top_k=top_k,
            type_equals=CHUNK_TYPE_HYPE,
            filters=query.filters,
        )
        resolved = resolve_synthetic_questions(
            hype_hits,
            self._lookup.get_by_id,  # type: ignore[attr-defined]
        )
        logger.debug("HyPE retrieval: %d hits → %d source chunks", len(hype_hits), len(resolved))
        return resolved[:top_k]
