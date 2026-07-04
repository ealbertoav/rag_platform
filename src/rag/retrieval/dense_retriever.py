from __future__ import annotations

import logging

from src.core.constants import CHUNK_TYPE_HYPE, CHUNK_TYPE_SUMMARY
from src.domain.entities.query import Query
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository

logger = logging.getLogger(__name__)

_HIERARCHICAL_EXCLUDE = frozenset({CHUNK_TYPE_HYPE, CHUNK_TYPE_SUMMARY})


class DenseRetriever:
    """Embed a query and search the dense (HNSW) vector index.

    Depends only on the domain-layer repository interfaces — no direct
    reference to BGE-M3 or Qdrant, so implementations can be swapped freely.
    """

    def __init__(
        self,
        embedder: EmbeddingRepository,
        vector_store: VectorStoreRepository,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    @property
    def vector_store(self) -> VectorStoreRepository:
        return self._vector_store

    # ── Public ─────────────────────────────────────────────────────────────────

    def retrieve(self, query: Query, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* (Chunk, score) pairs sorted by cosine similarity.

        If "query.embedding" is already populated, it is used directly,
        avoiding a redundant embedding call.
        """
        embedding = query.embedding or self._embedder.embed_query([query.text])[0]
        results = self._vector_store.search_dense(
            embedding,
            top_k=top_k,
            exclude_types=_HIERARCHICAL_EXCLUDE,
            filters=query.filters,
        )
        logger.debug("Dense retrieval: %d results for %r", len(results), query.text[:60])
        return results

    def embed_query(self, query: Query) -> Query:
        """Return *query* with "embedding" populated (no-op if already set)."""
        if query.embedding is not None:
            return query
        embedding = self._embedder.embed_query([query.text])[0]
        return query.model_copy(update={"embedding": embedding})
