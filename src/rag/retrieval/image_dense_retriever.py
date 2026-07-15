from __future__ import annotations

import logging

from src.domain.entities.query import Query
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.vector_store_repository import SearchResult
from src.infrastructure.vectordb.qdrant import QdrantVectorStore

logger = logging.getLogger(__name__)


class ImageDenseRetriever:
    """Embed a query and search the optional `image_dense` named vector (T-260).

    Only multimodal embedding providers (CLIP, Voyage-multimodal — T-251)
    embed text and images into one shared space, so a text query embedding
    can retrieve `image_dense` chunks directly. Same embed-then-search shape
    as `DenseRetriever`, but targets `QdrantVectorStore.search_image_dense()`
    (T-252/T-260) instead of the default `dense` vector — no `exclude_types`
    filter, since only image-bearing chunk types (e.g. figure) ever carry an
    `image_dense` vector at all.

    Depends on the concrete `QdrantVectorStore` rather than
    `VectorStoreRepository` — `search_image_dense()` is Qdrant-specific and
    not part of that domain interface (only `search_dense`/`search_sparse`/
    `search_hybrid` are).
    """

    def __init__(
        self,
        embedder: EmbeddingRepository,
        vector_store: QdrantVectorStore,
    ) -> None:
        self._embedder: EmbeddingRepository = embedder
        self._vector_store: QdrantVectorStore = vector_store

    @property
    def vector_store(self) -> QdrantVectorStore:
        return self._vector_store

    @property
    def enabled(self) -> bool:
        """True when the backing collection has an `image_dense` vector space."""
        return self._vector_store.image_dense_dim is not None

    # ── Public ─────────────────────────────────────────────────────────────────

    def retrieve(self, query: Query, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* (Chunk, score) pairs from the image_dense index.

        Returns an empty list when the active embedding provider has no
        image space (`enabled` is False), so callers can wire this retriever
        unconditionally without checking the provider first.

        If "query.embedding" is already populated, it is used directly,
        avoiding a redundant embedding call.
        """
        if not self.enabled:
            return []
        embedding = query.embedding or self._embedder.embed_query([query.text])[0]
        results = self._vector_store.search_image_dense(
            embedding,
            top_k=top_k,
            filters=query.filters,
        )
        logger.debug("Image-dense retrieval: %d results for %r", len(results), query.text[:60])
        return results

    def embed_query(self, query: Query) -> Query:
        """Return *query* with "embedding" populated (no-op if already set)."""
        if query.embedding is not None:
            return query
        embedding = self._embedder.embed_query([query.text])[0]
        return query.model_copy(update={"embedding": embedding})
