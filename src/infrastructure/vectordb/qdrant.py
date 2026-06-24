from __future__ import annotations

import logging
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointIdsList,
    PointStruct,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QSparseVector,
)

from src.core.constants import RRF_K
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.embedding_repository import DenseVector, SparseVector
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository

logger = logging.getLogger(__name__)

_DENSE = "dense"
_SPARSE = "sparse"
_EXPANSION = 3  # multiplier for hybrid search candidate pool


# ── RRF fusion (module-level so it can be tested independently) ────────────────


def rrf_fuse(
    dense: list[SearchResult],
    sparse: list[SearchResult],
    top_k: int,
    k: int = RRF_K,
) -> list[SearchResult]:
    """Reciprocal Rank Fusion: score = Σ 1 / (k + rank_i) across all lists."""
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}

    for rank, (chunk, _) in enumerate(dense):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunks[chunk.id] = chunk

    for rank, (chunk, _) in enumerate(sparse):
        scores[chunk.id] = scores.get(chunk.id, 0.0) + 1.0 / (k + rank + 1)
        chunks.setdefault(chunk.id, chunk)

    return [
        (chunks[cid], scores[cid])
        for cid in sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    ]


# ── Provider ───────────────────────────────────────────────────────────────────


class QdrantVectorStore(VectorStoreRepository):
    """VectorStoreRepository backed by Qdrant.

    Stores dense (HNSW cosine) and sparse (lexical) vectors per chunk.
    The collection is created automatically on first use if it does not exist.
    Uses the modern `query_points` API (qdrant-client ≥ 1.7).
    """

    def __init__(
        self,
        url: str = "http://localhost:6333",
        collection: str = "rag_documents",
        api_key: str = "",
        dense_dim: int = 1024,
    ) -> None:
        self.collection = collection
        self.dense_dim = dense_dim
        self._client = QdrantClient(url=url, api_key=api_key or None, check_compatibility=False)
        self._collection_ready = False

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> QdrantVectorStore:
        from src.core.settings import settings

        return cls(
            url=settings.qdrant.url,
            collection=settings.qdrant.collection,
            api_key=settings.qdrant.api_key,
            dense_dim=settings.embeddings.dense_dim,
        )

    # ── VectorStoreRepository interface ────────────────────────────────────────

    def upsert(self, chunks: list[Chunk]) -> None:
        self._ensure_collection()
        points: list[PointStruct] = []
        for chunk in chunks:
            if chunk.embedding is None or chunk.sparse_vector is None:
                raise VectorStoreError(
                    f"Chunk {chunk.id!r} must have embedding and sparse_vector before upsert"
                )
            points.append(self._to_point(chunk))
        if points:
            try:
                self._client.upsert(collection_name=self.collection, points=points)
            except Exception as exc:
                raise VectorStoreError("Qdrant upsert failed", cause=exc) from exc

    def search_dense(self, query_vector: DenseVector, top_k: int) -> list[SearchResult]:
        self._ensure_collection()
        try:
            response = self._client.query_points(
                collection_name=self.collection,
                query=query_vector,  # type: ignore[arg-type]
                using=_DENSE,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant dense search failed", cause=exc) from exc
        return [self._to_result(h) for h in response.points]

    def search_sparse(self, query_sparse: SparseVector, top_k: int) -> list[SearchResult]:
        self._ensure_collection()
        try:
            response = self._client.query_points(
                collection_name=self.collection,
                query=QSparseVector(  # type: ignore[arg-type]
                    indices=list(query_sparse.keys()),
                    values=list(query_sparse.values()),
                ),
                using=_SPARSE,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant sparse search failed", cause=exc) from exc
        return [self._to_result(h) for h in response.points]

    def search_hybrid(
        self,
        query_vector: DenseVector,
        query_sparse: SparseVector,
        alpha: float,
        top_k: int,
    ) -> list[SearchResult]:
        expansion = min(top_k * _EXPANSION, 50)
        dense = self.search_dense(query_vector, top_k=expansion)
        sparse = self.search_sparse(query_sparse, top_k=expansion)
        return rrf_fuse(dense, sparse, top_k=top_k)

    def delete(self, chunk_ids: list[str]) -> None:
        try:
            self._client.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=chunk_ids),  # type: ignore[arg-type]
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant delete failed", cause=exc) from exc

    def count(self) -> int:
        try:
            return self._client.count(collection_name=self.collection).count
        except Exception as exc:
            raise VectorStoreError("Qdrant count failed", cause=exc) from exc

    # ── Internals ──────────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        if self._collection_ready:
            return
        try:
            if not self._client.collection_exists(self.collection):
                self._client.create_collection(
                    collection_name=self.collection,
                    vectors_config={
                        _DENSE: VectorParams(size=self.dense_dim, distance=Distance.COSINE),
                    },
                    sparse_vectors_config={
                        _SPARSE: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                    },
                )
                logger.info(
                    "Created Qdrant collection %r (%d-dim)", self.collection, self.dense_dim
                )
        except Exception as exc:
            raise VectorStoreError("Qdrant collection setup failed", cause=exc) from exc
        self._collection_ready = True

    @staticmethod
    def _to_point(chunk: Chunk) -> PointStruct:
        # Bind to locals so pyright can narrow the types through the assertions.
        embedding = chunk.embedding
        sparse_vector = chunk.sparse_vector
        assert embedding is not None
        assert sparse_vector is not None
        return PointStruct(
            id=chunk.id,
            vector={
                _DENSE: embedding,
                _SPARSE: QSparseVector(
                    indices=list(sparse_vector.keys()),  # type: ignore[union-attr]
                    values=list(sparse_vector.values()),  # type: ignore[union-attr]
                ),
            },
            payload=chunk.model_dump(exclude={"embedding", "sparse_vector"}),
        )

    @staticmethod
    def _to_result(point: Any) -> SearchResult:
        chunk = Chunk.model_validate(point.payload or {})
        return chunk, float(point.score)
