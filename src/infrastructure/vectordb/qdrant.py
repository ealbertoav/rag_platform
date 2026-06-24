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
        embedding_model_name: str = "",
    ) -> None:
        self.collection = collection
        self.dense_dim = dense_dim
        self.embedding_model_name = embedding_model_name
        self._client = QdrantClient(url=url, api_key=api_key or None, check_compatibility=False)
        self._collection_ready = False
        self._model_validated = False  # set True after first successful _validate_embedding_model

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> QdrantVectorStore:
        from src.core.settings import settings
        from src.infrastructure.embeddings import embedding_model_identifier

        return cls(
            url=settings.qdrant.url,
            collection=settings.qdrant.collection,
            api_key=settings.qdrant.api_key,
            dense_dim=settings.embeddings.dense_dim,
            embedding_model_name=embedding_model_identifier(settings.embeddings.provider, settings),
        )

    def drop_collection(self) -> None:
        """Delete the Qdrant collection and reset the ready flag."""
        self._client.delete_collection(self.collection)
        self._collection_ready = False

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
        collection_existed: bool
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
                collection_existed = False
            else:
                collection_existed = True
        except Exception as exc:
            raise VectorStoreError("Qdrant collection setup failed", cause=exc) from exc
        if collection_existed and not self._model_validated:
            self._validate_embedding_model()
        self._collection_ready = True

    def get_collection_embedding_model(self) -> str | None:
        """Return the embedding model name stored in the collection's payload.

        Samples a few existing points and returns the "embedding_model_name"
        field from the first point that has it.  Returns "None" when:
        - The collection does not exist or cannot be queried.
        - The collection is empty.
        - Points pre-date model tracking (field absent from payload).
        """
        try:
            if not self._client.collection_exists(self.collection):
                return None
            points, _ = self._client.scroll(
                collection_name=self.collection,
                limit=5,
                with_payload=True,
                with_vectors=False,
            )
        except Exception as exc:  # noqa: BLE001 — intentional fail-open probe
            logger.debug("Cannot probe collection %r for model tracking: %s", self.collection, exc)
            return None

        for point in points:
            payload = point.payload or {}
            if "embedding_model_name" in payload:
                return str(payload["embedding_model_name"])
        return None

    def validate_embedding_model(self) -> None:
        """Raise VectorStoreError when the collection's model differs from the current config.

        Call this before ingestion to surface provider mismatches early, before
        any batches are processed.  No-op when the collection is empty or has no
        model tracking payload (legacy data).
        """
        self._validate_embedding_model()

    def _validate_embedding_model(self) -> None:
        """Raise VectorStoreError when the collection's model differs from the current config."""
        if not self.embedding_model_name:
            self._model_validated = True
            return
        existing_model = self.get_collection_embedding_model()
        if existing_model is not None and existing_model != self.embedding_model_name:
            raise VectorStoreError(
                f"Embedding model mismatch: collection '{self.collection}' was built with "
                f"'{existing_model}' but current config is '{self.embedding_model_name}'. "
                f"Run: python scripts/rebuild_embeddings.py --recreate-collection"
            )
        self._model_validated = True

    def _to_point(self, chunk: Chunk) -> PointStruct:
        embedding = chunk.embedding
        sparse_vector = chunk.sparse_vector
        assert embedding is not None
        assert sparse_vector is not None
        payload = chunk.model_dump(exclude={"embedding", "sparse_vector"})
        if self.embedding_model_name:
            payload["embedding_model_name"] = self.embedding_model_name
        return PointStruct(
            id=chunk.id,
            vector={
                _DENSE: embedding,
                _SPARSE: QSparseVector(
                    indices=list(sparse_vector.keys()),  # type: ignore[union-attr]
                    values=list(sparse_vector.values()),  # type: ignore[union-attr]
                ),
            },
            payload=payload,
        )

    @staticmethod
    def _to_result(point: Any) -> SearchResult:
        chunk = Chunk.model_validate(point.payload or {})
        return chunk, float(point.score)
