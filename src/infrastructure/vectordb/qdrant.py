from __future__ import annotations

import logging
import threading
import uuid
from types import TracebackType
from typing import Any, Protocol, TypeAlias, cast

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    Document,
    FieldCondition,
    Filter,
    HasIdCondition,
    Image,
    InferenceObject,
    IsNullCondition,
    MatchValue,
    PayloadField,
    PointIdsList,
    PointStruct,
    PointVectors,
    Range,
    SetPayload,
    SetPayloadOperation,
    SparseIndexParams,
    SparseVectorParams,
    UpdateVectors,
    UpdateVectorsOperation,
    VectorParams,
)
from qdrant_client.models import (
    SparseVector as QSparseVector,
)

from src.core.constants import (
    CHUNK_TYPE_KEY,
    FEEDBACK_REVISION_KEY,
    FEEDBACK_SCORE_KEY,
    RRF_K,
)
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.domain.repositories.embedding_repository import DenseVector, SparseVector
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository
from src.rag.retrieval.filters import build_qdrant_filter

logger = logging.getLogger(__name__)

_DENSE = "dense"
_SPARSE = "sparse"
_EXPANSION = 3  # multiplier for hybrid search candidate pool
_EMBEDDING_MODEL_METADATA_KEY = "embedding_model_name"
_FEEDBACK_SCORE_EPSILON = 1e-9
_MAX_FEEDBACK_UPDATE_RETRIES = 20
_FEEDBACK_UPDATE_ID_KEY = "feedback_update_id"

QdrantNamedVectors: TypeAlias = dict[
    str,
    DenseVector | QSparseVector | list[list[float]] | Document | Image | InferenceObject,
]


class _SynchronizedLock(Protocol):
    def __enter__(self) -> bool: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
        /,
    ) -> None: ...


class ThreadSafeQdrantClient:
    """Serialize Qdrant HTTP calls — the underlying httpx client is not thread-safe."""

    __slots__ = ("_client", "_lock")

    def __init__(self, client: QdrantClient, lock: _SynchronizedLock) -> None:
        self._client = client
        self._lock = lock

    def __getattr__(self, name: str) -> Any:
        attr = getattr(self._client, name)
        if not callable(attr):
            return attr

        def locked_call(*args: Any, **kwargs: Any) -> Any:
            with self._lock:
                return attr(*args, **kwargs)

        return locked_call


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
        self._client_lock = threading.RLock()
        raw_client = QdrantClient(url=url, api_key=api_key or None, check_compatibility=False)
        self._client = ThreadSafeQdrantClient(raw_client, self._client_lock)
        self._collection_ready = False
        self._model_validated = False  # set True after first successful _validate_embedding_model

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> QdrantVectorStore:
        from src.core.settings import settings
        from src.infrastructure.embeddings import embedding_model_identifier, provider_dense_dim

        provider = settings.embeddings.provider
        return cls(
            url=settings.qdrant.url,
            collection=settings.qdrant.collection,
            api_key=settings.qdrant.api_key,
            dense_dim=provider_dense_dim(provider, settings),
            embedding_model_name=embedding_model_identifier(provider, settings),
        )

    def drop_collection(self) -> None:
        """Delete the Qdrant collection and reset the ready flag."""
        with self._client_lock:
            self._client.delete_collection(self.collection)
            self._collection_ready = False

    # ── VectorStoreRepository interface ────────────────────────────────────────

    def upsert(self, chunks: list[Chunk]) -> None:
        self._ensure_collection()
        if not chunks:
            return
        for chunk in chunks:
            if chunk.embedding is None or chunk.sparse_vector is None:
                raise VectorStoreError(
                    f"Chunk {chunk.id!r} must have embedding and sparse_vector before upsert"
                )

        chunk_ids = [chunk.id for chunk in chunks]
        existing_ids = {str(point.id) for point in self._retrieve_points(chunk_ids)}
        new_chunks = [chunk for chunk in chunks if chunk.id not in existing_ids]
        existing_chunks = [chunk for chunk in chunks if chunk.id in existing_ids]

        snapshots: dict[str, PointStruct] = {}
        existing_committed = False

        try:
            if existing_chunks:
                snapshots = self._snapshot_existing_points([chunk.id for chunk in existing_chunks])
                self._update_existing_chunks(existing_chunks, snapshots=snapshots)
                existing_committed = True

            if new_chunks:
                self._insert_new_chunks(new_chunks)
        except Exception as exc:
            if existing_committed and snapshots:
                self._rollback_points(snapshots, context="upsert")
            raise VectorStoreError("Qdrant upsert failed", cause=exc) from exc

    def search_dense(
        self,
        query_vector: DenseVector,
        top_k: int,
        *,
        type_equals: str | None = None,
        exclude_types: frozenset[str] | None = None,
        document_ids: frozenset[str] | None = None,
        filters: RetrievalFilter | None = None,
    ) -> list[SearchResult]:
        self._ensure_collection()
        query_filter = build_qdrant_filter(
            type_equals=type_equals,
            exclude_types=exclude_types,
            document_ids=document_ids,
            filters=filters,
        )
        try:
            response = self._client.query_points(
                collection_name=self.collection,
                query=query_vector,  # type: ignore[arg-type]
                using=_DENSE,
                limit=top_k,
                with_payload=True,
                query_filter=query_filter,
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

    def delete_by_document_id(self, document_id: str) -> list[str]:
        """Delete all points whose payload document_id matches. Returns deleted chunk IDs."""
        self._ensure_collection()
        deleted: list[str] = []
        offset: Any | None = None
        while True:
            points, offset = self._client.scroll(
                collection_name=self.collection,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key="document_id",
                            match=MatchValue(value=document_id),
                        )
                    ]
                ),
                limit=100,
                offset=offset,
                with_payload=False,
                with_vectors=False,
            )
            if not points:
                break
            ids = [str(p.id) for p in points]
            if ids:
                self.delete(ids)
                deleted.extend(ids)
            if offset is None:
                break
        return deleted

    def count(self) -> int:
        try:
            return cast(int, self._client.count(collection_name=self.collection).count)
        except Exception as exc:
            raise VectorStoreError("Qdrant count failed", cause=exc) from exc

    def get_feedback_score(self, chunk_id: str) -> float:
        """Return accumulated user feedback score stored in chunk metadata."""
        points = self._retrieve_points([chunk_id])
        if not points:
            return 0.0
        return self._feedback_score_from_metadata(self._metadata_from_point(points[0]))

    def set_feedback_score(self, chunk_id: str, feedback_score: float) -> None:
        """Persist *feedback_score* under chunk metadata in the Qdrant payload."""
        metadata = self._require_chunk_metadata(chunk_id)
        metadata[FEEDBACK_SCORE_KEY] = feedback_score
        metadata[FEEDBACK_REVISION_KEY] = self._feedback_revision_from_metadata(metadata) + 1
        self._set_chunk_metadata(chunk_id, metadata)

    def accumulate_feedback_score(self, chunk_id: str, delta: float) -> float:
        """Add *delta* to the stored feedback score with compare-and-set retries."""
        if not self._retrieve_points([chunk_id]):
            raise VectorStoreError(
                f"Chunk {chunk_id!r} not found in collection {self.collection!r}"
            )
        self._ensure_feedback_fields_initialized(chunk_id)
        for attempt in range(_MAX_FEEDBACK_UPDATE_RETRIES):
            if not self._retrieve_points([chunk_id]):
                raise VectorStoreError(
                    f"Chunk {chunk_id!r} not found in collection {self.collection!r}"
                )
            current_score = self.get_feedback_score(chunk_id)
            current_revision = self.get_feedback_revision(chunk_id)
            updated = current_score + delta
            next_revision = current_revision + 1
            if self._try_set_feedback_score_if_current(
                chunk_id,
                expected_score=current_score,
                expected_revision=current_revision,
                feedback_score=updated,
                feedback_revision=next_revision,
            ):
                return updated
            logger.debug(
                "Feedback accumulate conflict for chunk_id=%r attempt=%d",
                chunk_id,
                attempt + 1,
            )
        raise VectorStoreError(
            "Failed to accumulate feedback for "
            f"{chunk_id!r} after {_MAX_FEEDBACK_UPDATE_RETRIES} attempts"
        )

    def get_feedback_revision(self, chunk_id: str) -> int:
        """Return the feedback revision counter stored in chunk metadata."""
        points = self._retrieve_points([chunk_id])
        if not points:
            return 0
        return self._feedback_revision_from_metadata(self._metadata_from_point(points[0]))

    def _feedback_cas_filter(
        self,
        chunk_id: str,
        expected_score: float,
        expected_revision: int,
    ) -> Filter:
        """Build a filter matching only the current feedback (score, revision) pair."""
        id_match = HasIdCondition(has_id=[chunk_id])
        score_match = self._feedback_score_match_condition(expected_score)
        revision_match = self._feedback_revision_match_condition(expected_revision)
        return Filter(must=[id_match, score_match, revision_match])

    def _feedback_score_match_condition(
        self,
        expected_score: float,
    ) -> Filter | FieldCondition | IsNullCondition:
        if self._feedback_scores_equal(expected_score, 0.0):
            return Filter(
                should=[
                    IsNullCondition(is_null=PayloadField(key="metadata.feedback_score")),
                    FieldCondition(
                        key="metadata.feedback_score",
                        range=Range(
                            gte=-_FEEDBACK_SCORE_EPSILON,
                            lte=_FEEDBACK_SCORE_EPSILON,
                        ),
                    ),
                ]
            )
        return FieldCondition(
            key="metadata.feedback_score",
            range=Range(
                gte=expected_score - _FEEDBACK_SCORE_EPSILON,
                lte=expected_score + _FEEDBACK_SCORE_EPSILON,
            ),
        )

    @staticmethod
    def _feedback_revision_match_condition(
        expected_revision: int,
    ) -> Filter | FieldCondition | IsNullCondition:
        if expected_revision == 0:
            return Filter(
                should=[
                    IsNullCondition(is_null=PayloadField(key="metadata.feedback_revision")),
                    FieldCondition(
                        key="metadata.feedback_revision",
                        range=Range(gte=0, lte=0),
                    ),
                ]
            )
        return FieldCondition(
            key="metadata.feedback_revision",
            range=Range(
                gte=expected_revision - _FEEDBACK_SCORE_EPSILON,
                lte=expected_revision + _FEEDBACK_SCORE_EPSILON,
            ),
        )

    def _feedback_cas_pre_check(
        self,
        chunk_id: str,
        expected_score: float,
        expected_revision: int,
    ) -> Filter | None:
        """Return a CAS filter when feedback state matches, else None on conflict."""
        cas_filter = self._feedback_cas_filter(
            chunk_id,
            expected_score,
            expected_revision,
        )
        current_metadata = self._metadata_from_point(self._require_point(chunk_id))
        actual_score = self._feedback_score_from_metadata(current_metadata)
        actual_revision = self._feedback_revision_from_metadata(current_metadata)
        if actual_revision != expected_revision or not self._feedback_scores_equal(
            actual_score, expected_score
        ):
            return None
        return cas_filter

    def _try_set_feedback_score_if_current(
        self,
        chunk_id: str,
        *,
        expected_score: float,
        expected_revision: int,
        feedback_score: float,
        feedback_revision: int,
    ) -> bool:
        """Persist feedback only when the stored (score, revision) pair still matches."""
        cas_filter = self._feedback_cas_pre_check(
            chunk_id,
            expected_score,
            expected_revision,
        )
        if cas_filter is None:
            return False
        update_id = str(uuid.uuid4())
        try:
            self._client.set_payload(
                collection_name=self.collection,
                payload={
                    FEEDBACK_SCORE_KEY: feedback_score,
                    FEEDBACK_REVISION_KEY: feedback_revision,
                    _FEEDBACK_UPDATE_ID_KEY: update_id,
                },
                key="metadata",
                points=cas_filter,
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant set_payload failed", cause=exc) from exc
        metadata = self._metadata_from_point(self._require_point(chunk_id))
        return (
            metadata.get(_FEEDBACK_UPDATE_ID_KEY) == update_id
            and self._feedback_revision_from_metadata(metadata) == feedback_revision
            and self._feedback_scores_equal(
                self._feedback_score_from_metadata(metadata),
                feedback_score,
            )
        )

    def _ensure_feedback_fields_initialized(self, chunk_id: str) -> None:
        """Backfill explicit feedback defaults under compare-and-set."""
        for attempt in range(_MAX_FEEDBACK_UPDATE_RETRIES):
            metadata = self._require_chunk_metadata(chunk_id)
            score = self._feedback_score_from_metadata(metadata)
            revision = self._feedback_revision_from_metadata(metadata)
            if (
                metadata.get(FEEDBACK_SCORE_KEY) == score
                and metadata.get(FEEDBACK_REVISION_KEY) == revision
            ):
                return
            if self._try_set_feedback_score_if_current(
                chunk_id,
                expected_score=score,
                expected_revision=revision,
                feedback_score=score,
                feedback_revision=revision,
            ):
                return
            logger.debug(
                "Feedback init conflict for chunk_id=%r attempt=%d",
                chunk_id,
                attempt + 1,
            )
        raise VectorStoreError(
            f"Failed to initialize feedback fields for {chunk_id!r} "
            f"after {_MAX_FEEDBACK_UPDATE_RETRIES} attempts"
        )

    def _insert_new_chunks(self, chunks: list[Chunk]) -> None:
        """Insert points absent at upsert start, re-checking each id before writing."""
        inserted_ids: list[str] = []
        try:
            for chunk in chunks:
                if self._insert_new_chunk(chunk):
                    inserted_ids.append(chunk.id)
        except Exception:
            if inserted_ids:
                self._rollback_inserted_chunks(inserted_ids, context="partial new-chunk insert")
            raise

    def _insert_new_chunk(self, chunk: Chunk) -> bool:
        """Insert one point when absent, or CAS-update if it appeared before writing.

        Returns True when a fresh point was inserted via upsert (eligible for insert rollback).
        """
        if self._retrieve_points([chunk.id]):
            self._update_existing_chunk(chunk)
            return False
        self._client.upsert(
            collection_name=self.collection,
            points=[self._to_point(chunk)],
        )
        points = self._retrieve_points([chunk.id])
        if not points:
            raise VectorStoreError(
                f"Chunk {chunk.id!r} not found after upsert in collection {self.collection!r}"
            )
        metadata = self._metadata_from_point(points[0])
        if self._feedback_revision_from_metadata(metadata) == 0 and self._feedback_scores_equal(
            self._feedback_score_from_metadata(metadata), 0.0
        ):
            return True
        # Concurrent writer recorded feedback; refresh vectors without clobbering it.
        self._update_existing_chunk(chunk)
        return True

    def _snapshot_existing_points(self, chunk_ids: list[str]) -> dict[str, PointStruct]:
        """Capture the full point state so upsert can roll back on partial failure."""
        unique_ids = list(dict.fromkeys(chunk_ids))
        points = self._retrieve_points(unique_ids, with_vectors=True)
        snapshots: dict[str, PointStruct] = {}
        for point in points:
            snapshots[str(point.id)] = PointStruct(
                id=point.id,
                vector=point.vector,
                payload=dict(point.payload or {}),
            )
        missing = [chunk_id for chunk_id in unique_ids if chunk_id not in snapshots]
        if missing:
            raise VectorStoreError(f"Cannot snapshot existing chunks for rollback: {missing!r}")
        return snapshots

    def _rollback_points(
        self,
        snapshots: dict[str, PointStruct],
        *,
        context: str,
    ) -> None:
        """Restore previously snapshotted points after a failed upsert."""
        if not snapshots:
            return
        try:
            self._client.upsert(
                collection_name=self.collection,
                points=list(snapshots.values()),
            )
        except Exception as exc:
            logger.error(
                "Qdrant %s rollback failed for chunk_ids=%r: %s",
                context,
                list(snapshots),
                exc,
            )
            raise VectorStoreError(
                f"Qdrant {context} rollback failed",
                cause=exc,
            ) from exc

    def _rollback_inserted_chunks(
        self,
        chunk_ids: list[str],
        *,
        context: str,
    ) -> None:
        """Delete freshly inserted points after a failed new-chunk batch."""
        if not chunk_ids:
            return
        try:
            self.delete(chunk_ids)
        except Exception as exc:
            logger.error(
                "Qdrant %s rollback failed for chunk_ids=%r: %s",
                context,
                chunk_ids,
                exc,
            )
            raise VectorStoreError(
                f"Qdrant {context} rollback failed",
                cause=exc,
            ) from exc

    def _update_existing_chunks(
        self,
        chunks: list[Chunk],
        *,
        snapshots: dict[str, PointStruct],
    ) -> None:
        """Refresh vectors and non-feedback payload for points that already exist."""
        updated_ids: list[str] = []
        try:
            for chunk in chunks:
                self._update_existing_chunk(chunk)
                updated_ids.append(chunk.id)
        except Exception:
            if updated_ids:
                partial = {
                    chunk_id: snapshots[chunk_id]
                    for chunk_id in updated_ids
                    if chunk_id in snapshots
                }
                self._rollback_points(partial, context="partial existing-chunk update")
            raise

    def _update_existing_chunk(self, chunk: Chunk) -> None:
        """Update one existing point atomically while preserving feedback under CAS."""
        for attempt in range(_MAX_FEEDBACK_UPDATE_RETRIES):
            existing = self._require_chunk_metadata(chunk.id)
            expected_score = self._feedback_score_from_metadata(existing)
            expected_revision = self._feedback_revision_from_metadata(existing)
            metadata = dict(chunk.metadata)
            metadata.update(self._feedback_metadata_from_stored(existing))
            if self._try_batch_update_existing_chunk_if_feedback_current(
                chunk,
                metadata=metadata,
                expected_score=expected_score,
                expected_revision=expected_revision,
            ):
                return
            logger.debug(
                "Upsert metadata conflict for chunk_id=%r attempt=%d",
                chunk.id,
                attempt + 1,
            )
        raise VectorStoreError(
            f"Failed to upsert existing chunk {chunk.id!r} "
            f"after {_MAX_FEEDBACK_UPDATE_RETRIES} attempts"
        )

    def _try_batch_update_existing_chunk_if_feedback_current(
        self,
        chunk: Chunk,
        *,
        metadata: dict[str, object],
        expected_score: float,
        expected_revision: int,
    ) -> bool:
        """Atomically refresh vectors and payload when the feedback state still matches."""
        cas_filter = self._feedback_cas_pre_check(
            chunk.id,
            expected_score,
            expected_revision,
        )
        if cas_filter is None:
            return False
        try:
            self._client.batch_update_points(
                collection_name=self.collection,
                update_operations=[
                    SetPayloadOperation(
                        set_payload=SetPayload(
                            payload=metadata,
                            key="metadata",
                            filter=cas_filter,
                        )
                    ),
                    UpdateVectorsOperation(
                        update_vectors=UpdateVectors(
                            points=[
                                PointVectors(
                                    id=chunk.id,
                                    vector=self._vectors_from_chunk(chunk),
                                )
                            ],
                            update_filter=cas_filter,
                        )
                    ),
                    SetPayloadOperation(
                        set_payload=SetPayload(
                            payload=self._top_level_payload(chunk),
                            filter=cas_filter,
                        )
                    ),
                ],
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant batch_update_points failed", cause=exc) from exc
        after = self._metadata_from_point(self._require_point(chunk.id))
        return self._feedback_revision_from_metadata(
            after
        ) == expected_revision and self._feedback_scores_equal(
            self._feedback_score_from_metadata(after),
            expected_score,
        )

    def _try_set_metadata_if_feedback_current(
        self,
        chunk_id: str,
        *,
        metadata: dict[str, object],
        expected_score: float,
        expected_revision: int,
    ) -> bool:
        """Persist metadata only when the stored feedback (score, revision) pair still matches."""
        cas_filter = self._feedback_cas_pre_check(
            chunk_id,
            expected_score,
            expected_revision,
        )
        if cas_filter is None:
            return False
        try:
            self._client.set_payload(
                collection_name=self.collection,
                payload=metadata,
                key="metadata",
                points=cas_filter,
                wait=True,
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant set_payload failed", cause=exc) from exc
        after = self._metadata_from_point(self._require_point(chunk_id))
        return self._feedback_revision_from_metadata(
            after
        ) == expected_revision and self._feedback_scores_equal(
            self._feedback_score_from_metadata(after),
            expected_score,
        )

    @staticmethod
    def _vectors_from_chunk(chunk: Chunk) -> QdrantNamedVectors:
        embedding = chunk.embedding
        sparse_vector = chunk.sparse_vector
        assert embedding is not None and sparse_vector is not None
        return cast(
            QdrantNamedVectors,
            {
                _DENSE: embedding,
                _SPARSE: QSparseVector(
                    indices=list(sparse_vector.keys()),  # type: ignore[union-attr]
                    values=list(sparse_vector.values()),  # type: ignore[union-attr]
                ),
            },
        )

    def _top_level_payload(self, chunk: Chunk) -> dict[str, object]:
        payload = chunk.model_dump(exclude={"embedding", "sparse_vector", "metadata"})
        if self.embedding_model_name:
            payload["embedding_model_name"] = self.embedding_model_name
        chunk_type = chunk.metadata.get(CHUNK_TYPE_KEY)
        if chunk_type:
            payload[CHUNK_TYPE_KEY] = chunk_type
        return payload

    @staticmethod
    def _feedback_metadata_from_stored(metadata: dict[str, object]) -> dict[str, object]:
        preserved: dict[str, object] = {}
        if FEEDBACK_SCORE_KEY in metadata:
            preserved[FEEDBACK_SCORE_KEY] = metadata[FEEDBACK_SCORE_KEY]
        if FEEDBACK_REVISION_KEY in metadata:
            preserved[FEEDBACK_REVISION_KEY] = metadata[FEEDBACK_REVISION_KEY]
        if _FEEDBACK_UPDATE_ID_KEY in metadata:
            preserved[_FEEDBACK_UPDATE_ID_KEY] = metadata[_FEEDBACK_UPDATE_ID_KEY]
        return preserved

    def _retrieve_points(
        self,
        chunk_ids: list[str],
        *,
        with_vectors: bool = False,
    ) -> list[Any]:
        self._ensure_collection()
        try:
            return cast(
                list[Any],
                self._client.retrieve(
                    collection_name=self.collection,
                    ids=chunk_ids,
                    with_payload=True,
                    with_vectors=with_vectors,
                ),
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant retrieve failed", cause=exc) from exc

    def _require_point(self, chunk_id: str) -> Any:
        points = self._retrieve_points([chunk_id])
        if not points:
            raise VectorStoreError(
                f"Chunk {chunk_id!r} not found in collection {self.collection!r}"
            )
        return points[0]

    def _require_chunk_metadata(self, chunk_id: str) -> dict[str, object]:
        return dict(self._metadata_from_point(self._require_point(chunk_id)))

    def _set_chunk_metadata(self, chunk_id: str, metadata: dict[str, object]) -> None:
        try:
            self._client.set_payload(
                collection_name=self.collection,
                payload={"metadata": metadata},
                points=[chunk_id],
            )
        except Exception as exc:
            raise VectorStoreError("Qdrant set_payload failed", cause=exc) from exc

    @staticmethod
    def _metadata_from_point(point: Any) -> dict[str, object]:
        return dict((point.payload or {}).get("metadata") or {})

    @staticmethod
    def _feedback_score_from_metadata(metadata: dict[str, object]) -> float:
        value = metadata.get(FEEDBACK_SCORE_KEY)
        if isinstance(value, bool):
            return 0.0
        if isinstance(value, int | float):
            return float(value)
        return 0.0

    @staticmethod
    def _feedback_revision_from_metadata(metadata: dict[str, object]) -> int:
        value = metadata.get(FEEDBACK_REVISION_KEY)
        if isinstance(value, bool):
            return 0
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
        return 0

    @staticmethod
    def _feedback_scores_equal(left: float, right: float) -> bool:
        return abs(left - right) <= _FEEDBACK_SCORE_EPSILON

    def get_feedback_scores(self, chunk_ids: list[str]) -> dict[str, float]:
        """Return feedback scores for *chunk_ids* in a single retrieve call."""
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        points = self._retrieve_points(unique_ids)
        scores = dict.fromkeys(unique_ids, 0.0)
        for point in points:
            scores[str(point.id)] = self._feedback_score_from_metadata(
                self._metadata_from_point(point)
            )
        return scores

    # ── Internals ──────────────────────────────────────────────────────────────

    def _ensure_collection(self) -> None:
        with self._client_lock:
            if self._collection_ready:
                return
            collection_existed: bool
            try:
                if not self._client.collection_exists(self.collection):
                    metadata = (
                        {_EMBEDDING_MODEL_METADATA_KEY: self.embedding_model_name}
                        if self.embedding_model_name
                        else None
                    )
                    self._client.create_collection(
                        collection_name=self.collection,
                        vectors_config={
                            _DENSE: VectorParams(size=self.dense_dim, distance=Distance.COSINE),
                        },
                        sparse_vectors_config={
                            _SPARSE: SparseVectorParams(index=SparseIndexParams(on_disk=False)),
                        },
                        metadata=metadata,
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
        """Return the embedding model tracked for this collection.

        Reads collection metadata first (O(1)).  Legacy collections without
        metadata fall back to the first tagged point payload.  Returns "None"
        when the collection is missing, empty, or has no model tracking.
        """
        metadata_model = self._read_collection_metadata_model()
        if metadata_model is not None:
            return metadata_model

        models = self._scan_payload_embedding_models(stop_after_first=True)
        if not models:
            return None
        return next(iter(models))

    def _read_collection_metadata_model(self) -> str | None:
        """Return embedding model name stored in collection metadata, if any."""
        try:
            if not self._client.collection_exists(self.collection):
                return None
            info = self._client.get_collection(self.collection)
        except Exception as exc:  # noqa: BLE001 — intentional fail-open probe
            logger.debug("Cannot read collection %r metadata: %s", self.collection, exc)
            return None

        metadata = info.config.metadata or {}
        value = metadata.get(_EMBEDDING_MODEL_METADATA_KEY)
        if isinstance(value, str) and value:
            return value
        return None

    def _write_collection_metadata_model(self, model_name: str) -> None:
        """Persist the active embedding model on the collection for O(1) checks."""
        if not model_name:
            return
        try:
            self._client.update_collection(
                collection_name=self.collection,
                metadata={_EMBEDDING_MODEL_METADATA_KEY: model_name},
            )
        except Exception as exc:  # noqa: BLE001 — metadata backfill is best-effort
            logger.debug("Cannot update collection %r metadata: %s", self.collection, exc)

    def _scan_payload_embedding_models(self, *, stop_after_first: bool = False) -> set[str]:
        """Return distinct embedding_model_name values found in point payloads."""
        try:
            if not self._client.collection_exists(self.collection):
                return set()
        except Exception as exc:  # noqa: BLE001 — intentional fail-open probe
            logger.debug("Cannot probe collection %r for model tracking: %s", self.collection, exc)
            return set()

        models: set[str] = set()
        offset: Any | None = None
        while True:
            try:
                points, offset = self._client.scroll(
                    collection_name=self.collection,
                    limit=100,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
            except Exception as exc:  # noqa: BLE001 — intentional fail-open probe
                logger.debug(
                    "Cannot probe collection %r for model tracking: %s", self.collection, exc
                )
                return set()

            for point in points:
                payload = point.payload or {}
                if _EMBEDDING_MODEL_METADATA_KEY in payload:
                    models.add(str(payload[_EMBEDDING_MODEL_METADATA_KEY]))

            if len(models) > 1:
                return models
            if stop_after_first and models:
                return models

            if offset is None:
                break
        return models

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

        metadata_model = self._read_collection_metadata_model()
        if metadata_model is not None:
            models = {metadata_model}
        else:
            models = self._scan_payload_embedding_models(stop_after_first=True)

        if len(models) > 1:
            raise VectorStoreError(
                f"Embedding model mismatch: collection '{self.collection}' contains "
                f"vectors from multiple models: {sorted(models)}. "
                f"Run: python scripts/rebuild_embeddings.py --recreate-collection"
            )
        existing_model = next(iter(models)) if models else None
        if existing_model is not None and existing_model != self.embedding_model_name:
            raise VectorStoreError(
                f"Embedding model mismatch: collection '{self.collection}' was built with "
                f"'{existing_model}' but current config is '{self.embedding_model_name}'. "
                f"Run: python scripts/rebuild_embeddings.py --recreate-collection"
            )
        if metadata_model is None and existing_model == self.embedding_model_name:
            self._write_collection_metadata_model(self.embedding_model_name)
        self._model_validated = True

    def _to_point(self, chunk: Chunk) -> PointStruct:
        payload = chunk.model_dump(exclude={"embedding", "sparse_vector"})
        metadata = dict(payload.get("metadata") or {})
        metadata.setdefault(FEEDBACK_SCORE_KEY, 0.0)
        metadata.setdefault(FEEDBACK_REVISION_KEY, 0)
        payload["metadata"] = metadata
        if self.embedding_model_name:
            payload["embedding_model_name"] = self.embedding_model_name
        chunk_type = chunk.metadata.get(CHUNK_TYPE_KEY)
        if chunk_type:
            payload[CHUNK_TYPE_KEY] = chunk_type
        return PointStruct(
            id=chunk.id,
            vector=self._vectors_from_chunk(chunk),
            payload=payload,
        )

    @staticmethod
    def _to_result(point: Any) -> SearchResult:
        chunk = Chunk.model_validate(point.payload or {})
        return chunk, float(point.score)
