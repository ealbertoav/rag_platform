"""T-013 unit tests — QdrantVectorStore (Qdrant client mocked)."""

from __future__ import annotations

import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import (
    FieldCondition,
    Filter,
    HasIdCondition,
    PointStruct,
    SetPayload,
    SetPayloadOperation,
    UpdateVectors,
    UpdateVectorsOperation,
)

from src.core.constants import FEEDBACK_REVISION_KEY, FEEDBACK_SCORE_KEY
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.infrastructure.vectordb.qdrant import QdrantVectorStore, ThreadSafeQdrantClient

FEEDBACK_SCORE_EPSILON = 1e-9
MAX_FEEDBACK_UPDATE_RETRIES = 20

BatchUpdateOperation = SetPayloadOperation | UpdateVectorsOperation


def _require_list(value: object) -> list[Any]:
    assert isinstance(value, list)
    return value


def _require_filter(value: object) -> Filter:
    assert isinstance(value, Filter)
    return value


def _require_has_id_condition(value: object) -> HasIdCondition:
    assert isinstance(value, HasIdCondition)
    return value


def _require_field_condition(value: object) -> FieldCondition:
    assert isinstance(value, FieldCondition)
    return value


def _require_update_operations(value: object) -> list[BatchUpdateOperation]:
    assert isinstance(value, list)
    return value


def _batch_metadata_ops(update_operations: Sequence[BatchUpdateOperation]) -> list[SetPayload]:
    ops: list[SetPayload] = []
    for op in update_operations:
        if (
            isinstance(op, SetPayloadOperation)
            and op.set_payload is not None
            and op.set_payload.key == "metadata"
        ):
            ops.append(op.set_payload)
    return ops


def _batch_top_level_ops(update_operations: Sequence[BatchUpdateOperation]) -> list[SetPayload]:
    ops: list[SetPayload] = []
    for op in update_operations:
        if (
            isinstance(op, SetPayloadOperation)
            and op.set_payload is not None
            and op.set_payload.key is None
        ):
            ops.append(op.set_payload)
    return ops


def _batch_vector_ops(update_operations: Sequence[BatchUpdateOperation]) -> list[UpdateVectors]:
    ops: list[UpdateVectors] = []
    for op in update_operations:
        if isinstance(op, UpdateVectorsOperation) and op.update_vectors is not None:
            ops.append(op.update_vectors)
    return ops


def _batch_update_existing_chunk_side_effect(point: MagicMock):
    """Return a mock batch_update_points handler that applies payload updates to *point*."""

    def batch_update_side_effect(**batch_kwargs: object) -> None:
        update_operations = _require_update_operations(batch_kwargs["update_operations"])
        payload = dict(point.payload or {})
        for operation in update_operations:
            if not isinstance(operation, SetPayloadOperation) or operation.set_payload is None:
                continue
            set_payload = operation.set_payload
            update = set_payload.payload
            assert isinstance(update, dict)
            if set_payload.key == "metadata":
                metadata = dict(payload.get("metadata") or {})
                metadata.update(update)
                payload["metadata"] = metadata
            else:
                metadata = payload.get("metadata")
                payload.update(update)
                if metadata is not None:
                    payload["metadata"] = metadata
        point.payload = payload

    return batch_update_side_effect


def _merge_metadata_set_payload_side_effect(point: MagicMock):
    """Return a mock set_payload handler that merges metadata updates into *point*."""

    def set_payload_side_effect(**set_payload_kwargs: object) -> None:
        payload = set_payload_kwargs["payload"]
        assert isinstance(payload, dict)
        metadata = dict(point.payload.get("metadata") or {})
        metadata.update(payload)
        point.payload = {"metadata": metadata}

    return set_payload_side_effect


def _zero_feedback_point(chunk: Chunk) -> MagicMock:
    point = MagicMock()
    point.id = chunk.id
    point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
    return point


def _wire_feedback_cas_conflict_batch_update(
    mock_client: MagicMock,
    point: MagicMock,
) -> dict[str, int]:
    """Simulate perpetual CAS loss by bumping feedback on every batch update."""
    metadata_writes = {"count": 0}

    def batch_update_side_effect(**_batch_kwargs: object) -> None:
        metadata_writes["count"] += 1
        bump = metadata_writes["count"]
        point.payload = {
            "metadata": {
                "feedback_score": float(bump + 100),
                "feedback_revision": bump + 100,
            },
        }

    mock_client.batch_update_points.side_effect = batch_update_side_effect
    return metadata_writes


def _batch_update_fail_on_chunk_id_side_effect(
    points_by_id: dict[str, MagicMock],
    failing_chunk_id: str,
    *,
    error_message: str = "batch update failed",
) -> object:
    """Return a batch_update_points handler that fails when updating *failing_chunk_id*."""

    def batch_update_side_effect(**batch_kwargs: object) -> None:
        vector_ops = _batch_vector_ops(
            _require_update_operations(batch_kwargs["update_operations"])
        )
        assert len(vector_ops) == 1
        chunk_id = str(vector_ops[0].points[0].id)
        if chunk_id == failing_chunk_id:
            raise RuntimeError(error_message)
        _batch_update_existing_chunk_side_effect(points_by_id[chunk_id])(**batch_kwargs)

    return batch_update_side_effect


# ── fixtures ───────────────────────────────────────────────────────────────────


def _chunk(i: int = 0, *, with_vectors: bool = True) -> Chunk:
    return Chunk(
        id=f"chunk-{i:04d}",
        document_id="doc-1",
        text=f"chunk text {i}",
        embedding=[float(i) * 0.01] * 4 if with_vectors else None,
        sparse_vector={i + 1: 0.9, i + 2: 0.5} if with_vectors else None,
        metadata={"source": "test.pdf"},
    )


def _existing_point(
    chunk: Chunk,
    *,
    feedback_score: float = 0.0,
    feedback_revision: int = 0,
    extra_metadata: dict[str, object] | None = None,
) -> MagicMock:
    """Build a Qdrant point mock with vectors for snapshot/rollback tests."""
    point = MagicMock()
    point.id = chunk.id
    metadata = dict(chunk.metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    if feedback_score:
        metadata[FEEDBACK_SCORE_KEY] = feedback_score
    if feedback_revision:
        metadata[FEEDBACK_REVISION_KEY] = feedback_revision
    point.payload = chunk.model_dump(exclude={"embedding", "sparse_vector"})
    point.payload["metadata"] = metadata
    point.vector = {
        "dense": chunk.embedding,
        "sparse": {
            "indices": list(chunk.sparse_vector.keys()),  # type: ignore[union-attr]
            "values": list(chunk.sparse_vector.values()),  # type: ignore[union-attr]
        },
    }
    return point


def _retrieve_by_id_side_effect(points_by_id: dict[str, MagicMock]):
    """Return a retrieve mock that serves points keyed by chunk id."""

    def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
        ids = kwargs.get("ids", [])
        assert isinstance(ids, list)
        return [points_by_id[str(chunk_id)] for chunk_id in ids if str(chunk_id) in points_by_id]

    return retrieve_side_effect


def _wire_existing_chunk_retrieve_and_batch_update(
    mock_client: MagicMock,
    existing_chunk: Chunk,
    existing: MagicMock,
) -> None:
    """Configure retrieve and batch-update mocks for a single existing point."""
    mock_client.retrieve.side_effect = _retrieve_by_id_side_effect({existing_chunk.id: existing})
    mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(existing)


def _wire_new_chunk_retrieve_and_upsert(
    mock_client: MagicMock,
    chunks: list[Chunk],
) -> None:
    """Simulate new-chunk upsert: absent at the start, readable after each insert."""
    points_by_id = {chunk.id: _zero_feedback_point(chunk) for chunk in chunks}
    inserted: set[str] = set()

    def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
        ids = kwargs.get("ids", [])
        assert isinstance(ids, list)
        if len(ids) > 1:
            return [points_by_id[str(chunk_id)] for chunk_id in ids if str(chunk_id) in inserted]
        if len(ids) == 1:
            chunk_id = str(ids[0])
            if chunk_id in inserted:
                return [points_by_id[chunk_id]]
        return []

    def upsert_side_effect(**kwargs: object) -> None:
        for point in _require_list(kwargs["points"]):
            inserted.add(str(point.id))

    mock_client.retrieve.side_effect = retrieve_side_effect
    mock_client.upsert.side_effect = upsert_side_effect


def _wire_mixed_upsert_new_insert_fails(
    mock_client: MagicMock,
    *,
    new_index: int = 0,
    existing_index: int = 1,
    feedback_score: float = 1.0,
    feedback_revision: int = 1,
) -> tuple[Chunk, Chunk]:
    """Configure mixed upsert where the new-chunk insert rises on upsert."""
    new_chunk = _chunk(new_index)
    existing_chunk = _chunk(existing_index)
    existing = _existing_point(
        existing_chunk,
        feedback_score=feedback_score,
        feedback_revision=feedback_revision,
    )
    _wire_existing_chunk_retrieve_and_batch_update(mock_client, existing_chunk, existing)

    def upsert_side_effect(**kwargs: object) -> None:
        points = _require_list(kwargs["points"])
        if len(points) == 1 and str(points[0].id) == new_chunk.id:
            raise RuntimeError("insert failed")

    mock_client.upsert.side_effect = upsert_side_effect
    return new_chunk, existing_chunk


def _scored_point(chunk_id: str, score: float, chunk: Chunk) -> MagicMock:
    point = MagicMock()
    point.id = chunk_id
    point.score = score
    point.payload = chunk.model_dump(exclude={"embedding", "sparse_vector"})
    return point


def _query_response(points: list[MagicMock]) -> MagicMock:
    """Wrap a list of scored points in a QueryResponse-like mock."""
    resp = MagicMock()
    resp.points = points
    return resp


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.collection_exists.return_value = True
    return client


@pytest.fixture
def store(mock_client: MagicMock) -> QdrantVectorStore:
    s = QdrantVectorStore(collection="test_col", dense_dim=4)
    s._client = mock_client
    s._collection_ready = True
    return s


# ── thread-safe client ─────────────────────────────────────────────────────────


class TestThreadSafeQdrantClient:
    def test_returns_non_callable_attribute_without_locking(self):
        raw = MagicMock()
        raw.collection_name = "raw-collection"
        wrapped = ThreadSafeQdrantClient(raw, threading.RLock())

        assert wrapped.collection_name == "raw-collection"

    def test_locks_callable_methods(self):
        raw = MagicMock()
        lock = threading.RLock()
        wrapped = ThreadSafeQdrantClient(raw, lock)

        wrapped.count(collection_name="test_col")

        raw.count.assert_called_once_with(collection_name="test_col")


# ── interface conformance ──────────────────────────────────────────────────────


class TestInterfaceConformance:
    def test_implements_vector_store_repository(self, store: QdrantVectorStore):
        assert isinstance(store, VectorStoreRepository)

    def test_from_settings_uses_full_model_identifier(self):
        with (
            patch("src.infrastructure.vectordb.qdrant.QdrantClient"),
            patch(
                "src.infrastructure.embeddings.embedding_model_identifier",
                return_value="openai:text-embedding-3-large",
            ) as mock_identifier,
        ):
            instance = QdrantVectorStore.from_settings()
        mock_identifier.assert_called_once()
        assert instance.embedding_model_name == "openai:text-embedding-3-large"

    def test_from_settings_returns_instance(self):
        with patch("src.infrastructure.vectordb.qdrant.QdrantClient"):
            instance = QdrantVectorStore.from_settings()
        assert isinstance(instance, QdrantVectorStore)

    def test_from_settings_uses_provider_dense_dim(self):
        from pydantic import SecretStr

        settings = MagicMock()
        settings.qdrant = MagicMock(
            url="http://localhost:6333", collection="rag_documents", api_key=""
        )
        emb = MagicMock()
        emb.provider = "openai"
        emb.dense_dim = 1024
        emb.openai = MagicMock(
            api_key=SecretStr("sk-test"),
            model="text-embedding-3-large",
            dimensions=3072,
        )
        settings.embeddings = emb

        with (
            patch("src.core.settings.settings", settings),
            patch("src.infrastructure.vectordb.qdrant.QdrantClient"),
        ):
            instance = QdrantVectorStore.from_settings()

        assert instance.dense_dim == 3072
        assert instance.dense_dim != emb.dense_dim


# ── _ensure_collection ─────────────────────────────────────────────────────────


class TestEnsureCollection:
    def test_creates_collection_when_missing(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = False
        s = QdrantVectorStore(collection="new_col", dense_dim=4)
        s._client = mock_client
        s._ensure_collection()
        mock_client.create_collection.assert_called_once()

    def test_create_collection_stores_embedding_model_metadata(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = False
        s = QdrantVectorStore(
            collection="new_col",
            dense_dim=4,
            embedding_model_name="openai:text-embedding-3-large@3072",
        )
        s._client = mock_client
        s._ensure_collection()
        _, kwargs = mock_client.create_collection.call_args
        assert kwargs["metadata"] == {"embedding_model_name": "openai:text-embedding-3-large@3072"}

    def test_skips_creation_when_exists(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        s = QdrantVectorStore(collection="existing", dense_dim=4)
        s._client = mock_client
        s._ensure_collection()
        mock_client.create_collection.assert_not_called()

    def test_only_checks_once_after_ready(self, store: QdrantVectorStore, mock_client: MagicMock):
        store._ensure_collection()
        store._ensure_collection()
        mock_client.collection_exists.assert_not_called()

    def test_wraps_exception_as_vector_store_error(self, mock_client: MagicMock):
        mock_client.collection_exists.side_effect = ConnectionError("no server")
        s = QdrantVectorStore(collection="col", dense_dim=4)
        s._client = mock_client
        with pytest.raises(VectorStoreError) as exc_info:
            s._ensure_collection()
        assert exc_info.value.cause is not None


# ── upsert ─────────────────────────────────────────────────────────────────────


class TestUpsert:
    def test_calls_client_upsert(self, store: QdrantVectorStore, mock_client: MagicMock):
        chunks = [_chunk(0), _chunk(1)]
        _wire_new_chunk_retrieve_and_upsert(mock_client, chunks)
        store.upsert(chunks)
        assert mock_client.upsert.call_count == 2

    def test_upsert_passes_collection_name(self, store: QdrantVectorStore, mock_client: MagicMock):
        chunk = _chunk()
        _wire_new_chunk_retrieve_and_upsert(mock_client, [chunk])
        store.upsert([chunk])
        _, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == "test_col"

    def test_upsert_point_count_matches_chunks(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunks = [_chunk(0), _chunk(1), _chunk(2)]
        _wire_new_chunk_retrieve_and_upsert(mock_client, chunks)
        store.upsert(chunks)
        assert mock_client.upsert.call_count == 3
        for call in mock_client.upsert.call_args_list:
            assert len(call.kwargs["points"]) == 1

    def test_upsert_preserves_existing_feedback_metadata(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        existing = MagicMock()
        existing.id = chunk.id
        existing.payload = {
            "metadata": {
                "feedback_score": 4.0,
                "feedback_revision": 3,
            }
        }
        mock_client.retrieve.return_value = [existing]
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            existing
        )
        store.upsert([chunk])
        mock_client.upsert.assert_not_called()
        mock_client.batch_update_points.assert_called_once()
        metadata_ops = _batch_metadata_ops(
            _require_update_operations(
                mock_client.batch_update_points.call_args.kwargs["update_operations"]
            )
        )
        assert len(metadata_ops) == 1
        assert metadata_ops[0].payload[FEEDBACK_SCORE_KEY] == 4.0
        assert metadata_ops[0].payload[FEEDBACK_REVISION_KEY] == 3

    def test_upsert_preserves_feedback_update_id(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        existing = MagicMock()
        existing.id = chunk.id
        existing.payload = {
            "metadata": {
                "feedback_score": 1.0,
                "feedback_revision": 1,
                "feedback_update_id": "update-123",
            }
        }
        mock_client.retrieve.return_value = [existing]
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            existing
        )
        store.upsert([chunk])
        metadata_ops = _batch_metadata_ops(
            _require_update_operations(
                mock_client.batch_update_points.call_args.kwargs["update_operations"]
            )
        )
        assert metadata_ops[0].payload["feedback_update_id"] == "update-123"

    def test_upsert_existing_chunk_uses_batch_update_points(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        existing = MagicMock()
        existing.id = chunk.id
        existing.payload = {"metadata": {}}
        mock_client.retrieve.return_value = [existing]
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            existing
        )
        store.upsert([chunk])
        mock_client.upsert.assert_not_called()
        batch_args = mock_client.batch_update_points.call_args.kwargs
        assert batch_args["collection_name"] == "test_col"
        vector_ops = _batch_vector_ops(_require_update_operations(batch_args["update_operations"]))
        assert len(vector_ops) == 1
        assert len(vector_ops[0].points) == 1
        assert str(vector_ops[0].points[0].id) == chunk.id

    def test_upsert_retries_metadata_when_feedback_changes_during_update(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 1}}
        mock_client.retrieve.return_value = [point]
        metadata_writes = {"count": 0}

        def batch_update_side_effect(**batch_kwargs: object) -> None:
            metadata_writes["count"] += 1
            if metadata_writes["count"] == 1:
                point.payload = {
                    "metadata": {"feedback_score": 2.0, "feedback_revision": 2},
                }
                return
            _batch_update_existing_chunk_side_effect(point)(**batch_kwargs)

        mock_client.batch_update_points.side_effect = batch_update_side_effect
        store.upsert([chunk])
        assert metadata_writes["count"] >= 2
        assert point.payload["metadata"][FEEDBACK_SCORE_KEY] == 2.0
        assert point.payload["metadata"][FEEDBACK_REVISION_KEY] == 2

    def test_upsert_mixed_new_and_existing_chunks(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        new_chunk = _chunk(0)
        existing_chunk = _chunk(1)
        existing = _existing_point(existing_chunk, feedback_score=2.0, feedback_revision=1)
        points_by_id = {existing_chunk.id: existing}
        inserted: set[str] = set()

        def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
            ids = kwargs.get("ids", [])
            assert isinstance(ids, list)
            if len(ids) > 1:
                return [
                    points_by_id[str(chunk_id)] for chunk_id in ids if str(chunk_id) in points_by_id
                ]
            if len(ids) == 1:
                chunk_id = str(ids[0])
                if chunk_id in points_by_id:
                    return [points_by_id[chunk_id]]
                if chunk_id in inserted:
                    return [_zero_feedback_point(new_chunk)]
            return []

        def upsert_side_effect(**kwargs: object) -> None:
            for point in _require_list(kwargs["points"]):
                inserted.add(str(point.id))

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.upsert.side_effect = upsert_side_effect
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            existing
        )
        store.upsert([new_chunk, existing_chunk])
        assert mock_client.batch_update_points.call_count == 1
        mock_client.upsert.assert_called_once()
        assert len(mock_client.upsert.call_args.kwargs["points"]) == 1
        vector_ops = _batch_vector_ops(
            _require_update_operations(
                mock_client.batch_update_points.call_args.kwargs["update_operations"]
            )
        )
        assert len(vector_ops[0].points) == 1
        assert mock_client.method_calls[0][0] == "retrieve"
        batch_update_index = next(
            i for i, call in enumerate(mock_client.method_calls) if call[0] == "batch_update_points"
        )
        upsert_index = next(
            i for i, call in enumerate(mock_client.method_calls) if call[0] == "upsert"
        )
        assert batch_update_index < upsert_index

    def test_upsert_adds_chunk_type_to_payload(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        from src.core.constants import CHUNK_TYPE_HYPE, CHUNK_TYPE_KEY

        chunk = _chunk(0).model_copy(update={"metadata": {CHUNK_TYPE_KEY: CHUNK_TYPE_HYPE}})
        _wire_new_chunk_retrieve_and_upsert(mock_client, [chunk])
        store.upsert([chunk])
        payload = mock_client.upsert.call_args.kwargs["points"][0].payload
        assert payload[CHUNK_TYPE_KEY] == CHUNK_TYPE_HYPE

    def test_upsert_skips_empty_list(self, store: QdrantVectorStore, mock_client: MagicMock):
        store.upsert([])
        mock_client.upsert.assert_not_called()

    def test_raises_on_missing_embedding(self, store: QdrantVectorStore):
        with pytest.raises(VectorStoreError, match="embedding"):
            store.upsert([_chunk(with_vectors=False)])

    def test_wraps_client_error(self, store: QdrantVectorStore, mock_client: MagicMock):
        chunk = _chunk()
        mock_client.retrieve.return_value = []
        mock_client.upsert.side_effect = RuntimeError("oops")
        with pytest.raises(VectorStoreError) as exc_info:
            store.upsert([chunk])
        assert exc_info.value.cause is not None

    def test_upsert_existing_chunk_sets_top_level_payload_fields(self, mock_client: MagicMock):
        from src.core.constants import CHUNK_TYPE_HYPE, CHUNK_TYPE_KEY

        store = QdrantVectorStore(
            collection="test_col",
            dense_dim=4,
            embedding_model_name="bge-m3",
        )
        store._client = mock_client
        store._collection_ready = True
        chunk = _chunk(0).model_copy(update={"metadata": {CHUNK_TYPE_KEY: CHUNK_TYPE_HYPE}})
        existing = MagicMock()
        existing.id = chunk.id
        existing.payload = {"metadata": {}}
        mock_client.retrieve.return_value = [existing]
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            existing
        )

        store.upsert([chunk])

        top_level_ops = _batch_top_level_ops(
            _require_update_operations(
                mock_client.batch_update_points.call_args.kwargs["update_operations"]
            )
        )
        assert top_level_ops[0].payload["embedding_model_name"] == "bge-m3"
        assert top_level_ops[0].payload[CHUNK_TYPE_KEY] == CHUNK_TYPE_HYPE

    def test_upsert_existing_chunk_raises_after_metadata_retry_exhaustion(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = _zero_feedback_point(chunk)
        mock_client.retrieve.return_value = [point]
        metadata_writes = _wire_feedback_cas_conflict_batch_update(mock_client, point)

        with pytest.raises(VectorStoreError, match="Failed to upsert existing chunk"):
            store.upsert([chunk])

        assert metadata_writes["count"] == MAX_FEEDBACK_UPDATE_RETRIES
        mock_client.update_vectors.assert_not_called()

    def test_upsert_existing_chunk_updates_per_chunk_in_order(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk_a = _chunk(0)
        chunk_b = _chunk(1)
        existing_a = _existing_point(chunk_a, feedback_score=1.0, feedback_revision=1)
        existing_b = _existing_point(chunk_b, feedback_score=2.0, feedback_revision=2)
        points_by_id = {chunk_a.id: existing_a, chunk_b.id: existing_b}

        mock_client.retrieve.side_effect = _retrieve_by_id_side_effect(points_by_id)

        mock_client.batch_update_points.side_effect = _batch_update_fail_on_chunk_id_side_effect(
            points_by_id,
            chunk_b.id,
            error_message="batch update failed for chunk-b",
        )

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            store.upsert([chunk_a, chunk_b])

        assert mock_client.batch_update_points.call_count == 2
        first_vector_ops = _batch_vector_ops(
            _require_update_operations(
                mock_client.batch_update_points.call_args_list[0].kwargs["update_operations"]
            )
        )
        assert str(first_vector_ops[0].points[0].id) == chunk_a.id
        second_vector_ops = _batch_vector_ops(
            _require_update_operations(
                mock_client.batch_update_points.call_args_list[1].kwargs["update_operations"]
            )
        )
        assert str(second_vector_ops[0].points[0].id) == chunk_b.id
        rollback_call = mock_client.upsert.call_args_list[0]
        assert len(rollback_call.kwargs["points"]) == 1
        assert str(rollback_call.kwargs["points"][0].id) == chunk_a.id

    def test_upsert_rolls_back_existing_when_new_insert_fails(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        new_chunk, existing_chunk = _wire_mixed_upsert_new_insert_fails(mock_client)

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            store.upsert([new_chunk, existing_chunk])

        assert mock_client.upsert.call_count == 2
        insert_points = mock_client.upsert.call_args_list[0].kwargs["points"]
        assert len(insert_points) == 1
        assert str(insert_points[0].id) == new_chunk.id
        rollback_points = mock_client.upsert.call_args_list[1].kwargs["points"]
        assert len(rollback_points) == 1
        assert str(rollback_points[0].id) == existing_chunk.id

    def test_snapshot_existing_points_requires_vectors(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        existing = _existing_point(chunk)
        mock_client.retrieve.return_value = [existing]

        snapshots = store._snapshot_existing_points([chunk.id])

        mock_client.retrieve.assert_called_once_with(
            collection_name="test_col",
            ids=[chunk.id],
            with_payload=True,
            with_vectors=True,
        )
        assert str(snapshots[chunk.id].id) == chunk.id
        assert snapshots[chunk.id].vector is not None
        assert snapshots[chunk.id].payload == existing.payload

    def test_snapshot_existing_points_raises_when_point_missing(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        mock_client.retrieve.return_value = []

        with pytest.raises(VectorStoreError, match="Cannot snapshot existing chunks"):
            store._snapshot_existing_points(["chunk-0001"])

    def test_rollback_points_skips_empty_snapshots(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        store._rollback_points({}, context="test")

        mock_client.upsert.assert_not_called()

    def test_rollback_points_raises_when_upsert_fails(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        snapshot = PointStruct(
            id=chunk.id,
            vector=_existing_point(chunk).vector,
            payload=_existing_point(chunk).payload,
        )
        mock_client.upsert.side_effect = RuntimeError("rollback failed")

        with pytest.raises(
            VectorStoreError, match="partial existing-chunk update rollback failed"
        ) as exc_info:
            store._rollback_points(
                {chunk.id: snapshot},
                context="partial existing-chunk update",
            )

        assert exc_info.value.cause is not None

    def test_rollback_points_preserves_concurrent_feedback(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        snapshot = PointStruct(
            id=chunk.id,
            vector=_existing_point(chunk, feedback_score=1.0, feedback_revision=1).vector,
            payload=_existing_point(chunk, feedback_score=1.0, feedback_revision=1).payload,
        )
        current = _existing_point(chunk, feedback_score=4.0, feedback_revision=5)
        mock_client.retrieve.return_value = [current]

        store._rollback_points({chunk.id: snapshot}, context="test")

        restored = mock_client.upsert.call_args.kwargs["points"][0]
        assert restored.payload["metadata"][FEEDBACK_SCORE_KEY] == 4.0
        assert restored.payload["metadata"][FEEDBACK_REVISION_KEY] == 5

    def test_snapshot_with_current_feedback_returns_snapshot_when_point_missing(
        self, store: QdrantVectorStore
    ):
        chunk = _chunk(0)
        snapshot = PointStruct(
            id=chunk.id,
            vector=_existing_point(chunk).vector,
            payload=_existing_point(chunk).payload,
        )
        assert store._snapshot_with_current_feedback(snapshot, None) == snapshot

    def test_upsert_rollback_preserves_concurrent_feedback_after_existing_update(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        new_chunk, existing_chunk = _wire_mixed_upsert_new_insert_fails(
            mock_client,
            feedback_score=1.0,
            feedback_revision=1,
        )
        existing = _existing_point(
            existing_chunk,
            feedback_score=1.0,
            feedback_revision=1,
        )
        reads = {"count": 0}

        def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            ids = kwargs.get("ids", [])
            assert isinstance(ids, list)
            if len(ids) > 1:
                return [existing]
            if len(ids) == 1 and str(ids[0]) == existing_chunk.id:
                if reads["count"] >= 6:
                    existing.payload = {
                        **existing.payload,
                        "metadata": {
                            **existing.payload["metadata"],
                            FEEDBACK_SCORE_KEY: 4.0,
                            FEEDBACK_REVISION_KEY: 5,
                        },
                    }
                return [existing]
            return []

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            existing
        )

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            store.upsert([new_chunk, existing_chunk])

        rollback_points = mock_client.upsert.call_args_list[-1].kwargs["points"]
        assert rollback_points[0].payload["metadata"][FEEDBACK_SCORE_KEY] == 4.0
        assert rollback_points[0].payload["metadata"][FEEDBACK_REVISION_KEY] == 5

    def test_partial_existing_chunk_rollback_preserves_concurrent_feedback(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk_a = _chunk(0)
        chunk_b = _chunk(1)
        existing_a = _existing_point(chunk_a, feedback_score=1.0, feedback_revision=1)
        existing_b = _existing_point(chunk_b, feedback_score=2.0, feedback_revision=2)
        points_by_id = {chunk_a.id: existing_a, chunk_b.id: existing_b}
        reads = {"count": 0}

        def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            ids = kwargs.get("ids", [])
            assert isinstance(ids, list)
            id_strs = [str(chunk_id) for chunk_id in ids]
            if len(id_strs) > 1:
                return [points_by_id[chunk_id] for chunk_id in id_strs if chunk_id in points_by_id]
            if chunk_a.id in id_strs and reads["count"] >= 5:
                existing_a.payload = {
                    **existing_a.payload,
                    "metadata": {
                        **existing_a.payload["metadata"],
                        FEEDBACK_SCORE_KEY: 6.0,
                        FEEDBACK_REVISION_KEY: 7,
                    },
                }
            if len(id_strs) == 1 and id_strs[0] in points_by_id:
                return [points_by_id[id_strs[0]]]
            return []

        mock_client.retrieve.side_effect = retrieve_side_effect

        mock_client.batch_update_points.side_effect = _batch_update_fail_on_chunk_id_side_effect(
            points_by_id,
            chunk_b.id,
            error_message="batch update failed for chunk-b",
        )

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            store.upsert([chunk_a, chunk_b])

        rollback_points = mock_client.upsert.call_args.kwargs["points"]
        assert rollback_points[0].payload["metadata"][FEEDBACK_SCORE_KEY] == 6.0
        assert rollback_points[0].payload["metadata"][FEEDBACK_REVISION_KEY] == 7

    def test_upsert_existing_chunk_wraps_batch_update_error(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        existing = MagicMock()
        existing.id = chunk.id
        existing.payload = {"metadata": {}}
        mock_client.retrieve.return_value = [existing]
        mock_client.batch_update_points.side_effect = RuntimeError("batch failed")

        with pytest.raises(VectorStoreError, match="batch_update_points failed") as exc_info:
            store.upsert([chunk])
        assert exc_info.value.cause is not None

    def test_upsert_routes_stale_new_chunk_to_update_path(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        """Chunk classified as new at upsert start but created before insert uses CAS update."""
        chunk = _chunk(0)
        existing = _existing_point(chunk, feedback_score=3.0, feedback_revision=2)
        reads = {"count": 0}

        def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            ids = kwargs.get("ids", [])
            assert isinstance(ids, list)
            if reads["count"] == 1:
                return []
            if chunk.id in [str(chunk_id) for chunk_id in ids]:
                return [existing]
            return []

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            existing
        )

        store.upsert([chunk])

        mock_client.upsert.assert_not_called()
        mock_client.batch_update_points.assert_called_once()
        metadata_ops = _batch_metadata_ops(
            _require_update_operations(
                mock_client.batch_update_points.call_args.kwargs["update_operations"]
            )
        )
        assert metadata_ops[0].payload[FEEDBACK_SCORE_KEY] == 3.0
        assert metadata_ops[0].payload[FEEDBACK_REVISION_KEY] == 2

    def test_upsert_new_chunk_updates_after_concurrent_feedback_on_insert(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        reads = {"count": 0}

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            if reads["count"] == 1:
                return []
            if reads["count"] == 2:
                return []
            point.payload = {
                "metadata": {"feedback_score": 1.0, "feedback_revision": 1},
            }
            return [point]

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            point
        )

        store.upsert([chunk])

        assert mock_client.upsert.call_count == 1
        mock_client.batch_update_points.assert_called_once()

    def test_upsert_new_chunk_returns_after_insert_when_feedback_still_default(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        reads = {"count": 0}

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            if reads["count"] <= 2:
                return []
            return [point]

        mock_client.retrieve.side_effect = retrieve_side_effect

        store.upsert([chunk])

        assert mock_client.upsert.call_count == 1
        mock_client.batch_update_points.assert_not_called()

    def test_insert_new_chunk_raises_after_retry_exhaustion(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = _zero_feedback_point(chunk)
        mock_client.retrieve.return_value = [point]
        metadata_writes = _wire_feedback_cas_conflict_batch_update(mock_client, point)

        with pytest.raises(VectorStoreError, match="Failed to upsert existing chunk"):
            store._insert_new_chunk(chunk)

        assert metadata_writes["count"] == MAX_FEEDBACK_UPDATE_RETRIES

    def test_insert_new_chunk_raises_when_point_missing_after_upsert(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        reads = {"count": 0}

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            if reads["count"] == 1:
                return []
            return []

        mock_client.retrieve.side_effect = retrieve_side_effect

        with pytest.raises(VectorStoreError, match="not found after upsert"):
            store._insert_new_chunk(chunk)

        mock_client.upsert.assert_called_once()

    def test_upsert_rolls_back_partial_new_chunk_inserts(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk_a = _chunk(0)
        chunk_b = _chunk(1)
        point_a = _zero_feedback_point(chunk_a)
        reads = {"count": 0}

        def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            ids = kwargs.get("ids", [])
            assert isinstance(ids, list)
            if reads["count"] == 1:
                return []
            if chunk_a.id in [str(chunk_id) for chunk_id in ids] and reads["count"] == 3:
                return [point_a]
            return []

        mock_client.retrieve.side_effect = retrieve_side_effect

        def upsert_side_effect(**kwargs: object) -> None:
            points = _require_list(kwargs["points"])
            if len(points) == 1 and str(points[0].id) == chunk_b.id:
                raise RuntimeError("insert failed for chunk-b")

        mock_client.upsert.side_effect = upsert_side_effect

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            store.upsert([chunk_a, chunk_b])

        assert mock_client.upsert.call_count == 2
        assert str(mock_client.upsert.call_args_list[0].kwargs["points"][0].id) == chunk_a.id
        assert str(mock_client.upsert.call_args_list[1].kwargs["points"][0].id) == chunk_b.id
        mock_client.delete.assert_called_once()
        deleted_ids = mock_client.delete.call_args.kwargs["points_selector"].points
        assert deleted_ids == [chunk_a.id]

    def test_insert_new_chunk_returns_false_when_concurrent_feedback_after_insert(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        reads = {"count": 0}

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            if reads["count"] == 1:
                return []
            point.payload = {
                "metadata": {"feedback_score": 1.0, "feedback_revision": 1},
            }
            return [point]

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            point
        )

        assert store._insert_new_chunk(chunk) is False

    def test_upsert_does_not_rollback_chunk_with_concurrent_feedback_on_insert(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk_a = _chunk(0)
        chunk_b = _chunk(1)
        point_a = _zero_feedback_point(chunk_a)
        point_b = MagicMock()
        point_b.id = chunk_b.id
        point_b.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        reads = {"count": 0}

        def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            ids = kwargs.get("ids", [])
            assert isinstance(ids, list)
            id_strs = [str(chunk_id) for chunk_id in ids]
            if reads["count"] == 1:
                return []
            if chunk_a.id in id_strs and reads["count"] == 3:
                return [point_a]
            if chunk_b.id in id_strs:
                if reads["count"] == 5:
                    return []
                point_b.payload = {
                    "metadata": {"feedback_score": 2.0, "feedback_revision": 1},
                }
                return [point_b]
            return []

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            point_b
        )

        chunk_c = _chunk(2)

        def upsert_side_effect(**kwargs: object) -> None:
            points = _require_list(kwargs["points"])
            if len(points) == 1 and str(points[0].id) == chunk_c.id:
                raise RuntimeError("insert failed for chunk-c")

        mock_client.upsert.side_effect = upsert_side_effect

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            store.upsert([chunk_a, chunk_b, chunk_c])

        mock_client.delete.assert_called_once()
        deleted_ids = mock_client.delete.call_args.kwargs["points_selector"].points
        assert deleted_ids == [chunk_a.id]

    def test_upsert_rolls_back_existing_and_partial_new_inserts(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        new_chunk, existing_chunk = _wire_mixed_upsert_new_insert_fails(mock_client)

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            store.upsert([new_chunk, existing_chunk])

        rollback_call = mock_client.upsert.call_args_list[-1]
        assert len(rollback_call.kwargs["points"]) == 1
        assert str(rollback_call.kwargs["points"][0].id) == existing_chunk.id
        mock_client.delete.assert_not_called()

    def test_rollback_inserted_chunks_skips_empty_ids(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        store._rollback_inserted_chunks([], context="test")
        mock_client.delete.assert_not_called()

    def test_rollback_inserted_chunks_raises_when_delete_fails(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        mock_client.delete.side_effect = RuntimeError("delete failed")

        with pytest.raises(
            VectorStoreError, match="partial new-chunk insert rollback failed"
        ) as exc_info:
            store._rollback_inserted_chunks(["chunk-0001"], context="partial new-chunk insert")

        assert exc_info.value.cause is not None


class TestExistingChunkMetadataCas:
    def test_try_set_metadata_if_feedback_current_returns_false_before_write(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 2}}
        mock_client.retrieve.return_value = [point]

        assert not store._try_set_metadata_if_feedback_current(
            "chunk-a",
            metadata={"source": "test.pdf"},
            expected_score=1.0,
            expected_revision=1,
        )
        mock_client.set_payload.assert_not_called()

    def test_try_set_metadata_if_feedback_current_wraps_set_payload_error(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 0}}
        mock_client.retrieve.return_value = [point]
        mock_client.set_payload.side_effect = RuntimeError("write failed")

        with pytest.raises(VectorStoreError, match="set_payload failed") as exc_info:
            store._try_set_metadata_if_feedback_current(
                "chunk-a",
                metadata={
                    "source": "test.pdf",
                    FEEDBACK_SCORE_KEY: 1.0,
                    FEEDBACK_REVISION_KEY: 0,
                },
                expected_score=1.0,
                expected_revision=0,
            )
        assert exc_info.value.cause is not None

    def test_try_set_metadata_if_feedback_current_raises_when_chunk_deleted_before_write(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        mock_client.retrieve.return_value = []
        with pytest.raises(VectorStoreError, match="not found"):
            store._try_set_metadata_if_feedback_current(
                "chunk-a",
                metadata={"source": "test.pdf"},
                expected_score=0.0,
                expected_revision=0,
            )
        mock_client.set_payload.assert_not_called()

    def test_try_set_metadata_if_feedback_current_raises_when_chunk_deleted_after_write(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 0}}
        mock_client.retrieve.side_effect = [[point], []]
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)
        with pytest.raises(VectorStoreError, match="not found"):
            store._try_set_metadata_if_feedback_current(
                "chunk-a",
                metadata={"source": "test.pdf", FEEDBACK_SCORE_KEY: 1.0, FEEDBACK_REVISION_KEY: 0},
                expected_score=1.0,
                expected_revision=0,
            )

    def test_try_set_metadata_if_feedback_current_returns_true_on_success(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 0}}
        mock_client.retrieve.return_value = [point]
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)

        assert store._try_set_metadata_if_feedback_current(
            "chunk-a",
            metadata={
                "source": "test.pdf",
                FEEDBACK_SCORE_KEY: 1.0,
                FEEDBACK_REVISION_KEY: 0,
            },
            expected_score=1.0,
            expected_revision=0,
        )
        mock_client.set_payload.assert_called_once()


class TestBatchUpdateExistingChunk:
    def test_try_batch_update_returns_false_when_cas_pre_check_fails(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 2.0, "feedback_revision": 1}}
        mock_client.retrieve.return_value = [point]

        assert not store._try_batch_update_existing_chunk_if_feedback_current(
            chunk,
            metadata={"source": "test.pdf"},
            expected_score=1.0,
            expected_revision=0,
        )
        mock_client.batch_update_points.assert_not_called()

    def test_try_batch_update_returns_true_on_success(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 1}}
        mock_client.retrieve.return_value = [point]
        mock_client.batch_update_points.side_effect = _batch_update_existing_chunk_side_effect(
            point
        )

        assert store._try_batch_update_existing_chunk_if_feedback_current(
            chunk,
            metadata={
                "source": "test.pdf",
                FEEDBACK_SCORE_KEY: 1.0,
                FEEDBACK_REVISION_KEY: 1,
            },
            expected_score=1.0,
            expected_revision=1,
        )
        mock_client.batch_update_points.assert_called_once()

    def test_try_batch_update_returns_false_when_feedback_changes_after_write(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 1}}
        mock_client.retrieve.return_value = [point]

        def batch_update_side_effect(**_batch_kwargs: object) -> None:
            point.payload = {
                "metadata": {"feedback_score": 2.0, "feedback_revision": 2},
            }

        mock_client.batch_update_points.side_effect = batch_update_side_effect

        assert not store._try_batch_update_existing_chunk_if_feedback_current(
            chunk,
            metadata={
                "source": "test.pdf",
                FEEDBACK_SCORE_KEY: 1.0,
                FEEDBACK_REVISION_KEY: 1,
            },
            expected_score=1.0,
            expected_revision=1,
        )

    def test_try_batch_update_wraps_batch_update_points_error(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        mock_client.retrieve.return_value = [point]
        mock_client.batch_update_points.side_effect = RuntimeError("batch failed")

        with pytest.raises(VectorStoreError, match="batch_update_points failed") as exc_info:
            store._try_batch_update_existing_chunk_if_feedback_current(
                chunk,
                metadata={"source": "test.pdf"},
                expected_score=0.0,
                expected_revision=0,
            )
        assert exc_info.value.cause is not None


# ── search_dense ───────────────────────────────────────────────────────────────


class TestSearchDense:
    def test_returns_list_of_tuples(self, store: QdrantVectorStore, mock_client: MagicMock):
        c = _chunk(0)
        mock_client.query_points.return_value = _query_response([_scored_point(c.id, 0.9, c)])
        results = store.search_dense([0.1, 0.2, 0.3, 0.4], top_k=5)
        assert isinstance(results, list)
        assert isinstance(results[0], tuple)

    def test_chunk_and_score_types(self, store: QdrantVectorStore, mock_client: MagicMock):
        c = _chunk(0)
        mock_client.query_points.return_value = _query_response([_scored_point(c.id, 0.85, c)])
        chunk, score = store.search_dense([0.1] * 4, top_k=1)[0]
        assert isinstance(chunk, Chunk)
        assert isinstance(score, float)

    def test_score_value_preserved(self, store: QdrantVectorStore, mock_client: MagicMock):
        c = _chunk(0)
        mock_client.query_points.return_value = _query_response([_scored_point(c.id, 0.77, c)])
        _, score = store.search_dense([0.1] * 4, top_k=1)[0]
        assert score == pytest.approx(0.77)

    def test_passes_top_k_as_limit(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.query_points.return_value = _query_response([])
        store.search_dense([0.1] * 4, top_k=7)
        _, kwargs = mock_client.query_points.call_args
        assert kwargs["limit"] == 7

    def test_type_equals_filter(self, store: QdrantVectorStore, mock_client: MagicMock):
        from src.core.constants import CHUNK_TYPE_HYPE

        mock_client.query_points.return_value = _query_response([])
        store.search_dense([0.1] * 4, top_k=5, type_equals=CHUNK_TYPE_HYPE)
        _, kwargs = mock_client.query_points.call_args
        query_filter = kwargs["query_filter"]
        assert query_filter.must[0].key == "type"
        assert query_filter.must[0].match.value == CHUNK_TYPE_HYPE

    def test_exclude_types_filter(self, store: QdrantVectorStore, mock_client: MagicMock):
        from src.core.constants import CHUNK_TYPE_HYPE

        mock_client.query_points.return_value = _query_response([])
        store.search_dense([0.1] * 4, top_k=5, exclude_types=frozenset({CHUNK_TYPE_HYPE}))
        _, kwargs = mock_client.query_points.call_args
        query_filter = kwargs["query_filter"]
        assert query_filter.must_not[0].match.value == CHUNK_TYPE_HYPE

    def test_document_ids_filter(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.query_points.return_value = _query_response([])
        store.search_dense(
            [0.1] * 4,
            top_k=5,
            document_ids=frozenset({"doc-a", "doc-b"}),
        )
        _, kwargs = mock_client.query_points.call_args
        query_filter = kwargs["query_filter"]
        assert query_filter.must[0].key == "document_id"
        assert set(query_filter.must[0].match.any) == {"doc-a", "doc-b"}

    def test_metadata_filter(self, store: QdrantVectorStore, mock_client: MagicMock):
        from src.rag.retrieval.filters import RetrievalFilter

        mock_client.query_points.return_value = _query_response([])
        store.search_dense(
            [0.1] * 4,
            top_k=5,
            filters=RetrievalFilter(metadata={"source": "report.pdf"}),
        )
        _, kwargs = mock_client.query_points.call_args
        query_filter = kwargs["query_filter"]
        assert query_filter.must[0].key == "metadata.source"
        assert query_filter.must[0].match.value == "report.pdf"


# ── search_sparse ──────────────────────────────────────────────────────────────


class TestSearchSparse:
    def test_returns_list(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.query_points.return_value = _query_response([])
        result = store.search_sparse({1: 0.9, 2: 0.5}, top_k=5)
        assert isinstance(result, list)

    def test_chunk_reconstructed_from_payload(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        c = _chunk(1)
        mock_client.query_points.return_value = _query_response([_scored_point(c.id, 0.6, c)])
        chunk, _ = store.search_sparse({1: 0.9}, top_k=1)[0]
        assert chunk.text == c.text
        assert chunk.document_id == c.document_id


# ── search_hybrid (RRF) ────────────────────────────────────────────────────────


class TestSearchHybrid:
    def test_combines_dense_and_sparse(self, store: QdrantVectorStore, mock_client: MagicMock):
        c0, c1, c2 = _chunk(0), _chunk(1), _chunk(2)
        # dense returns c0, c1; sparse returns c1, c2
        mock_client.query_points.side_effect = [
            _query_response([_scored_point(c0.id, 0.9, c0), _scored_point(c1.id, 0.8, c1)]),
            _query_response([_scored_point(c1.id, 0.7, c1), _scored_point(c2.id, 0.6, c2)]),
        ]
        results = store.search_hybrid([0.1] * 4, {1: 0.9}, alpha=0.7, top_k=3)
        ids = [c.id for c, _ in results]
        assert c0.id in ids
        assert c1.id in ids
        assert c2.id in ids

    def test_c1_ranks_higher_due_to_double_hit(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        c0, c1, c2 = _chunk(0), _chunk(1), _chunk(2)
        mock_client.query_points.side_effect = [
            _query_response([_scored_point(c0.id, 0.9, c0), _scored_point(c1.id, 0.8, c1)]),
            _query_response([_scored_point(c1.id, 0.95, c1), _scored_point(c2.id, 0.6, c2)]),
        ]
        results = store.search_hybrid([0.1] * 4, {1: 0.9}, alpha=0.7, top_k=3)
        # c1 appears in both lists, so it should have the highest fused score
        assert results[0][0].id == c1.id

    def test_respects_top_k(self, store: QdrantVectorStore, mock_client: MagicMock):
        chunks = [_chunk(i) for i in range(6)]

        def _pts(cs):
            pts = [_scored_point(c.id, 0.9 - i * 0.1, c) for i, c in enumerate(cs)]
            return _query_response(pts)

        mock_client.query_points.side_effect = [_pts(chunks[:3]), _pts(chunks[3:])]
        results = store.search_hybrid([0.1] * 4, {1: 0.9}, alpha=0.7, top_k=2)
        assert len(results) == 2


# ── delete & count ─────────────────────────────────────────────────────────────


class TestDeleteAndCount:
    def test_delete_calls_client(self, store: QdrantVectorStore, mock_client: MagicMock):
        store.delete(["id-1", "id-2"])
        mock_client.delete.assert_called_once()

    def test_count_returns_int(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.count.return_value = MagicMock(count=42)
        assert store.count() == 42

    def test_count_wraps_error(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.count.side_effect = RuntimeError("down")
        with pytest.raises(VectorStoreError):
            store.count()

    def test_chunk_exists_true(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.retrieve.return_value = [MagicMock()]
        assert store.chunk_exists("chunk-a") is True

    def test_chunk_exists_false(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.retrieve.return_value = []
        assert store.chunk_exists("missing") is False


class TestQdrantFeedbackScore:
    def test_get_feedback_score_reads_metadata(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 2.5}}
        mock_client.retrieve.return_value = [point]
        assert store.get_feedback_score("chunk-a") == 2.5

    def test_get_feedback_score_retrieve_failure_raises(self, store, mock_client):
        mock_client.retrieve.side_effect = RuntimeError("down")
        with pytest.raises(VectorStoreError, match="retrieve failed"):
            store.get_feedback_score("chunk-a")

    def test_get_feedback_score_bool_returns_zero(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": True}}
        mock_client.retrieve.return_value = [point]
        assert store.get_feedback_score("chunk-a") == 0.0

    def test_get_feedback_score_missing_chunk_returns_zero(self, store, mock_client):
        mock_client.retrieve.return_value = []
        assert store.get_feedback_score("missing") == 0.0

    def test_get_feedback_revision_reads_metadata(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_revision": 3}}
        mock_client.retrieve.return_value = [point]
        assert store.get_feedback_revision("chunk-a") == 3

    def test_get_feedback_revision_missing_chunk_returns_zero(self, store, mock_client):
        mock_client.retrieve.return_value = []
        assert store.get_feedback_revision("missing") == 0

    def test_get_feedback_revision_bool_returns_zero(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_revision": True}}
        mock_client.retrieve.return_value = [point]
        assert store.get_feedback_revision("chunk-a") == 0

    def test_get_feedback_revision_integer_float(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_revision": 4.0}}
        mock_client.retrieve.return_value = [point]
        assert store.get_feedback_revision("chunk-a") == 4

    def test_ensure_feedback_fields_initialized_backfills_defaults(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"source": "doc.pdf"}}
        mock_client.retrieve.return_value = [point]
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)
        store._ensure_feedback_fields_initialized("chunk-a")
        mock_client.set_payload.assert_called_once()
        _, set_payload_args = mock_client.set_payload.call_args
        assert set_payload_args["collection_name"] == "test_col"
        assert set_payload_args["key"] == "metadata"
        assert set_payload_args["payload"]["feedback_score"] == 0.0
        assert set_payload_args["payload"]["feedback_revision"] == 0
        assert "feedback_update_id" in set_payload_args["payload"]
        points = _require_filter(set_payload_args["points"])
        assert points.must is not None

    def test_ensure_feedback_fields_initialized_retries_without_clobbering_score(
        self, store, mock_client
    ):
        point = MagicMock()
        point.payload = {"metadata": {"source": "doc.pdf"}}
        reads = {"count": 0}

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            if reads["count"] >= 3:
                point.payload = {
                    "metadata": {
                        "source": "doc.pdf",
                        "feedback_score": 2.0,
                        "feedback_revision": 1,
                    }
                }
            return [point]

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)
        store._ensure_feedback_fields_initialized("chunk-a")
        assert point.payload["metadata"][FEEDBACK_SCORE_KEY] == 2.0
        assert point.payload["metadata"][FEEDBACK_REVISION_KEY] == 1

    def test_ensure_feedback_fields_initialized_raises_after_retry_exhaustion(
        self, store, mock_client
    ):
        point = MagicMock()
        point.payload = {"metadata": {"source": "doc.pdf"}}
        mock_client.retrieve.return_value = [point]
        mock_client.set_payload.side_effect = lambda **_kwargs: None

        with pytest.raises(VectorStoreError, match="Failed to initialize feedback fields"):
            store._ensure_feedback_fields_initialized("chunk-a")

        assert mock_client.set_payload.call_count == MAX_FEEDBACK_UPDATE_RETRIES

    def test_set_feedback_score_updates_metadata(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"source": "doc.pdf"}}
        mock_client.retrieve.return_value = [point]
        store.set_feedback_score("chunk-a", 1.0)
        mock_client.set_payload.assert_called_once_with(
            collection_name="test_col",
            payload={
                "metadata": {
                    "source": "doc.pdf",
                    "feedback_score": 1.0,
                    "feedback_revision": 1,
                }
            },
            points=["chunk-a"],
        )

    def test_set_feedback_score_missing_chunk_raises(self, store, mock_client):
        mock_client.retrieve.return_value = []
        with pytest.raises(VectorStoreError, match="not found"):
            store.set_feedback_score("missing", 1.0)

    def test_set_feedback_score_set_payload_failure_raises(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"source": "doc.pdf"}}
        mock_client.retrieve.return_value = [point]
        mock_client.set_payload.side_effect = RuntimeError("write failed")
        with pytest.raises(VectorStoreError, match="set_payload failed"):
            store.set_feedback_score("chunk-a", 1.0)

    def test_get_feedback_scores_empty_ids_returns_empty(self, store, mock_client):
        assert store.get_feedback_scores([]) == {}
        mock_client.retrieve.assert_not_called()

    def test_try_set_feedback_score_if_current_uses_conditional_filter(self, store, mock_client):
        point = MagicMock()
        point.payload = {
            "metadata": {"feedback_score": 1.0, "feedback_revision": 0},
        }
        mock_client.retrieve.return_value = [point]

        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)
        assert store._try_set_feedback_score_if_current(
            "chunk-a",
            expected_score=1.0,
            expected_revision=0,
            feedback_score=3.0,
            feedback_revision=1,
        )
        mock_client.set_payload.assert_called_once()
        _, set_payload_args = mock_client.set_payload.call_args
        assert set_payload_args["collection_name"] == "test_col"
        assert set_payload_args["payload"]["feedback_score"] == 3.0
        assert set_payload_args["payload"]["feedback_revision"] == 1
        assert "feedback_update_id" in set_payload_args["payload"]
        points = _require_filter(set_payload_args["points"])
        must_conditions = _require_list(points.must)
        id_match = _require_has_id_condition(must_conditions[0])
        assert id_match.has_id == ["chunk-a"]

    def test_try_set_feedback_score_if_current_returns_false_when_cas_misses(
        self, store, mock_client
    ):
        point = MagicMock()
        point.payload = {
            "metadata": {"feedback_score": 2.0, "feedback_revision": 1},
        }
        mock_client.retrieve.return_value = [point]
        assert not store._try_set_feedback_score_if_current(
            "chunk-a",
            expected_score=1.0,
            expected_revision=0,
            feedback_score=2.0,
            feedback_revision=1,
        )
        mock_client.set_payload.assert_not_called()

    def test_try_set_feedback_score_if_current_wraps_set_payload_error(self, store, mock_client):
        point = MagicMock()
        point.payload = {
            "metadata": {"feedback_score": 1.0, "feedback_revision": 0},
        }
        mock_client.retrieve.return_value = [point]
        mock_client.set_payload.side_effect = RuntimeError("set failed")
        with pytest.raises(VectorStoreError, match="Qdrant set_payload failed"):
            store._try_set_feedback_score_if_current(
                "chunk-a",
                expected_score=1.0,
                expected_revision=0,
                feedback_score=3.0,
                feedback_revision=1,
            )

    def test_try_set_feedback_score_if_current_raises_when_chunk_deleted_before_write(
        self, store, mock_client
    ):
        mock_client.retrieve.return_value = []
        with pytest.raises(VectorStoreError, match="not found"):
            store._try_set_feedback_score_if_current(
                "chunk-a",
                expected_score=0.0,
                expected_revision=0,
                feedback_score=1.0,
                feedback_revision=1,
            )
        mock_client.set_payload.assert_not_called()

    def test_try_set_feedback_score_if_current_raises_when_chunk_deleted_after_write(
        self, store, mock_client
    ):
        point = MagicMock()
        point.payload = {
            "metadata": {"feedback_score": 1.0, "feedback_revision": 0},
        }
        mock_client.retrieve.side_effect = [[point], []]
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)
        with pytest.raises(VectorStoreError, match="not found"):
            store._try_set_feedback_score_if_current(
                "chunk-a",
                expected_score=1.0,
                expected_revision=0,
                feedback_score=3.0,
                feedback_revision=1,
            )

    def test_verify_feedback_write_retries_after_stale_read(self, store, mock_client):
        """Retry verify reads so a successful writing is not applied twice."""
        reads = {"count": 0}
        update_id = "test-update-id"

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            point = MagicMock()
            if reads["count"] == 1:
                point.payload = {"metadata": {"feedback_score": 4.0, "feedback_revision": 4}}
            else:
                point.payload = {
                    "metadata": {
                        "feedback_score": 5.0,
                        "feedback_revision": 5,
                        "feedback_update_id": update_id,
                    }
                }
            return [point]

        mock_client.retrieve.side_effect = retrieve_side_effect

        assert store._verify_feedback_write(
            "chunk-a",
            update_id=update_id,
            expected_score=4.0,
            expected_revision=4,
            feedback_score=5.0,
            feedback_revision=5,
        )
        assert reads["count"] >= 2

    def test_accumulate_feedback_score_raises_when_chunk_deleted_during_retry(
        self, store, mock_client
    ):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 2}}
        reads = {"count": 0}

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            if reads["count"] <= 2:
                return [point]
            return []

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)
        with pytest.raises(VectorStoreError, match="not found"):
            store.accumulate_feedback_score("chunk-a", 0.5)

    def test_feedback_cas_filter_matches_zero_or_missing(self, store):
        match_filter = store._feedback_cas_filter("chunk-a", 0.0, 0)
        must_conditions = _require_list(match_filter.must)
        score_match = _require_filter(must_conditions[1])
        assert score_match.should is not None
        revision_match = _require_filter(must_conditions[2])
        assert revision_match.should is not None

    def test_feedback_cas_filter_matches_nonzero_with_range(self, store):
        match_filter = store._feedback_cas_filter("chunk-a", 2.0, 3)
        must_conditions = _require_list(match_filter.must)
        score_match = _require_field_condition(must_conditions[1])
        assert score_match.range is not None
        revision_match = _require_field_condition(must_conditions[2])
        assert revision_match.range is not None

    def test_get_feedback_scores_batch(self, store, mock_client):
        point_a = MagicMock()
        point_a.id = "chunk-a"
        point_a.payload = {"metadata": {"feedback_score": 2.0}}
        point_b = MagicMock()
        point_b.id = "chunk-b"
        point_b.payload = {"metadata": {}}
        mock_client.retrieve.return_value = [point_a, point_b]
        scores = store.get_feedback_scores(["chunk-a", "chunk-b", "chunk-a"])
        assert scores == {"chunk-a": 2.0, "chunk-b": 0.0}
        mock_client.retrieve.assert_called_once_with(
            collection_name="test_col",
            ids=["chunk-a", "chunk-b"],
            with_payload=True,
            with_vectors=False,
        )

    def test_accumulate_feedback_score_adds_delta(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 2}}
        mock_client.retrieve.return_value = [point]

        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)
        result = store.accumulate_feedback_score("chunk-a", 0.5)
        assert result == 1.5
        mock_client.set_payload.assert_called_once()
        _, set_payload_args = mock_client.set_payload.call_args
        assert set_payload_args["payload"]["feedback_score"] == 1.5
        assert set_payload_args["payload"]["feedback_revision"] == 3
        assert "feedback_update_id" in set_payload_args["payload"]

    def test_accumulate_feedback_score_missing_chunk_raises(self, store, mock_client):
        mock_client.retrieve.return_value = []
        with pytest.raises(VectorStoreError, match="not found"):
            store.accumulate_feedback_score("missing", 1.0)

    def test_accumulate_feedback_score_retries_after_concurrent_write(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        reads = {"count": 0}

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            reads["count"] += 1
            if reads["count"] == 4:
                point.payload = {
                    "metadata": {"feedback_score": 1.0, "feedback_revision": 1},
                }
            return [point]

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(point)
        result = store.accumulate_feedback_score("chunk-a", 1.0)
        assert result == 2.0
        assert point.payload["metadata"][FEEDBACK_SCORE_KEY] == 2.0
        assert point.payload["metadata"][FEEDBACK_REVISION_KEY] == 2

    def test_accumulate_feedback_score_serializes_concurrent_updates(self, store, mock_client):
        current = {"score": 0.0, "revision": 0, "update_id": None}
        lock = threading.Lock()

        def retrieve_side_effect(**_kwargs: object) -> list[MagicMock]:
            point = MagicMock()
            with lock:
                metadata: dict[str, object] = {
                    "feedback_score": current["score"],
                    "feedback_revision": current["revision"],
                }
                if current["update_id"] is not None:
                    metadata["feedback_update_id"] = current["update_id"]
                point.payload = {"metadata": metadata}
            return [point]

        def set_payload_side_effect(**set_payload_kwargs: object) -> None:
            payload = set_payload_kwargs["payload"]
            assert isinstance(payload, dict)
            with lock:
                expected_score = current["score"] + 1.0
                expected_revision = current["revision"] + 1
                requested_score = float(payload["feedback_score"])
                requested_revision = int(payload["feedback_revision"])
                if abs(requested_score - expected_score) > FEEDBACK_SCORE_EPSILON:
                    return
                if requested_revision != expected_revision:
                    return
                current["score"] = requested_score
                current["revision"] = requested_revision
                current["update_id"] = payload.get("feedback_update_id")

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.set_payload.side_effect = set_payload_side_effect

        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(
                pool.map(lambda _: store.accumulate_feedback_score("chunk-a", 1.0), range(10))
            )

        assert current["score"] == 10.0
        assert current["revision"] == 10
        assert sorted(results) == list(range(1, 11))

    def test_accumulate_feedback_score_raises_after_retry_exhaustion(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        mock_client.retrieve.return_value = [point]
        mock_client.set_payload.side_effect = lambda **_kwargs: None

        with pytest.raises(VectorStoreError, match="Failed to accumulate feedback"):
            store.accumulate_feedback_score("chunk-a", 1.0)

        assert mock_client.set_payload.call_count == MAX_FEEDBACK_UPDATE_RETRIES


# ── embedding model validation ─────────────────────────────────────────────────


class TestEmbeddingModelValidation:
    @staticmethod
    def _point(model_name: str | None) -> MagicMock:
        point = MagicMock()
        point.payload = {"embedding_model_name": model_name} if model_name else {}
        return point

    @staticmethod
    def _legacy_collection(mock_client: MagicMock) -> None:
        mock_client.get_collection.return_value = MagicMock(config=MagicMock(metadata={}))

    def test_finds_tagged_model_beyond_first_scroll_page(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        self._legacy_collection(mock_client)
        mock_client.scroll.side_effect = [
            ([self._point(None)] * 5, "page-2"),
            ([self._point("openai:text-embedding-3-large")], None),
        ]
        store = QdrantVectorStore(
            collection="test_col",
            dense_dim=4,
            embedding_model_name="voyage:voyage-large-2",
        )
        store._client = mock_client
        store._collection_ready = True

        with pytest.raises(VectorStoreError, match="Embedding model mismatch"):
            store.validate_embedding_model()

        assert mock_client.scroll.call_count == 2

    def test_raises_when_collection_has_multiple_models(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        self._legacy_collection(mock_client)
        mock_client.scroll.return_value = (
            [
                self._point("openai:text-embedding-3-large"),
                self._point("voyage:voyage-large-2"),
            ],
            None,
        )
        store = QdrantVectorStore(
            collection="test_col",
            dense_dim=4,
            embedding_model_name="openai:text-embedding-3-large",
        )
        store._client = mock_client
        store._collection_ready = True

        with pytest.raises(VectorStoreError, match="multiple models"):
            store.validate_embedding_model()

    def test_passes_when_untagged_points_precede_tagged_match(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        self._legacy_collection(mock_client)
        mock_client.scroll.side_effect = [
            ([self._point(None)] * 3, "page-2"),
            ([self._point("openai:text-embedding-3-large")], None),
        ]
        store = QdrantVectorStore(
            collection="test_col",
            dense_dim=4,
            embedding_model_name="openai:text-embedding-3-large",
        )
        store._client = mock_client
        store._collection_ready = True

        store.validate_embedding_model()
        assert store._model_validated


class TestQdrantMisc:
    def test_drop_collection(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store._collection_ready = True
        store._model_validated = True
        store.drop_collection()
        mock_client.delete_collection.assert_called_once_with("test_col")
        assert not store._collection_ready
        assert not store._model_validated

    def test_drop_collection_skips_missing(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = False
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store.drop_collection()
        mock_client.delete_collection.assert_not_called()

    def test_drop_collection_wraps_errors(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.delete_collection.side_effect = RuntimeError("down")
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        with pytest.raises(VectorStoreError, match="drop collection failed"):
            store.drop_collection()

    def test_recreate_collection_noop_when_missing(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = False
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store._collection_ready = True
        store._model_validated = True
        store.recreate_collection()
        mock_client.delete_collection.assert_not_called()
        assert not store._collection_ready
        assert not store._model_validated

    def test_recreate_collection_drops_existing(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store._collection_ready = True
        store.recreate_collection()
        mock_client.delete_collection.assert_called_once_with("test_col")
        assert not store._collection_ready

    def test_recreate_collection_purges_on_drop_failure(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.delete_collection.side_effect = RuntimeError("locked")
        point = MagicMock(id="p1")
        mock_client.scroll.side_effect = [([point], None), ([], None)]
        mock_client.count.return_value = MagicMock(count=0)
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store.recreate_collection()
        mock_client.delete.assert_called_once()
        assert store._collection_ready

    def test_recreate_collection_raises_when_purge_leaves_points(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.delete_collection.side_effect = RuntimeError("locked")
        mock_client.scroll.return_value = ([], None)
        mock_client.count.return_value = MagicMock(count=3)
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        with pytest.raises(VectorStoreError, match="still has 3 point"):
            store.recreate_collection()

    def test_recreate_collection_purges_multiple_scroll_pages(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.delete_collection.side_effect = RuntimeError("locked")
        point_a = MagicMock(id="p1")
        point_b = MagicMock(id="p2")
        mock_client.scroll.side_effect = [
            ([point_a], "page-2"),
            ([point_b], None),
            ([], None),
        ]
        mock_client.count.return_value = MagicMock(count=0)
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store.recreate_collection()
        assert mock_client.delete.call_count == 2

    def test_recreate_collection_raises_on_exists_check_failure(self, mock_client: MagicMock):
        mock_client.collection_exists.side_effect = RuntimeError("down")
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        with pytest.raises(VectorStoreError, match="status check failed"):
            store.recreate_collection()

    def test_recreate_collection_raises_on_purge_failure(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.delete_collection.side_effect = RuntimeError("locked")
        mock_client.scroll.side_effect = RuntimeError("scroll down")
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        with pytest.raises(VectorStoreError, match="Could not clear Qdrant collection"):
            store.recreate_collection()

    def test_get_collection_embedding_model_when_missing(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = False
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        assert store.get_collection_embedding_model() is None

    def test_scan_models_scroll_failure_returns_empty(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = MagicMock(config=MagicMock(metadata={}))
        mock_client.scroll.side_effect = RuntimeError("down")
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        assert store._scan_payload_embedding_models() == set()

    def test_scan_models_collection_exists_failure(self, mock_client: MagicMock):
        mock_client.collection_exists.side_effect = RuntimeError("down")
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        assert store._scan_payload_embedding_models() == set()

    def test_validate_uses_collection_metadata_without_scrolling(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(
                metadata={"embedding_model_name": "openai:text-embedding-3-large@3072"}
            )
        )
        store = QdrantVectorStore(
            collection="test_col",
            dense_dim=4,
            embedding_model_name="openai:text-embedding-3-large@3072",
        )
        store._client = mock_client
        store._collection_ready = True

        store.validate_embedding_model()

        mock_client.scroll.assert_not_called()
        assert store._model_validated

    def test_search_dense_wraps_error(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.query_points.side_effect = RuntimeError("search failed")
        with pytest.raises(VectorStoreError, match="dense search"):
            store.search_dense([0.1] * 4, top_k=1)

    def test_search_sparse_wraps_error(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.query_points.side_effect = RuntimeError("search failed")
        with pytest.raises(VectorStoreError, match="sparse search"):
            store.search_sparse({1: 0.9}, top_k=1)

    def test_delete_wraps_error(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.delete.side_effect = RuntimeError("delete failed")
        with pytest.raises(VectorStoreError, match="delete"):
            store.delete(["id-1"])

    def test_upsert_adds_embedding_model_to_payload(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        store.embedding_model_name = "bge_m3:models/bge-m3"
        chunk = _chunk(0)
        _wire_new_chunk_retrieve_and_upsert(mock_client, [chunk])
        store.upsert([chunk])
        _, kwargs = mock_client.upsert.call_args
        payload = kwargs["points"][0].payload
        assert payload["embedding_model_name"] == "bge_m3:models/bge-m3"


class TestDeleteByDocumentId:
    def test_deletes_matching_points(self, store: QdrantVectorStore, mock_client: MagicMock):
        p1 = MagicMock(id="c1")
        mock_client.scroll.return_value = ([p1], None)
        deleted = store.delete_by_document_id("doc-1")
        assert deleted == ["c1"]
        mock_client.delete.assert_called_once()

    def test_paginates_until_empty(self, store: QdrantVectorStore, mock_client: MagicMock):
        p1, p2 = MagicMock(id="c1"), MagicMock(id="c2")
        mock_client.scroll.side_effect = [
            ([p1], "page-2"),
            ([p2], None),
        ]
        deleted = store.delete_by_document_id("doc-1")
        assert deleted == ["c1", "c2"]
        assert mock_client.delete.call_count == 2

    def test_stops_when_no_points(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.scroll.return_value = ([], None)
        assert store.delete_by_document_id("doc-1") == []


class TestCollectionEmbeddingModelHelpers:
    def test_get_model_from_collection_metadata(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = MagicMock(
            config=MagicMock(metadata={"embedding_model_name": "openai:text-embedding-3-large"})
        )
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        assert store.get_collection_embedding_model() == "openai:text-embedding-3-large"
        mock_client.scroll.assert_not_called()

    def test_get_model_from_payload_when_no_metadata(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        mock_client.get_collection.return_value = MagicMock(config=MagicMock(metadata={}))
        point = MagicMock()
        point.payload = {"embedding_model_name": "bge_m3:local"}
        mock_client.scroll.return_value = ([point], None)
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        assert store.get_collection_embedding_model() == "bge_m3:local"

    def test_read_metadata_returns_none_on_probe_error(self, mock_client: MagicMock):
        mock_client.collection_exists.side_effect = RuntimeError("down")
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        assert store._read_collection_metadata_model() is None

    def test_write_metadata_skips_empty_name(self, mock_client: MagicMock):
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store._write_collection_metadata_model("")
        mock_client.update_collection.assert_not_called()

    def test_write_metadata_logs_on_failure(self, mock_client: MagicMock, caplog):
        import logging

        mock_client.update_collection.side_effect = RuntimeError("fail")
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        with caplog.at_level(logging.DEBUG):
            store._write_collection_metadata_model("bge_m3:local")
        assert "Cannot update collection" in caplog.text

    def test_scan_models_paginates_until_offset_none(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = True
        p1 = MagicMock()
        p1.payload = {"embedding_model_name": "model-a"}
        p2 = MagicMock()
        p2.payload = {"embedding_model_name": "model-a"}
        mock_client.scroll.side_effect = [
            ([p1], "page-2"),
            ([p2], None),
        ]
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        assert store._scan_payload_embedding_models() == {"model-a"}
