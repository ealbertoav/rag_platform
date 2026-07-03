"""T-013 unit tests — QdrantVectorStore (Qdrant client mocked)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from qdrant_client.models import FieldCondition, Filter, HasIdCondition

from src.core.constants import FEEDBACK_REVISION_KEY, FEEDBACK_SCORE_KEY
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.infrastructure.vectordb.qdrant import QdrantVectorStore

FEEDBACK_SCORE_EPSILON = 1e-9
MAX_FEEDBACK_UPDATE_RETRIES = 20


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


def _merge_metadata_set_payload_side_effect(point: MagicMock):
    """Return a mock set_payload handler that merges metadata updates into *point*."""

    def set_payload_side_effect(**set_payload_kwargs: object) -> None:
        payload = set_payload_kwargs["payload"]
        assert isinstance(payload, dict)
        metadata = dict(point.payload.get("metadata") or {})
        metadata.update(payload)
        point.payload = {"metadata": metadata}

    return set_payload_side_effect


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
        mock_client.retrieve.return_value = []
        store.upsert(chunks)
        mock_client.upsert.assert_called_once()

    def test_upsert_passes_collection_name(self, store: QdrantVectorStore, mock_client: MagicMock):
        mock_client.retrieve.return_value = []
        store.upsert([_chunk()])
        _, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == "test_col"

    def test_upsert_point_count_matches_chunks(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        mock_client.retrieve.return_value = []
        store.upsert([_chunk(0), _chunk(1), _chunk(2)])
        _, kwargs = mock_client.upsert.call_args
        assert len(kwargs["points"]) == 3

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
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(existing)
        store.upsert([chunk])
        mock_client.upsert.assert_not_called()
        mock_client.update_vectors.assert_called_once()
        metadata_calls = [
            call.kwargs
            for call in mock_client.set_payload.call_args_list
            if call.kwargs.get("key") == "metadata"
        ]
        assert len(metadata_calls) == 1
        assert metadata_calls[0]["payload"][FEEDBACK_SCORE_KEY] == 4.0
        assert metadata_calls[0]["payload"][FEEDBACK_REVISION_KEY] == 3

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
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(existing)
        store.upsert([chunk])
        metadata_calls = [
            call.kwargs
            for call in mock_client.set_payload.call_args_list
            if call.kwargs.get("key") == "metadata"
        ]
        assert metadata_calls[0]["payload"]["feedback_update_id"] == "update-123"

    def test_upsert_existing_chunk_uses_update_vectors(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        existing = MagicMock()
        existing.id = chunk.id
        existing.payload = {"metadata": {}}
        mock_client.retrieve.return_value = [existing]
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(existing)
        store.upsert([chunk])
        mock_client.upsert.assert_not_called()
        update_args = mock_client.update_vectors.call_args.kwargs
        assert update_args["collection_name"] == "test_col"
        assert len(update_args["points"]) == 1
        assert str(update_args["points"][0].id) == chunk.id

    def test_upsert_retries_metadata_when_feedback_changes_during_update(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 1}}
        mock_client.retrieve.return_value = [point]
        metadata_writes = {"count": 0}

        def set_payload_side_effect(**set_payload_kwargs: object) -> None:
            if set_payload_kwargs.get("key") != "metadata":
                return
            metadata_writes["count"] += 1
            if metadata_writes["count"] == 1:
                point.payload = {
                    "metadata": {"feedback_score": 2.0, "feedback_revision": 2},
                }
                return
            _merge_metadata_set_payload_side_effect(point)(**set_payload_kwargs)

        mock_client.set_payload.side_effect = set_payload_side_effect
        store.upsert([chunk])
        assert metadata_writes["count"] >= 2
        assert point.payload["metadata"][FEEDBACK_SCORE_KEY] == 2.0
        assert point.payload["metadata"][FEEDBACK_REVISION_KEY] == 2

    def test_upsert_mixed_new_and_existing_chunks(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        new_chunk = _chunk(0)
        existing_chunk = _chunk(1)
        existing = MagicMock()
        existing.id = existing_chunk.id
        existing.payload = {"metadata": {"feedback_score": 2.0, "feedback_revision": 1}}

        def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
            ids = kwargs.get("ids", [])
            assert isinstance(ids, list)
            if existing_chunk.id in ids:
                return [existing]
            return []

        mock_client.retrieve.side_effect = retrieve_side_effect
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(existing)
        store.upsert([new_chunk, existing_chunk])
        mock_client.upsert.assert_called_once()
        assert len(mock_client.upsert.call_args.kwargs["points"]) == 1
        mock_client.update_vectors.assert_called_once()
        assert len(mock_client.update_vectors.call_args.kwargs["points"]) == 1

    def test_upsert_adds_chunk_type_to_payload(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        from src.core.constants import CHUNK_TYPE_HYPE, CHUNK_TYPE_KEY

        chunk = _chunk(0).model_copy(update={"metadata": {CHUNK_TYPE_KEY: CHUNK_TYPE_HYPE}})
        mock_client.retrieve.return_value = []
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
        mock_client.upsert.side_effect = RuntimeError("oops")
        with pytest.raises(VectorStoreError) as exc_info:
            store.upsert([_chunk()])
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
        mock_client.set_payload.side_effect = _merge_metadata_set_payload_side_effect(existing)

        store.upsert([chunk])

        top_level_calls = [
            call.kwargs
            for call in mock_client.set_payload.call_args_list
            if call.kwargs.get("key") is None
        ]
        assert top_level_calls[0]["payload"]["embedding_model_name"] == "bge-m3"
        assert top_level_calls[0]["payload"][CHUNK_TYPE_KEY] == CHUNK_TYPE_HYPE

    def test_upsert_existing_chunk_raises_after_metadata_retry_exhaustion(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk = _chunk(0)
        point = MagicMock()
        point.id = chunk.id
        point.payload = {"metadata": {"feedback_score": 0.0, "feedback_revision": 0}}
        mock_client.retrieve.return_value = [point]
        metadata_writes = {"count": 0}

        def set_payload_side_effect(**set_payload_kwargs: object) -> None:
            if set_payload_kwargs.get("key") != "metadata":
                return
            metadata_writes["count"] += 1
            bump = metadata_writes["count"]
            point.payload = {
                "metadata": {
                    "feedback_score": float(bump + 100),
                    "feedback_revision": bump + 100,
                },
            }

        mock_client.set_payload.side_effect = set_payload_side_effect

        with pytest.raises(VectorStoreError, match="Failed to upsert metadata"):
            store.upsert([chunk])

        assert metadata_writes["count"] == MAX_FEEDBACK_UPDATE_RETRIES
        mock_client.update_vectors.assert_not_called()

    def test_upsert_existing_chunk_updates_per_chunk_in_order(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        chunk_a = _chunk(0)
        chunk_b = _chunk(1)
        existing_a = MagicMock()
        existing_a.id = chunk_a.id
        existing_a.payload = {"metadata": {"feedback_score": 1.0, "feedback_revision": 1}}
        existing_b = MagicMock()
        existing_b.id = chunk_b.id
        existing_b.payload = {"metadata": {"feedback_score": 2.0, "feedback_revision": 2}}
        points_by_id = {chunk_a.id: existing_a, chunk_b.id: existing_b}

        def retrieve_side_effect(**kwargs: object) -> list[MagicMock]:
            ids = kwargs.get("ids", [])
            assert isinstance(ids, list)
            return [
                points_by_id[str(chunk_id)] for chunk_id in ids if str(chunk_id) in points_by_id
            ]

        mock_client.retrieve.side_effect = retrieve_side_effect

        def set_payload_side_effect(**set_payload_kwargs: object) -> None:
            if set_payload_kwargs.get("key") != "metadata":
                return
            points = set_payload_kwargs["points"]
            point_filter = _require_filter(points)
            must_conditions = _require_list(point_filter.must)
            id_match = _require_has_id_condition(must_conditions[0])
            if id_match.has_id == [chunk_b.id]:
                raise RuntimeError("metadata write failed for chunk-b")
            point = points_by_id[chunk_a.id]
            _merge_metadata_set_payload_side_effect(point)(**set_payload_kwargs)

        mock_client.set_payload.side_effect = set_payload_side_effect

        with pytest.raises(VectorStoreError, match="Qdrant upsert failed"):
            store.upsert([chunk_a, chunk_b])

        assert mock_client.update_vectors.call_count == 1
        assert str(mock_client.update_vectors.call_args.kwargs["points"][0].id) == chunk_a.id


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
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store._collection_ready = True
        store.drop_collection()
        mock_client.delete_collection.assert_called_once_with("test_col")
        assert not store._collection_ready

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
        store.upsert([_chunk(0)])
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
