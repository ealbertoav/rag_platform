"""T-013 unit tests — QdrantVectorStore (Qdrant client mocked)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.vector_store_repository import VectorStoreRepository
from src.infrastructure.vectordb.qdrant import QdrantVectorStore

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

    def test_from_settings_returns_instance(self):
        with patch("src.infrastructure.vectordb.qdrant.QdrantClient"):
            instance = QdrantVectorStore.from_settings()
        assert isinstance(instance, QdrantVectorStore)


# ── _ensure_collection ─────────────────────────────────────────────────────────


class TestEnsureCollection:
    def test_creates_collection_when_missing(self, mock_client: MagicMock):
        mock_client.collection_exists.return_value = False
        s = QdrantVectorStore(collection="new_col", dense_dim=4)
        s._client = mock_client
        s._ensure_collection()
        mock_client.create_collection.assert_called_once()

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
        store.upsert(chunks)
        mock_client.upsert.assert_called_once()

    def test_upsert_passes_collection_name(self, store: QdrantVectorStore, mock_client: MagicMock):
        store.upsert([_chunk()])
        _, kwargs = mock_client.upsert.call_args
        assert kwargs["collection_name"] == "test_col"

    def test_upsert_point_count_matches_chunks(
        self, store: QdrantVectorStore, mock_client: MagicMock
    ):
        store.upsert([_chunk(0), _chunk(1), _chunk(2)])
        _, kwargs = mock_client.upsert.call_args
        assert len(kwargs["points"]) == 3

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


