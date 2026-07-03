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

    def test_get_feedback_score_missing_chunk_returns_zero(self, store, mock_client):
        mock_client.retrieve.return_value = []
        assert store.get_feedback_score("missing") == 0.0

    def test_set_feedback_score_updates_metadata(self, store, mock_client):
        point = MagicMock()
        point.payload = {"metadata": {"source": "doc.pdf"}}
        mock_client.retrieve.return_value = [point]
        store.set_feedback_score("chunk-a", 1.0)
        mock_client.set_payload.assert_called_once_with(
            collection_name="test_col",
            payload={"metadata": {"source": "doc.pdf", "feedback_score": 1.0}},
            points=["chunk-a"],
        )

    def test_set_feedback_score_missing_chunk_raises(self, store, mock_client):
        mock_client.retrieve.return_value = []
        with pytest.raises(VectorStoreError, match="not found"):
            store.set_feedback_score("missing", 1.0)

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
        assert store._model_validated is True


class TestQdrantMisc:
    def test_drop_collection(self, mock_client: MagicMock):
        store = QdrantVectorStore(collection="test_col", dense_dim=4)
        store._client = mock_client
        store._collection_ready = True
        store.drop_collection()
        mock_client.delete_collection.assert_called_once_with("test_col")
        assert store._collection_ready is False

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
        assert store._model_validated is True

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
