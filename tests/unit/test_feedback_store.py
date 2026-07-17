"""T-146 — pluggable feedback store backends and vector store delegation."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.core.exceptions import VectorStoreError
from src.infrastructure.vectordb.feedback_store import (
    FeedbackDelegatingVectorStore,
    FeedbackStore,
    QdrantFeedbackStore,
    RedisFeedbackStore,
    SqlFeedbackStore,
    build_vector_store_from_settings,
    create_feedback_store,
    wrap_vector_store_with_feedback,
)


class TestQdrantFeedbackStore:
    def test_delegates_to_vector_store(self):
        store = MagicMock()
        store.accumulate_feedback_score.return_value = 2.0
        store.get_feedback_score.return_value = 1.5
        store.get_feedback_scores.return_value = {"c0": 1.5}

        feedback = QdrantFeedbackStore(store)
        assert feedback.accumulate("c0", 1.0) == 2.0
        assert feedback.get_score("c0") == 1.5
        assert feedback.get_scores(["c0"]) == {"c0": 1.5}
        feedback.set_score("c0", 3.0)
        store.set_feedback_score.assert_called_once_with("c0", 3.0)


class TestRedisFeedbackStore:
    def test_accumulate_uses_hincrbyfloat(self):
        redis_client = MagicMock()
        redis_client.hincrbyfloat.return_value = 2.5
        store = RedisFeedbackStore(redis_client)
        assert store.accumulate("chunk-a", 1.0) == 2.5
        redis_client.hincrbyfloat.assert_called_once_with("rag:feedback:scores", "chunk-a", 1.0)

    def test_accumulate_wraps_errors(self):
        redis_client = MagicMock()
        redis_client.hincrbyfloat.side_effect = OSError("down")
        store = RedisFeedbackStore(redis_client)
        with pytest.raises(VectorStoreError, match="Redis feedback accumulate"):
            store.accumulate("chunk-a", 1.0)

    def test_get_score_missing_returns_zero(self):
        redis_client = MagicMock()
        redis_client.hget.return_value = None
        store = RedisFeedbackStore(redis_client)
        assert store.get_score("chunk-a") == 0.0

    def test_get_score_decodes_bytes(self):
        redis_client = MagicMock()
        redis_client.hget.return_value = b"1.25"
        store = RedisFeedbackStore(redis_client)
        assert store.get_score("chunk-a") == 1.25

    def test_get_score_invalid_value_returns_zero(self):
        redis_client = MagicMock()
        redis_client.hget.return_value = "not-a-number"
        store = RedisFeedbackStore(redis_client)
        assert store.get_score("chunk-a") == 0.0

    def test_get_score_wraps_errors(self):
        redis_client = MagicMock()
        redis_client.hget.side_effect = OSError("down")
        store = RedisFeedbackStore(redis_client)
        with pytest.raises(VectorStoreError, match="Redis feedback lookup"):
            store.get_score("chunk-a")

    def test_get_scores_batch(self):
        redis_client = MagicMock()
        redis_client.hmget.return_value = ["2.0", None, b"1.5", "bad"]
        store = RedisFeedbackStore(redis_client)
        assert store.get_scores(["a", "b", "c", "d"]) == {
            "a": 2.0,
            "b": 0.0,
            "c": 1.5,
            "d": 0.0,
        }

    def test_get_scores_empty(self):
        store = RedisFeedbackStore(MagicMock())
        assert store.get_scores([]) == {}

    def test_get_scores_wraps_errors(self):
        redis_client = MagicMock()
        redis_client.hmget.side_effect = OSError("down")
        store = RedisFeedbackStore(redis_client)
        with pytest.raises(VectorStoreError, match="batch lookup"):
            store.get_scores(["a"])

    def test_set_score(self):
        redis_client = MagicMock()
        store = RedisFeedbackStore(redis_client)
        store.set_score("chunk-a", 4.0)
        redis_client.hset.assert_called_once_with("rag:feedback:scores", "chunk-a", 4.0)

    def test_set_score_wraps_errors(self):
        redis_client = MagicMock()
        redis_client.hset.side_effect = OSError("down")
        store = RedisFeedbackStore(redis_client)
        with pytest.raises(VectorStoreError, match="Redis feedback set"):
            store.set_score("chunk-a", 1.0)


class TestSqlFeedbackStore:
    def test_accumulate_and_get(self, tmp_path: Path):
        db_path = tmp_path / "feedback.db"
        store = SqlFeedbackStore(db_path)
        assert store.accumulate("chunk-a", 1.0) == 1.0
        assert store.accumulate("chunk-a", 0.5) == 1.5
        assert store.get_score("chunk-a") == 1.5
        assert store.get_score("missing") == 0.0

    def test_accumulate_raises_when_row_missing(self, tmp_path: Path, monkeypatch):
        store = SqlFeedbackStore(tmp_path / "feedback.db")

        class FakeCursor:
            def execute(self, *_args, **_kwargs):
                return self

            @staticmethod
            def fetchone():
                return None

        class FakeConn:
            @staticmethod
            def execute(*_args, **_kwargs):
                return FakeCursor()

            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return None

        monkeypatch.setattr(store, "_connect", lambda: FakeConn())
        with pytest.raises(VectorStoreError, match="Feedback accumulate failed"):
            store.accumulate("chunk-a", 1.0)

    def test_get_scores(self, tmp_path: Path):
        store = SqlFeedbackStore(tmp_path / "feedback.db")
        store.set_score("a", 2.0)
        assert store.get_scores(["a", "b", "a"]) == {"a": 2.0, "b": 0.0}

    def test_get_scores_empty(self, tmp_path: Path):
        store = SqlFeedbackStore(tmp_path / "feedback.db")
        assert store.get_scores([]) == {}

    def test_connect_rolls_back_on_sql_error(self, tmp_path: Path):
        store = SqlFeedbackStore(tmp_path / "feedback.db")
        with pytest.raises(sqlite3.OperationalError), store._connect() as conn:
            conn.execute("NOT VALID SQL")


class TestFeedbackDelegatingVectorStore:
    def test_routes_feedback_ops(self):
        inner = MagicMock()
        inner.chunk_exists.return_value = True
        feedback = MagicMock()
        feedback.accumulate.return_value = 3.0
        feedback.get_scores.return_value = {"c0": 3.0}
        feedback.get_score.return_value = 3.0

        wrapped = FeedbackDelegatingVectorStore(inner, feedback)
        assert wrapped.accumulate_feedback_score("c0", 1.0) == 3.0
        assert wrapped.get_feedback_scores(["c0"]) == {"c0": 3.0}
        assert wrapped.get_feedback_score("c0") == 3.0
        wrapped.set_feedback_score("c0", 2.0)
        feedback.set_score.assert_called_once_with("c0", 2.0)
        inner.chunk_exists.assert_called_with("c0")
        wrapped.count()
        inner.count.assert_called_once()

    def test_accumulate_rejects_missing_chunk(self):
        inner = MagicMock()
        inner.chunk_exists.return_value = False
        inner.collection = "test-collection"
        feedback = MagicMock()
        wrapped = FeedbackDelegatingVectorStore(inner, feedback)
        with pytest.raises(VectorStoreError, match="not found"):
            wrapped.accumulate_feedback_score("missing", 1.0)
        feedback.accumulate.assert_not_called()

    def test_set_feedback_rejects_missing_chunk(self):
        inner = MagicMock()
        inner.chunk_exists.return_value = False
        inner.collection = "test-collection"
        feedback = MagicMock()
        wrapped = FeedbackDelegatingVectorStore(inner, feedback)
        with pytest.raises(VectorStoreError, match="not found"):
            wrapped.set_feedback_score("missing", 1.0)
        feedback.set_score.assert_not_called()

    def test_chunk_exists_delegates_to_inner(self):
        inner = MagicMock()
        inner.chunk_exists.return_value = True
        wrapped = FeedbackDelegatingVectorStore(inner, MagicMock())
        assert wrapped.chunk_exists("chunk-a") is True
        inner.chunk_exists.assert_called_once_with("chunk-a")

    def test_get_chunk_delegates_to_inner(self):
        inner = MagicMock()
        inner.get_chunk.return_value = None
        wrapped = FeedbackDelegatingVectorStore(inner, MagicMock())
        assert wrapped.get_chunk("chunk-a") is None
        inner.get_chunk.assert_called_once_with("chunk-a")

    def test_delegates_vector_operations(self):
        inner = MagicMock()
        inner.chunk_exists.return_value = True
        feedback = MagicMock()
        wrapped = FeedbackDelegatingVectorStore(inner, feedback)
        chunks = [MagicMock()]
        wrapped.upsert(chunks)
        inner.upsert.assert_called_once_with(chunks)
        wrapped.delete(["c0"])
        inner.delete.assert_called_once_with(["c0"])
        wrapped.search_dense([0.1], 5)
        inner.search_dense.assert_called_once()
        wrapped.search_sparse({1: 0.5}, 3)
        inner.search_sparse.assert_called_once()
        wrapped.search_hybrid([0.1], {1: 0.5}, 0.7, 5)
        inner.search_hybrid.assert_called_once()


class TestFeedbackStoreFactory:
    def test_qdrant_backend_returns_qdrant_store(self):
        vector_store = MagicMock()
        store = create_feedback_store(
            "qdrant",
            vector_store,
            redis_url="redis://localhost",
            default_sqlite_path=Path("/tmp/feedback.db"),
        )
        assert isinstance(store, QdrantFeedbackStore)

    def test_redis_backend(self):
        vector_store = MagicMock()
        with patch(
            "src.infrastructure.vectordb.feedback_store._build_redis_client",
            return_value=MagicMock(),
        ):
            store = create_feedback_store(
                "redis",
                vector_store,
                redis_url="redis://localhost",
                default_sqlite_path=Path("/tmp/feedback.db"),
            )
        assert isinstance(store, RedisFeedbackStore)

    def test_postgres_backend_uses_default_sqlite_path(self, tmp_path: Path):
        vector_store = MagicMock()
        default_path = tmp_path / "default-feedback.db"
        store = create_feedback_store(
            "postgres",
            vector_store,
            redis_url="redis://localhost",
            postgres_url="",
            default_sqlite_path=default_path,
        )
        assert isinstance(store, SqlFeedbackStore)
        assert store._path == default_path

    def test_postgres_backend_uses_explicit_sqlite_path(self, tmp_path: Path):
        vector_store = MagicMock()
        db_path = tmp_path / "feedback.db"
        store = create_feedback_store(
            "postgres",
            vector_store,
            redis_url="redis://localhost",
            postgres_url=str(db_path),
            default_sqlite_path=Path("/tmp/unused.db"),
        )
        assert isinstance(store, SqlFeedbackStore)
        assert store._path == db_path

    def test_postgres_backend_rejects_postgresql_dsn(self):
        vector_store = MagicMock()
        with pytest.raises(ValueError, match="Postgres feedback backend"):
            create_feedback_store(
                "postgres",
                vector_store,
                redis_url="redis://localhost",
                postgres_url="postgresql://localhost/rag",
                default_sqlite_path=Path("/tmp/feedback.db"),
            )

    def test_invalid_backend_raises(self):
        with pytest.raises(ValueError, match="Unsupported feedback backend"):
            create_feedback_store(
                "invalid",  # type: ignore[arg-type]
                MagicMock(),
                redis_url="redis://localhost",
                default_sqlite_path=Path("/tmp/feedback.db"),
            )

    def test_wrap_qdrant_returns_inner(self):
        inner = MagicMock()
        wrapped = wrap_vector_store_with_feedback(
            inner,
            backend="qdrant",
            redis_url="redis://localhost",
            default_sqlite_path=Path("/tmp/feedback.db"),
        )
        assert wrapped is inner

    def test_wrap_redis_returns_delegating_store(self):
        inner = MagicMock()
        with patch(
            "src.infrastructure.vectordb.feedback_store._build_redis_client",
            return_value=MagicMock(),
        ):
            wrapped = wrap_vector_store_with_feedback(
                inner,
                backend="redis",
                redis_url="redis://localhost",
                default_sqlite_path=Path("/tmp/feedback.db"),
            )
        assert isinstance(wrapped, FeedbackDelegatingVectorStore)

    def test_build_redis_client(self):
        with patch("redis.from_url", return_value=MagicMock()) as from_url:
            from src.infrastructure.vectordb.feedback_store import _build_redis_client

            _build_redis_client("redis://localhost:6379", "secret")
        from_url.assert_called_once_with(
            "redis://localhost:6379",
            password="secret",
            decode_responses=True,
        )

    def test_build_vector_store_from_settings_qdrant_backend(self):
        inner = MagicMock()
        with (
            patch(
                "src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings",
                return_value=inner,
            ),
            patch("src.core.settings.settings") as mock_settings,
        ):
            mock_settings.quality.feedback_loop.backend = "qdrant"
            mock_settings.quality.feedback_loop.postgres_url = ""
            mock_settings.redis.url = "redis://localhost"
            mock_settings.redis.password.get_secret_value.return_value = ""
            result = build_vector_store_from_settings()
        assert result is inner

    def test_build_vector_store_from_settings_redis_backend(self):
        inner = MagicMock()
        with (
            patch(
                "src.infrastructure.vectordb.qdrant.QdrantVectorStore.from_settings",
                return_value=inner,
            ),
            patch("src.core.settings.settings") as mock_settings,
            patch(
                "src.infrastructure.vectordb.feedback_store._build_redis_client",
                return_value=MagicMock(),
            ),
        ):
            mock_settings.quality.feedback_loop.backend = "redis"
            mock_settings.quality.feedback_loop.postgres_url = ""
            mock_settings.redis.url = "redis://localhost"
            mock_settings.redis.password.get_secret_value.return_value = ""
            result = build_vector_store_from_settings()
        assert isinstance(result, FeedbackDelegatingVectorStore)

    def test_build_vector_store_from_settings_reuses_injected_store(self):
        inner = MagicMock()
        with patch("src.core.settings.settings") as mock_settings:
            mock_settings.quality.feedback_loop.backend = "qdrant"
            mock_settings.quality.feedback_loop.postgres_url = ""
            mock_settings.redis.url = "redis://localhost"
            mock_settings.redis.password.get_secret_value.return_value = ""
            result = build_vector_store_from_settings(vector_store=inner)
        assert result is inner


class TestFeedbackStoreBaseGetScores:
    def test_default_get_scores_uses_get_score(self):
        class StubStore(FeedbackStore):
            def __init__(self) -> None:
                self.scores = {"a": 1.0, "b": 2.0}

            def accumulate(self, chunk_id: str, delta: float) -> float:
                return 0.0

            def get_score(self, chunk_id: str) -> float:
                return self.scores.get(chunk_id, 0.0)

            def set_score(self, chunk_id: str, score: float) -> None:
                self.scores[chunk_id] = score

        store = StubStore()
        assert store.get_scores(["a", "b", "a"]) == {"a": 1.0, "b": 2.0}
