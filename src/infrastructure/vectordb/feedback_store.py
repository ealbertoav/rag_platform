from __future__ import annotations

import logging
import sqlite3
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Literal

from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.domain.repositories.embedding_repository import DenseVector, SparseVector
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository
from src.infrastructure.cache.redis_client import RedisHashClient, build_redis_client

logger = logging.getLogger(__name__)

FeedbackBackend = Literal["qdrant", "redis", "postgres"]

_REDIS_HASH_KEY = "rag:feedback:scores"
_SQL_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_scores (
    chunk_id TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0.0
);
"""


class FeedbackStore(ABC):
    """Atomic feedback score persistence for multi-replica deployments."""

    @abstractmethod
    def accumulate(self, chunk_id: str, delta: float) -> float:
        """Add *delta* to the stored score and return the new total."""

    @abstractmethod
    def get_score(self, chunk_id: str) -> float:
        """Return the accumulated score for *chunk_id* (0.0 when unset)."""

    def get_scores(self, chunk_ids: list[str]) -> dict[str, float]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        return {chunk_id: self.get_score(chunk_id) for chunk_id in unique_ids}

    @abstractmethod
    def set_score(self, chunk_id: str, score: float) -> None:
        """Overwrite the stored score for *chunk_id*."""


class QdrantFeedbackStore(FeedbackStore):
    """Default backend — delegates to Qdrant compare-and-set accumulation."""

    def __init__(self, vector_store: VectorStoreRepository) -> None:
        self._vector_store = vector_store

    def accumulate(self, chunk_id: str, delta: float) -> float:
        return self._vector_store.accumulate_feedback_score(chunk_id, delta)

    def get_score(self, chunk_id: str) -> float:
        return self._vector_store.get_feedback_score(chunk_id)

    def get_scores(self, chunk_ids: list[str]) -> dict[str, float]:
        return self._vector_store.get_feedback_scores(chunk_ids)

    def set_score(self, chunk_id: str, score: float) -> None:
        self._vector_store.set_feedback_score(chunk_id, score)


class RedisFeedbackStore(FeedbackStore):
    """Redis hash with HINCRBYFLOAT for atomic multi-pod increments."""

    def __init__(self, redis_client: RedisHashClient, *, hash_key: str = _REDIS_HASH_KEY) -> None:
        self._redis = redis_client
        self._hash_key = hash_key

    def accumulate(self, chunk_id: str, delta: float) -> float:
        try:
            raw = self._redis.hincrbyfloat(self._hash_key, chunk_id, delta)
        except Exception as exc:
            raise VectorStoreError(
                f"Redis feedback accumulate failed for {chunk_id!r}",
                cause=exc,
            ) from exc
        return float(raw)

    def get_score(self, chunk_id: str) -> float:
        try:
            raw = self._redis.hget(self._hash_key, chunk_id)
        except Exception as exc:
            raise VectorStoreError(
                f"Redis feedback lookup failed for {chunk_id!r}",
                cause=exc,
            ) from exc
        if raw is None:
            return 0.0
        if isinstance(raw, bytes):
            raw = raw.decode()
        try:
            return float(raw)
        except (TypeError, ValueError):
            return 0.0

    def get_scores(self, chunk_ids: list[str]) -> dict[str, float]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        try:
            values = self._redis.hmget(self._hash_key, unique_ids)
        except Exception as exc:
            raise VectorStoreError("Redis feedback batch lookup failed", cause=exc) from exc
        scores: dict[str, float] = {}
        for chunk_id, raw in zip(unique_ids, values, strict=True):
            if raw is None:
                scores[chunk_id] = 0.0
            elif isinstance(raw, bytes):
                scores[chunk_id] = float(raw.decode())
            else:
                try:
                    scores[chunk_id] = float(raw)
                except (TypeError, ValueError):
                    scores[chunk_id] = 0.0
        return scores

    def set_score(self, chunk_id: str, score: float) -> None:
        try:
            self._redis.hset(self._hash_key, chunk_id, score)
        except Exception as exc:
            raise VectorStoreError(
                f"Redis feedback set failed for {chunk_id!r}",
                cause=exc,
            ) from exc


class SqlFeedbackStore(FeedbackStore):
    """SQL-backed atomic increment (SQLite for local dev; Postgres DSN in production)."""

    def __init__(self, db_path: Path) -> None:
        self._path = db_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SQL_SCHEMA)

    def accumulate(self, chunk_id: str, delta: float) -> float:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback_scores (chunk_id, score) VALUES (?, ?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET score = score + excluded.score",
                (chunk_id, delta),
            )
            row = conn.execute(
                "SELECT score FROM feedback_scores WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        if row is None:
            raise VectorStoreError(f"Feedback accumulate failed for {chunk_id!r}")
        return float(row[0])

    def get_score(self, chunk_id: str) -> float:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT score FROM feedback_scores WHERE chunk_id = ?",
                (chunk_id,),
            ).fetchone()
        return float(row[0]) if row is not None else 0.0

    def get_scores(self, chunk_ids: list[str]) -> dict[str, float]:
        unique_ids = list(dict.fromkeys(chunk_ids))
        if not unique_ids:
            return {}
        placeholders = ",".join("?" for _ in unique_ids)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT chunk_id, score FROM feedback_scores WHERE chunk_id IN ({placeholders})",
                unique_ids,
            ).fetchall()
        found = {str(chunk_id): float(score) for chunk_id, score in rows}
        return {chunk_id: found.get(chunk_id, 0.0) for chunk_id in unique_ids}

    def set_score(self, chunk_id: str, score: float) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO feedback_scores (chunk_id, score) VALUES (?, ?) "
                "ON CONFLICT(chunk_id) DO UPDATE SET score = excluded.score",
                (chunk_id, score),
            )


class FeedbackDelegatingVectorStore(VectorStoreRepository):
    """Route feedback operations to a pluggable store; delegate all vector ops to *inner*."""

    def __init__(self, inner: VectorStoreRepository, feedback: FeedbackStore) -> None:
        self._inner = inner
        self._feedback = feedback

    def _require_chunk(self, chunk_id: str) -> None:
        if not self._inner.chunk_exists(chunk_id):
            collection = getattr(self._inner, "collection", "vector store")
            raise VectorStoreError(f"Chunk {chunk_id!r} not found in collection {collection!r}")

    def upsert(self, chunks: list[Chunk]) -> None:
        self._inner.upsert(chunks)

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
        return self._inner.search_dense(
            query_vector,
            top_k,
            type_equals=type_equals,
            exclude_types=exclude_types,
            document_ids=document_ids,
            filters=filters,
        )

    def search_sparse(self, query_sparse: SparseVector, top_k: int) -> list[SearchResult]:
        return self._inner.search_sparse(query_sparse, top_k)

    def search_hybrid(
        self,
        query_vector: DenseVector,
        query_sparse: SparseVector,
        alpha: float,
        top_k: int,
    ) -> list[SearchResult]:
        return self._inner.search_hybrid(query_vector, query_sparse, alpha, top_k)

    def delete(self, chunk_ids: list[str]) -> None:
        self._inner.delete(chunk_ids)

    def count(self) -> int:
        return self._inner.count()

    def chunk_exists(self, chunk_id: str) -> bool:
        return self._inner.chunk_exists(chunk_id)

    def get_feedback_score(self, chunk_id: str) -> float:
        return self._feedback.get_score(chunk_id)

    def set_feedback_score(self, chunk_id: str, feedback_score: float) -> None:
        self._require_chunk(chunk_id)
        self._feedback.set_score(chunk_id, feedback_score)

    def accumulate_feedback_score(self, chunk_id: str, delta: float) -> float:
        self._require_chunk(chunk_id)
        return self._feedback.accumulate(chunk_id, delta)

    def get_feedback_scores(self, chunk_ids: list[str]) -> dict[str, float]:
        return self._feedback.get_scores(chunk_ids)


def _build_redis_client(redis_url: str, password: str) -> RedisHashClient:
    return build_redis_client(redis_url, password)


def create_feedback_store(
    backend: FeedbackBackend,
    vector_store: VectorStoreRepository,
    *,
    redis_url: str,
    redis_password: str = "",
    postgres_url: str = "",
    default_sqlite_path: Path,
) -> FeedbackStore:
    if backend == "qdrant":
        return QdrantFeedbackStore(vector_store)
    if backend == "redis":
        client = _build_redis_client(redis_url, redis_password)
        return RedisFeedbackStore(client)
    if backend == "postgres":
        db_path = Path(postgres_url) if postgres_url else default_sqlite_path
        if postgres_url.startswith("postgresql://") or postgres_url.startswith("postgres://"):
            msg = (
                "Postgres feedback backend requires a file path or sqlite URL for now; "
                "use backend=redis for multi-replica atomic increments"
            )
            raise ValueError(msg)
        return SqlFeedbackStore(db_path)
    msg = f"Unsupported feedback backend: {backend!r}"
    raise ValueError(msg)


def wrap_vector_store_with_feedback(
    vector_store: VectorStoreRepository,
    *,
    backend: FeedbackBackend,
    redis_url: str,
    redis_password: str = "",
    postgres_url: str = "",
    default_sqlite_path: Path,
) -> VectorStoreRepository:
    """Return *vector_store* unchanged for Qdrant, else a delegating wrapper."""
    if backend == "qdrant":
        return vector_store
    feedback = create_feedback_store(
        backend,
        vector_store,
        redis_url=redis_url,
        redis_password=redis_password,
        postgres_url=postgres_url,
        default_sqlite_path=default_sqlite_path,
    )
    logger.info("Feedback backend=%r enabled for multi-replica atomic increments", backend)
    return FeedbackDelegatingVectorStore(vector_store, feedback)


def build_vector_store_from_settings(
    *,
    vector_store: VectorStoreRepository | None = None,
    default_sqlite_path: Path | None = None,
) -> VectorStoreRepository:
    """Build Qdrant plus an optional feedback wrapper from application settings."""
    from src.core.constants import ROOT
    from src.core.settings import settings
    from src.infrastructure.vectordb.qdrant import QdrantVectorStore

    base = vector_store or QdrantVectorStore.from_settings()
    feedback_cfg = settings.quality.feedback_loop
    sqlite_path = default_sqlite_path or (ROOT / "data" / "processed" / "feedback.db")
    return wrap_vector_store_with_feedback(
        base,
        backend=feedback_cfg.backend,
        redis_url=settings.redis.url,
        redis_password=settings.redis.password.get_secret_value(),
        postgres_url=feedback_cfg.postgres_url,
        default_sqlite_path=sqlite_path,
    )
