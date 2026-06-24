"""Redis-backed caching decorator for any EmbeddingRepository.

Caches dense vectors only — sparse vectors are either zero-cost (BM25)
or model-native and not worth the Redis round-trip overhead.

Cache key: SHA-256 (text | model_identifier)
Value    : JSON-encoded list[float], stored as a Redis string with TTL.

Fail-open: if Redis is unavailable, the provider falls through to the
inner implementation without raising. Connection is retried at most once
per 60 seconds, so a temporarily unavailable Redis is picked back up
automatically after it recovers.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from typing import TYPE_CHECKING

from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

if TYPE_CHECKING:
    from redis import Redis
    from redis.client import Pipeline

logger = logging.getLogger(__name__)

_RECONNECT_COOLDOWN = 60.0  # seconds between Redis reconnect attempts


class CachedEmbeddingProvider(EmbeddingRepository):
    """Transparent caching layer around any EmbeddingRepository.

    Usage (handled automatically by the factory when cache.enabled=True):

        inner = BGEM3EmbeddingProvider.from_settings()
        cached = CachedEmbeddingProvider(inner, redis_url="redis://localhost:6379")
    """

    def __init__(
        self,
        inner: EmbeddingRepository,
        redis_url: str = "redis://localhost:6379",
        redis_password: str = "",
        ttl_seconds: int = 604800,
        model_identifier: str = "",
    ) -> None:
        self._inner = inner
        self._redis_url = redis_url
        self._redis_password = redis_password
        self._ttl = ttl_seconds
        self._model_id = model_identifier
        self._redis: Redis | None = None
        self._next_retry_at: float = 0.0  # monotonic timestamp; 0 = connect immediately

    # ── EmbeddingRepository interface ──────────────────────────────────────────

    def embed(self, texts: list[str]) -> list[DenseVector]:
        if not texts:
            return []
        client = self._get_client()
        if client is None:
            return self._inner.embed(texts)

        keys = [self._make_key(t) for t in texts]
        cached_raw = self._mget(client, keys)

        hits: dict[int, DenseVector] = {}
        misses: list[tuple[int, str]] = []
        for i, raw in enumerate(cached_raw):
            if raw is not None:
                hits[i] = json.loads(raw)
            else:
                misses.append((i, texts[i]))

        if misses:
            miss_vecs = self._inner.embed([t for _, t in misses])
            pipeline: Pipeline = client.pipeline()
            for (idx, text), vec in zip(misses, miss_vecs, strict=True):
                hits[idx] = vec
                pipeline.set(self._make_key(text), json.dumps(vec), ex=self._ttl)
            pipeline.execute()

        n_hits = len(texts) - len(misses)
        logger.debug(
            "embed cache: %d hits, %d misses (model=%s)", n_hits, len(misses), self._model_id
        )
        _record_cache_metrics(hits=n_hits, misses=len(misses))
        return [hits[i] for i in range(len(texts))]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        # Sparse vectors are not cached — delegate directly to inner provider.
        return self._inner.embed_sparse(texts)

    def embed_both(self, texts: list[str]) -> tuple[list[DenseVector], list[SparseVector]]:
        # Cache dense only; inner computes sparse.
        dense = self.embed(texts)
        sparse = self._inner.embed_sparse(texts)
        return dense, sparse

    # ── Internals ──────────────────────────────────────────────────────────────

    def _make_key(self, text: str) -> str:
        digest = hashlib.sha256(f"{text}|{self._model_id}".encode()).hexdigest()
        return f"emb:{digest}"

    def _get_client(self) -> Redis | None:
        if self._redis is not None:
            return self._redis
        # Respect the reconnection cooldown so a temporarily down Redis is
        # retried periodically rather than giving up for the process lifetime.
        if time.monotonic() < self._next_retry_at:
            return None
        try:
            from redis import Redis as _Redis
            from redis.exceptions import ConnectionError as _ConnError
            from redis.exceptions import RedisError as _RedisError

            client: Redis = _Redis.from_url(  # type: ignore[assignment]
                self._redis_url,
                password=self._redis_password or None,
                decode_responses=True,
            )
            client.ping()
            self._redis = client
            self._next_retry_at = 0.0
            logger.info("Embedding cache connected to Redis at %s", self._redis_url)
            return client
        except ImportError:
            logger.error(
                "redis package not installed; embedding cache disabled. "
                "Run: uv sync --extra api-embeddings"
            )
            self._next_retry_at = float("inf")  # don't retry — package won't appear
            return None
        except (_ConnError, _RedisError, OSError) as exc:  # type: ignore[misc]
            self._next_retry_at = time.monotonic() + _RECONNECT_COOLDOWN
            logger.warning("Redis unavailable (%s); will retry in %.0fs", exc, _RECONNECT_COOLDOWN)
            return None

    @staticmethod
    def _mget(client: Redis, keys: list[str]) -> list[str | None]:
        try:
            from redis.exceptions import RedisError as _RedisError

            return client.mget(keys)  # type: ignore[return-value]
        except _RedisError as exc:  # type: ignore[misc]
            logger.warning("Redis mget failed (%s); bypassing cache for this batch", exc)
            return [None] * len(keys)


# ── Prometheus metrics (optional, best-effort) ─────────────────────────────────


def _record_cache_metrics(hits: int, misses: int) -> None:
    try:
        from src.observability.metrics import EMBEDDING_CACHE_HITS, EMBEDDING_CACHE_MISSES

        EMBEDDING_CACHE_HITS.inc(hits)
        EMBEDDING_CACHE_MISSES.inc(misses)
    except (ImportError, AttributeError):
        pass
