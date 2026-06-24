"""Redis-backed caching decorator for any EmbeddingRepository.

Caches dense vectors only — sparse vectors are either zero-cost (BM25)
or model-native and not worth the Redis round-trip overhead.

Cache key: SHA-256 (text | provider:model_name)  — includes both provider and
           specific model so that switching models within the same provider
           (e.g. text-embedding-3-large → text-embedding-3-small) uses
           distinct keys and never returns stale vectors.
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
            self._execute_pipeline(pipeline)

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
        """Cache-aware combined embed.

        For cache misses, calls inner.embed_both() so providers like BGE-M3 can
        compute dense + sparse in a single forward pass.  For cache hits (dense
        already stored), calls inner.embed_sparse() only for those texts.
        """
        if not texts:
            return [], []

        client = self._get_client()
        if client is None:
            return self._inner.embed_both(texts)

        keys = [self._make_key(t) for t in texts]
        cached_raw = self._mget(client, keys)

        dense_out: dict[int, DenseVector] = {}
        sparse_out: dict[int, SparseVector] = {}
        hit_indices: list[int] = []
        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, raw in enumerate(cached_raw):
            if raw is not None:
                dense_out[i] = json.loads(raw)
                hit_indices.append(i)
            else:
                miss_indices.append(i)
                miss_texts.append(texts[i])

        # Misses: single inner.embed_both() — one forward pass for BGE-M3.
        if miss_texts:
            miss_dense, miss_sparse = self._inner.embed_both(miss_texts)
            pipeline: Pipeline = client.pipeline()
            for list_pos, orig_idx in enumerate(miss_indices):
                vec = miss_dense[list_pos]
                dense_out[orig_idx] = vec
                sparse_out[orig_idx] = miss_sparse[list_pos]
                pipeline.set(self._make_key(miss_texts[list_pos]), json.dumps(vec), ex=self._ttl)
            self._execute_pipeline(pipeline)

        # Hits: dense served from cache; sparse still requires an inner call.
        if hit_indices:
            hit_sparse = self._inner.embed_sparse([texts[i] for i in hit_indices])
            for orig_idx, sp in zip(hit_indices, hit_sparse, strict=True):
                sparse_out[orig_idx] = sp

        n_hits = len(hit_indices)
        n_misses = len(miss_indices)
        logger.debug(
            "embed_both cache: %d hits, %d misses (model=%s)", n_hits, n_misses, self._model_id
        )
        _record_cache_metrics(hits=n_hits, misses=n_misses)
        return (
            [dense_out[i] for i in range(len(texts))],
            [sparse_out[i] for i in range(len(texts))],
        )

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
        # Import separately so the exception names are always bound when the
        # connection-error except clause runs.
        try:
            from redis import Redis as _Redis
            from redis.exceptions import ConnectionError as _ConnError
            from redis.exceptions import RedisError as _RedisError
        except ImportError:
            logger.error(
                "redis package not installed; embedding cache disabled. "
                "Run: uv sync --extra api-embeddings"
            )
            self._next_retry_at = float("inf")  # don't retry — package won't appear
            return None
        try:
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
        except (_ConnError, _RedisError, OSError) as exc:  # type: ignore[misc]
            self._next_retry_at = time.monotonic() + _RECONNECT_COOLDOWN
            logger.warning("Redis unavailable (%s); will retry in %.0fs", exc, _RECONNECT_COOLDOWN)
            return None

    def _execute_pipeline(self, pipeline: Pipeline) -> None:
        """Execute a Redis pipeline, resetting the client on RedisError."""
        try:
            pipeline.execute()
        except Exception as exc:  # noqa: BLE001
            from redis.exceptions import RedisError as _RedisError  # type: ignore[import-untyped]

            if isinstance(exc, _RedisError):
                logger.warning(
                    "Redis pipeline write failed (%s); cache write skipped, will retry in %.0fs",
                    exc,
                    _RECONNECT_COOLDOWN,
                )
                self._redis = None
                self._next_retry_at = time.monotonic() + _RECONNECT_COOLDOWN
            else:
                raise

    def _mget(self, client: Redis, keys: list[str]) -> list[str | None]:
        from redis.exceptions import RedisError as _RedisError  # type: ignore[import-untyped]

        try:
            return client.mget(keys)  # type: ignore[return-value]
        except _RedisError as exc:  # type: ignore[misc]
            logger.warning(
                "Redis mget failed (%s); bypassing cache for this batch, will retry in %.0fs",
                exc,
                _RECONNECT_COOLDOWN,
            )
            self._redis = None
            self._next_retry_at = time.monotonic() + _RECONNECT_COOLDOWN
            return [None] * len(keys)


# ── Prometheus metrics (optional, best-effort) ─────────────────────────────────


def _record_cache_metrics(hits: int, misses: int) -> None:
    try:
        from src.observability.metrics import EMBEDDING_CACHE_HITS, EMBEDDING_CACHE_MISSES

        EMBEDDING_CACHE_HITS.inc(hits)
        EMBEDDING_CACHE_MISSES.inc(misses)
    except (ImportError, AttributeError):
        pass
