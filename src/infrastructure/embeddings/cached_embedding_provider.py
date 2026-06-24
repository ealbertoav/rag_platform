"""Redis-backed caching decorator for any EmbeddingRepository.

Caches dense vectors only — sparse vectors are either zero-cost (BM25)
or model-native and not worth the Redis round-trip overhead.

Cache key: SHA-256(text | model_identifier)
Value    : JSON-encoded list[float], stored as a Redis string with TTL.

Fail-open: if Redis is unavailable the provider falls through to the
inner implementation without raising.
"""

from __future__ import annotations

import hashlib
import json
import logging
from typing import TYPE_CHECKING

from redis import Redis
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import RedisError

from src.domain.repositories.embedding_repository import (
    DenseVector,
    EmbeddingRepository,
    SparseVector,
)

if TYPE_CHECKING:
    from redis.client import Pipeline

logger = logging.getLogger(__name__)


class CachedEmbeddingProvider(EmbeddingRepository):
    """Transparent caching layer around any EmbeddingRepository.

    Usage (handled automatically by the factory when cache.enabled=True):

        inner = BGEM3EmbeddingProvider.from_settings()
        cached = CachedEmbeddingProvider(inner, redis_client, ttl_seconds=604800)
    """

    def __init__(
        self,
        inner: EmbeddingRepository,
        redis_url: str = "redis://localhost:6379",
        ttl_seconds: int = 604800,
        model_identifier: str = "",
    ) -> None:
        self._inner = inner
        self._redis_url = redis_url
        self._ttl = ttl_seconds
        self._model_id = model_identifier
        self._redis: Redis | None = None  # lazy connection; stays None if Redis is down

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

        n_hits = len(hits) - len(misses)
        logger.debug(
            "embed cache: %d hits, %d misses (model=%s)", n_hits, len(misses), self._model_id
        )
        _record_cache_metrics(hits=n_hits, misses=len(misses))
        return [hits[i] for i in range(len(texts))]

    def embed_sparse(self, texts: list[str]) -> list[SparseVector]:
        # Sparse vectors are not cached — delegate directly to inner provider.
        return self._inner.embed_sparse(texts)

    def embed_both(self, texts: list[str]) -> tuple[list[DenseVector], list[SparseVector]]:
        # Cache dense only; sparse is computed by inner.
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
        try:
            client: Redis = Redis.from_url(self._redis_url, decode_responses=True)  # type: ignore[assignment]
            client.ping()
            self._redis = client
            logger.info("Embedding cache connected to Redis at %s", self._redis_url)
            return client
        except (RedisConnectionError, RedisError, OSError) as exc:
            logger.warning("Redis unavailable (%s); embedding cache disabled", exc)
            return None

    @staticmethod
    def _mget(client: Redis, keys: list[str]) -> list[str | None]:
        try:
            return client.mget(keys)  # type: ignore[return-value]
        except RedisError as exc:
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
