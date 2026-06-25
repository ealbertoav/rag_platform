"""Unit tests for CachedEmbeddingProvider.

All Redis interactions are mocked — no real Redis required.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch

import pytest
from redis.exceptions import ConnectionError as RedisConnError

from src.infrastructure.embeddings.cached_embedding_provider import (
    CachedEmbeddingProvider,
)

_TEXTS = ["hello", "world"]
_VECS = [[0.1, 0.2], [0.3, 0.4]]


def _make_inner() -> MagicMock:
    inner = MagicMock()
    inner.embed.return_value = _VECS
    inner.embed_sparse.return_value = [{}, {}]
    return inner


def _make_provider(inner: MagicMock | None = None) -> CachedEmbeddingProvider:
    return CachedEmbeddingProvider(
        inner=inner or _make_inner(),
        redis_url="redis://localhost:6379",
        model_identifier="test-model",
    )


# ── Cache hit / miss basics ────────────────────────────────────────────────────


class TestCacheHitMiss:
    def test_cache_miss_calls_inner_and_writes(self) -> None:
        inner = _make_inner()
        provider = _make_provider(inner)
        mock_redis = MagicMock()
        mock_redis.mget.return_value = [None, None]
        mock_pipeline = MagicMock()
        mock_redis.pipeline.return_value = mock_pipeline
        provider._redis = mock_redis

        result = provider.embed(_TEXTS)

        inner.embed.assert_called_once_with(_TEXTS)
        assert mock_pipeline.set.call_count == 2
        mock_pipeline.execute.assert_called_once()
        assert result == _VECS

    def test_cache_hit_skips_inner(self) -> None:
        import json

        inner = _make_inner()
        provider = _make_provider(inner)
        mock_redis = MagicMock()
        mock_redis.mget.return_value = [json.dumps(v) for v in _VECS]
        provider._redis = mock_redis

        result = provider.embed(_TEXTS)

        inner.embed.assert_not_called()
        assert result == _VECS

    def test_empty_texts_returns_empty(self) -> None:
        provider = _make_provider()
        assert provider.embed([]) == []


# ── Mid-session Redis failure recovery ────────────────────────────────────────


class TestRedisRecovery:
    def test_mget_failure_resets_client_and_schedules_retry(self) -> None:
        provider = _make_provider()
        mock_redis = MagicMock()
        mock_redis.mget.side_effect = RedisConnError("connection lost")
        provider._redis = mock_redis

        # embed() should still succeed (falls through to inner)
        result = provider.embed(_TEXTS)

        assert result == _VECS
        assert provider._redis is None
        assert provider._next_retry_at > time.monotonic()

    def test_pipeline_execute_failure_resets_client(self) -> None:
        provider = _make_provider()
        mock_redis = MagicMock()
        mock_redis.mget.return_value = [None, None]
        mock_pipeline = MagicMock()
        mock_pipeline.execute.side_effect = RedisConnError("connection lost")
        mock_redis.pipeline.return_value = mock_pipeline
        provider._redis = mock_redis

        # embed() should still succeed — vectors come from inner even if write fails
        result = provider.embed(_TEXTS)

        assert result == _VECS
        assert provider._redis is None
        assert provider._next_retry_at > time.monotonic()

    def test_client_retried_after_cooldown(self) -> None:
        provider = _make_provider()
        mock_redis = MagicMock()
        mock_redis.mget.side_effect = RedisConnError("connection lost")
        provider._redis = mock_redis

        # First call: mget fails, client reset
        provider.embed(_TEXTS)
        assert provider._redis is None

        # During cooldown: _get_client returns None without attempting reconnection
        with patch(
            "src.infrastructure.embeddings.cached_embedding_provider.time.monotonic",
            return_value=provider._next_retry_at - 1,
        ):
            client = provider._get_client()
        assert client is None

        # After cooldown: _get_client attempts reconnection
        provider._next_retry_at = 0.0
        new_redis = MagicMock()
        new_redis.mget.return_value = [None, None]
        new_pipeline = MagicMock()
        new_redis.pipeline.return_value = new_pipeline

        with patch(
            "src.infrastructure.embeddings.cached_embedding_provider.CachedEmbeddingProvider._get_client",
            return_value=new_redis,
        ):
            result = provider.embed(_TEXTS)

        assert result == _VECS

    def test_non_redis_error_in_pipeline_propagates(self) -> None:
        provider = _make_provider()
        mock_redis = MagicMock()
        mock_redis.mget.return_value = [None, None]
        mock_pipeline = MagicMock()
        mock_pipeline.execute.side_effect = ValueError("unexpected")
        mock_redis.pipeline.return_value = mock_pipeline
        provider._redis = mock_redis

        with pytest.raises(ValueError, match="unexpected"):
            provider.embed(_TEXTS)

        # Client should NOT be reset for non-Redis errors
        assert provider._redis is mock_redis


# ── embed_both() ───────────────────────────────────────────────────────────────


_SPARSE_VECS = [{1: 0.5, 2: 0.3}, {3: 0.8}]


def _make_inner_with_both() -> MagicMock:
    inner = MagicMock()
    inner.embed.return_value = _VECS
    inner.embed_both.return_value = (_VECS, _SPARSE_VECS)
    inner.embed_sparse.return_value = _SPARSE_VECS
    return inner


class TestEmbedBoth:
    def test_full_miss_calls_inner_embed_both_once(self) -> None:
        """On a full cache miss, inner.embed_both() is called — not embed() + embed_sparse()."""
        inner = _make_inner_with_both()
        provider = _make_provider(inner)
        mock_redis = MagicMock()
        mock_redis.mget.return_value = [None, None]
        mock_redis.pipeline.return_value = MagicMock()
        provider._redis = mock_redis

        dense, sparse = provider.embed_both(_TEXTS)

        inner.embed_both.assert_called_once_with(_TEXTS)
        inner.embed.assert_not_called()
        # embed_sparse only called for hits; no hits here
        inner.embed_sparse.assert_not_called()
        assert dense == _VECS
        assert sparse == _SPARSE_VECS

    def test_full_hit_calls_embed_sparse_only(self) -> None:
        """On a full cache hit, only embed_sparse() is called — no model forward pass for dense."""
        import json

        inner = _make_inner_with_both()
        provider = _make_provider(inner)
        mock_redis = MagicMock()
        mock_redis.mget.return_value = [json.dumps(v) for v in _VECS]
        provider._redis = mock_redis

        dense, sparse = provider.embed_both(_TEXTS)

        inner.embed_both.assert_not_called()
        inner.embed.assert_not_called()
        inner.embed_sparse.assert_called_once_with(_TEXTS)
        assert dense == _VECS
        assert sparse == _SPARSE_VECS

    def test_partial_hit_uses_embed_both_for_misses_and_embed_sparse_for_hits(self) -> None:
        """Partial hit: embed_both() for miss texts, embed_sparse() for hit texts."""
        import json

        texts = ["hit-text", "miss-text"]
        hit_vec = [0.1, 0.2]
        miss_vec = [0.3, 0.4]
        miss_sparse = {5: 0.9}
        hit_sparse = {7: 0.1}

        inner = MagicMock()
        inner.embed_both.return_value = ([miss_vec], [miss_sparse])
        inner.embed_sparse.return_value = [hit_sparse]

        provider = _make_provider(inner)
        mock_redis = MagicMock()
        mock_redis.mget.return_value = [json.dumps(hit_vec), None]
        mock_redis.pipeline.return_value = MagicMock()
        provider._redis = mock_redis

        dense, sparse = provider.embed_both(texts)

        inner.embed_both.assert_called_once_with(["miss-text"])
        inner.embed_sparse.assert_called_once_with(["hit-text"])
        assert dense == [hit_vec, miss_vec]
        assert sparse == [hit_sparse, miss_sparse]

    def test_redis_unavailable_delegates_to_inner_embed_both(self) -> None:
        inner = _make_inner_with_both()
        provider = _make_provider(inner)
        # No Redis client set — _get_client() returns None
        provider._redis = None
        provider._next_retry_at = float("inf")

        dense, sparse = provider.embed_both(_TEXTS)

        inner.embed_both.assert_called_once_with(_TEXTS)
        assert dense == _VECS
        assert sparse == _SPARSE_VECS

    def test_empty_texts_returns_empty(self) -> None:
        provider = _make_provider()
        assert provider.embed_both([]) == ([], [])


# ── embed_query / embed_sparse delegation ─────────────────────────────────────


class TestDelegation:
    def test_embed_query_delegates_to_inner(self) -> None:
        inner = _make_inner()
        inner.embed_query.return_value = _VECS
        provider = _make_provider(inner)
        result = provider.embed_query(_TEXTS)
        inner.embed_query.assert_called_once_with(_TEXTS)
        assert result == _VECS

    def test_embed_sparse_delegates_to_inner(self) -> None:
        inner = _make_inner()
        provider = _make_provider(inner)
        provider.embed_sparse(_TEXTS)
        inner.embed_sparse.assert_called_once_with(_TEXTS)


# ── Lazy Redis client connection ───────────────────────────────────────────────


class TestGetClient:
    def test_lazy_connect_success(self) -> None:
        provider = _make_provider()
        mock_client = MagicMock()
        mock_client.ping.return_value = True

        with patch(
            "redis.Redis.from_url",
            return_value=mock_client,
        ):
            client = provider._get_client()

        assert client is mock_client
        assert provider._redis is mock_client

    def test_import_error_disables_cache_permanently(self) -> None:
        provider = _make_provider()
        with patch.dict("sys.modules", {"redis": None}):
            client = provider._get_client()
        assert client is None
        assert provider._next_retry_at == float("inf")

    def test_connection_failure_sets_cooldown(self) -> None:
        provider = _make_provider()
        with patch("redis.Redis.from_url", side_effect=RedisConnError("down")):
            client = provider._get_client()
        assert client is None
        assert provider._next_retry_at > time.monotonic()


# ── Cache metrics ──────────────────────────────────────────────────────────────


class TestCacheMetrics:
    def test_records_prometheus_metrics_on_embed(self) -> None:
        provider = _make_provider()
        mock_redis = MagicMock()
        mock_redis.mget.return_value = [None, None]
        mock_redis.pipeline.return_value = MagicMock()
        provider._redis = mock_redis

        hits = MagicMock()
        misses = MagicMock()
        with (
            patch("src.observability.metrics.EMBEDDING_CACHE_HITS", hits),
            patch("src.observability.metrics.EMBEDDING_CACHE_MISSES", misses),
        ):
            provider.embed(_TEXTS)

        hits.inc.assert_called_once_with(0)
        misses.inc.assert_called_once_with(2)

    def test_record_metrics_import_error_is_noop(self) -> None:
        from src.infrastructure.embeddings import cached_embedding_provider as mod

        with patch.dict("sys.modules", {"src.observability.metrics": None}):
            mod._record_cache_metrics(hits=1, misses=2)
