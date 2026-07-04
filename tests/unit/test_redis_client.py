"""Tests for shared Redis client helper."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.infrastructure.cache.redis_client import build_redis_client


def test_build_redis_client_uses_from_url() -> None:
    with patch("redis.from_url", return_value=MagicMock()) as from_url:
        client = build_redis_client("redis://localhost:6379", "secret")
    from_url.assert_called_once_with(
        "redis://localhost:6379",
        password="secret",
        decode_responses=True,
    )
    assert client is from_url.return_value
