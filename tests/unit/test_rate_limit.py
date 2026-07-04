"""T-160 / T-146 — API rate limiting middleware tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient

from src.api.rate_limit import (
    EXEMPT_PATHS,
    PROTECTED_PREFIXES,
    InMemoryRateLimiter,
    client_key,
    configure_rate_limit,
    is_protected_path,
    rate_limit_middleware,
    try_build_redis_rate_limiter,
)


def _app_with_middleware(
    *,
    enabled: bool = True,
    rpm: int = 2,
    burst: int = 0,
    limiter: InMemoryRateLimiter | None = None,
) -> FastAPI:
    configure_rate_limit(enabled=enabled, requests_per_minute=rpm, burst=burst, limiter=limiter)
    app = FastAPI()
    app.middleware("http")(rate_limit_middleware)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/feedback")
    async def feedback() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/chat")
    async def chat() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/ingest/path")
    async def ingest_path() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/evals/run")
    async def evals_run() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/public")
    async def public() -> dict[str, str]:
        return {"status": "ok"}

    return app


class TestRateLimitHelpers:
    def test_protected_paths(self):
        assert is_protected_path("/feedback") is True
        assert is_protected_path("/feedback/extra") is True
        assert is_protected_path("/chat/stream") is True
        assert is_protected_path("/health") is False
        assert is_protected_path("/metrics") is False
        assert is_protected_path("/public") is False

    def test_constants_include_feedback(self):
        assert "/feedback" in PROTECTED_PREFIXES
        assert "/health" in EXEMPT_PATHS

    def test_client_key_prefers_api_key(self):
        request = MagicMock()
        request.headers = {"X-API-Key": "secret"}
        request.client = MagicMock(host="127.0.0.1")
        assert client_key(request) == "key:secret"

    def test_client_key_uses_forwarded_for(self):
        request = MagicMock()
        request.headers = {"X-Forwarded-For": "203.0.113.1, 10.0.0.1"}
        request.client = None
        assert client_key(request) == "ip:203.0.113.1"

    def test_client_key_unknown_when_no_client(self):
        request = MagicMock()
        request.headers = {}
        request.client = None
        assert client_key(request) == "ip:unknown"


class TestInMemoryRateLimiter:
    def test_allows_under_limit(self):
        limiter = InMemoryRateLimiter(requests_per_minute=2, burst=0)
        assert limiter.allow("client-a") == (True, 0)
        assert limiter.allow("client-a") == (True, 0)

    def test_blocks_over_limit(self):
        limiter = InMemoryRateLimiter(requests_per_minute=1, burst=0)
        assert limiter.allow("client-a")[0] is True
        allowed, retry_after = limiter.allow("client-a")
        assert allowed is False
        assert retry_after >= 1

    def test_expires_old_events_from_window(self, monkeypatch):
        limiter = InMemoryRateLimiter(requests_per_minute=1, burst=0)
        times = iter([0.0, 70.0, 70.0])
        monkeypatch.setattr("src.api.rate_limit.time.monotonic", lambda: next(times))
        assert limiter.allow("client-a")[0] is True
        assert limiter.allow("client-a")[0] is True


@pytest.mark.asyncio
class TestRateLimitMiddleware:
    async def test_disabled_passes_through(self):
        app = _app_with_middleware(enabled=False, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(5):
                resp = await client.post("/feedback")
                assert resp.status_code == 200

    async def test_exempt_paths_never_limited(self):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            for _ in range(5):
                assert (await client.get("/health")).status_code == 200
                assert (await client.get("/metrics")).status_code == 200

    async def test_feedback_returns_429_with_retry_after(self):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/feedback")).status_code == 200
            resp = await client.post("/feedback")
            assert resp.status_code == 429
            assert resp.headers.get("Retry-After") is not None

    async def test_records_metric_on_rejection(self):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        with patch("src.api.rate_limit.record_rate_limit_rejection") as record:
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://test",
            ) as client:
                await client.post("/feedback")
                await client.post("/feedback")
            record.assert_called_once_with("/feedback")


class TestRedisRateLimiterFallback:
    def test_returns_none_when_redis_unavailable(self):
        with patch(
            "src.api.rate_limit.build_redis_client",
            side_effect=OSError("down"),
        ):
            assert try_build_redis_rate_limiter(60, 10) is None

    def test_returns_limiter_when_redis_available(self):
        mock_client = MagicMock()
        with patch(
            "src.api.rate_limit.build_redis_client",
            return_value=mock_client,
        ):
            limiter = try_build_redis_rate_limiter(60, 10)
        assert isinstance(limiter, InMemoryRateLimiter)
        mock_client.ping.assert_called_once()


class TestRateLimitConfiguration:
    def test_logs_when_enabled(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="src.api.rate_limit"):
            configure_rate_limit(enabled=True, requests_per_minute=30, burst=5)
        assert "API rate limiting enabled" in caplog.text
