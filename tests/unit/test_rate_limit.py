"""T-160 / T-146 — API rate limiting middleware tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from httpx import ASGITransport, AsyncClient

from src.api.rate_limit import (
    EXEMPT_PATHS,
    PROTECTED_PREFIXES,
    InMemoryRateLimiter,
    RateLimitHTTPMiddleware,
    RedisRateLimiter,
    client_key,
    configure_rate_limit,
    is_protected_path,
    should_rate_limit_request,
    try_build_redis_rate_limiter,
)


def _app_with_middleware(
    *,
    enabled: bool = True,
    rpm: int = 2,
    burst: int = 0,
    limiter: InMemoryRateLimiter | None = None,
    cors_origins: list[str] | None = None,
) -> FastAPI:
    configure_rate_limit(enabled=enabled, requests_per_minute=rpm, burst=burst, limiter=limiter)
    app = FastAPI()
    app.add_middleware(RateLimitHTTPMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins or ["http://localhost:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/metrics")
    async def metrics() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/feedback")
    async def feedback() -> dict[str, str]:
        return {"status": "ok"}

    @app.options("/feedback")
    async def feedback_options() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/chat")
    async def chat() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/chat/agent")
    async def chat_agent() -> dict[str, str]:
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
        assert is_protected_path("/chat/agent") is True
        assert is_protected_path("/chat/agent/full") is True
        assert is_protected_path("/ingest") is True
        assert is_protected_path("/evals/run") is True
        assert is_protected_path("/health") is False
        assert is_protected_path("/metrics") is False
        assert is_protected_path("/public") is False

    def test_constants_include_feedback(self):
        assert "/feedback" in PROTECTED_PREFIXES
        assert "/health" in EXEMPT_PATHS

    def test_should_rate_limit_request_skips_options(self):
        request = MagicMock()
        request.method = "OPTIONS"
        request.url.path = "/feedback"
        assert should_rate_limit_request(request) is False

    def test_should_rate_limit_request_protects_post(self):
        request = MagicMock()
        request.method = "POST"
        request.url.path = "/feedback"
        assert should_rate_limit_request(request) is True

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

    def test_client_key_uses_direct_ip(self):
        request = MagicMock()
        request.headers = {}
        request.client = MagicMock(host="192.168.1.50")
        assert client_key(request) == "ip:192.168.1.50"

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

    def test_burst_allows_extra_requests(self):
        limiter = InMemoryRateLimiter(requests_per_minute=1, burst=2)
        assert limiter.allow("client-a")[0] is True
        assert limiter.allow("client-a")[0] is True
        assert limiter.allow("client-a")[0] is True
        assert limiter.allow("client-a")[0] is False

    def test_separate_keys_have_independent_limits(self):
        limiter = InMemoryRateLimiter(requests_per_minute=1, burst=0)
        assert limiter.allow("client-a")[0] is True
        assert limiter.allow("client-b")[0] is True
        assert limiter.allow("client-a")[0] is False
        assert limiter.allow("client-b")[0] is False

    def test_expires_old_events_from_window(self, monkeypatch):
        limiter = InMemoryRateLimiter(requests_per_minute=1, burst=0)
        times = iter([0.0, 70.0, 70.0])
        monkeypatch.setattr("src.api.rate_limit.time.monotonic", lambda: next(times))
        assert limiter.allow("client-a")[0] is True
        assert limiter.allow("client-a")[0] is True


class TestRedisRateLimiter:
    def test_allow_returns_script_result(self):
        mock_script = MagicMock(return_value=[1, 0])
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script
        limiter = RedisRateLimiter(mock_client, requests_per_minute=2, burst=0)
        assert limiter.allow("client-a") == (True, 0)
        mock_script.assert_called_once()

    def test_allow_blocks_when_script_denies(self):
        mock_script = MagicMock(return_value=[0, 42])
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script
        limiter = RedisRateLimiter(mock_client, requests_per_minute=1, burst=0)
        assert limiter.allow("client-a") == (False, 42)

    def test_allow_fails_open_on_redis_error(self, caplog):
        import logging

        mock_script = MagicMock(side_effect=OSError("redis down"))
        mock_client = MagicMock()
        mock_client.register_script.return_value = mock_script
        limiter = RedisRateLimiter(mock_client, requests_per_minute=1, burst=0)
        with caplog.at_level(logging.WARNING, logger="src.api.rate_limit"):
            assert limiter.allow("client-a") == (True, 0)
        assert "Redis rate limit check failed" in caplog.text


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

    async def test_options_preflight_not_limited(self):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/feedback")
            for _ in range(5):
                resp = await client.options(
                    "/feedback",
                    headers={
                        "Origin": "http://localhost:3000",
                        "Access-Control-Request-Method": "POST",
                    },
                )
                assert resp.status_code != 429

    async def test_feedback_returns_429_with_retry_after(self):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (await client.post("/feedback")).status_code == 200
            resp = await client.post("/feedback")
            assert resp.status_code == 429
            assert resp.headers.get("Retry-After") is not None
            assert resp.json() == {"detail": "Rate limit exceeded"}

    @pytest.mark.parametrize(
        "method,path",
        [
            ("post", "/feedback"),
            ("get", "/chat"),
            ("post", "/chat/agent"),
            ("get", "/ingest/path"),
            ("get", "/evals/run"),
        ],
    )
    async def test_protected_routes_return_429_when_exceeded(self, method: str, path: str):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            first = await client.request(method, path)
            assert first.status_code == 200
            second = await client.request(method, path)
            assert second.status_code == 429

    async def test_public_route_not_rate_limited(self):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/feedback")
            for _ in range(5):
                assert (await client.get("/public")).status_code == 200

    async def test_per_api_key_isolation(self):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            assert (
                await client.post("/feedback", headers={"X-API-Key": "key-a"})
            ).status_code == 200
            assert (
                await client.post("/feedback", headers={"X-API-Key": "key-b"})
            ).status_code == 200
            assert (
                await client.post("/feedback", headers={"X-API-Key": "key-a"})
            ).status_code == 429
            assert (
                await client.post("/feedback", headers={"X-API-Key": "key-b"})
            ).status_code == 429

    async def test_429_includes_cors_headers_for_cross_origin(self):
        app = _app_with_middleware(enabled=True, rpm=1, burst=0)
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            await client.post("/feedback")
            resp = await client.post(
                "/feedback",
                headers={"Origin": "http://localhost:3000"},
            )
            assert resp.status_code == 429
            assert resp.headers.get("access-control-allow-origin") == "http://localhost:3000"

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

    def test_returns_redis_limiter_when_redis_available(self):
        mock_client = MagicMock()
        with patch(
            "src.api.rate_limit.build_redis_client",
            return_value=mock_client,
        ):
            limiter = try_build_redis_rate_limiter(60, 10)
        assert isinstance(limiter, RedisRateLimiter)
        mock_client.ping.assert_called_once()


class TestRateLimitConfiguration:
    def test_logs_when_enabled(self, caplog):
        import logging

        with caplog.at_level(logging.INFO, logger="src.api.rate_limit"):
            configure_rate_limit(enabled=True, requests_per_minute=30, burst=5)
        assert "API rate limiting enabled" in caplog.text

    def test_prefers_redis_limiter_when_available(self):
        mock_client = MagicMock()
        with patch(
            "src.api.rate_limit.build_redis_client",
            return_value=mock_client,
        ):
            config = configure_rate_limit(enabled=True, requests_per_minute=30, burst=5)
        assert isinstance(config.limiter, RedisRateLimiter)

    def test_falls_back_to_in_memory_when_redis_unavailable(self):
        with patch(
            "src.api.rate_limit.build_redis_client",
            side_effect=OSError("down"),
        ):
            config = configure_rate_limit(enabled=True, requests_per_minute=30, burst=5)
        assert isinstance(config.limiter, InMemoryRateLimiter)

    def test_uses_settings_defaults_when_no_overrides(self):
        with patch(
            "src.api.rate_limit.build_redis_client",
            side_effect=OSError("down"),
        ):
            config = configure_rate_limit()
        assert config.enabled is False
        assert isinstance(config.limiter, InMemoryRateLimiter)
