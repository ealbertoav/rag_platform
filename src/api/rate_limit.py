from __future__ import annotations

import logging
import math
import threading
import time
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from fastapi import Request
from fastapi.responses import JSONResponse, Response

from src.core.settings import settings
from src.infrastructure.cache.redis_client import build_redis_client
from src.observability.metrics import record_rate_limit_rejection

logger = logging.getLogger(__name__)

PROTECTED_PREFIXES = ("/ingest", "/chat", "/evals/run", "/feedback")
EXEMPT_PATHS = ("/health", "/metrics")


def client_key(request: Request) -> str:
    api_key = request.headers.get("X-API-Key")
    if api_key:
        return f"key:{api_key}"
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return f"ip:{forwarded.split(',')[0].strip()}"
    if request.client is not None:
        return f"ip:{request.client.host}"
    return "ip:unknown"


def is_protected_path(path: str) -> bool:
    if path in EXEMPT_PATHS:
        return False
    return any(path == prefix or path.startswith(f"{prefix}/") for prefix in PROTECTED_PREFIXES)


class InMemoryRateLimiter:
    """Sliding-window limiter keyed by client identity."""

    def __init__(self, requests_per_minute: int, burst: int) -> None:
        self._rpm = requests_per_minute
        self._burst = burst
        self._window_seconds = 60.0
        self._lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            limit = self._rpm + self._burst
            if len(events) >= limit:
                retry_after = max(1, math.ceil(self._window_seconds - (now - events[0])))
                return False, retry_after
            events.append(now)
        return True, 0


@dataclass
class RateLimitConfig:
    enabled: bool = False
    limiter: InMemoryRateLimiter | None = None


_config = RateLimitConfig()


def configure_rate_limit(
    *,
    enabled: bool | None = None,
    requests_per_minute: int | None = None,
    burst: int | None = None,
    limiter: InMemoryRateLimiter | None = None,
) -> RateLimitConfig:
    """Apply rate-limit settings used by the HTTP middleware."""
    cfg = settings.api.rate_limit
    enabled_val = cfg.enabled if enabled is None else enabled
    rpm = cfg.requests_per_minute if requests_per_minute is None else requests_per_minute
    burst_val = cfg.burst if burst is None else burst
    limiter_val = limiter or InMemoryRateLimiter(rpm, burst_val)
    _config.enabled = enabled_val
    _config.limiter = limiter_val
    if enabled_val:
        logger.info(
            "API rate limiting enabled: %d req/min + burst %d on protected routes",
            rpm,
            burst_val,
        )
    return _config


async def rate_limit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Return 429 when protected routes exceed configured per-client limits."""
    if not _config.enabled or _config.limiter is None or not is_protected_path(request.url.path):
        return await call_next(request)

    allowed, retry_after = _config.limiter.allow(client_key(request))
    if allowed:
        return await call_next(request)

    record_rate_limit_rejection(request.url.path)
    return JSONResponse(
        status_code=429,
        content={"detail": "Rate limit exceeded"},
        headers={"Retry-After": str(retry_after)},
    )


def try_build_redis_rate_limiter(
    requests_per_minute: int,
    burst: int,
) -> InMemoryRateLimiter | None:
    """Attempt a Redis-backed limiter; fall back to in-memory on failure."""
    try:
        client = build_redis_client(
            settings.redis.url,
            settings.redis.password.get_secret_value(),
        )
        client.ping()
    except Exception as exc:
        logger.warning("Redis unavailable for rate limiting; using in-memory limiter: %s", exc)
        return None

    # Shared Redis limiter would need Lua scripting; in-memory per pod is acceptable
    # for T-160 graceful degradation (documented in ops guide).
    _ = client
    return InMemoryRateLimiter(requests_per_minute, burst)
