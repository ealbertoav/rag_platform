from __future__ import annotations

import logging
import math
import threading
import time
import uuid
from collections import defaultdict, deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, cast, override

from fastapi import Request
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.settings import settings
from src.infrastructure.cache.redis_client import build_redis_client
from src.observability.metrics import record_rate_limit_rejection

logger = logging.getLogger(__name__)

PROTECTED_PREFIXES = ("/ingest", "/chat", "/evals/run", "/feedback")
EXEMPT_PATHS = ("/health", "/metrics")

SLIDING_WINDOW_LUA = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window_ms = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]

redis.call('ZREMRANGEBYSCORE', key, 0, now - window_ms)
local count = redis.call('ZCARD', key)
if count >= limit then
    local oldest = redis.call('ZRANGE', key, 0, 0, 'WITHSCORES')
    if oldest[2] then
        local retry_after = math.ceil((window_ms - (now - tonumber(oldest[2]))) / 1000)
        return {0, math.max(1, retry_after)}
    end
    return {0, 1}
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, window_ms)
return {1, 0}
"""


class RateLimiter(Protocol):
    def allow(self, key: str) -> tuple[bool, int]: ...


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


def should_rate_limit_request(request: Request) -> bool:
    if request.method == "OPTIONS":
        return False
    return is_protected_path(request.url.path)


class InMemoryRateLimiter:
    """Sliding-window limiter keyed by client identity."""

    def __init__(self, requests_per_minute: int, burst: int) -> None:
        self._rpm: int = requests_per_minute
        self._burst: int = burst
        self._window_seconds: float = 60.0
        self._lock: threading.Lock = threading.Lock()
        self._events: dict[str, deque[float]] = defaultdict(deque)

    def allow(self, key: str) -> tuple[bool, int]:
        now = time.monotonic()
        cutoff = now - self._window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                _ = events.popleft()
            limit = self._rpm + self._burst
            if len(events) >= limit:
                retry_after = max(1, math.ceil(self._window_seconds - (now - events[0])))
                return False, retry_after
            events.append(now)
        return True, 0


class RedisRateLimiter:
    """Shared sliding-window limiter backed by Redis sorted sets."""

    KEY_PREFIX: ClassVar[str] = "rag:rate_limit:"

    def __init__(self, client: Any, requests_per_minute: int, burst: int) -> None:
        self._window_ms: Any = 60_000
        self._limit: Any = requests_per_minute + burst
        self._client: Any = client
        self._script: Any = client.register_script(SLIDING_WINDOW_LUA)

    def allow(self, key: str) -> tuple[bool, int]:
        now_ms = int(time.time() * 1000)
        member = f"{now_ms}:{uuid.uuid4().hex}"
        try:
            allowed, retry_after = self._script(
                keys=[f"{self.KEY_PREFIX}{key}"],
                args=[now_ms, self._window_ms, self._limit, member],
            )
            return bool(allowed), int(retry_after)
        except Exception as exc:
            logger.warning("Redis rate limit check failed; allowing request: %s", exc)
            return True, 0


@dataclass
class RateLimitConfig:
    enabled: bool = False
    limiter: RateLimiter | None = None


_config = RateLimitConfig()


def configure_rate_limit(
    *,
    enabled: bool | None = None,
    requests_per_minute: int | None = None,
    burst: int | None = None,
    limiter: RateLimiter | None = None,
) -> RateLimitConfig:
    """Apply rate-limit settings used by the HTTP middleware."""
    cfg = settings.api.rate_limit
    enabled_val = cfg.enabled if enabled is None else enabled
    rpm = cfg.requests_per_minute if requests_per_minute is None else requests_per_minute
    burst_val = cfg.burst if burst is None else burst
    limiter_val = limiter or try_build_redis_rate_limiter(rpm, burst_val)
    if limiter_val is None:
        limiter_val = InMemoryRateLimiter(rpm, burst_val)
    _config.enabled = enabled_val
    _config.limiter = limiter_val
    if enabled_val:
        backend = "redis" if isinstance(limiter_val, RedisRateLimiter) else "in-memory"
        logger.info(
            "API rate limiting enabled (%s): %d req/min + burst %d on protected routes",
            backend,
            rpm,
            burst_val,
        )
    return _config


async def rate_limit_middleware(
    request: Request,
    call_next: Callable[[Request], Awaitable[Response]],
) -> Response:
    """Return 429 when protected routes exceed configured per-client limits."""
    if not _config.enabled or _config.limiter is None or not should_rate_limit_request(request):
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


class RateLimitHTTPMiddleware(BaseHTTPMiddleware):
    @override
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        return await rate_limit_middleware(request, call_next)


def try_build_redis_rate_limiter(
    requests_per_minute: int,
    burst: int,
) -> RedisRateLimiter | None:
    """Attempt a Redis-backed limiter; fall back to in-memory on failure."""
    try:
        client = build_redis_client(
            settings.redis.url,
            settings.redis.password.get_secret_value(),
        )
        _ = client.ping()
    except Exception as exc:
        logger.warning("Redis unavailable for rate limiting; using in-memory limiter: %s", exc)
        return None

    return RedisRateLimiter(cast(Any, client), requests_per_minute, burst)
