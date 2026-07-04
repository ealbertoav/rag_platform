from __future__ import annotations

from typing import Any, Protocol, cast


class RedisHashClient(Protocol):
    def hincrbyfloat(self, name: str, key: str, amount: float) -> float: ...

    def hget(self, name: str, key: str) -> Any: ...

    def hmget(self, name: str, keys: list[str]) -> list[Any]: ...

    def hset(self, name: str, key: str, value: float) -> int: ...

    def ping(self) -> bool: ...


def build_redis_client(redis_url: str, password: str = "") -> RedisHashClient:
    """Return a decode-responses Redis client (lazy import keeps optional paths isolated)."""
    import redis

    client = redis.from_url(
        redis_url,
        password=password or None,
        decode_responses=True,
    )
    return cast(RedisHashClient, client)
