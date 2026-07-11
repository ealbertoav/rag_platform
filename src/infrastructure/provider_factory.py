"""Shared helpers for feature-flagged, cached infrastructure factories."""

from __future__ import annotations

from collections.abc import Callable, Hashable
from typing import Any, Generic, Protocol, TypeVar

from src.core.settings import Settings

T = TypeVar("T")


class EnabledProviderConfig(Protocol):
    """Minimal settings shape for enabled/provider factories."""

    enabled: bool
    provider: str


def load_settings() -> Settings:
    """Read settings lazily so env reloads apply without re-importing callers."""
    from src.core.settings import settings

    return settings


class EnabledProviderCache(Generic[T]):
    """Cache for feature-flagged providers keyed by "(enabled, provider, identity)".

    "identity" is an optional fingerprint of provider-specific config (for
    example, Azure DI credentials). When the key changes, any previous value
    with a "close()" method is disposed of before the new entry is stored.
    """

    def __init__(self) -> None:
        self._key: tuple[Any, ...] | None = None
        self._value: T | None = None

    def clear(self) -> None:
        """Drop the cached instance (for test and settings reloads)."""
        self._dispose(self._value)
        self._key = None
        self._value = None

    def get(
        self,
        cfg: EnabledProviderConfig,
        create: Callable[[str], T],
        *,
        identity: Hashable | None = None,
    ) -> T | None:
        """Return a cached provider, "None" when disabled, or "create(provider)"."""
        cache_key: tuple[Any, ...] = (cfg.enabled, cfg.provider, identity)
        if self._key == cache_key:
            return self._value

        # Drop the previous entry before building a replacement, so a failed
        # creation cannot leave a disposed instance under the old key, and so
        # credential/config rotations never keep serving the prior client.
        self._dispose(self._value)
        self._key = None
        self._value = None

        if not cfg.enabled:
            self._key = cache_key
            self._value = None
            return None

        value = create(cfg.provider)
        self._key = cache_key
        self._value = value
        return value

    @staticmethod
    def _dispose(value: T | None) -> None:
        if value is None:
            return
        close = getattr(value, "close", None)
        if callable(close):
            close()
