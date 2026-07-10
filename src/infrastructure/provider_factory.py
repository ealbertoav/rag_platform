"""Shared helpers for feature-flagged, cached infrastructure factories."""

from __future__ import annotations

from collections.abc import Callable
from typing import Generic, Protocol, TypeVar

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
    """Cache for feature-flagged providers keyed by "(enabled, provider)"."""

    def __init__(self) -> None:
        self._key: tuple[bool, str] | None = None
        self._value: T | None = None

    def clear(self) -> None:
        """Drop the cached instance (for test and settings reloads)."""
        self._key = None
        self._value = None

    def get(self, cfg: EnabledProviderConfig, create: Callable[[str], T]) -> T | None:
        """Return a cached provider, "None" when disabled, or "create(provider)"."""
        cache_key = (cfg.enabled, cfg.provider)
        if self._key == cache_key:
            return self._value

        if not cfg.enabled:
            self._key = cache_key
            self._value = None
            return None

        value = create(cfg.provider)
        self._key = cache_key
        self._value = value
        return value
