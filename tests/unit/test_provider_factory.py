"""Unit tests for shared EnabledProviderCache / load_settings."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.infrastructure.provider_factory import EnabledProviderCache, load_settings


class TestLoadSettings:
    def test_returns_settings_singleton(self) -> None:
        from src.core.settings import settings

        assert load_settings() is settings


class TestEnabledProviderCache:
    def test_returns_none_when_disabled(self) -> None:
        cache: EnabledProviderCache[str] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=False, provider="x")
        assert cache.get(cfg, lambda _p: "created") is None

    def test_caches_none_for_disabled(self) -> None:
        cache: EnabledProviderCache[str] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=False, provider="x")
        calls = 0

        def create(_provider: str) -> str:
            nonlocal calls
            calls += 1
            return "created"

        assert cache.get(cfg, create) is None
        assert cache.get(cfg, create) is None
        assert calls == 0

    def test_creates_and_caches_when_enabled(self) -> None:
        cache: EnabledProviderCache[object] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="docling")
        first = cache.get(cfg, lambda _p: object())
        second = cache.get(cfg, lambda _p: object())
        assert first is second

    def test_clear_invalidates_cache(self) -> None:
        cache: EnabledProviderCache[object] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="docling")
        first = cache.get(cfg, lambda _p: object())
        cache.clear()
        second = cache.get(cfg, lambda _p: object())
        assert first is not second

    def test_create_failure_does_not_poison_cache(self) -> None:
        cache: EnabledProviderCache[str] = EnabledProviderCache()
        enabled = SimpleNamespace(enabled=True, provider="bad")
        disabled = SimpleNamespace(enabled=False, provider="bad")

        with pytest.raises(ValueError, match="boom"):
            cache.get(enabled, lambda _p: (_ for _ in ()).throw(ValueError("boom")))

        assert cache.get(disabled, lambda _p: "created") is None

    def test_passes_provider_name_to_create(self) -> None:
        cache: EnabledProviderCache[str] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="tesseract")
        assert cache.get(cfg, lambda provider: f"ok:{provider}") == "ok:tesseract"
