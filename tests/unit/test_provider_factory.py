"""Unit tests for shared EnabledProviderCache / load_settings."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

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

    def test_identity_change_rebuilds_provider(self) -> None:
        cache: EnabledProviderCache[object] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="azure_di")
        first = cache.get(cfg, lambda _p: object(), identity=("ep-a", "key-a"))
        second = cache.get(cfg, lambda _p: object(), identity=("ep-b", "key-b"))
        third = cache.get(cfg, lambda _p: object(), identity=("ep-b", "key-b"))
        assert first is not second
        assert second is third

    def test_same_identity_reuses_provider(self) -> None:
        cache: EnabledProviderCache[object] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="azure_di")
        identity = ("ep", "key", "2024-11-30")
        first = cache.get(cfg, lambda _p: object(), identity=identity)
        second = cache.get(cfg, lambda _p: object(), identity=identity)
        assert first is second

    def test_clear_closes_cached_value(self) -> None:
        cache: EnabledProviderCache[MagicMock] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="azure_di")
        provider = MagicMock()
        assert cache.get(cfg, lambda _p: provider) is provider
        cache.clear()
        provider.close.assert_called_once_with()

    def test_identity_change_closes_previous_value(self) -> None:
        cache: EnabledProviderCache[MagicMock] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="azure_di")
        first = MagicMock()
        second = MagicMock()
        assert cache.get(cfg, lambda _p: first, identity="a") is first
        assert cache.get(cfg, lambda _p: second, identity="b") is second
        first.close.assert_called_once_with()
        second.close.assert_not_called()

    def test_failed_create_after_identity_change_does_not_keep_old_value(self) -> None:
        cache: EnabledProviderCache[MagicMock] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="azure_di")
        first = MagicMock()
        assert cache.get(cfg, lambda _p: first, identity="a") is first

        with pytest.raises(ValueError, match="boom"):
            cache.get(
                cfg,
                lambda _p: (_ for _ in ()).throw(ValueError("boom")),
                identity="b",
            )

        first.close.assert_called_once_with()
        recreated = MagicMock()
        assert cache.get(cfg, lambda _p: recreated, identity="a") is recreated
        assert recreated is not first

    def test_disable_closes_previous_value(self) -> None:
        cache: EnabledProviderCache[MagicMock] = EnabledProviderCache()
        enabled = SimpleNamespace(enabled=True, provider="azure_di")
        disabled = SimpleNamespace(enabled=False, provider="azure_di")
        provider = MagicMock()
        assert cache.get(enabled, lambda _p: provider) is provider
        assert cache.get(disabled, lambda _p: MagicMock()) is None
        provider.close.assert_called_once_with()

    def test_values_without_close_are_replaced_safely(self) -> None:
        cache: EnabledProviderCache[object] = EnabledProviderCache()
        cfg = SimpleNamespace(enabled=True, provider="docling")
        first = object()
        second = object()
        assert cache.get(cfg, lambda _p: first, identity=1) is first
        assert cache.get(cfg, lambda _p: second, identity=2) is second
