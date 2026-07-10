"""T-220 — OCR provider factory tests."""

from __future__ import annotations

from collections.abc import Generator
from unittest.mock import patch

import pytest

from src.core.exceptions import ConfigurationError
from src.core.settings import Settings
from src.infrastructure import ocr as ocr_module
from src.infrastructure.ocr import clear_ocr_provider_cache, get_ocr_provider


@pytest.fixture(autouse=True)
def _clear_ocr_cache() -> Generator[None]:
    clear_ocr_provider_cache()
    yield
    clear_ocr_provider_cache()


def _ocr_settings(*, enabled: bool = False, provider: str = "tesseract") -> Settings:
    return Settings(parsing={"ocr": {"enabled": enabled, "provider": provider}})


class TestGetOcrProvider:
    def test_returns_none_when_disabled(self) -> None:
        assert get_ocr_provider(_ocr_settings(enabled=False)) is None

    def test_returns_none_when_disabled_without_explicit_settings(self) -> None:
        disabled = _ocr_settings(enabled=False)
        with patch("src.infrastructure.ocr._settings", return_value=disabled):
            assert get_ocr_provider() is None

    def test_reads_live_settings_when_not_passed(self) -> None:
        from src.evals.e2e.technique_benchmark import temporary_config

        with temporary_config({"PARSING__OCR__ENABLED": "false"}):
            assert get_ocr_provider() is None

    def test_cache_returns_none_for_same_disabled_settings(self) -> None:
        settings = _ocr_settings(enabled=False, provider="tesseract")
        first = get_ocr_provider(settings)
        second = get_ocr_provider(settings)
        assert first is None
        assert second is None

    def test_clear_ocr_provider_cache_allows_reload(self) -> None:
        settings = _ocr_settings(enabled=False)
        assert get_ocr_provider(settings) is None
        clear_ocr_provider_cache()
        assert get_ocr_provider(settings) is None

    @pytest.mark.parametrize("provider", ["tesseract", "easyocr", "docling"])
    def test_known_self_hosted_provider_not_implemented(self, provider: str) -> None:
        settings = _ocr_settings(enabled=True, provider=provider)
        with pytest.raises(ConfigurationError, match=r"not implemented yet \(T-221\)"):
            get_ocr_provider(settings)

    def test_azure_di_not_implemented(self) -> None:
        settings = _ocr_settings(enabled=True, provider="azure_di")
        with pytest.raises(ConfigurationError, match=r"not implemented yet \(T-222\)"):
            get_ocr_provider(settings)

    def test_unknown_provider_raises_configuration_error(self) -> None:
        settings = _ocr_settings(enabled=True, provider="unknown")
        with pytest.raises(ConfigurationError, match="Unknown OCR provider"):
            get_ocr_provider(settings)

    def test_failed_enabled_lookup_does_not_poison_disabled_cache(self) -> None:
        enabled = _ocr_settings(enabled=True, provider="tesseract")
        with pytest.raises(ConfigurationError):
            get_ocr_provider(enabled)
        assert get_ocr_provider(_ocr_settings(enabled=False)) is None

    def test_module_exports(self) -> None:
        assert set(ocr_module.__all__) == {"clear_ocr_provider_cache", "get_ocr_provider"}
