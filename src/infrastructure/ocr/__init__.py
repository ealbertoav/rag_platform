"""OCR provider factory (T-220).

Concrete providers land in T-221 (self-hosted / Docling-backed) and T-222 (Azure DI).
Until then, enabling OCR with a known provider raises :class:`ConfigurationError`.
"""

from __future__ import annotations

from src.core.exceptions import ConfigurationError
from src.core.settings import Settings
from src.domain.repositories.ocr_repository import OcrRepository
from src.infrastructure.provider_factory import EnabledProviderCache, load_settings

__all__ = [
    "clear_ocr_provider_cache",
    "get_ocr_provider",
]

_KNOWN_PROVIDERS = frozenset({"tesseract", "easyocr", "docling", "azure_di"})

_settings = load_settings
_cache: EnabledProviderCache[OcrRepository] = EnabledProviderCache()


def clear_ocr_provider_cache() -> None:
    """Reset the cached OCR provider instance (for tests and settings reloads)."""
    _cache.clear()


def _create_ocr_provider(provider: str) -> OcrRepository:
    if provider not in _KNOWN_PROVIDERS:
        raise ConfigurationError(f"Unknown OCR provider: {provider!r}")

    if provider == "azure_di":
        raise ConfigurationError("OCR provider 'azure_di' is not implemented yet (T-222)")

    raise ConfigurationError(f"OCR provider {provider!r} is not implemented yet (T-221)")


def get_ocr_provider(app_settings: Settings | None = None) -> OcrRepository | None:
    """Return the configured OCR provider, or None when disabled.

    When OCR is enabled, known providers raise :class:`ConfigurationError` until
    T-221 (tesseract / easyocr / docling) or T-222 (azure_di) implement them.
    """
    if app_settings is None:
        app_settings = _settings()
    return _cache.get(app_settings.parsing.ocr, _create_ocr_provider)
