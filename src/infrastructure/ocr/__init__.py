"""OCR provider factory (T-220 / T-221).

Self-hosted providers (tesseract / easyocr / docling) are Docling-backed.
Azure Document Intelligence lands in T-222.
"""

from __future__ import annotations

from src.core.exceptions import ConfigurationError
from src.core.settings import Settings
from src.domain.repositories.ocr_repository import OcrRepository
from src.infrastructure.ocr.docling_provider import DoclingOcrProvider
from src.infrastructure.ocr.easyocr_provider import EasyOcrProvider
from src.infrastructure.ocr.tesseract_provider import TesseractOcrProvider
from src.infrastructure.provider_factory import EnabledProviderCache, load_settings

__all__ = [
    "DoclingOcrProvider",
    "EasyOcrProvider",
    "TesseractOcrProvider",
    "clear_ocr_provider_cache",
    "get_ocr_provider",
]

_settings = load_settings
_cache: EnabledProviderCache[OcrRepository] = EnabledProviderCache()


def clear_ocr_provider_cache() -> None:
    """Reset the cached OCR provider instance (for tests and settings reloads)."""
    _cache.clear()


def _create_ocr_provider(provider: str) -> OcrRepository:
    match provider:
        case "tesseract":
            return TesseractOcrProvider()
        case "easyocr":
            return EasyOcrProvider()
        case "docling":
            return DoclingOcrProvider()
        case "azure_di":
            raise ConfigurationError("OCR provider 'azure_di' is not implemented yet (T-222)")
        case _:
            raise ConfigurationError(f"Unknown OCR provider: {provider!r}")


def get_ocr_provider(app_settings: Settings | None = None) -> OcrRepository | None:
    """Return the configured OCR provider, or None when disabled.

    When OCR is enabled, self-hosted providers (tesseract / easyocr / docling)
    return Docling-backed implementations. ``azure_di`` raises until T-222.
    """
    if app_settings is None:
        app_settings = _settings()
    return _cache.get(app_settings.parsing.ocr, _create_ocr_provider)
