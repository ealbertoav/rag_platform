"""OCR provider factory (T-220 / T-221 / T-222).

Self-hosted providers (tesseract / easyocr / docling) are Docling-backed.
Azure Document Intelligence uses the REST API via ``AzureDocumentIntelligenceOcr``.
"""

from __future__ import annotations

from src.core.exceptions import ConfigurationError
from src.core.settings import Settings
from src.domain.repositories.ocr_repository import OcrRepository
from src.infrastructure.ocr.azure_di_provider import AzureDocumentIntelligenceOcr
from src.infrastructure.ocr.docling_provider import DoclingOcrProvider
from src.infrastructure.ocr.easyocr_provider import EasyOcrProvider
from src.infrastructure.ocr.tesseract_provider import TesseractOcrProvider
from src.infrastructure.provider_factory import EnabledProviderCache, load_settings

__all__ = [
    "AzureDocumentIntelligenceOcr",
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


def _azure_di_from_settings(app_settings: Settings) -> AzureDocumentIntelligenceOcr:
    cfg = app_settings.parsing.ocr.azure_di
    return AzureDocumentIntelligenceOcr(
        endpoint=cfg.endpoint,
        api_key=cfg.api_key.get_secret_value(),
        api_version=cfg.api_version,
        model_id=cfg.model_id,
        timeout_seconds=cfg.timeout_seconds,
        poll_interval_seconds=cfg.poll_interval_seconds,
    )


def get_ocr_provider(app_settings: Settings | None = None) -> OcrRepository | None:
    """Return the configured OCR provider, or None when disabled.

    When OCR is enabled, self-hosted providers (tesseract / easyocr / docling)
    return Docling-backed implementations. ``azure_di`` returns
    ``AzureDocumentIntelligenceOcr`` when endpoint and API key are set.
    """
    if app_settings is None:
        app_settings = _settings()

    def _create(provider: str) -> OcrRepository:
        match provider:
            case "tesseract":
                return TesseractOcrProvider()
            case "easyocr":
                return EasyOcrProvider()
            case "docling":
                return DoclingOcrProvider()
            case "azure_di":
                return _azure_di_from_settings(app_settings)
            case _:
                raise ConfigurationError(f"Unknown OCR provider: {provider!r}")

    return _cache.get(app_settings.parsing.ocr, _create)
