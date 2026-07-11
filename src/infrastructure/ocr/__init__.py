"""OCR provider factory (T-220 / T-221 / T-222).

Self-hosted providers (tesseract / easyocr / docling) are Docling-backed.
Azure Document Intelligence uses the REST API via ``AzureDocumentIntelligenceOcr``.
"""

from __future__ import annotations

from collections.abc import Hashable

from src.core.exceptions import ConfigurationError
from src.core.settings import OcrSettings, Settings
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


def _ocr_provider_identity(ocr_cfg: OcrSettings) -> Hashable | None:
    """Fingerprint config captured at provider construction (Azure DI only)."""
    if ocr_cfg.provider != "azure_di":
        return None
    azure = ocr_cfg.azure_di
    return (
        azure.endpoint,
        azure.api_key.get_secret_value(),
        azure.api_version,
        azure.model_id,
        azure.timeout_seconds,
        azure.poll_interval_seconds,
    )


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

    Cache key is ``(enabled, provider, azure_di identity)`` so credential and
    Azure config rotations rebuild the client without requiring an explicit
    ``clear_ocr_provider_cache()`` first.
    """
    if app_settings is None:
        app_settings = _settings()

    ocr_cfg = app_settings.parsing.ocr

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

    return _cache.get(ocr_cfg, _create, identity=_ocr_provider_identity(ocr_cfg))
