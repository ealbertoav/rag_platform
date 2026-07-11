"""Scanned-PDF OCR fallback for ingestion (T-223).

When "parsing.ocr.enabled" is true, low-text / empty PDF loads are re-read
via "get_ocr_provider()" and the extracted text replaces document content.
"Low-text" means fewer than "parsing.ocr.min_chars" non-whitespace characters
(per page when "metadata["pages"]" is present). Born-digital PDFs with enough
extractable text (or mixed born-digital + scanned pages) skip OCR so existing
text is not overwritten.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from src.core.constants import OCR_APPLIED_KEY
from src.core.exceptions import ConfigurationError, DocumentLoadError
from src.core.settings import OcrSettings, Settings
from src.domain.entities.document import Document
from src.domain.repositories.ocr_repository import OcrRepository

logger = logging.getLogger(__name__)

_PDF_EXTENSION = ".pdf"


def text_is_low(text: str, min_chars: int) -> bool:
    """Return True when *text* has fewer than *min_chars* non-whitespace characters."""
    return sum(1 for char in text if not char.isspace()) < min_chars


def document_needs_ocr(document: Document, min_chars: int) -> bool:
    """Return True when a *document* should be sent through whole-file OCR.

    Prefer page-level signals when "metadata["pages"]" is present so mixed
    born-digital and scanned PDFs are not overwritten. Without page texts, fall
    back to overall document content length.
    """
    pages = document.metadata.get("pages")
    if isinstance(pages, list) and pages:
        page_texts = [page if isinstance(page, str) else "" for page in pages]
        # Whole-file OCR only when every page is low-text.
        return all(text_is_low(page, min_chars) for page in page_texts)
    return text_is_low(document.content, min_chars)


def should_attempt_ocr(
    document: Document,
    path: Path,
    *,
    app_settings: Settings | None = None,
) -> bool:
    """Return True when ingest would run whole-file OCR for *path*.

    Matches the preconditions in "apply_ocr_fallback" (PDF suffix, OCR enabled,
    low extractable text) so callers can cheaply gate work before provider I/O.
    """
    if path.suffix.lower() != _PDF_EXTENSION:
        return False
    ocr_cfg = _ocr_settings(app_settings)
    if not ocr_cfg.enabled:
        return False
    return document_needs_ocr(document, ocr_cfg.min_chars)


def apply_ocr_fallback(
    document: Document,
    path: Path,
    *,
    app_settings: Settings | None = None,
    ocr_provider: OcrRepository | None = None,
) -> Document:
    """Replace low-text PDF content with OCR text when OCR is enabled.

    No-op when:
      - path is not a PDF
      - document has enough extractable text (including mixed pages)
      - OCR is disabled
      - OCR provider is misconfigured ("ConfigurationError")
      - OCR returns empty text
      - OCR raises "DocumentLoadError" (logged; an original document kept)

    Provider resolution (and "get_ocr_provider()") runs only after the
    low-text check, so born-digital PDFs never pay for — or fail on —
    OCR factory construction.
    """
    if path.suffix.lower() != _PDF_EXTENSION:
        return document

    ocr_cfg = _ocr_settings(app_settings)
    min_chars = ocr_cfg.min_chars
    if not document_needs_ocr(document, min_chars):
        logger.debug("Skipping OCR for %s (sufficient extractable text)", path.name)
        return document

    provider = ocr_provider
    if provider is None:
        if not ocr_cfg.enabled:
            return document
        provider = _resolve_ocr_provider(path, app_settings)
        if provider is None:
            return document

    try:
        ocr_text = provider.ocr(path).strip()
    except DocumentLoadError as exc:
        logger.warning("OCR failed for %s; keeping extractable text: %s", path.name, exc)
        return document

    if not ocr_text:
        logger.warning("OCR returned empty text for %s; keeping extractable text", path.name)
        return document

    metadata: dict[str, Any] = dict(document.metadata)
    metadata[OCR_APPLIED_KEY] = True
    logger.info(
        "Applied OCR fallback for %s (%d → %d non-whitespace chars)",
        path.name,
        sum(1 for char in document.content if not char.isspace()),
        sum(1 for char in ocr_text if not char.isspace()),
    )
    return document.model_copy(update={"content": ocr_text, "metadata": metadata})


def _resolve_ocr_provider(
    path: Path,
    app_settings: Settings | None,
) -> OcrRepository | None:
    """Return the configured OCR provider, or None when disabled / misconfigured."""
    from src.infrastructure.ocr import get_ocr_provider

    try:
        return get_ocr_provider(app_settings)
    except ConfigurationError as exc:
        logger.warning(
            "OCR provider misconfigured for %s; keeping extractable text: %s",
            path.name,
            exc,
        )
        return None


def _ocr_settings(app_settings: Settings | None) -> OcrSettings:
    if app_settings is not None:
        return app_settings.parsing.ocr
    from src.core.settings import settings

    return settings.parsing.ocr
