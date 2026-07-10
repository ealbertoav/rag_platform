"""Shared Docling-backed OCR implementation used by self-hosted providers (T-221)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar, Protocol, cast, override

from src.core.exceptions import ConfigurationError, DocumentLoadError
from src.domain.repositories.ocr_repository import OcrRepository

logger = logging.getLogger(__name__)

_OCR_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}
)


class _DoclingConversionStatus(Protocol):
    name: str


class _DoclingDocument(Protocol):
    def export_to_markdown(self) -> str: ...


class _DoclingConversionResult(Protocol):
    status: _DoclingConversionStatus | None
    document: _DoclingDocument


class _DoclingConverter(Protocol):
    def convert(self, source: str) -> _DoclingConversionResult: ...


def _ocr_options_for_engine(engine: str) -> Any:
    """Build Docling OCR options for a self-hosted engine name."""
    from docling.datamodel.pipeline_options import (
        EasyOcrOptions,
        OcrAutoOptions,
        TesseractCliOcrOptions,
    )

    match engine:
        case "tesseract":
            return TesseractCliOcrOptions()
        case "easyocr":
            return EasyOcrOptions()
        case "docling":
            return OcrAutoOptions()
        case _:
            raise ConfigurationError(f"Unknown Docling OCR engine: {engine!r}")


def create_ocr_converter(engine: str) -> _DoclingConverter:
    """Create a Docling ``DocumentConverter`` with OCR enabled for *engine*."""
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, ImageFormatOption, PdfFormatOption

        pipeline_options = PdfPipelineOptions(
            do_ocr=True,
            ocr_options=_ocr_options_for_engine(engine),
        )
        return cast(
            _DoclingConverter,
            DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options),
                    InputFormat.IMAGE: ImageFormatOption(pipeline_options=pipeline_options),
                }
            ),
        )
    except ConfigurationError:
        raise
    except ImportError as exc:
        raise ConfigurationError(
            f"OCR provider {engine!r} requires the docling package. "
            "Install with: uv pip install docling"
        ) from exc


class DoclingBackedOcr(OcrRepository):
    """``OcrRepository`` backed by Docling with a selectable OCR engine.

    Subclasses set :attr:`engine` to ``tesseract``, ``easyocr``, or ``docling``.
    """

    engine: ClassVar[str] = "docling"

    def __init__(self, converter: _DoclingConverter | None = None) -> None:
        self._converter: _DoclingConverter | None = converter

    @override
    def ocr(self, path: Path) -> str:
        ext = path.suffix.lower()
        if ext not in _OCR_EXTENSIONS:
            raise DocumentLoadError(
                f"OCR provider {self.engine!r} does not support '{ext}'. "
                f"Supported: {sorted(_OCR_EXTENSIONS)}"
            )

        try:
            if self._converter is None:
                self._converter = create_ocr_converter(self.engine)
            result = self._converter.convert(str(path))
            status = getattr(result, "status", None)
            if status is not None and getattr(status, "name", str(status)) == "FAILURE":
                raise DocumentLoadError(f"OCR conversion failed for {path}")

            text = result.document.export_to_markdown().strip()
            if not text:
                logger.warning("No OCR text extracted from %s", path.name)
            return text
        except DocumentLoadError:
            raise
        except ConfigurationError as exc:
            raise DocumentLoadError(
                f"OCR provider {self.engine!r} is not configured for {path.name}",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise DocumentLoadError(f"Cannot OCR with {self.engine!r}: {path}", cause=exc) from exc
