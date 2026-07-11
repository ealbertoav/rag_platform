"""Tesseract OCR provider via Docling (T-221)."""

from __future__ import annotations

from typing import ClassVar

from src.infrastructure.ocr.docling_backed import DoclingBackedOcr

__all__ = ["TesseractOcrProvider"]


class TesseractOcrProvider(DoclingBackedOcr):
    """Self-hosted OCR using Docling + Tesseract CLI."""

    engine: ClassVar[str] = "tesseract"
