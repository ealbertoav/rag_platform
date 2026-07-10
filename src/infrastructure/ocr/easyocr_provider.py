"""EasyOCR provider via Docling (T-221)."""

from __future__ import annotations

from typing import ClassVar

from src.infrastructure.ocr.docling_backed import DoclingBackedOcr

__all__ = ["EasyOcrProvider"]


class EasyOcrProvider(DoclingBackedOcr):
    """Self-hosted OCR using Docling + EasyOCR."""

    engine: ClassVar[str] = "easyocr"
