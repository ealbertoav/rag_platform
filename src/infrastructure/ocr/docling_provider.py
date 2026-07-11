"""Docling auto-OCR provider (T-221)."""

from __future__ import annotations

from typing import ClassVar

from src.infrastructure.ocr.docling_backed import DoclingBackedOcr

__all__ = ["DoclingOcrProvider"]


class DoclingOcrProvider(DoclingBackedOcr):
    """Self-hosted OCR using Docling with automatic engine selection."""

    engine: ClassVar[str] = "docling"
