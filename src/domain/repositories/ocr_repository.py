from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class OcrRepository(ABC):
    """Contract for optical character recognition on image or scanned pages."""

    @abstractmethod
    def ocr(self, path: Path) -> str:
        """Extract plain text from the file at a *path*."""
