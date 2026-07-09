from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from src.domain.entities.parsed_document import ParsedDocument


class LayoutParserRepository(ABC):
    """Contract for layout-aware document parsing (PDF, DOCX, PPTX, …)."""

    @abstractmethod
    def parse(self, path: Path) -> ParsedDocument:
        """Parse *path* into a :class:`ParsedDocument`."""
