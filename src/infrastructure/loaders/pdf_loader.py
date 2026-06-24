from __future__ import annotations

import logging
from pathlib import Path

from pypdf import PdfReader

from src.core.exceptions import DocumentLoadError
from src.domain.entities.document import Document

logger = logging.getLogger(__name__)


class PdfLoader:
    """Loads text from PDF files using pypdf."""

    @staticmethod
    def load(path: Path) -> Document:
        try:
            reader = PdfReader(str(path))
            page_texts: list[str] = []
            for page in reader.pages:
                text = page.extract_text() or ""
                page_texts.append(text)

            content = "\n\n".join(page_texts).strip()
            if not content:
                logger.warning("No extractable text in %s (may be scanned)", path.name)

            return Document(
                source=str(path.resolve()),
                content=content,
                metadata={
                    "filename": path.name,
                    "extension": path.suffix.lower(),
                    "loader": "pdf",
                    "page_count": len(reader.pages),
                    "pages": page_texts,
                },
            )
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Cannot load PDF: {path}", cause=exc) from exc
