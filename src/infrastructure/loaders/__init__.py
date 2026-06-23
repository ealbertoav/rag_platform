from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.core.constants import SUPPORTED_EXTENSIONS
from src.core.exceptions import DocumentLoadError
from src.domain.entities.document import Document
from src.infrastructure.loaders.docx_loader import DocxLoader
from src.infrastructure.loaders.html_loader import HtmlLoader
from src.infrastructure.loaders.markdown_loader import MarkdownLoader
from src.infrastructure.loaders.pdf_loader import PdfLoader


class DocumentLoader(Protocol):
    @staticmethod
    def load(path: Path) -> Document: ...


_LOADERS: dict[str, DocumentLoader] = {
    ".pdf": PdfLoader(),
    ".docx": DocxLoader(),
    ".html": HtmlLoader(),
    ".htm": HtmlLoader(),
    ".md": MarkdownLoader(),
    ".markdown": MarkdownLoader(),
}


def load_document(path: Path) -> Document:
    """Load *path* using the appropriate loader, chosen by file extension.

    Raises:
        DocumentLoadError: if the extension is unsupported or loading fails.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    return _LOADERS[ext].load(path)


__all__ = [
    "DocxLoader",
    "DocumentLoader",
    "HtmlLoader",
    "MarkdownLoader",
    "PdfLoader",
    "load_document",
]
