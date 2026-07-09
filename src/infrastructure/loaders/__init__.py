from __future__ import annotations

from pathlib import Path
from typing import Protocol

from src.core.constants import SUPPORTED_EXTENSIONS
from src.core.exceptions import ConfigurationError, DocumentLoadError
from src.core.settings import Settings
from src.domain.entities.document import Document
from src.infrastructure.loaders.docx_loader import DocxLoader
from src.infrastructure.loaders.html_loader import HtmlLoader
from src.infrastructure.loaders.markdown_loader import MarkdownLoader
from src.infrastructure.loaders.pdf_loader import PdfLoader

_LAYOUT_PARSER_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})


def _settings() -> Settings:
    """Read settings lazily so env reloads apply without re-importing this module."""
    from src.core.settings import settings

    return settings


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


def _load_with_layout_parser(path: Path) -> Document:
    from src.infrastructure.parsers import get_layout_parser, parsed_to_document

    app_settings = _settings()
    try:
        parser = get_layout_parser(app_settings)
    except ConfigurationError as exc:
        raise DocumentLoadError(
            f"Layout parser misconfigured for {path.name}",
            cause=exc,
        ) from exc
    if parser is None:
        raise DocumentLoadError(
            f"Layout parser requested for {path.name} but parsing.layout_parser.enabled is false"
        )
    return parsed_to_document(parser.parse(path))


def load_document(path: Path) -> Document:
    """Load *path* using the appropriate loader, chosen by file extension.

    When "parsing.layout_parser.enabled" is true, PDF and DOCX files are
    routed through :class:`DoclingLayoutParser` instead of the plain-text loaders.

    Raises:
        DocumentLoadError: if the extension is unsupported or loading fails.
    """
    ext = path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        raise DocumentLoadError(
            f"Unsupported file type '{ext}'. Supported: {sorted(SUPPORTED_EXTENSIONS)}"
        )
    if ext in _LAYOUT_PARSER_EXTENSIONS and _settings().parsing.layout_parser.enabled:
        return _load_with_layout_parser(path)
    return _LOADERS[ext].load(path)


__all__ = [
    "DocxLoader",
    "DocumentLoader",
    "HtmlLoader",
    "MarkdownLoader",
    "PdfLoader",
    "load_document",
]
