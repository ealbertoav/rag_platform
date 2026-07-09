from __future__ import annotations

from pathlib import Path

from src.core.exceptions import ConfigurationError
from src.core.settings import Settings
from src.core.settings import settings as default_settings
from src.domain.entities.document import Document
from src.domain.entities.parsed_document import ParsedDocument
from src.domain.repositories.layout_parser_repository import LayoutParserRepository
from src.infrastructure.parsers.docling_parser import DoclingLayoutParser

__all__ = [
    "DoclingLayoutParser",
    "get_layout_parser",
    "parsed_to_document",
]


def get_layout_parser(app_settings: Settings | None = None) -> LayoutParserRepository | None:
    """Return the configured layout parser, or None when disabled."""
    if app_settings is None:
        app_settings = default_settings

    cfg = app_settings.parsing.layout_parser
    if not cfg.enabled:
        return None

    match cfg.provider:
        case "docling":
            return DoclingLayoutParser()
        case _:
            raise ConfigurationError(f"Unknown layout parser provider: {cfg.provider!r}")


def parsed_to_document(parsed: ParsedDocument) -> Document:
    """Convert a :class:`ParsedDocument` into a :class:`Document` for ingestion."""
    source = parsed.source
    if not Path(source).is_absolute():
        source = str(Path(source).resolve())

    metadata = dict(parsed.metadata)
    metadata.setdefault("loader", "docling")
    return Document(source=source, content=parsed.content, metadata=metadata)
