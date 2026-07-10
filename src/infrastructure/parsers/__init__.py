from __future__ import annotations

from pathlib import Path

from src.core.exceptions import ConfigurationError
from src.core.settings import Settings
from src.domain.entities.document import Document
from src.domain.entities.parsed_document import ParsedDocument
from src.domain.repositories.layout_parser_repository import LayoutParserRepository
from src.infrastructure.parsers.docling_parser import DoclingLayoutParser
from src.infrastructure.provider_factory import EnabledProviderCache, load_settings

__all__ = [
    "DoclingLayoutParser",
    "clear_layout_parser_cache",
    "get_layout_parser",
    "parsed_to_document",
]

_settings = load_settings
_cache: EnabledProviderCache[LayoutParserRepository] = EnabledProviderCache()


def clear_layout_parser_cache() -> None:
    """Reset the cached layout parser instance (for tests and settings reloads)."""
    _cache.clear()


def _create_layout_parser(provider: str) -> LayoutParserRepository:
    match provider:
        case "docling":
            return DoclingLayoutParser()
        case _:
            raise ConfigurationError(f"Unknown layout parser provider: {provider!r}")


def get_layout_parser(app_settings: Settings | None = None) -> LayoutParserRepository | None:
    """Return the configured layout parser, or None when disabled."""
    if app_settings is None:
        app_settings = _settings()
    return _cache.get(app_settings.parsing.layout_parser, _create_layout_parser)


def parsed_to_document(parsed: ParsedDocument) -> Document:
    """Convert a :class:`ParsedDocument` into a :class:`Document` for ingestion."""
    source = parsed.source
    if not Path(source).is_absolute():
        source = str(Path(source).resolve())

    metadata = dict(parsed.metadata)
    metadata.setdefault("loader", "docling")
    return Document(source=source, content=parsed.content, metadata=metadata)
