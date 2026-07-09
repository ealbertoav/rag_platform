from __future__ import annotations

from pathlib import Path

from src.core.exceptions import ConfigurationError
from src.core.settings import Settings
from src.domain.entities.document import Document
from src.domain.entities.parsed_document import ParsedDocument
from src.domain.repositories.layout_parser_repository import LayoutParserRepository
from src.infrastructure.parsers.docling_parser import DoclingLayoutParser

__all__ = [
    "DoclingLayoutParser",
    "clear_layout_parser_cache",
    "get_layout_parser",
    "parsed_to_document",
]

_cached_parser_key: tuple[bool, str] | None = None
_cached_parser: LayoutParserRepository | None = None


def _settings() -> Settings:
    """Read settings lazily so env reloads apply without re-importing this module."""
    from src.core.settings import settings

    return settings


def clear_layout_parser_cache() -> None:
    """Reset the cached layout parser instance (for tests and settings reloads)."""
    global _cached_parser_key, _cached_parser
    _cached_parser_key = None
    _cached_parser = None


def get_layout_parser(app_settings: Settings | None = None) -> LayoutParserRepository | None:
    """Return the configured layout parser, or None when disabled."""
    global _cached_parser_key, _cached_parser

    if app_settings is None:
        app_settings = _settings()

    cfg = app_settings.parsing.layout_parser
    cache_key = (cfg.enabled, cfg.provider)
    if _cached_parser_key == cache_key:
        return _cached_parser

    if not cfg.enabled:
        _cached_parser_key = cache_key
        _cached_parser = None
        return None

    match cfg.provider:
        case "docling":
            parser: LayoutParserRepository = DoclingLayoutParser()
        case _:
            raise ConfigurationError(f"Unknown layout parser provider: {cfg.provider!r}")

    _cached_parser_key = cache_key
    _cached_parser = parser
    return parser


def parsed_to_document(parsed: ParsedDocument) -> Document:
    """Convert a :class:`ParsedDocument` into a :class:`Document` for ingestion."""
    source = parsed.source
    if not Path(source).is_absolute():
        source = str(Path(source).resolve())

    metadata = dict(parsed.metadata)
    metadata.setdefault("loader", "docling")
    return Document(source=source, content=parsed.content, metadata=metadata)
