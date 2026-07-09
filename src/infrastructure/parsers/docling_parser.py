from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol, cast, override

from src.core.constants import (
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SECTION_KEY,
    FIGURE_ID_KEY,
    TABLE_ID_KEY,
)
from src.core.exceptions import ConfigurationError, DocumentLoadError
from src.domain.entities.parsed_document import ParsedDocument
from src.domain.repositories.layout_parser_repository import LayoutParserRepository

logger = logging.getLogger(__name__)

_LAYOUT_PARSER_EXTENSIONS: frozenset[str] = frozenset({".pdf", ".docx"})


class _DoclingProvenance(Protocol):
    page_no: int
    bbox: Any | None


class _ProvItem(Protocol):
    prov: list[_DoclingProvenance]


class _DoclingLabel(Protocol):
    name: str


class _SectionItem(Protocol):
    label: _DoclingLabel | None
    text: str


class _DoclingTable(_ProvItem, Protocol):
    def export_to_markdown(self) -> str: ...


class _DoclingPicture(_ProvItem, Protocol):
    def caption_text(self, doc: _DoclingDocument) -> str: ...


class _DoclingDocument(Protocol):
    pages: dict[Any, Any] | None
    tables: list[_DoclingTable]
    pictures: list[_DoclingPicture]

    def iterate_items(self) -> Iterator[tuple[_SectionItem, int]]: ...

    def export_to_markdown(self) -> str: ...


class _DoclingConversionStatus(Protocol):
    name: str


class _DoclingConversionResult(Protocol):
    status: _DoclingConversionStatus | None
    document: _DoclingDocument


class _DoclingConverter(Protocol):
    def convert(self, source: str) -> _DoclingConversionResult: ...


def _page_no(item: _ProvItem) -> int | None:
    prov = getattr(item, "prov", None)
    if prov and len(prov) > 0:
        return int(prov[0].page_no)
    return None


def _bbox(item: _ProvItem) -> list[float] | None:
    prov = getattr(item, "prov", None)
    if not prov or len(prov) == 0:
        return None
    box = prov[0].bbox
    if box is None:
        return None
    return [
        float(box.l),  # noqa: E741 — Docling bbox coordinate
        float(box.t),
        float(box.r),
        float(box.b),
    ]


def _extract_sections(doc: _DoclingDocument) -> list[str]:
    sections: list[str] = []
    for item, _level in doc.iterate_items():
        label = item.label
        if label is not None and label.name == "SECTION_HEADER" and item.text:
            sections.append(item.text)
    return sections


def _extract_tables(doc: _DoclingDocument) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for index, table in enumerate(doc.tables, start=1):
        entry: dict[str, Any] = {
            TABLE_ID_KEY: f"table-{index}",
            "markdown": table.export_to_markdown(),
        }
        page = _page_no(table)
        if page is not None:
            entry[CHUNK_PAGE_KEY] = page
        bbox = _bbox(table)
        if bbox is not None:
            entry[BBOX_KEY] = bbox
        tables.append(entry)
    return tables


def _extract_figures(doc: _DoclingDocument) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    for index, picture in enumerate(doc.pictures, start=1):
        entry: dict[str, Any] = {FIGURE_ID_KEY: f"figure-{index}"}
        page = _page_no(picture)
        if page is not None:
            entry[CHUNK_PAGE_KEY] = page
        bbox = _bbox(picture)
        if bbox is not None:
            entry[BBOX_KEY] = bbox
        caption = picture.caption_text(doc)
        if caption:
            entry["caption"] = caption
        figures.append(entry)
    return figures


def build_docling_metadata(path: Path, doc: _DoclingDocument) -> dict[str, Any]:
    """Build ParsedDocument metadata from a Docling document."""
    pages = doc.pages
    page_count = len(pages) if pages else 0
    sections = _extract_sections(doc)
    metadata: dict[str, Any] = {
        "filename": path.name,
        "extension": path.suffix.lower(),
        "loader": "docling",
        "page_count": page_count,
        "sections": sections,
        "tables": _extract_tables(doc),
        "figures": _extract_figures(doc),
    }
    if sections:
        metadata[CHUNK_SECTION_KEY] = sections[0]
    return metadata


class DoclingLayoutParser(LayoutParserRepository):
    """Layout-aware PDF/DOCX parser backed by Docling."""

    def __init__(self, converter: _DoclingConverter | None = None) -> None:
        self._converter: _DoclingConverter | None = converter

    @override
    def parse(self, path: Path) -> ParsedDocument:
        ext = path.suffix.lower()
        if ext not in _LAYOUT_PARSER_EXTENSIONS:
            raise DocumentLoadError(
                f"DoclingLayoutParser does not support '{ext}'. "
                f"Supported: {sorted(_LAYOUT_PARSER_EXTENSIONS)}"
            )

        try:
            converter = self._converter or _create_converter()
            result = converter.convert(str(path))
            status = getattr(result, "status", None)
            if status is not None and getattr(status, "name", str(status)) == "FAILURE":
                raise DocumentLoadError(f"Docling conversion failed for {path}")

            doc = result.document
            content = doc.export_to_markdown().strip()
            if not content:
                logger.warning("No extractable text in %s (may be scanned)", path.name)

            return ParsedDocument(
                source=str(path.resolve()),
                content=content,
                metadata=build_docling_metadata(path, doc),
            )
        except DocumentLoadError:
            raise
        except Exception as exc:
            raise DocumentLoadError(f"Cannot parse with Docling: {path}", cause=exc) from exc


def _create_converter() -> _DoclingConverter:
    try:
        from docling.document_converter import DocumentConverter
    except ImportError as exc:
        raise ConfigurationError(
            "Layout parser 'docling' requires the docling package. "
            "Install with: uv pip install docling"
        ) from exc
    return cast(_DoclingConverter, DocumentConverter())
