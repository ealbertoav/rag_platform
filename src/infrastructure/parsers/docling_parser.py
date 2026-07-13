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


def _provenance_metadata(item: _ProvItem) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    page = _page_no(item)
    if page is not None:
        metadata[CHUNK_PAGE_KEY] = page
    bbox = _bbox(item)
    if bbox is not None:
        metadata[BBOX_KEY] = bbox
    return metadata


def _extract_tables(doc: _DoclingDocument) -> list[dict[str, Any]]:
    tables: list[dict[str, Any]] = []
    for index, table in enumerate(doc.tables, start=1):
        entry: dict[str, Any] = {
            TABLE_ID_KEY: f"table-{index}",
            **_provenance_metadata(table),
        }
        try:
            markdown = table.export_to_markdown().strip()
        except Exception as exc:
            logger.warning(
                "Failed to export table %s to markdown: %s",
                entry[TABLE_ID_KEY],
                exc,
            )
            markdown = ""
        if markdown:
            entry["text"] = markdown
        tables.append(entry)
    return tables


def _extract_figures(doc: _DoclingDocument) -> list[dict[str, Any]]:
    figures: list[dict[str, Any]] = []
    for index, picture in enumerate(doc.pictures, start=1):
        entry: dict[str, Any] = {
            FIGURE_ID_KEY: f"figure-{index}",
            **_provenance_metadata(picture),
        }
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
            if self._converter is None:
                self._converter = create_docling_converter()
            result = self._converter.convert(str(path))
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
        except ConfigurationError as exc:
            raise DocumentLoadError(
                f"Docling layout parser is not configured for {path.name}",
                cause=exc,
            ) from exc
        except Exception as exc:
            raise DocumentLoadError(f"Cannot parse with Docling: {path}", cause=exc) from exc


def create_docling_converter() -> _DoclingConverter:
    """Create a Docling converter with picture rasterization enabled.

    Shared by :class:`DoclingLayoutParser` and figure-asset extraction so layout
    "figures[]" metadata and "PictureItem.get_image()" rasters come from the
    same pipeline options (PDF "PdfPipelineOptions" / DOCX
    "PaginatedPipelineOptions" with "generate_picture_images=True").
    """
    try:
        # Optional runtime dependency — install separately: uv pip install docling
        import importlib

        base_models = importlib.import_module("docling.datamodel.base_models")
        pipeline_options_mod = importlib.import_module("docling.datamodel.pipeline_options")
        document_converter_mod = importlib.import_module("docling.document_converter")
        input_format = base_models.InputFormat
        pdf_pipeline_options = pipeline_options_mod.PdfPipelineOptions
        paginated_pipeline_options = pipeline_options_mod.PaginatedPipelineOptions
        document_converter = document_converter_mod.DocumentConverter
        pdf_format_option = document_converter_mod.PdfFormatOption
        word_format_option = document_converter_mod.WordFormatOption
    except ImportError as exc:
        raise ConfigurationError(
            "Layout parser 'docling' requires the docling package. "
            "Install with: uv pip install docling"
        ) from exc

    pdf_options = pdf_pipeline_options()
    pdf_options.generate_picture_images = True
    # DOCX uses SimplePipeline; PaginatedPipelineOptions carries generate_picture_images
    # so PictureItem ImageRefs / get_image() can yield raster bytes when supported.
    docx_options = paginated_pipeline_options()
    docx_options.generate_picture_images = True
    return cast(
        _DoclingConverter,
        document_converter(
            allowed_formats=[input_format.PDF, input_format.DOCX],
            format_options={
                input_format.PDF: pdf_format_option(pipeline_options=pdf_options),
                input_format.DOCX: word_format_option(pipeline_options=docx_options),
            },
        ),
    )
