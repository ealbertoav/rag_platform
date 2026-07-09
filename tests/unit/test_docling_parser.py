"""T-200 — DoclingLayoutParser, factory, and loader delegation tests.

Docling is mocked at the unit boundary (heavy ML dependency).
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import (
    BBOX_KEY,
    CHUNK_PAGE_KEY,
    CHUNK_SECTION_KEY,
    FIGURE_ID_KEY,
    TABLE_ID_KEY,
)
from src.core.exceptions import ConfigurationError, DocumentLoadError
from src.core.settings import Settings
from src.domain.entities.document import Document
from src.domain.entities.parsed_document import ParsedDocument
from src.domain.repositories.layout_parser_repository import LayoutParserRepository
from src.infrastructure.loaders import load_document
from src.infrastructure.parsers import get_layout_parser, parsed_to_document
from src.infrastructure.parsers.docling_parser import DoclingLayoutParser, build_docling_metadata

_DOCLING_PARSER = "src.infrastructure.parsers.docling_parser"


def _internal(name: str) -> object:
    return getattr(importlib.import_module(_DOCLING_PARSER), name)


docling_page_no = cast(Callable[[object], int | None], _internal("_page_no"))
docling_bbox = cast(Callable[[object], list[float] | None], _internal("_bbox"))
docling_extract_sections = cast(Callable[[object], list[str]], _internal("_extract_sections"))
docling_extract_tables = cast(
    Callable[[object], list[dict[str, Any]]],
    _internal("_extract_tables"),
)
docling_extract_figures = cast(
    Callable[[object], list[dict[str, Any]]],
    _internal("_extract_figures"),
)
docling_create_converter = cast(Callable[[], object], _internal("_create_converter"))

# ── Helpers ────────────────────────────────────────────────────────────────────

_LAYOUT_PARSER_ENABLED = "src.infrastructure.loaders.settings.parsing.layout_parser.enabled"


def _enable_layout_parser(monkeypatch: pytest.MonkeyPatch, *, enabled: bool = True) -> None:
    monkeypatch.setattr(_LAYOUT_PARSER_ENABLED, enabled)


def _prov(page_no: int = 1, bbox: tuple[float, float, float, float] | None = None) -> list[object]:
    box = None
    if bbox is not None:
        box = SimpleNamespace(l=bbox[0], t=bbox[1], r=bbox[2], b=bbox[3])
    return [SimpleNamespace(page_no=page_no, bbox=box)]


def _section_item(text: str) -> object:
    return SimpleNamespace(
        label=SimpleNamespace(name="SECTION_HEADER"),
        text=text,
        prov=_prov(page_no=2),
    )


def _make_docling_doc(
    *,
    markdown: str = "# Title\n\nBody text.",
    pages: dict[int, object] | None = None,
    sections: list[str] | None = None,
    tables: list[object] | None = None,
    pictures: list[object] | None = None,
) -> MagicMock:
    doc = MagicMock()
    doc.export_to_markdown.return_value = markdown
    doc.pages = pages if pages is not None else {1: SimpleNamespace(page_no=1)}

    items: list[tuple[object, int]] = []
    for title in sections or []:
        items.append((_section_item(title), 1))
    doc.iterate_items.return_value = iter(items)

    if tables is None:
        table = MagicMock()
        table.export_to_markdown.return_value = "| A | B |\n|---|---|\n| 1 | 2 |"
        table.prov = _prov(page_no=3, bbox=(10.0, 20.0, 100.0, 80.0))
        tables = [table]
    doc.tables = tables

    if pictures is None:
        picture = MagicMock()
        picture.prov = _prov(page_no=4, bbox=(5.0, 5.0, 50.0, 40.0))
        picture.caption_text.return_value = "Figure caption"
        pictures = [picture]
    doc.pictures = pictures

    return doc


def _make_converter(doc: MagicMock, *, status_name: str = "SUCCESS") -> MagicMock:
    converter = MagicMock()
    converter.convert.return_value = SimpleNamespace(
        status=SimpleNamespace(name=status_name),
        document=doc,
    )
    return converter


# ── Metadata helpers ───────────────────────────────────────────────────────────


class TestDoclingMetadataHelpers:
    def test_page_no_returns_none_without_provenance(self) -> None:
        assert docling_page_no(SimpleNamespace()) is None

    def test_page_no_reads_first_provenance(self) -> None:
        item = SimpleNamespace(prov=_prov(page_no=7))
        assert docling_page_no(item) == 7

    def test_bbox_returns_none_without_provenance(self) -> None:
        assert docling_bbox(SimpleNamespace()) is None

    def test_bbox_returns_none_when_box_missing(self) -> None:
        item = SimpleNamespace(prov=[SimpleNamespace(page_no=1, bbox=None)])
        assert docling_bbox(item) is None

    def test_bbox_extracts_coordinates(self) -> None:
        item = SimpleNamespace(prov=_prov(bbox=(1.0, 2.0, 3.0, 4.0)))
        assert docling_bbox(item) == [1.0, 2.0, 3.0, 4.0]

    def test_extract_sections_skips_non_headers(self) -> None:
        doc = MagicMock()
        doc.iterate_items.return_value = iter(
            [
                (SimpleNamespace(label=SimpleNamespace(name="PARAGRAPH"), text="x"), 0),
                (_section_item("Intro"), 1),
            ]
        )
        assert docling_extract_sections(doc) == ["Intro"]

    def test_extract_sections_skips_empty_text(self) -> None:
        doc = MagicMock()
        doc.iterate_items.return_value = iter(
            [(SimpleNamespace(label=SimpleNamespace(name="SECTION_HEADER"), text=""), 1)]
        )
        assert docling_extract_sections(doc) == []

    def test_extract_tables_without_page_or_bbox(self) -> None:
        table = MagicMock()
        table.export_to_markdown.return_value = "table-md"
        table.prov = []
        doc = MagicMock()
        doc.tables = [table]
        result = docling_extract_tables(doc)
        assert result == [{TABLE_ID_KEY: "table-1", "markdown": "table-md"}]

    def test_extract_figures_without_caption(self) -> None:
        picture = MagicMock()
        picture.prov = []
        picture.caption_text.return_value = ""
        doc = MagicMock()
        doc.pictures = [picture]
        result = docling_extract_figures(doc)
        assert result == [{FIGURE_ID_KEY: "figure-1"}]

    def test_build_docling_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "report.pdf"
        doc = _make_docling_doc(sections=["Introduction"])
        metadata = build_docling_metadata(path, doc)
        assert metadata["filename"] == "report.pdf"
        assert metadata["extension"] == ".pdf"
        assert metadata["loader"] == "docling"
        assert metadata["page_count"] == 1
        assert metadata["sections"] == ["Introduction"]
        assert metadata[CHUNK_SECTION_KEY] == "Introduction"
        assert len(metadata["tables"]) == 1
        assert metadata["tables"][0][TABLE_ID_KEY] == "table-1"
        assert metadata["tables"][0][CHUNK_PAGE_KEY] == 3
        assert metadata["tables"][0][BBOX_KEY] == [10.0, 20.0, 100.0, 80.0]
        assert metadata["figures"][0][FIGURE_ID_KEY] == "figure-1"
        assert metadata["figures"][0]["caption"] == "Figure caption"

    def test_build_docling_metadata_no_sections(self, tmp_path: Path) -> None:
        doc = _make_docling_doc(sections=[])
        metadata = build_docling_metadata(tmp_path / "a.docx", doc)
        assert CHUNK_SECTION_KEY not in metadata


# ── DoclingLayoutParser ────────────────────────────────────────────────────────


class TestDoclingLayoutParser:
    def test_implements_layout_parser_repository(self) -> None:
        assert isinstance(DoclingLayoutParser(converter=MagicMock()), LayoutParserRepository)

    def test_parse_pdf_returns_parsed_document(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")
        doc = _make_docling_doc(markdown="Parsed markdown body.")
        parser = DoclingLayoutParser(converter=_make_converter(doc))
        result = parser.parse(path)
        assert isinstance(result, ParsedDocument)
        assert result.content == "Parsed markdown body."
        assert result.metadata["loader"] == "docling"

    def test_parse_docx_supported(self, tmp_path: Path) -> None:
        path = tmp_path / "sample.docx"
        path.write_bytes(b"PK placeholder")
        doc = _make_docling_doc(markdown="DOCX content")
        parser = DoclingLayoutParser(converter=_make_converter(doc))
        result = parser.parse(path)
        assert result.content == "DOCX content"

    def test_unsupported_extension_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "notes.md"
        path.write_text("# hello")
        with pytest.raises(DocumentLoadError, match="does not support"):
            DoclingLayoutParser(converter=MagicMock()).parse(path)

    def test_conversion_failure_raises_document_load_error(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _make_docling_doc()
        parser = DoclingLayoutParser(converter=_make_converter(doc, status_name="FAILURE"))
        with pytest.raises(DocumentLoadError, match="conversion failed"):
            parser.parse(path)

    def test_empty_content_allowed(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _make_docling_doc(markdown="   ")
        parser = DoclingLayoutParser(converter=_make_converter(doc))
        result = parser.parse(path)
        assert result.content == ""

    def test_wraps_unexpected_exception(self, tmp_path: Path) -> None:
        path = tmp_path / "err.pdf"
        path.write_bytes(b"%PDF-1.4")
        converter = MagicMock()
        converter.convert.side_effect = RuntimeError("boom")
        with pytest.raises(DocumentLoadError) as exc_info:
            DoclingLayoutParser(converter=converter).parse(path)
        assert exc_info.value.cause is not None

    def test_re_raises_document_load_error(self, tmp_path: Path) -> None:
        path = tmp_path / "err.pdf"
        path.write_bytes(b"%PDF-1.4")
        converter = MagicMock()
        converter.convert.side_effect = DocumentLoadError("already wrapped")
        with pytest.raises(DocumentLoadError, match="already wrapped"):
            DoclingLayoutParser(converter=converter).parse(path)

    def test_status_none_treated_as_success(self, tmp_path: Path) -> None:
        path = tmp_path / "ok.pdf"
        path.write_bytes(b"%PDF-1.4")
        doc = _make_docling_doc(markdown="ok")
        converter = MagicMock()
        converter.convert.return_value = SimpleNamespace(status=None, document=doc)
        result = DoclingLayoutParser(converter=converter).parse(path)
        assert result.content == "ok"

    def test_create_converter_returns_document_converter(self) -> None:
        mock_converter_cls = MagicMock(return_value="converter-instance")
        doc_converter_mod = MagicMock()
        doc_converter_mod.DocumentConverter = mock_converter_cls
        with patch.dict(
            "sys.modules",
            {
                "docling": MagicMock(),
                "docling.document_converter": doc_converter_mod,
            },
        ):
            assert docling_create_converter() == "converter-instance"
        mock_converter_cls.assert_called_once()

    def test_create_converter_raises_when_docling_missing(self) -> None:
        import builtins

        real_import = builtins.__import__

        def _blocked(name, *args, **kwargs):
            if name == "docling.document_converter" or name.startswith("docling"):
                raise ImportError("blocked")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=_blocked),
            pytest.raises(ConfigurationError, match="uv pip install docling"),
        ):
            docling_create_converter()


# ── Factory ────────────────────────────────────────────────────────────────────


class TestGetLayoutParser:
    def test_returns_none_when_disabled(self) -> None:
        settings = Settings(parsing={"layout_parser": {"enabled": False, "provider": "docling"}})
        assert get_layout_parser(settings) is None

    def test_returns_none_when_disabled_without_explicit_settings(self) -> None:
        with patch("src.infrastructure.parsers.default_settings") as mock_settings:
            mock_settings.parsing.layout_parser.enabled = False
            assert get_layout_parser() is None

    def test_returns_docling_parser_when_enabled(self) -> None:
        settings = Settings(parsing={"layout_parser": {"enabled": True, "provider": "docling"}})
        parser = get_layout_parser(settings)
        assert isinstance(parser, DoclingLayoutParser)

    def test_unknown_provider_raises_configuration_error(self) -> None:
        settings = Settings(parsing={"layout_parser": {"enabled": True, "provider": "unknown"}})
        with pytest.raises(ConfigurationError, match="Unknown layout parser provider"):
            get_layout_parser(settings)


class TestParsedToDocument:
    def test_converts_parsed_document(self) -> None:
        parsed = ParsedDocument(
            source="data/raw/manual.pdf",
            content="hello",
            metadata={"filename": "manual.pdf", "loader": "docling"},
        )
        doc = parsed_to_document(parsed)
        assert isinstance(doc, Document)
        assert doc.content == "hello"
        assert doc.metadata["loader"] == "docling"

    def test_resolves_relative_source(self, tmp_path: Path) -> None:
        rel = Path("relative.docx")
        parsed = ParsedDocument(source=str(rel), content="x")
        doc = parsed_to_document(parsed)
        assert Path(doc.source).is_absolute()

    def test_defaults_loader_when_missing(self) -> None:
        parsed = ParsedDocument(source="/abs/file.pdf", content="x", metadata={})
        doc = parsed_to_document(parsed)
        assert doc.metadata["loader"] == "docling"


# ── Loader delegation ──────────────────────────────────────────────────────────


class TestLayoutParserLoaderDelegation:
    def test_pdf_uses_docling_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_layout_parser(monkeypatch)
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF-1.4")

        parsed = ParsedDocument(
            source=str(path.resolve()),
            content="layout text",
            metadata={"loader": "docling", "filename": "report.pdf"},
        )
        mock_parser = MagicMock()
        mock_parser.parse.return_value = parsed

        with patch(
            "src.infrastructure.parsers.get_layout_parser",
            return_value=mock_parser,
        ):
            doc = load_document(path)

        mock_parser.parse.assert_called_once_with(path)
        assert doc.content == "layout text"
        assert doc.metadata["loader"] == "docling"

    def test_docx_uses_docling_when_enabled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_layout_parser(monkeypatch)
        path = tmp_path / "report.docx"
        path.write_bytes(b"PK")

        parsed = ParsedDocument(source=str(path.resolve()), content="docx layout", metadata={})
        mock_parser = MagicMock()
        mock_parser.parse.return_value = parsed

        with patch(
            "src.infrastructure.parsers.get_layout_parser",
            return_value=mock_parser,
        ):
            doc = load_document(path)

        assert doc.content == "docx layout"

    def test_disabled_uses_plain_pdf_loader(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_layout_parser(monkeypatch, enabled=False)
        path = tmp_path / "plain.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")

        reader = MagicMock()
        page = MagicMock()
        page.extract_text.return_value = "plain pdf text"
        reader.pages = [page]

        with patch("src.infrastructure.loaders.pdf_loader.PdfReader", return_value=reader):
            doc = load_document(path)

        assert doc.metadata["loader"] == "pdf"
        assert "plain pdf text" in doc.content

    def test_enabled_but_parser_none_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_layout_parser(monkeypatch)
        path = tmp_path / "report.pdf"
        path.write_bytes(b"%PDF-1.4")

        with (
            patch("src.infrastructure.parsers.get_layout_parser", return_value=None),
            pytest.raises(DocumentLoadError, match="enabled is false"),
        ):
            load_document(path)

    def test_html_not_delegated_when_layout_parser_enabled(
        self, html_file: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        _enable_layout_parser(monkeypatch)
        doc = load_document(html_file)
        assert doc.metadata["loader"] == "html"


@pytest.fixture
def html_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.html"
    path.write_text("<html><body><p>hello</p></body></html>", encoding="utf-8")
    return path
