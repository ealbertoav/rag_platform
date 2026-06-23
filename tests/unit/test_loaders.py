"""T-010 — document loader tests.

PDF is tested via mocking (pypdf is a unit boundary, not under test here).
DOCX, HTML, and Markdown use real fixture files created in tmp_path.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import docx as python_docx
import pytest

from src.core.exceptions import DocumentLoadError
from src.domain.entities.document import Document
from src.infrastructure.loaders import load_document
from src.infrastructure.loaders.docx_loader import DocxLoader
from src.infrastructure.loaders.html_loader import HtmlLoader
from src.infrastructure.loaders.markdown_loader import MarkdownLoader
from src.infrastructure.loaders.pdf_loader import PdfLoader

# ── Fixtures ───────────────────────────────────────────────────────────────────


@pytest.fixture
def docx_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.docx"
    doc = python_docx.Document()
    doc.add_heading("Introduction", level=1)
    doc.add_paragraph("This is the first paragraph.")
    doc.add_heading("Details", level=2)
    doc.add_paragraph("This is the second paragraph with more detail.")
    doc.save(str(path))
    return path


@pytest.fixture
def html_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.html"
    path.write_text(
        """<!DOCTYPE html>
<html>
<head><title>Test Page</title><style>body{color:red}</style></head>
<body>
  <nav>Navigation links</nav>
  <header>Site Header</header>
  <main>
    <h1>Main Content</h1>
    <p>This is the main paragraph.</p>
    <p>And a second one.</p>
  </main>
  <footer>Footer text</footer>
  <script>alert('hello')</script>
</body>
</html>""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def markdown_file(tmp_path: Path) -> Path:
    path = tmp_path / "sample.md"
    path.write_text(
        """# Title

First paragraph with some content.

## Section One

Content of section one.

## Section Two

Content of section two.
""",
        encoding="utf-8",
    )
    return path


@pytest.fixture
def latin1_file(tmp_path: Path) -> Path:
    path = tmp_path / "latin1.md"
    path.write_bytes("# Héllo Wörld\n\nSome cöntent.".encode("latin-1"))
    return path


# ── PdfLoader ──────────────────────────────────────────────────────────────────


class TestPdfLoader:
    @staticmethod
    def _make_mock_reader(page_texts: list[str]) -> MagicMock:
        pages = []
        for text in page_texts:
            page = MagicMock()
            page.extract_text.return_value = text
            pages.append(page)
        reader = MagicMock()
        reader.pages = pages
        return reader

    def test_returns_document(self, tmp_path: Path):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")
        reader = self._make_mock_reader(["Page one content.", "Page two content."])
        with patch("src.infrastructure.loaders.pdf_loader.PdfReader", return_value=reader):
            doc = PdfLoader().load(path)
        assert isinstance(doc, Document)

    def test_content_joins_pages(self, tmp_path: Path):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")
        reader = self._make_mock_reader(["First page.", "Second page."])
        with patch("src.infrastructure.loaders.pdf_loader.PdfReader", return_value=reader):
            doc = PdfLoader().load(path)
        assert "First page." in doc.content
        assert "Second page." in doc.content

    def test_metadata_has_required_keys(self, tmp_path: Path):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")
        reader = self._make_mock_reader(["text"])
        with patch("src.infrastructure.loaders.pdf_loader.PdfReader", return_value=reader):
            doc = PdfLoader().load(path)
        assert doc.metadata["filename"] == "test.pdf"
        assert doc.metadata["extension"] == ".pdf"
        assert doc.metadata["loader"] == "pdf"
        assert doc.metadata["page_count"] == 1

    def test_source_is_absolute_path(self, tmp_path: Path):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")
        reader = self._make_mock_reader(["text"])
        with patch("src.infrastructure.loaders.pdf_loader.PdfReader", return_value=reader):
            doc = PdfLoader().load(path)
        assert Path(doc.source).is_absolute()

    def test_none_page_text_handled(self, tmp_path: Path):
        path = tmp_path / "test.pdf"
        path.write_bytes(b"%PDF-1.4 placeholder")
        page = MagicMock()
        page.extract_text.return_value = None
        reader = MagicMock()
        reader.pages = [page]
        with patch("src.infrastructure.loaders.pdf_loader.PdfReader", return_value=reader):
            doc = PdfLoader().load(path)
        assert isinstance(doc.content, str)

    def test_wraps_exception_as_document_load_error(self, tmp_path: Path):
        path = tmp_path / "bad.pdf"
        path.write_bytes(b"not a pdf")
        with patch(
            "src.infrastructure.loaders.pdf_loader.PdfReader",
            side_effect=ValueError("bad pdf"),
        ), pytest.raises(DocumentLoadError) as exc_info:
            PdfLoader().load(path)
        assert exc_info.value.cause is not None


# ── DocxLoader ─────────────────────────────────────────────────────────────────


class TestDocxLoader:
    def test_returns_document(self, docx_file: Path):
        assert isinstance(DocxLoader().load(docx_file), Document)

    def test_content_contains_paragraphs(self, docx_file: Path):
        doc = DocxLoader().load(docx_file)
        assert "first paragraph" in doc.content.lower()
        assert "second paragraph" in doc.content.lower()

    def test_metadata_filename(self, docx_file: Path):
        doc = DocxLoader().load(docx_file)
        assert doc.metadata["filename"] == "sample.docx"
        assert doc.metadata["extension"] == ".docx"
        assert doc.metadata["loader"] == "docx"

    def test_metadata_sections(self, docx_file: Path):
        doc = DocxLoader().load(docx_file)
        assert "Introduction" in doc.metadata["sections"]
        assert "Details" in doc.metadata["sections"]

    def test_source_is_absolute(self, docx_file: Path):
        doc = DocxLoader().load(docx_file)
        assert Path(doc.source).is_absolute()

    def test_missing_file_raises_document_load_error(self, tmp_path: Path):
        with pytest.raises(DocumentLoadError):
            DocxLoader().load(tmp_path / "ghost.docx")


# ── HtmlLoader ─────────────────────────────────────────────────────────────────


class TestHtmlLoader:
    def test_returns_document(self, html_file: Path):
        assert isinstance(HtmlLoader().load(html_file), Document)

    def test_main_content_preserved(self, html_file: Path):
        doc = HtmlLoader().load(html_file)
        assert "Main Content" in doc.content
        assert "main paragraph" in doc.content.lower()

    def test_boilerplate_stripped(self, html_file: Path):
        doc = HtmlLoader().load(html_file)
        assert "Navigation links" not in doc.content
        assert "Site Header" not in doc.content
        assert "Footer text" not in doc.content
        assert "alert" not in doc.content
        assert "color:red" not in doc.content

    def test_title_in_metadata(self, html_file: Path):
        doc = HtmlLoader().load(html_file)
        assert doc.metadata["title"] == "Test Page"

    def test_metadata_keys(self, html_file: Path):
        doc = HtmlLoader().load(html_file)
        assert doc.metadata["loader"] == "html"
        assert doc.metadata["extension"] == ".html"

    def test_latin1_fallback(self, tmp_path: Path):
        path = tmp_path / "latin1.html"
        path.write_bytes(b"<html><body><p>Caf\xe9</p></body></html>")
        doc = HtmlLoader().load(path)
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0

    def test_missing_file_raises_document_load_error(self, tmp_path: Path):
        with pytest.raises(DocumentLoadError):
            HtmlLoader().load(tmp_path / "ghost.html")


# ── MarkdownLoader ─────────────────────────────────────────────────────────────


class TestMarkdownLoader:
    def test_returns_document(self, markdown_file: Path):
        assert isinstance(MarkdownLoader().load(markdown_file), Document)

    def test_raw_content_preserved(self, markdown_file: Path):
        doc = MarkdownLoader().load(markdown_file)
        assert "# Title" in doc.content
        assert "## Section One" in doc.content

    def test_headings_in_metadata(self, markdown_file: Path):
        doc = MarkdownLoader().load(markdown_file)
        assert doc.metadata["heading_count"] == 3
        assert "Title" in doc.metadata["headings"]
        assert "Section One" in doc.metadata["headings"]
        assert "Section Two" in doc.metadata["headings"]

    def test_metadata_keys(self, markdown_file: Path):
        doc = MarkdownLoader().load(markdown_file)
        assert doc.metadata["loader"] == "markdown"
        assert doc.metadata["extension"] == ".md"

    def test_latin1_fallback(self, latin1_file: Path):
        doc = MarkdownLoader().load(latin1_file)
        assert isinstance(doc.content, str)
        assert len(doc.content) > 0

    def test_missing_file_raises_document_load_error(self, tmp_path: Path):
        with pytest.raises(DocumentLoadError):
            MarkdownLoader().load(tmp_path / "ghost.md")


# ── load_document factory ──────────────────────────────────────────────────────


class TestLoadDocument:
    def test_dispatches_to_markdown_loader(self, markdown_file: Path):
        doc = load_document(markdown_file)
        assert doc.metadata["loader"] == "markdown"

    def test_dispatches_to_html_loader(self, html_file: Path):
        doc = load_document(html_file)
        assert doc.metadata["loader"] == "html"

    def test_dispatches_to_docx_loader(self, docx_file: Path):
        doc = load_document(docx_file)
        assert doc.metadata["loader"] == "docx"

    def test_unsupported_extension_raises(self, tmp_path: Path):
        path = tmp_path / "file.xyz"
        path.write_text("content")
        with pytest.raises(DocumentLoadError, match="Unsupported"):
            load_document(path)

    def test_htm_extension_uses_html_loader(self, tmp_path: Path):
        path = tmp_path / "page.htm"
        path.write_text("<html><body><p>hello</p></body></html>")
        doc = load_document(path)
        assert doc.metadata["loader"] == "html"

    def test_dot_markdown_extension(self, tmp_path: Path):
        path = tmp_path / "readme.markdown"
        path.write_text("# Hello\n\nWorld.")
        doc = load_document(path)
        assert doc.metadata["loader"] == "markdown"
