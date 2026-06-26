from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Protocol

from src.core.constants import CHUNK_RAW_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document

_TEMPLATE_PATH = Path(__file__).parents[2] / "prompts" / "ingestion" / "chunk_header_template.txt"
_MISSING = "—"


class _Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...


def load_header_template(path: Path | None = None) -> Template:
    """Load the contextual header line template from the disk."""
    template_path = path or _TEMPLATE_PATH
    return Template(template_path.read_text(encoding="utf-8").strip())


def _document_title(document: Document, chunk: Chunk) -> str:
    return str(chunk.metadata.get("filename") or Path(document.source).name)


def _section_label(chunk: Chunk) -> str:
    section = chunk.metadata.get("section")
    if section:
        return str(section)
    sections = chunk.metadata.get("sections")
    if isinstance(sections, list) and sections:
        return str(sections[0])
    headings = chunk.metadata.get("headings")
    if isinstance(headings, list) and headings:
        return str(headings[0])
    return _MISSING


def _page_label(chunk: Chunk) -> str:
    page = chunk.metadata.get("page")
    if page is None:
        return _MISSING
    return str(page)


def build_header_line(
    document: Document,
    chunk: Chunk,
    template: Template | None = None,
) -> str:
    """Build the bracketed header line from document/chunk metadata."""
    tmpl = template or load_header_template()
    return tmpl.substitute(
        document=_document_title(document, chunk),
        section=_section_label(chunk),
        page=_page_label(chunk),
    ).strip()


def prepend_headers(
    document: Document,
    chunk: Chunk,
    template: Template | None = None,
) -> str:
    """Return a chunk text prefixed with a contextual header for embedding."""
    header = build_header_line(document, chunk, template)
    return f"{header}\n{chunk.text}"


def chunk_context_text(
    chunk: Chunk,
    *,
    exclude_from_llm_context: bool | None = None,
) -> str:
    """Return text for LLM context, optionally stripping the CCH prefix."""
    if exclude_from_llm_context is None:
        from src.core.settings import settings

        exclude_from_llm_context = settings.chunking.contextual_headers.exclude_from_llm_context

    if not exclude_from_llm_context:
        return chunk.text

    raw = chunk.metadata.get(CHUNK_RAW_TEXT_KEY)
    return raw if isinstance(raw, str) else chunk.text


class ContextualHeadersChunker:
    """Decorator that prepends document context headers before embedding."""

    def __init__(self, inner: _Chunker, template: Template | None = None) -> None:
        self._inner = inner
        self._template = template

    def chunk(self, document: Document) -> list[Chunk]:
        return [self._apply_headers(document, chunk) for chunk in self._inner.chunk(document)]

    def _apply_headers(self, document: Document, chunk: Chunk) -> Chunk:
        raw_text = chunk.text
        prefixed = prepend_headers(document, chunk, self._template)
        return chunk.model_copy(
            update={
                "text": prefixed,
                "metadata": {**chunk.metadata, CHUNK_RAW_TEXT_KEY: raw_text},
            }
        )
