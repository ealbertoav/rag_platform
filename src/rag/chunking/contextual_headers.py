from __future__ import annotations

from pathlib import Path
from string import Template
from typing import Protocol

from src.core.constants import (
    CHUNK_PARENT_ID_KEY,
    CHUNK_RAW_TEXT_KEY,
    MERGED_CHUNK_IDS_KEY,
    PARENT_CONTEXT_TEXT_KEY,
)
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
    parent_context = chunk.metadata.get(PARENT_CONTEXT_TEXT_KEY)
    if isinstance(parent_context, str) and parent_context:
        return parent_context

    if exclude_from_llm_context is None:
        from src.core.settings import settings

        exclude_from_llm_context = settings.chunking.contextual_headers.exclude_from_llm_context

    if not exclude_from_llm_context:
        return chunk.text

    raw = chunk.metadata.get(CHUNK_RAW_TEXT_KEY)
    return raw if isinstance(raw, str) else chunk.text


def passage_context_key(chunk: Chunk) -> str:
    """Stable key for chunks that share the same LLM-facing passage text."""
    parent_id = chunk.metadata.get(CHUNK_PARENT_ID_KEY)
    parent_context = chunk.metadata.get(PARENT_CONTEXT_TEXT_KEY)
    if (
        isinstance(parent_id, str)
        and parent_id
        and isinstance(parent_context, str)
        and parent_context
    ):
        return parent_id

    merged_ids = chunk.metadata.get(MERGED_CHUNK_IDS_KEY)
    if isinstance(merged_ids, list) and merged_ids:
        return "merged:" + ",".join(sorted(str(chunk_id) for chunk_id in merged_ids))

    return chunk.id


def group_chunks_by_passage(chunks: list[Chunk]) -> list[tuple[Chunk, list[Chunk]]]:
    """Group chunks that map to the same generator-facing passage, preserving order."""
    groups: dict[str, list[Chunk]] = {}
    order: list[str] = []
    for chunk in chunks:
        key = passage_context_key(chunk)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(chunk)
    return [(groups[key][0], groups[key]) for key in order]


def join_chunk_context(chunks: list[Chunk]) -> str:
    """Join chunk passages for LLM prompts, deduplicating shared parent context."""
    parts: list[str] = []
    for representative, _group in group_chunks_by_passage(chunks):
        text = chunk_context_text(representative)
        if text:
            parts.append(text)
    return "\n\n".join(parts)


def format_passages_for_llm(
    chunks: list[Chunk],
    *,
    normalize_newlines: bool = False,
) -> str:
    """Format deduplicated passages for batched quality LLM prompts.

    When *normalize_newlines* is True, newlines are collapsed to spaces (readable
    prose prompts for explain/grade). When False, passage text is preserved for
    verbatim span matching (source highlighting).
    """
    lines: list[str] = []
    for index, (representative, _) in enumerate(group_chunks_by_passage(chunks), start=1):
        text = chunk_context_text(representative).strip()
        if normalize_newlines:
            text = text.replace("\n", " ")
        lines.append(f"[{index}] chunk_id={representative.id}\n{text}")
    return "\n\n".join(lines)


class ContextualHeadersChunker:
    """Decorator that prepends document context headers before embedding."""

    def __init__(self, inner: _Chunker, template: Template | None = None) -> None:
        self._inner: _Chunker = inner
        self._template: Template | None = template

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
