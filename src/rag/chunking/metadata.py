from __future__ import annotations

from typing import Any

from src.core.constants import CHUNK_SECTION_KEY, LAYOUT_DOCUMENT_METADATA_KEYS


def _resolve_section_label(document_metadata: dict[str, Any]) -> str | None:
    """Derive a single chunk-safe section label from document metadata."""
    existing = document_metadata.get(CHUNK_SECTION_KEY)
    if existing:
        return str(existing)

    sections = document_metadata.get("sections")
    if isinstance(sections, list) and sections:
        return str(sections[0])

    headings = document_metadata.get("headings")
    if isinstance(headings, list) and headings:
        return str(headings[0])

    return None


def chunk_metadata(document_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return document metadata safe to spread onto every indexed chunk.

    Layout parsers attach document-level structures (tables, figures, section
    outlines) for downstream multimodal chunking (T-202+). Those keys must not
    be copied into each text chunk payload. Plain loaders (DOCX, Markdown) store
    section outlines in "sections" / "headings"; the first title is promoted
    to "CHUNK_SECTION_KEY" so contextual headers keep working when layout
    parsing is disabled.
    """
    filtered = {
        key: value
        for key, value in document_metadata.items()
        if key not in LAYOUT_DOCUMENT_METADATA_KEYS
    }
    section = _resolve_section_label(document_metadata)
    if section is not None:
        filtered[CHUNK_SECTION_KEY] = section
    return filtered
