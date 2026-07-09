from __future__ import annotations

from typing import Any

from src.core.constants import LAYOUT_DOCUMENT_METADATA_KEYS


def chunk_metadata(document_metadata: dict[str, Any]) -> dict[str, Any]:
    """Return document metadata safe to spread onto every indexed chunk.

    Layout parsers attach document-level structures (tables, figures, section
    outlines) for downstream multimodal chunking (T-202+). Those keys must not
    be copied into each text chunk payload.
    """
    return {
        key: value
        for key, value in document_metadata.items()
        if key not in LAYOUT_DOCUMENT_METADATA_KEYS
    }
