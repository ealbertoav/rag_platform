"""Shared Markdown ATX heading helpers (loaders + section chunking)."""

from __future__ import annotations

import re

# ATX headings (# … ######). Shared by MarkdownLoader and SectionChunker (T-240).
HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def extract_markdown_headings(text: str) -> list[str]:
    """Return bare heading titles from ATX markdown in *text*."""
    return HEADING_RE.findall(text)
