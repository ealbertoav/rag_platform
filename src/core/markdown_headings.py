"""Shared Markdown ATX heading helpers (loaders and section chunking)."""

from __future__ import annotations

import re
from collections.abc import Iterator

# ATX headings (# … ######). Shared by MarkdownLoader and SectionChunker (T-240).
HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)

# CommonMark fenced code: 0–3 spaces, then ≥3 backticks or tildes.
_FENCE_LINE_RE = re.compile(r"^( {0,3})(`{3,}|~{3,})(.*)$")


def _fenced_code_ranges(text: str) -> list[tuple[int, int]]:
    """Return "[start, end)" spans for fenced code blocks (incl. unclosed)."""
    ranges: list[tuple[int, int]] = []
    in_fence = False
    fence_char = ""
    fence_len = 0
    fence_start = 0
    offset = 0

    for line in text.splitlines(keepends=True):
        line_body = line.rstrip("\r\n")
        match = _FENCE_LINE_RE.match(line_body)
        if match:
            marker = match.group(2)
            char, length = marker[0], len(marker)
            info = match.group(3)
            if not in_fence:
                # Backtick fences reject info strings that contain backticks.
                if not (char == "`" and "`" in info):
                    in_fence = True
                    fence_char = char
                    fence_len = length
                    fence_start = offset
            elif char == fence_char and length >= fence_len and not info.strip():
                ranges.append((fence_start, offset + len(line)))
                in_fence = False
        offset += len(line)

    if in_fence:
        ranges.append((fence_start, len(text)))
    return ranges


def _in_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= pos < end for start, end in ranges)


def iter_atx_heading_matches(text: str) -> Iterator[re.Match[str]]:
    """Yield ATX heading matches that fall outside fenced code blocks."""
    fence_ranges = _fenced_code_ranges(text)
    for match in HEADING_RE.finditer(text):
        if not _in_ranges(match.start(), fence_ranges):
            yield match


def extract_markdown_headings(text: str) -> list[str]:
    """Return bare heading titles from ATX Markdown in *text* (skip fences)."""
    return [match.group(1).strip() for match in iter_atx_heading_matches(text)]
