"""Section-boundary splitting helpers for SectionChunker (T-240)."""

from __future__ import annotations

from typing import Any, NamedTuple

from src.core.markdown_headings import HEADING_RE

_SLIDE_SEPARATOR = "\n\n---\n\n"


class SectionSegment(NamedTuple):
    """A contiguous document span with an optional section title."""

    title: str | None
    body: str


def _outline_titles(metadata: dict[str, Any]) -> list[str]:
    """Prefer layout "sections" then markdown "headings" outlines."""
    for key in ("sections", "headings"):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            return [str(item).strip() for item in value if str(item).strip()]
    return []


def split_markdown_sections(content: str) -> list[SectionSegment] | None:
    """Split *content* on ATX headings. Returns "None" when no headings exist."""
    matches = list(HEADING_RE.finditer(content))
    if not matches:
        return None

    segments: list[SectionSegment] = []
    preamble = content[: matches[0].start()].strip()
    if preamble:
        segments.append(SectionSegment(title=None, body=preamble))

    for index, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.start()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(content)
        body = content[start:end].strip()
        if body:
            segments.append(SectionSegment(title=title, body=body))
    return segments


def split_outline_title_sections(
    content: str,
    titles: list[str],
) -> list[SectionSegment] | None:
    """Split when outline titles appear as whole lines (plain DOCX-style text).

    Titles are matched in outline order by scanning forward for exact whole-line
    matches. Returns "None" when no title can be located.
    """
    if not titles or not content.strip():
        return None

    lines = content.splitlines(keepends=True)
    positions: list[tuple[int, str]] = []
    search_from = 0

    for title in titles:
        for line_index in range(search_from, len(lines)):
            if lines[line_index].strip() == title:
                positions.append((line_index, title))
                search_from = line_index + 1
                break

    if not positions:
        return None

    segments: list[SectionSegment] = []
    first_line, _ = positions[0]
    preamble = "".join(lines[:first_line]).strip()
    if preamble:
        segments.append(SectionSegment(title=None, body=preamble))

    for index, (line_index, title) in enumerate(positions):
        end_line = positions[index + 1][0] if index + 1 < len(positions) else len(lines)
        body = "".join(lines[line_index:end_line]).strip()
        if body:
            segments.append(SectionSegment(title=title, body=body))
    return segments


def split_slide_sections(
    content: str,
    titles: list[str],
) -> list[SectionSegment] | None:
    """Split PPTX-style content on "---" slide separators."""
    if _SLIDE_SEPARATOR not in content:
        return None

    slides = content.split(_SLIDE_SEPARATOR)
    segments: list[SectionSegment] = []
    for index, slide in enumerate(slides):
        body = slide.strip()
        if not body:
            continue
        title: str | None = titles[index] if index < len(titles) else None
        if title is None:
            first_line = next(
                (line.strip() for line in body.splitlines() if line.strip()),
                "",
            )
            title = first_line or None
        segments.append(SectionSegment(title=title, body=body))
    return segments or None


def iter_section_segments(
    content: str,
    metadata: dict[str, Any] | None = None,
) -> list[SectionSegment]:
    """Split *content* into section segments using the best available boundaries.

    Priority:
    1. Markdown ATX headings in the body (Markdown / Docling export)
    2. Outline titles as whole lines (plain DOCX)
    3. PPTX "---" slide separators
    4. Single segment spanning the full document
    """
    text = content if content is not None else ""
    meta = metadata or {}

    markdown = split_markdown_sections(text)
    if markdown is not None:
        return markdown

    titles = _outline_titles(meta)
    outline = split_outline_title_sections(text, titles)
    if outline is not None:
        return outline

    slides = split_slide_sections(text, titles)
    if slides is not None:
        return slides

    body = text.strip()
    if not body:
        return []
    return [SectionSegment(title=None, body=body)]
