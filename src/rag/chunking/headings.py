"""Section-boundary splitting helpers for SectionChunker (T-240)."""

from __future__ import annotations

from typing import Any, NamedTuple

from src.core.markdown_headings import iter_atx_heading_matches

_SLIDE_SEPARATOR = "\n\n---\n\n"


class SectionSegment(NamedTuple):
    """A contiguous document span with an optional section title."""

    title: str | None
    body: str


def _outline_titles(metadata: dict[str, Any]) -> list[str]:
    """Prefer layout "sections" then Markdown "headings" outlines."""
    for key in ("sections", "headings"):
        value = metadata.get(key)
        if isinstance(value, list) and value:
            return [str(item).strip() for item in value if str(item).strip()]
    return []


def split_markdown_sections(content: str) -> list[SectionSegment] | None:
    """Split *content* on ATX headings outside fenced code blocks.

    Returns "None" when no real (non-fenced) headings exist.
    """
    matches = list(iter_atx_heading_matches(content))
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


def _first_nonempty_line(body: str) -> str:
    return next((line.strip() for line in body.splitlines() if line.strip()), "")


def split_pptx_slide_records(slides: Any) -> list[SectionSegment] | None:
    """Build segments from PptxLoader "slides" records (title and text per slide).

    Titles are loader-authoritative, so agenda lines cannot steal later titles, and
    intra-slide "---" text cannot invent fake boundaries.
    """
    if not isinstance(slides, list) or not slides:
        return None

    segments: list[SectionSegment] = []
    for item in slides:
        if not isinstance(item, dict):
            continue
        body = str(item.get("text") or "").strip()
        if not body:
            continue
        raw_title = item.get("title")
        title = str(raw_title).strip() if raw_title else None
        if not title:
            title = _first_nonempty_line(body) or None
        segments.append(SectionSegment(title=title, body=body))
    return segments or None


def split_slide_sections(
    content: str,
    titles: list[str],
) -> list[SectionSegment] | None:
    """Split PPTX-style content on "---" slide separators (string fallback).

    Prefer "split_pptx_slide_records" when loader metadata includes "slides".
    Titles are matched only when the slide's first non-empty line equals the next
    unused outline title — never by scanning deeper body lines (agenda slides).
    """
    if _SLIDE_SEPARATOR not in content:
        return None

    pending = [title for title in titles if title]
    segments: list[SectionSegment] = []
    for slide in content.split(_SLIDE_SEPARATOR):
        body = slide.strip()
        if not body:
            continue
        first_line = _first_nonempty_line(body)
        if pending and first_line == pending[0]:
            title: str | None = pending.pop(0)
        else:
            title = first_line or None
        segments.append(SectionSegment(title=title, body=body))
    return segments or None


def iter_section_segments(
    content: str,
    metadata: dict[str, Any] | None = None,
) -> list[SectionSegment]:
    """Split *content* into section segments using the best available boundaries.

    Priority:
    1. PptxLoader "slides" records (authoritative per-slide titles and bodies)
    2. Markdown ATX headings in the body (Markdown / Docling export; skips fences)
    3. PPTX "---" slide separators when "loader" is "pptx" (string fallback)
    4. Outline titles as whole lines (plain DOCX)
    5. Single segment spanning the full document
    """
    text = content if content is not None else ""
    meta = metadata or {}

    # Prefer loader-authored slide records over ATX scans of joined deck text.
    # PPTX body lines like "# Key Points" must not invent Markdown sections.
    pptx_records = split_pptx_slide_records(meta.get("slides") or [])
    if pptx_records is not None:
        return pptx_records

    markdown = split_markdown_sections(text)
    if markdown is not None:
        return markdown

    titles = _outline_titles(meta)

    # Gate string "---" splitting on PPTX provenance, so DOCX/Markdown horizontal
    # rules (`---` paragraphs joined as "\\n\\n---\\n\\n") are not treated as slides.
    if meta.get("loader") == "pptx":
        slides = split_slide_sections(text, titles)
        if slides is not None:
            return slides

    outline = split_outline_title_sections(text, titles)
    if outline is not None:
        return outline

    body = text.strip()
    if not body:
        return []
    return [SectionSegment(title=None, body=body)]
