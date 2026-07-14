"""T-240 — section-boundary chunker tests."""

from __future__ import annotations

import pytest

from src.core.constants import CHUNK_INDEX_KEY, CHUNK_SECTION_KEY, CHUNK_SOURCE_KEY
from src.core.markdown_headings import extract_markdown_headings
from src.domain.entities.document import Document
from src.rag.chunking import get_chunker
from src.rag.chunking.contextual_headers import build_header_line
from src.rag.chunking.headings import (
    SectionSegment,
    iter_section_segments,
    split_markdown_sections,
    split_outline_title_sections,
    split_pptx_slide_records,
    split_slide_sections,
)
from src.rag.chunking.section_chunker import SectionChunker

_PARA = "word " * 120  # ~120 tokens


def _doc(
    content: str,
    *,
    source: str = "test.md",
    metadata: dict[str, object] | None = None,
) -> Document:
    return Document(source=source, content=content, metadata=metadata or {})


def _approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ── markdown heading helpers ───────────────────────────────────────────────────


class TestMarkdownHeadings:
    def test_extract_titles(self):
        text = "# Title\n\nbody\n\n## Section\n\nmore"
        assert extract_markdown_headings(text) == ["Title", "Section"]


class TestSectionSplitHelpers:
    def test_markdown_split_with_preamble(self):
        content = "Intro blurb.\n\n# Alpha\n\nA text.\n\n## Beta\n\nB text."
        segments = split_markdown_sections(content)
        assert segments is not None
        assert segments[0] == SectionSegment(title=None, body="Intro blurb.")
        assert segments[1].title == "Alpha"
        assert "# Alpha" in segments[1].body
        assert segments[2].title == "Beta"

    def test_markdown_split_returns_none_without_headings(self):
        assert split_markdown_sections("plain paragraph") is None

    def test_outline_split_docx_style(self):
        content = "Preface.\n\nIntroduction\n\nIntro body.\n\nDetails\n\nDetail body."
        segments = split_outline_title_sections(
            content,
            ["Introduction", "Details"],
        )
        assert segments is not None
        assert segments[0].title is None
        assert segments[0].body == "Preface."
        assert segments[1].title == "Introduction"
        assert "Intro body." in segments[1].body
        assert segments[2].title == "Details"

    def test_outline_split_returns_none_when_titles_missing(self):
        assert split_outline_title_sections("no titles here", ["Introduction"]) is None
        assert split_outline_title_sections("", ["Introduction"]) is None
        assert split_outline_title_sections("x", []) is None

    def test_outline_skips_unmatched_mid_titles(self):
        content = "Introduction\n\nbody\n\nDetails\n\nmore"
        segments = split_outline_title_sections(
            content,
            ["Introduction", "Missing", "Details"],
        )
        assert segments is not None
        assert [s.title for s in segments] == ["Introduction", "Details"]

    def test_slide_split_with_titles(self):
        content = "Slide One\n\nbody A\n\n---\n\nSlide Two\n\nbody B"
        segments = split_slide_sections(content, ["Slide One", "Slide Two"])
        assert segments is not None
        assert len(segments) == 2
        assert segments[0].title == "Slide One"
        assert segments[1].title == "Slide Two"

    def test_slide_split_infers_title_from_first_line(self):
        content = "Alpha Title\n\nbody\n\n---\n\nBeta Title\n\nmore"
        segments = split_slide_sections(content, [])
        assert segments is not None
        assert segments[0].title == "Alpha Title"
        assert segments[1].title == "Beta Title"

    def test_slide_split_returns_none_without_separator(self):
        assert split_slide_sections("no slides", ["A"]) is None

    def test_slide_split_skips_empty_slides(self):
        # Compact title lists omit empty slides; "Mid" must not attach to "Last".
        content = "Only\n\n---\n\n\n\n---\n\nLast"
        segments = split_slide_sections(content, ["Only", "Mid", "Last"])
        assert segments is not None
        assert [(s.title, s.body) for s in segments] == [
            ("Only", "Only"),
            ("Last", "Last"),
        ]

    def test_slide_split_untitled_middle_does_not_steal_next_title(self):
        # PptxLoader sections omit untitled slides → titles are not index-aligned.
        content = (
            "Introduction\n\nWelcome.\n\n"
            "---\n\n"
            "Agenda bullet one\n\n"
            "---\n\n"
            "Details\n\nMore detail."
        )
        segments = split_slide_sections(content, ["Introduction", "Details"])
        assert segments is not None
        assert [s.title for s in segments] == [
            "Introduction",
            "Agenda bullet one",
            "Details",
        ]
        assert "Welcome." in segments[0].body
        assert "Agenda bullet one" in segments[1].body
        assert "More detail." in segments[2].body

    def test_slide_split_ignores_pending_title_deeper_in_body(self):
        # Agenda / outline slides list upcoming titles as body lines — must not steal.
        content = (
            "Introduction\n\nWelcome.\n\n"
            "---\n\n"
            "Agenda\n\nDetails\n\nNext steps\n\n"
            "---\n\n"
            "Details\n\nMore detail."
        )
        segments = split_slide_sections(content, ["Introduction", "Details"])
        assert segments is not None
        assert [s.title for s in segments] == [
            "Introduction",
            "Agenda",
            "Details",
        ]
        assert "More detail." in segments[2].body

    def test_slide_split_first_line_only_for_pending_titles(self):
        # Subtitle-first slides keep the first-line label; pending titles stay unused
        # unless the first line matches (loader "slides" records handle real titles).
        content = "Subtitle\n\nIntroduction\n\nbody\n\n---\n\nDetails\n\nmore"
        segments = split_slide_sections(content, ["Introduction", "Details"])
        assert segments is not None
        assert segments[0].title == "Subtitle"
        assert segments[1].title == "Details"

    def test_pptx_records_use_loader_titles(self):
        segments = split_pptx_slide_records(
            [
                {"title": "Introduction", "text": "Subtitle\n\nIntroduction\n\nbody"},
                {"title": None, "text": "Agenda\n\nDetails\n\nNext"},
                {"title": "Details", "text": "Details\n\nMore."},
            ]
        )
        assert segments is not None
        assert [s.title for s in segments] == [
            "Introduction",
            "Agenda",
            "Details",
        ]
        assert "Agenda\n\nDetails" in segments[1].body

    def test_pptx_records_keep_intra_slide_hr_intact(self):
        body = "Title\n\nBefore\n\n---\n\nAfter"
        segments = split_pptx_slide_records([{"title": "Title", "text": body}])
        assert segments is not None
        assert len(segments) == 1
        assert segments[0].body == body
        assert segments[0].title == "Title"

    def test_pptx_records_skip_invalid_and_empty(self):
        assert split_pptx_slide_records([]) is None
        assert split_pptx_slide_records("not-a-list") is None
        segments = split_pptx_slide_records(
            [
                "skip-me",
                {"title": "Keep", "text": ""},
                {"title": "Real", "text": "  body  "},
            ]
        )
        assert segments == [SectionSegment(title="Real", body="body")]

    def test_iter_prefers_markdown_over_outline(self):
        content = "# MD Heading\n\nbody"
        segments = iter_section_segments(
            content,
            {"sections": ["Docx Title"]},
        )
        assert segments[0].title == "MD Heading"

    def test_iter_prefers_pptx_records_over_string_separators(self):
        content = "IgnoredJoined\n\n---\n\nShouldNotSplit"
        segments = iter_section_segments(
            content,
            {
                "loader": "pptx",
                "sections": ["Introduction", "Details"],
                "slides": [
                    {
                        "title": "Introduction",
                        "text": "Introduction\n\nWelcome.\n\n---\n\nStill one slide",
                    },
                    {"title": None, "text": "Agenda\n\nDetails"},
                    {"title": "Details", "text": "Details\n\nMore."},
                ],
            },
        )
        assert len(segments) == 3
        assert [s.title for s in segments] == [
            "Introduction",
            "Agenda",
            "Details",
        ]
        assert "Still one slide" in segments[0].body
        assert "---" in segments[0].body

    def test_iter_prefers_slides_over_outline(self):
        # Titled whole lines would make outline absorb the untitled middle slide.
        content = (
            "Introduction\n\nWelcome.\n\n"
            "---\n\n"
            "Untitled body only\n\n"
            "---\n\n"
            "Details\n\nMore."
        )
        segments = iter_section_segments(
            content,
            {"sections": ["Introduction", "Details"], "loader": "pptx"},
        )
        assert len(segments) == 3
        assert [s.title for s in segments] == [
            "Introduction",
            "Untitled body only",
            "Details",
        ]
        assert "---" not in segments[0].body
        assert "Untitled body only" in segments[1].body

    def test_iter_docx_hr_does_not_trigger_slide_split(self):
        content = "Introduction\n\nBefore rule.\n\n---\n\nAfter rule.\n\nDetails\n\nMore."
        segments = iter_section_segments(
            content,
            {"sections": ["Introduction", "Details"], "loader": "docx"},
        )
        assert [s.title for s in segments] == ["Introduction", "Details"]
        assert "Before rule." in segments[0].body
        assert "---" in segments[0].body
        assert "After rule." in segments[0].body

    def test_iter_falls_back_to_outline(self):
        content = "Introduction\n\nbody text here"
        segments = iter_section_segments(content, {"sections": ["Introduction"]})
        assert segments[0].title == "Introduction"

    def test_iter_falls_back_to_headings_metadata(self):
        content = "Section One\n\nbody"
        segments = iter_section_segments(content, {"headings": ["Section One"]})
        assert segments[0].title == "Section One"

    def test_iter_falls_back_to_slides_when_pptx_loader(self):
        content = "Slide body A\n\n---\n\nSlide body B"
        segments = iter_section_segments(content, {"loader": "pptx"})
        assert len(segments) == 2
        assert segments[0].title == "Slide body A"

    def test_iter_ignores_separators_without_pptx_loader(self):
        content = "Slide body A\n\n---\n\nSlide body B"
        segments = iter_section_segments(content, {})
        assert segments == [
            SectionSegment(title=None, body="Slide body A\n\n---\n\nSlide body B")
        ]

    def test_iter_single_segment_without_boundaries(self):
        segments = iter_section_segments("plain text only", {})
        assert segments == [SectionSegment(title=None, body="plain text only")]

    def test_iter_empty_content(self):
        assert iter_section_segments("   ", {}) == []
        assert iter_section_segments("", None) == []


# ── SectionChunker ─────────────────────────────────────────────────────────────


class TestSectionChunker:
    def test_get_chunker_returns_section(self):
        chunker = get_chunker("section", chunk_size=300, overlap=20)
        assert isinstance(chunker, SectionChunker)

    def test_splits_markdown_with_per_chunk_section(self):
        content = "# Alpha\n\nAlpha body.\n\n## Beta\n\nBeta body."
        chunks = SectionChunker().chunk(_doc(content, metadata={"headings": ["Alpha", "Beta"]}))
        sections = [c.metadata.get(CHUNK_SECTION_KEY) for c in chunks]
        assert "Alpha" in sections
        assert "Beta" in sections
        alpha = next(c for c in chunks if c.metadata.get(CHUNK_SECTION_KEY) == "Alpha")
        beta = next(c for c in chunks if c.metadata.get(CHUNK_SECTION_KEY) == "Beta")
        assert "Alpha body" in alpha.text
        assert "Beta body" in beta.text

    def test_preamble_omits_section_metadata(self):
        content = "Front matter.\n\n# Main\n\nMain body."
        chunks = SectionChunker().chunk(
            _doc(
                content,
                metadata={"section": "Main", "headings": ["Main"]},
            )
        )
        preamble = next(c for c in chunks if "Front matter" in c.text)
        assert CHUNK_SECTION_KEY not in preamble.metadata
        main = next(c for c in chunks if c.metadata.get(CHUNK_SECTION_KEY) == "Main")
        assert "Main body" in main.text

    def test_oversized_section_is_recursively_split(self):
        body = (_PARA + "\n\n") * 8
        content = f"# Huge\n\n{body}"
        chunks = SectionChunker(chunk_size=200, overlap=20).chunk(_doc(content))
        assert len(chunks) > 1
        assert all(c.metadata.get(CHUNK_SECTION_KEY) == "Huge" for c in chunks)
        assert all(_approx_tokens(c.text) <= 200 for c in chunks)

    def test_no_headings_behaves_like_recursive(self):
        content = "Short plain document."
        chunks = SectionChunker().chunk(_doc(content))
        assert len(chunks) == 1
        assert chunks[0].text == content
        assert CHUNK_SECTION_KEY not in chunks[0].metadata

    def test_empty_document_returns_empty(self):
        assert SectionChunker().chunk(_doc("")) == []

    def test_document_id_and_source_and_index(self):
        content = "# A\n\none\n\n# B\n\ntwo"
        doc = _doc(content, source="docs/guide.md")
        chunks = SectionChunker().chunk(doc)
        assert all(c.document_id == doc.id for c in chunks)
        assert all(c.metadata[CHUNK_SOURCE_KEY] == "docs/guide.md" for c in chunks)
        assert [c.metadata[CHUNK_INDEX_KEY] for c in chunks] == list(range(len(chunks)))

    def test_docx_outline_fallback(self):
        content = "Introduction\n\nIntro paragraph.\n\nDetails\n\nDetail paragraph."
        doc = _doc(
            content,
            source="report.docx",
            metadata={"sections": ["Introduction", "Details"], "loader": "docx"},
        )
        chunks = SectionChunker().chunk(doc)
        by_section = {c.metadata.get(CHUNK_SECTION_KEY): c.text for c in chunks}
        assert "Intro paragraph" in by_section["Introduction"]
        assert "Detail paragraph" in by_section["Details"]

    def test_pptx_slide_fallback(self):
        content = "Intro Title\n\nslide body\n\n---\n\nDetails Title\n\nmore body"
        doc = _doc(
            content,
            source="deck.pptx",
            metadata={"sections": ["Intro Title", "Details Title"], "loader": "pptx"},
        )
        chunks = SectionChunker().chunk(doc)
        sections = {c.metadata[CHUNK_SECTION_KEY] for c in chunks}
        assert sections == {"Intro Title", "Details Title"}

    def test_pptx_untitled_middle_slide_keeps_own_section(self):
        content = (
            "Intro Title\n\nslide body\n\n"
            "---\n\n"
            "Bullet without a title placeholder\n\n"
            "---\n\n"
            "Details Title\n\nmore body"
        )
        doc = _doc(
            content,
            source="deck.pptx",
            metadata={"sections": ["Intro Title", "Details Title"], "loader": "pptx"},
        )
        chunks = SectionChunker().chunk(doc)
        by_section = {c.metadata.get(CHUNK_SECTION_KEY): c.text for c in chunks}
        assert "slide body" in by_section["Intro Title"]
        assert "Bullet without a title placeholder" in by_section[
            "Bullet without a title placeholder"
        ]
        assert "more body" in by_section["Details Title"]
        # Untitled middle must not be swallowed by Intro Title (outline path bug).
        assert "Bullet without a title placeholder" not in by_section["Intro Title"]

    def test_pptx_slides_metadata_preserves_hr_and_agenda_titles(self):
        doc = _doc(
            "joined-content-unused-when-slides-present",
            source="deck.pptx",
            metadata={
                "loader": "pptx",
                "sections": ["Intro Title", "Details Title"],
                "slides": [
                    {
                        "title": "Intro Title",
                        "text": "Intro Title\n\nslide body\n\n---\n\nstill intro",
                    },
                    {
                        "title": None,
                        "text": "Agenda\n\nDetails Title\n\nNext",
                    },
                    {"title": "Details Title", "text": "Details Title\n\nmore body"},
                ],
            },
        )
        chunks = SectionChunker().chunk(doc)
        by_section = {c.metadata.get(CHUNK_SECTION_KEY): c.text for c in chunks}
        assert "still intro" in by_section["Intro Title"]
        assert "---" in by_section["Intro Title"]
        assert "Details Title" in by_section["Agenda"]
        assert "more body" in by_section["Details Title"]
        assert "slides" not in chunks[0].metadata

    def test_overlap_validation_delegates_to_recursive(self):
        with pytest.raises(ValueError, match="overlap"):
            SectionChunker(chunk_size=100, overlap=100)

    def test_contextual_headers_use_per_chunk_section(self):
        content = "# Alpha\n\nA.\n\n# Beta\n\nB."
        chunker = get_chunker("section", use_contextual_headers=True, chunk_size=500)
        doc = _doc(content, metadata={"filename": "guide.md", "headings": ["Alpha", "Beta"]})
        chunks = chunker.chunk(doc)
        alpha = next(c for c in chunks if "Alpha" in c.metadata.get(CHUNK_SECTION_KEY, ""))
        header = build_header_line(doc, alpha)
        assert "Section: Alpha" in header
        assert "Section: Beta" not in header


class TestSectionChunkerFactoryErrors:
    def test_unknown_strategy_still_raises(self):
        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            get_chunker("not-a-strategy")
