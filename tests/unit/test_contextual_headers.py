"""T-120 — contextual chunk headers tests."""

from __future__ import annotations

from src.core.constants import CHUNK_RAW_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking import get_chunker
from src.rag.chunking.contextual_headers import (
    ContextualHeadersChunker,
    build_header_line,
    chunk_context_text,
    group_chunks_by_passage,
    has_mixed_modality,
    join_chunk_context,
    join_chunk_context_multimodal,
    passage_context_key,
    prepend_headers,
)
from src.rag.chunking.recursive_chunker import RecursiveChunker
from src.type_regression.contextual_headers import (
    check_contextual_headers_api_types,
    check_contextual_headers_chunker_returns_chunks,
    check_multimodal_context_types,
)


def _doc(
    content: str = "Revenue grew 12% year over year.",
    *,
    source: str = "/data/raw/annual_report_2023.pdf",
    metadata: dict | None = None,
) -> Document:
    base_meta = {"filename": "annual_report_2023.pdf", "section": "Revenue", "page": 42}
    if metadata:
        base_meta.update(metadata)
    return Document(source=source, content=content, metadata=base_meta)


def _chunk(text: str = "Revenue grew 12% year over year.", metadata: dict | None = None) -> Chunk:
    meta = {"filename": "annual_report_2023.pdf", "section": "Revenue", "page": 42}
    if metadata:
        meta.update(metadata)
    return Chunk(document_id="doc-1", text=text, metadata=meta)


class TestBuildHeaderLine:
    def test_uses_loader_metadata(self):
        doc = _doc()
        chunk = _chunk()
        header = build_header_line(doc, chunk)
        assert "annual_report_2023.pdf" in header
        assert "Revenue" in header
        assert "42" in header

    def test_falls_back_to_source_basename(self):
        doc = Document(source="/tmp/report.md", content="text", metadata={})
        chunk = Chunk(document_id=doc.id, text="text", metadata={})
        header = build_header_line(doc, chunk)
        assert "report.md" in header


class TestPrependHeaders:
    def test_prefixes_chunk_text(self):
        doc = _doc()
        chunk = _chunk()
        result = prepend_headers(doc, chunk)
        assert result.startswith("[Document:")
        assert result.endswith(chunk.text)

    def test_example_format(self):
        doc = _doc()
        chunk = _chunk()
        result = prepend_headers(doc, chunk)
        assert "[Document: annual_report_2023.pdf | Section: Revenue | Page: 42]" in result


class TestContextualHeadersChunker:
    def test_embedded_text_includes_header(self):
        doc = _doc("Body text here.")
        inner = RecursiveChunker(chunk_size=500)
        chunker = ContextualHeadersChunker(inner)
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert chunks[0].text.startswith("[Document:")
        assert "Body text here." in chunks[0].text

    def test_raw_text_preserved_in_metadata(self):
        doc = _doc("Body text here.")
        chunker = ContextualHeadersChunker(RecursiveChunker(chunk_size=500))
        chunks = chunker.chunk(doc)
        assert chunks[0].metadata[CHUNK_RAW_TEXT_KEY] == "Body text here."

    def test_disabled_via_factory_leaves_text_unchanged(self):
        doc = _doc("Plain chunk.")
        chunker = get_chunker("recursive", use_contextual_headers=False, chunk_size=500)
        chunks = chunker.chunk(doc)
        assert chunks[0].text == "Plain chunk."
        assert CHUNK_RAW_TEXT_KEY not in chunks[0].metadata

    def test_enabled_via_factory_applies_headers(self):
        doc = _doc("Plain chunk.")
        chunker = get_chunker("recursive", use_contextual_headers=True, chunk_size=500)
        chunks = chunker.chunk(doc)
        assert chunks[0].text.startswith("[Document:")
        assert chunks[0].metadata[CHUNK_RAW_TEXT_KEY] == "Plain chunk."


class TestChunkContextText:
    def test_prefers_raw_text_when_present(self):
        chunk = Chunk(
            document_id="d1",
            text="[Document: x]\nActual content.",
            metadata={CHUNK_RAW_TEXT_KEY: "Actual content."},
        )
        assert chunk_context_text(chunk) == "Actual content."

    def test_falls_back_to_chunk_text(self):
        chunk = Chunk(document_id="d1", text="No header applied.")
        assert chunk_context_text(chunk) == "No header applied."

    def test_empty_parent_context_falls_back_to_child_text(self):
        from src.core.constants import CHUNK_PARENT_ID_KEY, PARENT_CONTEXT_TEXT_KEY

        chunk = Chunk(
            document_id="d1",
            text="child fallback.",
            metadata={
                CHUNK_PARENT_ID_KEY: "parent-0",
                PARENT_CONTEXT_TEXT_KEY: "",
            },
        )
        assert chunk_context_text(chunk) == "child fallback."

    def test_exclude_false_returns_prefixed_text(self):
        chunk = Chunk(
            document_id="d1",
            text="[Document: x]\nActual content.",
            metadata={CHUNK_RAW_TEXT_KEY: "Actual content."},
        )
        assert (
            chunk_context_text(chunk, exclude_from_llm_context=False)
            == "[Document: x]\nActual content."
        )

    def test_exclude_true_strips_header(self):
        chunk = Chunk(
            document_id="d1",
            text="[Document: x]\nActual content.",
            metadata={CHUNK_RAW_TEXT_KEY: "Actual content."},
        )
        assert chunk_context_text(chunk, exclude_from_llm_context=True) == "Actual content."


class TestJoinChunkContext:
    def test_joins_distinct_chunks(self):
        chunks = [
            Chunk(document_id="d1", text="first passage."),
            Chunk(document_id="d1", text="second passage."),
        ]
        assert join_chunk_context(chunks) == "first passage.\n\nsecond passage."

    def test_deduplicates_siblings_with_same_parent(self):
        from src.core.constants import CHUNK_PARENT_ID_KEY, PARENT_CONTEXT_TEXT_KEY

        parent_text = "shared parent passage."
        chunks = [
            Chunk(
                id="child-0",
                document_id="d1",
                text="child one.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
            Chunk(
                id="child-1",
                document_id="d1",
                text="child two.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
        ]
        assert join_chunk_context(chunks) == parent_text

    def test_keeps_distinct_children_without_parent_context(self):
        from src.core.constants import CHUNK_PARENT_ID_KEY

        chunks = [
            Chunk(
                id="child-0",
                document_id="d1",
                text="child one slice.",
                metadata={CHUNK_PARENT_ID_KEY: "parent-0"},
            ),
            Chunk(
                id="child-1",
                document_id="d1",
                text="child two slice.",
                metadata={CHUNK_PARENT_ID_KEY: "parent-0"},
            ),
        ]
        assert join_chunk_context(chunks) == "child one slice.\n\nchild two slice."

    def test_deduplicates_parent_hit_when_enriched_child_present(self):
        from src.core.constants import CHUNK_PARENT_ID_KEY, PARENT_CONTEXT_TEXT_KEY

        parent_text = "shared parent passage."
        chunks = [
            Chunk(
                id="parent-0",
                document_id="d1",
                text=parent_text,
            ),
            Chunk(
                id="child-0",
                document_id="d1",
                text="child slice.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
        ]
        assert join_chunk_context(chunks) == parent_text

    def test_deduplicates_parent_hit_after_enriched_child(self):
        from src.core.constants import CHUNK_PARENT_ID_KEY, PARENT_CONTEXT_TEXT_KEY

        parent_text = "shared parent passage."
        chunks = [
            Chunk(
                id="child-0",
                document_id="d1",
                text="child slice.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: parent_text,
                },
            ),
            Chunk(
                id="parent-0",
                document_id="d1",
                text=parent_text,
            ),
        ]
        assert join_chunk_context(chunks) == parent_text

    def test_empty_parent_context_keeps_sibling_child_text(self):
        from src.core.constants import CHUNK_PARENT_ID_KEY, PARENT_CONTEXT_TEXT_KEY

        chunks = [
            Chunk(
                id="child-0",
                document_id="d1",
                text="child one slice.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: "",
                },
            ),
            Chunk(
                id="child-1",
                document_id="d1",
                text="child two slice.",
                metadata={
                    CHUNK_PARENT_ID_KEY: "parent-0",
                    PARENT_CONTEXT_TEXT_KEY: "",
                },
            ),
        ]
        assert join_chunk_context(chunks) == "child one slice.\n\nchild two slice."


class TestModalityContext:
    """T-270 — mixed-modality context helpers."""

    def test_single_modality_is_not_mixed(self):
        chunks = [
            Chunk(document_id="d1", text="first passage."),
            Chunk(document_id="d1", text="second passage."),
        ]
        assert has_mixed_modality(chunks) is False

    def test_multiple_modalities_is_mixed(self):
        from src.core.constants import MODALITY_TABLE

        chunks = [
            Chunk(document_id="d1", text="prose passage."),
            Chunk(document_id="d1", text="| a | b |", modality=MODALITY_TABLE),
        ]
        assert has_mixed_modality(chunks) is True

    def test_single_chunk_is_not_mixed(self):
        assert has_mixed_modality([Chunk(document_id="d1", text="solo passage.")]) is False

    def test_labels_each_passage_by_modality(self):
        from src.core.constants import MODALITY_CAPTION, MODALITY_TABLE

        chunks = [
            Chunk(document_id="d1", text="prose passage."),
            Chunk(document_id="d1", text="| a | b |", modality=MODALITY_TABLE),
            Chunk(document_id="d1", text="a chart of revenue.", modality=MODALITY_CAPTION),
        ]
        assert join_chunk_context_multimodal(chunks) == (
            "[TEXT]\nprose passage.\n\n[TABLE]\n| a | b |\n\n[FIGURE CAPTION]\na chart of revenue."
        )

    def test_unknown_modality_falls_back_to_upper_name(self):
        chunk = Chunk(document_id="d1", text="custom passage.", modality="diagram")
        assert join_chunk_context_multimodal([chunk]) == "[DIAGRAM]\ncustom passage."


class TestPassageContextKey:
    def test_merged_chunks_share_key(self):
        from src.core.constants import MERGED_CHUNK_IDS_KEY

        chunk = Chunk(
            id="merged-1",
            document_id="d1",
            text="combined passage.",
            metadata={MERGED_CHUNK_IDS_KEY: ["c1", "c0"]},
        )
        assert passage_context_key(chunk) == "merged:c0,c1"


class TestGroupChunksByPassage:
    def test_groups_merged_source_copies(self):
        from src.core.constants import MERGED_CHUNK_IDS_KEY

        chunk_a = Chunk(
            id="c0",
            document_id="d1",
            text="combined passage.",
            metadata={MERGED_CHUNK_IDS_KEY: ["c0", "c1"]},
        )
        chunk_b = Chunk(
            id="c1",
            document_id="d1",
            text="combined passage.",
            metadata={MERGED_CHUNK_IDS_KEY: ["c0", "c1"]},
        )
        groups = group_chunks_by_passage([chunk_a, chunk_b])
        assert len(groups) == 1
        assert {chunk.id for chunk in groups[0][1]} == {"c0", "c1"}


class TestTypeRegressionFixtures:
    """Runtime checks for src/type_regression — mypy validates those modules at lint time."""

    def test_contextual_headers_api_types(self) -> None:
        header, prefixed, context, key, groups, joined = check_contextual_headers_api_types(
            _doc(), _chunk()
        )
        assert header and prefixed and context and key and groups and joined

    def test_contextual_headers_chunker_returns_chunks(self) -> None:
        chunks = check_contextual_headers_chunker_returns_chunks(_doc("Body text here."))
        assert len(chunks) >= 1
        assert isinstance(chunks[0].text, str)

    def test_multimodal_context_types(self) -> None:
        mixed, joined = check_multimodal_context_types([_chunk()])
        assert mixed is False
        assert joined.startswith("[TEXT]")
