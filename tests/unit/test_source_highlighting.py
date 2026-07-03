"""T-144 — Source highlighting in answers tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.core.constants import CHUNK_PARENT_ID_KEY, MERGED_CHUNK_IDS_KEY, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.rag.chunking.contextual_headers import chunk_context_text
from src.rag.quality.source_highlighting import (
    extract_highlights,
    parse_source_highlighting,
)


def _chunk(
    chunk_id: str,
    text: str = "sample text",
    *,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", text=text, metadata=metadata or {})


def _answer(text: str = "Revenue grew 12% in Q3.") -> Answer:
    return Answer(query_id="q-1", text=text, sources=["c0", "c1"])


def _highlights_json(items: list[dict[str, object]]) -> str:
    return json.dumps({"highlights": items})


class TestParseSourceHighlighting:
    def test_parses_clean_json(self):
        payload = _highlights_json(
            [
                {"chunk_id": "c0", "spans": ["Revenue grew 12% in Q3."]},
                {"chunk_id": "c1", "spans": ["Operating margin improved."]},
            ]
        )
        output = parse_source_highlighting(payload)
        assert len(output.highlights) == 2
        assert output.highlights[0].chunk_id == "c0"

    def test_extracts_json_from_prose(self):
        payload = (
            "Highlights:\n"
            + _highlights_json([{"chunk_id": "c0", "spans": ["Revenue grew 12% in Q3."]}])
            + "\nDone."
        )
        output = parse_source_highlighting(payload)
        assert output.highlights[0].spans == ["Revenue grew 12% in Q3."]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse source highlighting"):
            parse_source_highlighting("not json at all")


class TestExtractHighlights:
    def test_returns_verbatim_spans_per_chunk(self):
        llm = MagicMock()
        passage_a = "Revenue grew 12% in Q3. Costs were stable."
        passage_b = "Operating margin improved to 18%."
        llm.generate.return_value = _highlights_json(
            [
                {"chunk_id": "c0", "spans": ["Revenue grew 12% in Q3."]},
                {"chunk_id": "c1", "spans": ["Operating margin improved to 18%."]},
            ]
        )
        chunks = [_chunk("c0", passage_a), _chunk("c1", passage_b)]
        highlights = extract_highlights(_answer(), chunks, llm)
        assert highlights == {
            "c0": ["Revenue grew 12% in Q3."],
            "c1": ["Operating margin improved to 18%."],
        }
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert "Revenue grew 12% in Q3." in prompt
        assert passage_a in prompt
        llm.generate.assert_called_once()

    def test_empty_chunks_returns_empty_dict(self):
        llm = MagicMock()
        assert extract_highlights(_answer(), [], llm) == {}
        llm.generate.assert_not_called()

    def test_empty_answer_returns_empty_dict(self):
        llm = MagicMock()
        assert extract_highlights(_answer(text="   "), [_chunk("c0")], llm) == {}
        llm.generate.assert_not_called()

    def test_llm_failure_returns_empty_dict(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("model unavailable")
        highlights = extract_highlights(_answer(), [_chunk("c0")], llm)
        assert highlights == {}

    def test_parse_failure_returns_empty_dict(self):
        llm = MagicMock()
        llm.generate.return_value = "garbage"
        highlights = extract_highlights(_answer(), [_chunk("c0")], llm)
        assert highlights == {}

    def test_non_verbatim_spans_are_dropped(self):
        llm = MagicMock()
        passage = "Revenue grew 12% in Q3."
        llm.generate.return_value = _highlights_json(
            [{"chunk_id": "c0", "spans": ["Revenue increased twelve percent."]}]
        )
        highlights = extract_highlights(_answer(), [_chunk("c0", passage)], llm)
        assert highlights == {}

    def test_whitespace_normalized_llm_span_returns_verbatim_passage_text(self):
        llm = MagicMock()
        passage = "Revenue grew\n12% in Q3."
        llm.generate.return_value = _highlights_json(
            [{"chunk_id": "c0", "spans": ["Revenue grew 12% in Q3."]}]
        )
        highlights = extract_highlights(_answer(), [_chunk("c0", passage)], llm)
        assert highlights == {"c0": ["Revenue grew\n12% in Q3."]}
        assert highlights["c0"][0] in passage

    def test_irregular_spacing_returns_verbatim_passage_text(self):
        llm = MagicMock()
        passage = "Revenue  grew  12% in Q3."
        llm.generate.return_value = _highlights_json(
            [{"chunk_id": "c0", "spans": ["Revenue grew 12% in Q3."]}]
        )
        highlights = extract_highlights(_answer(), [_chunk("c0", passage)], llm)
        assert highlights == {"c0": ["Revenue  grew  12% in Q3."]}
        assert highlights["c0"][0] in passage

    def test_missing_highlights_omits_chunk(self):
        llm = MagicMock()
        llm.generate.return_value = _highlights_json(
            [{"chunk_id": "c0", "spans": ["Revenue grew 12% in Q3."]}]
        )
        chunks = [
            _chunk("c0", "Revenue grew 12% in Q3."),
            _chunk("c1", "Unrelated text."),
        ]
        highlights = extract_highlights(_answer(), chunks, llm)
        assert highlights == {"c0": ["Revenue grew 12% in Q3."]}

    def test_parent_context_siblings_share_one_highlight(self):
        llm = MagicMock()
        parent_text = "Shared parent body with supporting facts."
        child_a = _chunk(
            "child-a",
            text="slice a",
            metadata={
                CHUNK_PARENT_ID_KEY: "parent-0",
                PARENT_CONTEXT_TEXT_KEY: parent_text,
            },
        )
        child_b = _chunk(
            "child-b",
            text="slice b",
            metadata={
                CHUNK_PARENT_ID_KEY: "parent-0",
                PARENT_CONTEXT_TEXT_KEY: parent_text,
            },
        )
        llm.generate.return_value = _highlights_json(
            [{"chunk_id": "child-a", "spans": ["Shared parent body with supporting facts."]}]
        )
        answer = Answer(query_id="q-1", text="Supporting facts.", sources=["child-a", "child-b"])
        highlights = extract_highlights(answer, [child_a, child_b], llm)
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert prompt.count(parent_text) == 1
        assert highlights == {
            "child-a": ["Shared parent body with supporting facts."],
            "child-b": ["Shared parent body with supporting facts."],
        }
        span = highlights["child-a"][0]
        assert span in chunk_context_text(child_a)
        assert span not in child_a.text

    def test_sibling_chunk_id_maps_to_group_highlights(self):
        llm = MagicMock()
        parent_text = "Shared parent body with supporting facts."
        child_a = _chunk(
            "child-a",
            text="slice a",
            metadata={
                CHUNK_PARENT_ID_KEY: "parent-0",
                PARENT_CONTEXT_TEXT_KEY: parent_text,
            },
        )
        child_b = _chunk(
            "child-b",
            text="slice b",
            metadata={
                CHUNK_PARENT_ID_KEY: "parent-0",
                PARENT_CONTEXT_TEXT_KEY: parent_text,
            },
        )
        llm.generate.return_value = _highlights_json(
            [{"chunk_id": "child-b", "spans": ["Shared parent body with supporting facts."]}]
        )
        answer = Answer(query_id="q-1", text="Supporting facts.", sources=["child-a", "child-b"])
        highlights = extract_highlights(answer, [child_a, child_b], llm)
        assert highlights == {
            "child-a": ["Shared parent body with supporting facts."],
            "child-b": ["Shared parent body with supporting facts."],
        }

    def test_merged_source_copies_share_one_highlight(self):
        llm = MagicMock()
        merged_text = "Combined passage with key detail."
        merged = _chunk(
            "merged-1",
            text=merged_text,
            metadata={MERGED_CHUNK_IDS_KEY: ["c0", "c1"]},
        )
        from src.rag.quality.explainable_retrieval import resolve_chunks_for_sources

        resolved = resolve_chunks_for_sources(["c0", "c1"], [merged])
        llm.generate.return_value = _highlights_json(
            [{"chunk_id": "c0", "spans": ["Combined passage with key detail."]}]
        )
        answer = Answer(query_id="q-1", text="Key detail.", sources=["c0", "c1"])
        highlights = extract_highlights(answer, resolved, llm)
        assert highlights == {
            "c0": ["Combined passage with key detail."],
            "c1": ["Combined passage with key detail."],
        }

    def test_deduplicates_identical_spans(self):
        llm = MagicMock()
        passage = "Revenue grew 12% in Q3. Revenue grew 12% in Q3."
        llm.generate.return_value = _highlights_json(
            [
                {
                    "chunk_id": "c0",
                    "spans": ["Revenue grew 12% in Q3.", "Revenue grew 12% in Q3."],
                }
            ]
        )
        highlights = extract_highlights(_answer(), [_chunk("c0", passage)], llm)
        assert highlights == {"c0": ["Revenue grew 12% in Q3."]}
