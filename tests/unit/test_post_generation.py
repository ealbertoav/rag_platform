"""Combined post-generation explain + highlight tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.rag.quality.post_generation import explain_and_highlight, parse_explain_and_highlight


def _chunk(chunk_id: str, text: str = "sample text") -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", text=text, metadata={})


def _answer(text: str = "Revenue grew 12% in Q3.") -> Answer:
    return Answer(query_id="q-1", text=text, sources=["c0", "c1"])


class TestParseExplainAndHighlight:
    def test_parses_combined_json(self):
        payload = json.dumps(
            {
                "explanations": [{"chunk_id": "c0", "reason": "Mentions revenue."}],
                "highlights": [{"chunk_id": "c0", "spans": ["Revenue grew 12% in Q3."]}],
            }
        )
        output = parse_explain_and_highlight(payload)
        assert output.explanations[0].reason == "Mentions revenue."
        assert output.highlights[0].spans == ["Revenue grew 12% in Q3."]

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse explain and highlight"):
            parse_explain_and_highlight("not json")


class TestExplainAndHighlight:
    def test_returns_both_explanations_and_highlights(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps(
            {
                "explanations": [
                    {"chunk_id": "c0", "reason": "Mentions revenue growth."},
                    {"chunk_id": "c1", "reason": "Adds margin detail."},
                ],
                "highlights": [
                    {"chunk_id": "c0", "spans": ["Revenue grew 12% in Q3."]},
                    {"chunk_id": "c1", "spans": ["Operating margin improved to 18%."]},
                ],
            }
        )
        chunks = [
            _chunk("c0", "Revenue grew 12% in Q3. Costs were stable."),
            _chunk("c1", "Operating margin improved to 18%."),
        ]
        explanations, highlights = explain_and_highlight(
            "What was Q3 revenue?", _answer(), chunks, llm
        )
        assert len(explanations) == 2
        assert explanations[0].chunk_id == "c0"
        assert highlights == {
            "c0": ["Revenue grew 12% in Q3."],
            "c1": ["Operating margin improved to 18%."],
        }
        llm.generate.assert_called_once()

    def test_llm_failure_returns_empty_results(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("model unavailable")
        explanations, highlights = explain_and_highlight("q", _answer(), [_chunk("c0")], llm)
        assert explanations == []
        assert highlights == {}

    def test_parse_failure_returns_empty_results(self):
        llm = MagicMock()
        llm.generate.return_value = "garbage"
        explanations, highlights = explain_and_highlight("q", _answer(), [_chunk("c0")], llm)
        assert explanations == []
        assert highlights == {}
