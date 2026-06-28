"""T-126 — proposition chunking tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.core.constants import (
    CHUNK_SOURCE_KEY,
    CHUNK_TYPE_KEY,
    CHUNK_TYPE_PROPOSITION,
    PROPOSITION_INDEX_KEY,
)
from src.domain.entities.chunk import Chunk
from src.domain.entities.document import Document
from src.rag.chunking import get_chunker
from src.rag.chunking.proposition_chunker import (
    PropositionChunker,
    extract_propositions,
    grade_proposition,
    load_extract_template,
    passes_quality_threshold,
)


def _doc(content: str, source: str = "policy.md") -> Document:
    return Document(source=source, content=content)


def _propositions_json(propositions: list[str]) -> str:
    return json.dumps(propositions)


def _scores_json(**scores: int) -> str:
    return json.dumps(scores)


def _chunker(llm: MagicMock, *, overlap: int = 0) -> PropositionChunker:
    return PropositionChunker(llm=llm, chunk_size=200, overlap=overlap, quality_threshold=7)


class TestPropositionParsing:
    def test_passes_quality_threshold(self):
        scores = {"accuracy": 8, "clarity": 7, "completeness": 9, "conciseness": 7}
        assert passes_quality_threshold(scores, 7)

    def test_fails_when_any_score_below_threshold(self):
        scores = {"accuracy": 8, "clarity": 6, "completeness": 9, "conciseness": 7}
        assert not passes_quality_threshold(scores, 7)

    def test_extract_propositions_parses_json_list(self):
        llm = MagicMock()
        llm.generate.return_value = _propositions_json(
            ["The policy covers dental care.", "Coverage starts on January 1."]
        )
        result = extract_propositions("Dental coverage begins January 1.", llm)
        assert result == ["The policy covers dental care.", "Coverage starts on January 1."]

    def test_grade_proposition_parses_json_scores(self):
        llm = MagicMock()
        llm.generate.return_value = _scores_json(
            accuracy=9, clarity=8, completeness=8, conciseness=7
        )
        scores = grade_proposition("Dental coverage begins January 1.", "source text", llm)
        assert scores == {
            "accuracy": 9,
            "clarity": 8,
            "completeness": 8,
            "conciseness": 7,
        }

    def test_grade_proposition_parses_whole_number_float_scores(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps(
            {"accuracy": 9.0, "clarity": 8.0, "completeness": 8.0, "conciseness": 7.0}
        )
        scores = grade_proposition("Dental coverage begins January 1.", "source text", llm)
        assert scores == {
            "accuracy": 9,
            "clarity": 8,
            "completeness": 8,
            "conciseness": 7,
        }

    def test_grade_proposition_parses_string_scores(self):
        llm = MagicMock()
        llm.generate.return_value = json.dumps(
            {"accuracy": "9", "clarity": "8", "completeness": "8", "conciseness": "7"}
        )
        scores = grade_proposition("Dental coverage begins January 1.", "source text", llm)
        assert scores == {
            "accuracy": 9,
            "clarity": 8,
            "completeness": 8,
            "conciseness": 7,
        }

    def test_grade_proposition_handles_braces_in_text(self):
        llm = MagicMock()
        llm.generate.return_value = _scores_json(
            accuracy=9, clarity=8, completeness=8, conciseness=7
        )
        original = 'Coverage applies when {"status": "active"} and {tier: "gold"}.'
        proposition = 'The policy covers members with {"status": "active"}.'
        scores = grade_proposition(proposition, original, llm)
        assert scores == {
            "accuracy": 9,
            "clarity": 8,
            "completeness": 8,
            "conciseness": 7,
        }
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert '{"status": "active"}' in prompt
        assert '{tier: "gold"}' in prompt


class TestPropositionChunker:
    def test_returns_standalone_proposition_chunks(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            _propositions_json(["Employees receive 20 days of paid leave per year."]),
            _scores_json(accuracy=9, clarity=8, completeness=8, conciseness=8),
        ]
        chunks = _chunker(llm).chunk(_doc("Employees receive twenty days of paid leave each year."))
        assert len(chunks) == 1
        assert chunks[0].text == "Employees receive 20 days of paid leave per year."
        assert isinstance(chunks[0], Chunk)

    def test_discards_low_quality_propositions(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            _propositions_json(
                [
                    "Good proposition with full context.",
                    "Bad proposition.",
                ]
            ),
            _scores_json(accuracy=9, clarity=8, completeness=8, conciseness=8),
            _scores_json(accuracy=5, clarity=6, completeness=5, conciseness=6),
        ]
        chunks = _chunker(llm).chunk(_doc("Long source text about employee benefits."))
        assert len(chunks) == 1
        assert chunks[0].text == "Good proposition with full context."

    def test_document_id_and_metadata_set(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            _propositions_json(["The contract term is 12 months."]),
            _scores_json(accuracy=8, clarity=8, completeness=8, conciseness=8),
        ]
        doc = _doc("Contract term: twelve months.", source="contracts/a.pdf")
        chunks = _chunker(llm).chunk(doc)
        assert chunks[0].document_id == doc.id
        assert chunks[0].metadata[CHUNK_SOURCE_KEY] == "contracts/a.pdf"
        assert chunks[0].metadata[PROPOSITION_INDEX_KEY] == 0
        assert chunks[0].metadata[CHUNK_TYPE_KEY] == CHUNK_TYPE_PROPOSITION
        assert "proposition_scores" in chunks[0].metadata
        assert "chunk_index" not in chunks[0].metadata

    def test_empty_document_returns_empty(self):
        llm = MagicMock()
        assert _chunker(llm).chunk(_doc("")) == []

    def test_extraction_failure_skips_segment(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        chunks = _chunker(llm).chunk(_doc("Some factual content about refunds."))
        assert chunks == []

    def test_grading_failure_skips_proposition_not_whole_document(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            _propositions_json(
                [
                    "First proposition fails grading.",
                    "Second proposition survives.",
                ]
            ),
            RuntimeError("grading failed"),
            _scores_json(accuracy=9, clarity=8, completeness=8, conciseness=8),
        ]
        chunks = _chunker(llm).chunk(_doc("Policy text about benefits and refunds."))
        assert len(chunks) == 1
        assert chunks[0].text == "Second proposition survives."

    def test_quality_threshold_validation(self):
        llm = MagicMock()
        with pytest.raises(ValueError, match="quality_threshold"):
            PropositionChunker(llm=llm, quality_threshold=11)

    def test_get_chunker_returns_proposition(self):
        llm = MagicMock()
        chunker = get_chunker("proposition", llm=llm)
        assert isinstance(chunker, PropositionChunker)

    def test_load_extract_template_reads_prompt_file(self):
        template = load_extract_template()
        assert "$text" in template.template

    def test_deduplicates_propositions_across_segments(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            _propositions_json(["Employees receive 20 days of paid leave per year."]),
            _scores_json(accuracy=9, clarity=8, completeness=8, conciseness=8),
            _propositions_json(["Employees receive 20 days of paid leave per year."]),
            _scores_json(accuracy=9, clarity=8, completeness=8, conciseness=8),
        ]
        long_doc = _doc(" ".join(["Employees receive twenty days of paid leave each year."] * 40))
        chunks = PropositionChunker(llm=llm, chunk_size=80, overlap=0, quality_threshold=7).chunk(
            long_doc
        )
        assert len(chunks) == 1
        assert chunks[0].text == "Employees receive 20 days of paid leave per year."

    def test_case_insensitive_deduplication(self):
        llm = MagicMock()
        llm.generate.side_effect = [
            _propositions_json(["The policy covers dental care."]),
            _scores_json(accuracy=9, clarity=8, completeness=8, conciseness=8),
            _propositions_json(["the policy covers dental care."]),
            _scores_json(accuracy=9, clarity=8, completeness=8, conciseness=8),
        ]
        long_doc = _doc(" ".join(["Dental coverage details for all employees."] * 40))
        chunks = PropositionChunker(llm=llm, chunk_size=80, overlap=0, quality_threshold=7).chunk(
            long_doc
        )
        assert len(chunks) == 1
