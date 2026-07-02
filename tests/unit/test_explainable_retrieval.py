"""T-143 — Explainable retrieval API tests."""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import pytest

from src.core.constants import CHUNK_PARENT_ID_KEY, MERGED_CHUNK_IDS_KEY, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.rag.quality.explainable_retrieval import (
    ChunkExplanation,
    explain_chunks,
    parse_explain_retrieval,
    resolve_chunks_for_sources,
)


def _chunk(
    chunk_id: str,
    text: str = "sample text",
    *,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", text=text, metadata=metadata or {})


def _explanations_json(items: list[dict[str, object]]) -> str:
    return json.dumps({"explanations": items})


class TestParseExplainRetrieval:
    def test_parses_clean_json(self):
        payload = _explanations_json(
            [
                {"chunk_id": "c0", "reason": "Contains Q3 revenue figures."},
                {"chunk_id": "c1", "reason": "Mentions the same fiscal period."},
            ]
        )
        output = parse_explain_retrieval(payload)
        assert len(output.explanations) == 2
        assert output.explanations[0].chunk_id == "c0"

    def test_extracts_json_from_prose(self):
        payload = (
            "Here are the explanations:\n"
            + _explanations_json([{"chunk_id": "c0", "reason": "Direct match on topic."}])
            + "\nDone."
        )
        output = parse_explain_retrieval(payload)
        assert output.explanations[0].reason == "Direct match on topic."

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse explain retrieval"):
            parse_explain_retrieval("not json at all")


class TestResolveChunksForSources:
    def test_maps_direct_chunk_ids(self):
        chunks = [_chunk("c0"), _chunk("c1")]
        resolved = resolve_chunks_for_sources(["c0", "c1"], chunks)
        assert [chunk.id for chunk in resolved] == ["c0", "c1"]

    def test_maps_merged_source_ids_to_parent_passage(self):
        merged = _chunk(
            "merged-1",
            text="combined passage",
            metadata={MERGED_CHUNK_IDS_KEY: ["c0", "c1"]},
        )
        resolved = resolve_chunks_for_sources(["c0", "c1"], [merged])
        assert [chunk.id for chunk in resolved] == ["c0", "c1"]
        assert all(chunk.text == "combined passage" for chunk in resolved)

    def test_deduplicates_repeated_source_ids(self):
        chunks = [_chunk("c0")]
        resolved = resolve_chunks_for_sources(["c0", "c0"], chunks)
        assert len(resolved) == 1


class TestExplainChunks:
    def test_returns_explanations_for_all_chunks(self):
        llm = MagicMock()
        llm.generate.return_value = _explanations_json(
            [
                {"chunk_id": "c0", "reason": "Mentions revenue."},
                {"chunk_id": "c1", "reason": "Covers the same quarter."},
            ]
        )
        chunks = [_chunk("c0"), _chunk("c1")]
        explanations = explain_chunks("What was revenue?", chunks, llm)
        assert explanations == [
            ChunkExplanation(chunk_id="c0", reason="Mentions revenue."),
            ChunkExplanation(chunk_id="c1", reason="Covers the same quarter."),
        ]
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert "chunk_id=c0" in prompt
        assert "chunk_id=c1" in prompt
        llm.generate.assert_called_once()

    def test_empty_chunks_returns_empty_list(self):
        llm = MagicMock()
        assert explain_chunks("q", [], llm) == []
        llm.generate.assert_not_called()

    def test_llm_failure_returns_empty_list(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("model unavailable")
        explanations = explain_chunks("q", [_chunk("c0")], llm)
        assert explanations == []

    def test_parse_failure_returns_empty_list(self):
        llm = MagicMock()
        llm.generate.return_value = "garbage"
        explanations = explain_chunks("q", [_chunk("c0")], llm)
        assert explanations == []

    def test_missing_explanation_omits_chunk(self):
        llm = MagicMock()
        llm.generate.return_value = _explanations_json(
            [{"chunk_id": "c0", "reason": "Only first chunk explained."}]
        )
        explanations = explain_chunks("q", [_chunk("c0"), _chunk("c1")], llm)
        assert len(explanations) == 1
        assert explanations[0].chunk_id == "c0"

    def test_parent_context_siblings_share_one_explanation(self):
        llm = MagicMock()
        parent_text = "shared parent body."
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
        llm.generate.return_value = _explanations_json(
            [{"chunk_id": "child-a", "reason": "Mentions the shared parent topic."}]
        )
        explanations = explain_chunks("q", [child_a, child_b], llm)
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert prompt.count("shared parent body.") == 1
        assert "chunk_id=child-a" in prompt
        assert "chunk_id=child-b" not in prompt
        assert explanations == [
            ChunkExplanation(
                chunk_id="child-a",
                reason="Mentions the shared parent topic.",
            ),
            ChunkExplanation(
                chunk_id="child-b",
                reason="Mentions the shared parent topic.",
            ),
        ]

    def test_merged_source_copies_share_one_explanation(self):
        llm = MagicMock()
        merged = _chunk(
            "merged-1",
            text="combined passage",
            metadata={MERGED_CHUNK_IDS_KEY: ["c0", "c1"]},
        )
        resolved = resolve_chunks_for_sources(["c0", "c1"], [merged])
        llm.generate.return_value = _explanations_json(
            [{"chunk_id": "c0", "reason": "Contains the merged segment."}]
        )
        explanations = explain_chunks("q", resolved, llm)
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert prompt.count("combined passage") == 1
        assert "chunk_id=c0" in prompt
        assert "chunk_id=c1" not in prompt
        assert explanations == [
            ChunkExplanation(chunk_id="c0", reason="Contains the merged segment."),
            ChunkExplanation(chunk_id="c1", reason="Contains the merged segment."),
        ]
