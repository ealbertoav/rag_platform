"""T-140 — Reliable RAG document relevancy grading tests."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.constants import CHUNK_PARENT_ID_KEY, CHUNK_RAW_TEXT_KEY, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.services.generation_service import GenerationService
from src.domain.services.retrieval_service import RetrievalService
from src.rag.quality.reliable_rag import (
    ChunkRelevance,
    _format_passages,
    _group_chunks_for_grading,
    grade_relevance,
    parse_relevance_grading,
)


def _chunk(
    chunk_id: str,
    text: str = "text",
    *,
    parent_id: str | None = None,
    metadata: dict[str, object] | None = None,
) -> Chunk:
    chunk_metadata: dict[str, object] = dict(metadata or {})
    if parent_id is not None:
        chunk_metadata[CHUNK_PARENT_ID_KEY] = parent_id
    return Chunk(id=chunk_id, document_id="doc-1", text=text, metadata=chunk_metadata)


class _StubLookup:
    def __init__(self, chunks: dict[str, Chunk]) -> None:
        self._chunks = chunks

    def get_by_id(self, chunk_id: str) -> Chunk | None:
        return self._chunks.get(chunk_id)


def _dense_mock() -> MagicMock:
    mock = MagicMock()
    mock.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
    return mock


def _grades_json(grades: list[dict[str, object]]) -> str:
    return json.dumps({"grades": grades})


class TestParseRelevanceGrading:
    def test_parses_clean_json(self):
        payload = _grades_json(
            [
                {"chunk_id": "c0", "relevance_score": 0.9, "supporting": True},
                {"chunk_id": "c1", "relevance_score": 0.2, "supporting": False},
            ]
        )
        output = parse_relevance_grading(payload)
        assert len(output.grades) == 2
        assert output.grades[0].chunk_id == "c0"
        assert output.grades[0].relevance_score == pytest.approx(0.9)

    def test_extracts_json_from_prose(self):
        payload = (
            "Here are the grades:\n"
            + _grades_json([{"chunk_id": "c0", "relevance_score": 0.8, "supporting": True}])
            + "\nDone."
        )
        output = parse_relevance_grading(payload)
        assert output.grades[0].chunk_id == "c0"

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse relevance grading"):
            parse_relevance_grading("not json at all")


class TestGradeRelevanceFormatting:
    def test_uses_parent_context_text_not_child_slice(self):
        parent = _chunk("parent-0", text="full parent passage about kubernetes deployments.")
        child = _chunk(
            "child-0",
            text="narrow child slice.",
            parent_id="parent-0",
            metadata={PARENT_CONTEXT_TEXT_KEY: parent.text},
        )
        passages = _format_passages([child])
        assert "full parent passage about kubernetes deployments." in passages
        assert "narrow child slice." not in passages

    def test_uses_cch_raw_text_for_llm_context(self):
        chunk = _chunk(
            "c0",
            text="[Doc: manual | Section: Ops | Page: 1]\nActual kubernetes guidance.",
            metadata={CHUNK_RAW_TEXT_KEY: "Actual kubernetes guidance."},
        )
        passages = _format_passages([chunk])
        assert "Actual kubernetes guidance." in passages
        assert "[Doc: manual" not in passages

    def test_parent_context_siblings_graded_once(self):
        parent_text = "shared parent body about kubernetes."
        child_a = _chunk(
            "child-a",
            text="slice a",
            parent_id="parent-0",
            metadata={PARENT_CONTEXT_TEXT_KEY: parent_text},
        )
        child_b = _chunk(
            "child-b",
            text="slice b",
            parent_id="parent-0",
            metadata={PARENT_CONTEXT_TEXT_KEY: parent_text},
        )
        groups = _group_chunks_for_grading([child_a, child_b])
        assert len(groups) == 1
        assert {chunk.id for chunk in groups[0][1]} == {"child-a", "child-b"}
        passages = _format_passages([child_a, child_b])
        assert passages.count("shared parent body about kubernetes.") == 1

    def test_sibling_group_kept_or_dropped_together(self):
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [{"chunk_id": "child-a", "relevance_score": 0.9, "supporting": True}]
        )
        parent_text = "shared parent body."
        child_a = _chunk(
            "child-a",
            text="slice a",
            parent_id="parent-0",
            metadata={PARENT_CONTEXT_TEXT_KEY: parent_text},
        )
        child_b = _chunk(
            "child-b",
            text="slice b",
            parent_id="parent-0",
            metadata={PARENT_CONTEXT_TEXT_KEY: parent_text},
        )
        kept, passed, failed = grade_relevance("query", [child_a, child_b], llm, min_score=0.5)
        assert {c.id for c in kept} == {"child-a", "child-b"}
        assert passed == 2
        assert failed == 0


class TestGradeRelevance:
    def test_empty_chunks(self):
        llm = MagicMock()
        kept, passed, failed = grade_relevance("query", [], llm)
        assert kept == []
        assert passed == 0
        assert failed == 0
        llm.generate.assert_not_called()

    def test_filters_below_min_score(self):
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [
                {"chunk_id": "c0", "relevance_score": 0.9, "supporting": True},
                {"chunk_id": "c1", "relevance_score": 0.3, "supporting": True},
            ]
        )
        chunks = [_chunk("c0", "relevant"), _chunk("c1", "irrelevant")]
        kept, passed, failed = grade_relevance("query", chunks, llm, min_score=0.5)
        assert [c.id for c in kept] == ["c0"]
        assert passed == 1
        assert failed == 1
        assert kept[0].metadata["relevance_score"] == pytest.approx(0.9)
        assert kept[0].metadata["relevance_supporting"] is True

    def test_missing_grade_excludes_chunk(self):
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [{"chunk_id": "c0", "relevance_score": 0.9, "supporting": True}]
        )
        chunks = [_chunk("c0"), _chunk("c1")]
        kept, passed, failed = grade_relevance("query", chunks, llm, min_score=0.5)
        assert [c.id for c in kept] == ["c0"]
        assert passed == 1
        assert failed == 1

    def test_llm_failure_returns_all_chunks(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("llm down")
        chunks = [_chunk("c0"), _chunk("c1")]
        kept, passed, failed = grade_relevance("query", chunks, llm, min_score=0.5)
        assert kept == chunks
        assert passed == 2
        assert failed == 0

    def test_parse_failure_returns_all_chunks(self):
        llm = MagicMock()
        llm.generate.return_value = "garbage"
        chunks = [_chunk("c0")]
        kept, passed, failed = grade_relevance("query", chunks, llm, min_score=0.5)
        assert kept == chunks
        assert passed == 1
        assert failed == 0


class TestRetrievalServiceReliableRAG:
    @pytest.fixture
    def reranked_chunks(self) -> list[Chunk]:
        return [
            _chunk("c0", "kubernetes deployment guide"),
            _chunk("c1", "unrelated cooking recipe"),
        ]

    def _service(
        self,
        *,
        reliable_rag_enabled: bool = False,
        reliable_rag_min_score: float = 0.5,
        reranked_chunks: list[Chunk] | None = None,
        llm: MagicMock | None = None,
    ) -> RetrievalService:
        chunks = reranked_chunks or []
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(c, 0.9) for c in chunks])
        reranker = MagicMock()
        reranker.rerank.return_value = chunks
        return RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=hybrid,
            reranker=reranker,
            top_k_retrieval=10,
            top_k_rerank=2,
            reliable_rag_enabled=reliable_rag_enabled,
            reliable_rag_min_score=reliable_rag_min_score,
            llm=llm,
        )

    @pytest.mark.asyncio
    async def test_disabled_preserves_reranked_chunks(self, reranked_chunks):
        llm = MagicMock()
        svc = self._service(
            reliable_rag_enabled=False,
            reranked_chunks=reranked_chunks,
            llm=llm,
        )
        result = await svc.retrieve(Query(text="kubernetes"))
        assert [c.id for c in result.chunks] == ["c0", "c1"]
        llm.generate.assert_not_called()

    @pytest.mark.asyncio
    async def test_enabled_filters_low_scoring_chunks(self, reranked_chunks):
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [
                {"chunk_id": "c0", "relevance_score": 0.95, "supporting": True},
                {"chunk_id": "c1", "relevance_score": 0.1, "supporting": False},
            ]
        )
        svc = self._service(
            reliable_rag_enabled=True,
            reranked_chunks=reranked_chunks,
            llm=llm,
        )
        result = await svc.retrieve(Query(text="kubernetes"))
        assert [c.id for c in result.chunks] == ["c0"]
        assert result.context.strip() != ""

    @pytest.mark.asyncio
    async def test_all_chunks_filtered_yields_empty_context(self, reranked_chunks):
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [
                {"chunk_id": "c0", "relevance_score": 0.1, "supporting": False},
                {"chunk_id": "c1", "relevance_score": 0.2, "supporting": False},
            ]
        )
        svc = self._service(
            reliable_rag_enabled=True,
            reranked_chunks=reranked_chunks,
            llm=llm,
        )
        result = await svc.retrieve(Query(text="kubernetes"))
        assert result.chunks == []
        assert result.context == ""

    @pytest.mark.asyncio
    async def test_empty_context_triggers_no_info_generation(self, reranked_chunks):
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [
                {"chunk_id": "c0", "relevance_score": 0.1, "supporting": False},
                {"chunk_id": "c1", "relevance_score": 0.2, "supporting": False},
            ]
        )
        svc = self._service(
            reliable_rag_enabled=True,
            reranked_chunks=reranked_chunks,
            llm=llm,
        )
        result = await svc.retrieve(Query(text="kubernetes"))
        generation = GenerationService(llm=MagicMock())
        answer = generation.generate("kubernetes", result.context, [])
        assert answer.text == "I don't have information about this."

    @pytest.mark.asyncio
    async def test_relevance_grading_span_records_pass_fail_counts(self, reranked_chunks):
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [
                {"chunk_id": "c0", "relevance_score": 0.9, "supporting": True},
                {"chunk_id": "c1", "relevance_score": 0.1, "supporting": False},
            ]
        )
        svc = self._service(
            reliable_rag_enabled=True,
            reranked_chunks=reranked_chunks,
            llm=llm,
        )
        mock_span = MagicMock()
        with patch(
            "src.domain.services.retrieval_service._tracer.start_as_current_span",
            return_value=MagicMock(__enter__=MagicMock(return_value=mock_span)),
        ) as start_span:
            await svc.retrieve(Query(text="kubernetes"))
            span_names = [call.args[0] for call in start_span.call_args_list]
            assert "retrieval.relevance_grading" in span_names
            grading_calls = [
                call
                for call in start_span.call_args_list
                if call.args and call.args[0] == "retrieval.relevance_grading"
            ]
            assert grading_calls
            mock_span.set_attribute.assert_any_call("relevance.pass_count", 1)
            mock_span.set_attribute.assert_any_call("relevance.fail_count", 1)


class TestRetrievalServiceReliableRAGPipelineOrder:
    @pytest.mark.asyncio
    async def test_grades_parent_context_after_enrichment(self):
        parent = _chunk("parent-0", text="irrelevant cooking encyclopedia with no kubernetes info.")
        child = _chunk("child-0", text="kubernetes deployment", parent_id="parent-0")
        lookup = _StubLookup({"parent-0": parent})
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [{"chunk_id": "child-0", "relevance_score": 0.1, "supporting": False}]
        )
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(child, 0.9)])
        reranker = MagicMock()
        reranker.rerank.return_value = [child]

        svc = RetrievalService(
            dense_retriever=_dense_mock(),
            hybrid_retriever=hybrid,
            reranker=reranker,
            parent_context_enabled=True,
            parent_child_strategy=True,
            chunk_lookup=lookup,
            reliable_rag_enabled=True,
            llm=llm,
        )
        result = await svc.retrieve(Query(text="kubernetes"))
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert "irrelevant cooking encyclopedia" in prompt
        assert "kubernetes deployment" not in prompt
        assert result.chunks == []
        assert result.context == ""

    @pytest.mark.asyncio
    async def test_grading_runs_after_rse_merge(self):
        child_a = _chunk(
            "child-a",
            text="part one about kubernetes.",
            parent_id="parent-0",
            metadata={"chunk_index": 0},
        )
        child_b = _chunk(
            "child-b",
            text="part two about kubernetes.",
            parent_id="parent-0",
            metadata={"chunk_index": 1},
        )
        parent = _chunk("parent-0", text="full parent body.")
        lookup = _StubLookup({"parent-0": parent})
        llm = MagicMock()
        llm.generate.return_value = _grades_json(
            [{"chunk_id": "child-a", "relevance_score": 0.95, "supporting": True}]
        )
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(child_a, 0.9), (child_b, 0.8)])
        reranker = MagicMock()
        reranker.rerank.return_value = [child_a, child_b]

        svc = RetrievalService(
            dense_retriever=_dense_mock(),
            hybrid_retriever=hybrid,
            reranker=reranker,
            parent_context_enabled=True,
            parent_child_strategy=True,
            chunk_lookup=lookup,
            rse_enabled=True,
            rse_max_segment_tokens=500,
            reliable_rag_enabled=True,
            llm=llm,
        )
        result = await svc.retrieve(Query(text="kubernetes"))
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert "full parent body." in prompt
        assert result.chunks
        assert result.context == "full parent body."


class TestChunkRelevanceModel:
    def test_score_bounds(self):
        ChunkRelevance(chunk_id="c0", relevance_score=0.0, supporting=False)
        ChunkRelevance(chunk_id="c0", relevance_score=1.0, supporting=True)
        with pytest.raises(ValueError):
            ChunkRelevance(chunk_id="c0", relevance_score=1.5, supporting=True)
