"""T-071 — AgentPipeline and parse_decision tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.constants import CHUNK_RAW_TEXT_KEY
from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.rag.pipelines.agent_pipeline import (
    AgentAction,
    AgentPipeline,
    AgentRunResult,
    parse_decision,
)

# ── helpers ────────────────────────────────────────────────────────────────────

_ANSWER = '{"action":"ANSWER","reasoning":"ok","refined_query":"","entities":[],"clarification":""}'
_RETRIEVE_MORE = (
    '{"action":"RETRIEVE_MORE","reasoning":"need more",'
    '"refined_query":"refined q","entities":[],"clarification":""}'
)
_CLARIFY = (
    '{"action":"CLARIFY","reasoning":"ambig","refined_query":"",'
    '"entities":[],"clarification":"Which region?"}'
)
_GRAPH_LOOKUP = (
    '{"action":"GRAPH_LOOKUP","reasoning":"entities needed",'
    '"refined_query":"","entities":["EKS","IAM"],"clarification":""}'
)
_RETRIEVE_MORE_LOOP = (
    '{"action":"RETRIEVE_MORE","reasoning":"still need more",'
    '"refined_query":"q2","entities":[],"clarification":""}'
)


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=f"relevant text {i}")


def _retrieval_result(chunks: list[Chunk] | None = None):
    from src.domain.entities.query import Query
    from src.domain.services.retrieval_service import RetrievalResult

    chunks = chunks or [_chunk(0), _chunk(1)]
    context = "\n\n".join(c.text for c in chunks)
    return RetrievalResult(query=Query(text="q"), chunks=chunks, context=context)


async def _token_stream(*tokens: str) -> AsyncIterator[str]:
    for t in tokens:
        yield t


def _chat_mock(
    decision_response: str = _ANSWER,
    retrieval_chunks: list[Chunk] | None = None,
) -> MagicMock:
    m = MagicMock()
    m.retrieval.retrieve = AsyncMock(return_value=_retrieval_result(retrieval_chunks))
    m.generation.stream.return_value = _token_stream("answer token")
    m.generation.generate.return_value = Answer(query_id="q1", text="final answer", sources=["c0"])
    m.generation.call_llm.return_value = decision_response
    m.retrieval.service.hybrid.graph = None
    return m


def _pipeline(
    decision: str = _ANSWER,
    chunks: list[Chunk] | None = None,
    max_iterations: int = 3,
) -> AgentPipeline:
    return AgentPipeline(pipeline=_chat_mock(decision, chunks), max_iterations=max_iterations)


# ── parse_decision ─────────────────────────────────────────────────────────────


class TestParseDecision:
    def test_valid_answer(self):
        d = parse_decision('{"action":"ANSWER","reasoning":"enough context"}')
        assert d.action == AgentAction.ANSWER

    def test_retrieve_more(self):
        raw = '{"action":"RETRIEVE_MORE","reasoning":"need more","refined_query":"EKS IAM"}'
        d = parse_decision(raw)
        assert d.action == AgentAction.RETRIEVE_MORE
        assert d.refined_query == "EKS IAM"

    def test_graph_lookup(self):
        raw = '{"action":"GRAPH_LOOKUP","reasoning":"entities","entities":["EKS","IAM"]}'
        d = parse_decision(raw)
        assert d.action == AgentAction.GRAPH_LOOKUP
        assert "EKS" in d.entities

    def test_clarify(self):
        raw = '{"action":"CLARIFY","reasoning":"ambiguous","clarification":"Which region?"}'
        d = parse_decision(raw)
        assert d.action == AgentAction.CLARIFY
        assert d.clarification == "Which region?"

    def test_embedded_json(self):
        text = 'Here is my decision:\n{"action":"ANSWER","reasoning":"ok"}\nDone.'
        d = parse_decision(text)
        assert d.action == AgentAction.ANSWER

    def test_invalid_json_fallback(self):
        d = parse_decision("not valid json")
        assert d.action == AgentAction.ANSWER
        assert "fallback" in d.reasoning

    def test_unknown_action_raises_fallback(self):
        d = parse_decision('{"action":"UNKNOWN","reasoning":"x"}')
        assert d.action == AgentAction.ANSWER

    def test_non_dict_json_falls_back(self):
        d = parse_decision("[1, 2, 3]")
        assert d.action == AgentAction.ANSWER
        assert "fallback" in d.reasoning


class TestAgentPipelineFactory:
    def test_from_settings_reads_self_rag_flag(self):
        from unittest.mock import patch

        from src.core.settings import settings

        with (
            patch(
                "src.rag.pipelines.agent_pipeline.ChatPipeline.from_settings",
                return_value=MagicMock(),
            ) as from_settings,
            patch.object(settings.quality.self_rag, "enabled", True),
        ):
            pipeline = AgentPipeline.from_settings()
        from_settings.assert_called_once()
        assert pipeline._self_rag_enabled is True


# ── AgentPipeline ──────────────────────────────────────────────────────────────


class TestAgentPipelineChat:
    @pytest.mark.asyncio
    async def test_chat_returns_async_iterator(self):
        p = _pipeline()
        stream = await p.chat("question")
        assert hasattr(stream, "__aiter__")

    @pytest.mark.asyncio
    async def test_chat_yields_tokens(self):
        p = _pipeline()
        tokens = [t async for t in await p.chat("question")]
        assert tokens == ["answer token"]

    @pytest.mark.asyncio
    async def test_chat_full_returns_answer(self):
        p = _pipeline()
        result = await p.chat_full("question")
        assert isinstance(result, AgentRunResult)
        assert result.answer.text == "final answer"

    @pytest.mark.asyncio
    async def test_answer_decision_calls_generation(self):
        chat = _chat_mock('{"action":"ANSWER","reasoning":"enough"}')
        p = AgentPipeline(pipeline=chat)
        await p.chat_full("q")
        chat.generation.generate.assert_called_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_chat_full_uses_chunk_context_text_not_embedded_headers(self):
        header_chunk = Chunk(
            id="c0",
            document_id="doc",
            text="[Document: report.pdf | Section: Revenue | Page: 3]\nRevenue grew 12%.",
            metadata={CHUNK_RAW_TEXT_KEY: "Revenue grew 12%."},
        )
        chat = _chat_mock(retrieval_chunks=[header_chunk])
        p = AgentPipeline(pipeline=chat)
        await p.chat_full("What was revenue growth?")

        _, context, _ = chat.generation.generate.call_args[0]  # type: ignore[attr-defined]
        assert "Revenue grew 12%." in context
        assert "[Document:" not in context

    @pytest.mark.asyncio
    async def test_decide_prompt_uses_chunk_context_text(self):
        header_chunk = Chunk(
            id="c0",
            document_id="doc",
            text="[Document: report.pdf]\nBody for decision.",
            metadata={CHUNK_RAW_TEXT_KEY: "Body for decision."},
        )
        chat = _chat_mock(retrieval_chunks=[header_chunk])
        p = AgentPipeline(pipeline=chat)
        await p.chat_full("q")

        prompt = chat.generation.call_llm.call_args[0][0]  # type: ignore[attr-defined]
        assert "Body for decision." in prompt
        assert "[Document:" not in prompt

    @pytest.mark.asyncio
    async def test_retrieve_more_triggers_second_retrieval(self):
        chat = _chat_mock()
        # First decision: RETRIEVE_MORE, then ANSWER
        chat.generation.call_llm.side_effect = [_RETRIEVE_MORE, _ANSWER]
        p = AgentPipeline(pipeline=chat)
        await p.chat_full("q")
        assert chat.retrieval.retrieve.call_count == 2  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_clarify_returns_empty_context(self):
        p = _pipeline(decision=_CLARIFY)
        # With no context, generation falls back to no-info reply
        await p.chat_full("ambiguous question")
        # Generation is still called (with empty context)
        p._pipeline.generation.generate.assert_called_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_max_iterations_limits_loop(self):
        chat = _chat_mock(decision_response=_RETRIEVE_MORE_LOOP)
        # Always returns RETRIEVE_MORE — should stop at max_iterations
        p = AgentPipeline(pipeline=chat, max_iterations=2)
        await p.chat_full("q")
        # Initial + 2 re-retrievals capped by max_iterations
        assert chat.retrieval.retrieve.call_count <= 3  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_llm_decision_failure_defaults_to_answer(self):
        chat = _chat_mock()
        chat.generation.call_llm.side_effect = RuntimeError("LLM down")
        p = AgentPipeline(pipeline=chat)
        result = await p.chat_full("q")
        assert isinstance(result, AgentRunResult)

    @pytest.mark.asyncio
    async def test_graph_lookup_uses_graph_retriever(self):
        chat = _chat_mock(decision_response=_GRAPH_LOOKUP)
        # Wire in a graph retriever mock
        graph_mock = MagicMock()
        graph_mock.search = AsyncMock(return_value=[(_chunk(99), 1.0)])
        chat.retrieval.service.hybrid.graph = graph_mock
        p = AgentPipeline(pipeline=chat)
        await p.chat_full("EKS IAM question")
        graph_mock.search.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_graph_lookup_without_retriever_returns_empty(self):
        chat = _chat_mock(decision_response=_GRAPH_LOOKUP)
        chat.retrieval.service.hybrid.graph = None
        p = AgentPipeline(pipeline=chat)
        result = await p.chat_full("EKS IAM question")
        assert result.actions == [AgentAction.GRAPH_LOOKUP]

    @pytest.mark.asyncio
    async def test_graph_lookup_exception_is_swallowed(self):
        chat = _chat_mock(decision_response=_GRAPH_LOOKUP)
        graph_mock = MagicMock()
        graph_mock.search = AsyncMock(side_effect=RuntimeError("graph down"))
        chat.retrieval.service.hybrid.graph = graph_mock
        p = AgentPipeline(pipeline=chat)
        result = await p.chat_full("EKS IAM question")
        assert result.actions == [AgentAction.GRAPH_LOOKUP]

    @pytest.mark.asyncio
    async def test_empty_initial_chunks_skips_decision_loop(self):
        from src.domain.entities.query import Query
        from src.domain.services.retrieval_service import RetrievalResult

        chat = _chat_mock()
        chat.retrieval.retrieve = AsyncMock(
            return_value=RetrievalResult(query=Query(text="q"), chunks=[], context="")
        )
        p = AgentPipeline(pipeline=chat)
        result = await p.chat_full("q")
        assert result.actions == []
        chat.generation.call_llm.assert_not_called()

    @pytest.mark.asyncio
    async def test_decision_template_is_cached(self):
        p = _pipeline()
        first = p._get_decision_template()
        second = p._get_decision_template()
        assert first is second
        assert p._decision_template is first

    @pytest.mark.asyncio
    async def test_clarify_action_returns_empty_chunks_in_retrieve(self):
        chat = _chat_mock(decision_response=_CLARIFY)
        p = AgentPipeline(pipeline=chat)
        run = await p._agentic_retrieve("ambiguous question")
        assert run.chunks == []
        assert run.actions == [AgentAction.CLARIFY]
