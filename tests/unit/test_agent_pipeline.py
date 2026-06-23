"""T-071 — AgentPipeline and _parse_decision tests."""
from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.rag.pipelines.agent_pipeline import (
    AgentAction,
    AgentDecision,
    AgentPipeline,
    _parse_decision,
)


# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=f"relevant text {i}")


def _retrieval_result(chunks: list[Chunk] | None = None):
    from src.domain.services.retrieval_service import RetrievalResult
    from src.domain.entities.query import Query

    chunks = chunks or [_chunk(0), _chunk(1)]
    context = "\n\n".join(c.text for c in chunks)
    return RetrievalResult(query=Query(text="q"), chunks=chunks, context=context)


async def _token_stream(*tokens: str) -> AsyncIterator[str]:
    for t in tokens:
        yield t


def _chat_mock(
    decision_response: str = '{"action":"ANSWER","reasoning":"ok","refined_query":"","entities":[],"clarification":""}',
    retrieval_chunks: list[Chunk] | None = None,
) -> MagicMock:
    m = MagicMock()
    # retrieval pipeline
    m._retrieval.retrieve = AsyncMock(return_value=_retrieval_result(retrieval_chunks))
    # generation service
    m._generation.stream.return_value = _token_stream("answer token")
    m._generation.generate.return_value = Answer(
        query_id="q1", text="final answer", sources=["c0"]
    )
    m._generation._llm.generate.return_value = decision_response
    # no graph retriever by default
    m._retrieval._service._hybrid._graph = None
    return m


def _pipeline(
    decision: str = '{"action":"ANSWER","reasoning":"ok","refined_query":"","entities":[],"clarification":""}',
    chunks: list[Chunk] | None = None,
    max_iterations: int = 3,
) -> AgentPipeline:
    return AgentPipeline(chat=_chat_mock(decision, chunks), max_iterations=max_iterations)


# ── _parse_decision ────────────────────────────────────────────────────────────


class TestParseDecision:
    def test_valid_answer(self):
        d = _parse_decision('{"action":"ANSWER","reasoning":"enough context"}')
        assert d.action == AgentAction.ANSWER

    def test_retrieve_more(self):
        d = _parse_decision('{"action":"RETRIEVE_MORE","reasoning":"need more","refined_query":"EKS IAM"}')
        assert d.action == AgentAction.RETRIEVE_MORE
        assert d.refined_query == "EKS IAM"

    def test_graph_lookup(self):
        d = _parse_decision('{"action":"GRAPH_LOOKUP","reasoning":"entities","entities":["EKS","IAM"]}')
        assert d.action == AgentAction.GRAPH_LOOKUP
        assert "EKS" in d.entities

    def test_clarify(self):
        d = _parse_decision('{"action":"CLARIFY","reasoning":"ambiguous","clarification":"Which region?"}')
        assert d.action == AgentAction.CLARIFY
        assert d.clarification == "Which region?"

    def test_embedded_json(self):
        text = 'Here is my decision:\n{"action":"ANSWER","reasoning":"ok"}\nDone.'
        d = _parse_decision(text)
        assert d.action == AgentAction.ANSWER

    def test_invalid_json_fallback(self):
        d = _parse_decision("not valid json")
        assert d.action == AgentAction.ANSWER
        assert "fallback" in d.reasoning

    def test_unknown_action_raises_fallback(self):
        d = _parse_decision('{"action":"UNKNOWN","reasoning":"x"}')
        assert d.action == AgentAction.ANSWER


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
        assert isinstance(result, Answer)
        assert result.text == "final answer"

    @pytest.mark.asyncio
    async def test_answer_decision_calls_generation(self):
        chat = _chat_mock('{"action":"ANSWER","reasoning":"enough"}')
        p = AgentPipeline(chat=chat)
        await p.chat_full("q")
        chat._generation.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrieve_more_triggers_second_retrieval(self):
        retrieve_more = '{"action":"RETRIEVE_MORE","reasoning":"need more","refined_query":"refined q","entities":[],"clarification":""}'
        answer = '{"action":"ANSWER","reasoning":"ok","refined_query":"","entities":[],"clarification":""}'
        chat = _chat_mock()
        # First decision: RETRIEVE_MORE, then ANSWER
        chat._generation._llm.generate.side_effect = [retrieve_more, answer]
        p = AgentPipeline(chat=chat)
        await p.chat_full("q")
        assert chat._retrieval.retrieve.call_count == 2

    @pytest.mark.asyncio
    async def test_clarify_returns_empty_context(self):
        clarify = '{"action":"CLARIFY","reasoning":"ambig","refined_query":"","entities":[],"clarification":"Which region?"}'
        p = _pipeline(decision=clarify)
        # With no context, generation falls back to no-info reply
        await p.chat_full("ambiguous question")
        # Generation is still called (with empty context)
        p._chat._generation.generate.assert_called_once()

    @pytest.mark.asyncio
    async def test_max_iterations_limits_loop(self):
        retrieve_more = '{"action":"RETRIEVE_MORE","reasoning":"still need more","refined_query":"q2","entities":[],"clarification":""}'
        chat = _chat_mock(decision_response=retrieve_more)
        # Always returns RETRIEVE_MORE — should stop at max_iterations
        p = AgentPipeline(chat=chat, max_iterations=2)
        await p.chat_full("q")
        # Initial + 2 re-retrievals capped by max_iterations
        assert chat._retrieval.retrieve.call_count <= 3

    @pytest.mark.asyncio
    async def test_llm_decision_failure_defaults_to_answer(self):
        chat = _chat_mock()
        chat._generation._llm.generate.side_effect = RuntimeError("LLM down")
        p = AgentPipeline(chat=chat)
        result = await p.chat_full("q")
        assert isinstance(result, Answer)

    @pytest.mark.asyncio
    async def test_graph_lookup_uses_graph_retriever(self):
        graph_decision = '{"action":"GRAPH_LOOKUP","reasoning":"entities needed","refined_query":"","entities":["EKS","IAM"],"clarification":""}'
        chat = _chat_mock(decision_response=graph_decision)
        # Wire in a graph retriever mock
        graph_mock = MagicMock()
        graph_mock.search.return_value = [(_chunk(99), 1.0)]
        chat._retrieval._service._hybrid._graph = graph_mock
        p = AgentPipeline(chat=chat)
        await p.chat_full("EKS IAM question")
        graph_mock.search.assert_called_once()
