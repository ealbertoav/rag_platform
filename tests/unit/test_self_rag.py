"""T-141 — Self-RAG decision loop tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.pipelines.agent_pipeline import AgentAction, AgentPipeline, AgentRunResult
from src.rag.quality.self_rag import (
    RetrievalDecision,
    UtilityAction,
    check_support,
    decide_retrieval,
    parse_retrieval_decision,
    parse_support_check,
    parse_utility_score,
    score_utility,
)


def _chunk(chunk_id: str, text: str = "kubernetes deployment guide") -> Chunk:
    return Chunk(id=chunk_id, document_id="doc-1", text=text)


def _retrieval_result(chunks: list[Chunk] | None = None, *, context: str | None = None):
    from src.domain.entities.query import Query
    from src.domain.services.retrieval_service import RetrievalResult

    chunks = chunks if chunks is not None else [_chunk("c0")]
    resolved_context = context if context is not None else "\n\n".join(c.text for c in chunks)
    return RetrievalResult(query=Query(text="q"), chunks=chunks, context=resolved_context)


class _StubLLM(LLMRepository):
    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._calls = 0

    def generate(self, prompt: str, context: str, **kwargs: Any) -> str:
        if self._responses:
            response = self._responses[self._calls]
            self._calls += 1
            return response
        return ""

    def generate_stream(self, prompt: str, context: str, **kwargs: Any) -> AsyncIterator[str]:
        async def _stream() -> AsyncIterator[str]:
            yield self.generate(prompt, context, **kwargs)

        return _stream()


class TestParseSelfRAG:
    def test_parse_retrieval_decision(self):
        payload = json.dumps({"need_retrieval": True, "reasoning": "domain question"})
        result = parse_retrieval_decision(payload)
        assert result.need_retrieval is True
        assert result.reasoning == "domain question"

    def test_parse_support_check(self):
        payload = json.dumps({"supported": False, "reasoning": "hallucinated fact"})
        result = parse_support_check(payload)
        assert result.supported is False

    def test_parse_utility_score(self):
        payload = json.dumps(
            {
                "score": 0.85,
                "action": "accept",
                "reasoning": "complete answer",
                "refined_query": "",
            }
        )
        result = parse_utility_score(payload)
        assert result.action == UtilityAction.ACCEPT
        assert result.score == pytest.approx(0.85)

    def test_invalid_json_raises(self):
        with pytest.raises(ValueError, match="Could not parse retrieval decision"):
            parse_retrieval_decision("not json")


class TestSelfRAGChains:
    def test_decide_retrieval_parses_structured_output(self):
        llm = _StubLLM(['{"need_retrieval": false, "reasoning": "greeting"}'])
        result = decide_retrieval("hello", llm)
        assert isinstance(result, RetrievalDecision)
        assert result.need_retrieval is False

    def test_check_support_failure_fallback(self):
        llm = _StubLLM(["garbage response"])
        result = check_support("q", "draft", "context", llm)
        assert result.supported is True

    def test_score_utility_refuse(self):
        llm = _StubLLM(
            [
                json.dumps(
                    {
                        "score": 0.1,
                        "action": "refuse",
                        "reasoning": "unsupported",
                        "refined_query": "",
                    }
                )
            ]
        )
        result = score_utility("q", "bad draft", "ctx", llm)
        assert result.action == UtilityAction.REFUSE


def _chat_mock(
    *,
    retrieval_chunks: list[Chunk] | None = None,
    draft_answer: str = "supported draft answer",
    direct_answer: str = "Hello!",
) -> MagicMock:
    m = MagicMock()
    m.retrieval.retrieve = AsyncMock(return_value=_retrieval_result(retrieval_chunks))
    m.generation.generate.return_value = Answer(
        query_id="q1",
        text=draft_answer,
        sources=["c0"],
    )
    m.generation.generate_direct.return_value = Answer(
        query_id="q1",
        text=direct_answer,
        sources=[],
    )
    m.generation.call_llm.return_value = ""
    m.retrieval.service.hybrid.graph = None
    return m


def _self_rag_responses(
    *,
    need_retrieval: bool = True,
    supported: bool = True,
    utility_action: str = "accept",
    refined_query: str = "",
) -> list[str]:
    return [
        _retrieval_gate(need_retrieval=need_retrieval, reasoning="retrieval gate"),
        _support_gate(supported=supported, reasoning="support gate"),
        _utility_gate(
            action=utility_action,
            score=0.9,
            reasoning="utility gate",
            refined_query=refined_query,
        ),
    ]


def _retrieval_gate(*, need_retrieval: bool, reasoning: str) -> str:
    return json.dumps({"need_retrieval": need_retrieval, "reasoning": reasoning})


def _support_gate(*, supported: bool, reasoning: str) -> str:
    return json.dumps({"supported": supported, "reasoning": reasoning})


def _utility_gate(
    *,
    action: str,
    score: float,
    reasoning: str,
    refined_query: str = "",
) -> str:
    return json.dumps(
        {
            "score": score,
            "action": action,
            "reasoning": reasoning,
            "refined_query": refined_query,
        }
    )


def _unsupported_reretrieve_accept_responses(
    *,
    refined_query: str = "kubernetes deployment steps",
) -> list[str]:
    """First iteration unsupported + reretrieve; second iteration grounded + accept."""
    return [
        _retrieval_gate(need_retrieval=True, reasoning="needs docs"),
        _support_gate(supported=False, reasoning="not grounded"),
        _utility_gate(
            action="reretrieve",
            score=0.2,
            reasoning="try better query",
            refined_query=refined_query,
        ),
        _support_gate(supported=True, reasoning="grounded"),
        _utility_gate(action="accept", score=0.9, reasoning="good"),
    ]


async def _run_self_rag(
    side_effect: list[str],
    *,
    question: str = "kubernetes",
    max_iterations: int = 3,
    chat: MagicMock | None = None,
) -> tuple[MagicMock, AgentRunResult]:
    chat = chat or _chat_mock()
    chat.generation.call_llm.side_effect = side_effect
    pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=max_iterations)
    return chat, await pipeline.chat_full(question)


async def _run_unsupported_reretrieve_accept(
    *,
    question: str = "kubernetes",
    refined_query: str = "kubernetes deployment steps",
) -> tuple[MagicMock, AgentRunResult]:
    chat, result = await _run_self_rag(
        _unsupported_reretrieve_accept_responses(refined_query=refined_query),
        question=question,
    )
    assert result.actions == [AgentAction.RETRIEVE_MORE, AgentAction.ANSWER]
    assert chat.retrieval.retrieve.await_count == 2
    return chat, result


def _chat_with_empty_then_populated_retrieval() -> MagicMock:
    chat = _chat_mock()
    chat.retrieval.retrieve = AsyncMock(
        side_effect=[
            _retrieval_result([], context=""),
            _retrieval_result([_chunk("c0")]),
        ]
    )
    return chat


def _empty_context_reretrieve_accept_responses(
    *,
    refined_query: str = "kubernetes deployment",
) -> list[str]:
    """Empty first retrieval + reretrieve; second iteration grounded + accept."""
    return [
        _retrieval_gate(need_retrieval=True, reasoning="needs docs"),
        _utility_gate(
            action="reretrieve",
            score=0.2,
            reasoning="empty index",
            refined_query=refined_query,
        ),
        _support_gate(supported=True, reasoning="grounded"),
        _utility_gate(action="accept", score=0.9, reasoning="good"),
    ]


async def _run_empty_context_reretrieve_accept(
    *,
    question: str = "k8s",
    refined_query: str = "kubernetes deployment",
) -> tuple[MagicMock, AgentRunResult]:
    chat, result = await _run_self_rag(
        _empty_context_reretrieve_accept_responses(refined_query=refined_query),
        question=question,
        chat=_chat_with_empty_then_populated_retrieval(),
    )
    assert result.actions == [AgentAction.RETRIEVE_MORE, AgentAction.ANSWER]
    assert chat.retrieval.retrieve.await_count == 2
    return chat, result


class TestAgentPipelineSelfRAG:
    @pytest.mark.asyncio
    async def test_disabled_uses_standard_agent_loop(self):
        chat = _chat_mock()
        chat.generation.call_llm.return_value = (
            '{"action":"ANSWER","reasoning":"ok","refined_query":"",'
            '"entities":[],"clarification":""}'
        )
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=False)
        result = await pipeline.chat_full("kubernetes question")
        assert result.self_rag_decisions == []
        assert result.actions == [AgentAction.ANSWER]

    @pytest.mark.asyncio
    async def test_enabled_accepts_supported_draft(self):
        chat = _chat_mock()
        side_effect = _self_rag_responses()
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=3)
        result = await pipeline.chat_full("kubernetes question")
        assert len(result.self_rag_decisions) == 1
        assert result.self_rag_decisions[0].need_retrieval is True
        assert result.self_rag_decisions[0].supported is True
        assert result.self_rag_decisions[0].utility_action == "accept"
        assert result.answer.text == "supported draft answer"

    @pytest.mark.asyncio
    async def test_refuses_after_max_iterations_when_unsupported(self):
        chat = _chat_mock()
        side_effect = [
            json.dumps({"need_retrieval": True, "reasoning": "needs docs"}),
            json.dumps({"supported": False, "reasoning": "not in context"}),
            json.dumps(
                {
                    "score": 0.2,
                    "action": "reretrieve",
                    "reasoning": "retry",
                    "refined_query": "kubernetes details",
                }
            ),
            json.dumps({"supported": False, "reasoning": "still unsupported"}),
            json.dumps(
                {
                    "score": 0.1,
                    "action": "reretrieve",
                    "reasoning": "still weak",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=2)
        result = await pipeline.chat_full("kubernetes question")
        assert result.answer.text == "I don't have information about this."
        assert len(result.self_rag_decisions) == 2
        assert all(step.supported is False for step in result.self_rag_decisions)
        assert result.actions == [AgentAction.RETRIEVE_MORE, AgentAction.RETRIEVE_MORE]

    @pytest.mark.asyncio
    async def test_no_retrieval_path(self):
        chat = _chat_mock(direct_answer="Hello!")
        side_effect = [
            json.dumps({"need_retrieval": False, "reasoning": "greeting"}),
            json.dumps(
                {
                    "score": 0.95,
                    "action": "accept",
                    "reasoning": "fine greeting",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True)
        result = await pipeline.chat_full("hello")
        assert result.self_rag_decisions[0].need_retrieval is False
        assert result.answer.text == "Hello!"
        assert result.actions == [AgentAction.ANSWER]
        chat.generation.generate_direct.assert_called_once_with("hello")
        chat.generation.generate.assert_not_called()
        chat.retrieval.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_retrieval_refuse_labels_clarify_action(self):
        chat = _chat_mock(direct_answer="Hello!")
        side_effect = [
            json.dumps({"need_retrieval": False, "reasoning": "greeting"}),
            json.dumps(
                {
                    "score": 0.1,
                    "action": "refuse",
                    "reasoning": "inappropriate",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True)
        result = await pipeline.chat_full("hello")
        assert result.actions == [AgentAction.CLARIFY]
        assert result.answer.text == "I don't have information about this."

    @pytest.mark.asyncio
    async def test_no_retrieval_reretrieve_skips_retrieval_gate(self):
        """After utility reretrieve, retrieval must run even if decide_retrieval says no."""
        chat = _chat_mock(direct_answer="partial guess")
        side_effect = [
            json.dumps({"need_retrieval": False, "reasoning": "maybe greeting"}),
            json.dumps(
                {
                    "score": 0.4,
                    "action": "reretrieve",
                    "reasoning": "needs docs",
                    "refined_query": "kubernetes docs",
                }
            ),
            json.dumps({"supported": True, "reasoning": "grounded"}),
            json.dumps(
                {
                    "score": 0.9,
                    "action": "accept",
                    "reasoning": "good",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=3)
        result = await pipeline.chat_full("tell me about k8s")
        assert result.actions == [AgentAction.RETRIEVE_MORE, AgentAction.ANSWER]
        assert chat.retrieval.retrieve.await_count == 1
        assert result.self_rag_decisions[1].need_retrieval is True
        assert "reretrieve" in result.self_rag_decisions[1].retrieval_reasoning
        chat.generation.generate_direct.assert_called_once()
        assert chat.generation.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_unsupported_reretrieve_skips_retrieval_gate(self):
        _, result = await _run_unsupported_reretrieve_accept()
        assert result.self_rag_decisions[1].need_retrieval is True
        assert "reretrieve" in result.self_rag_decisions[1].retrieval_reasoning

    @pytest.mark.asyncio
    async def test_supported_reretrieve_skips_retrieval_gate(self):
        chat = _chat_mock()
        side_effect = [
            json.dumps({"need_retrieval": True, "reasoning": "needs docs"}),
            json.dumps({"supported": True, "reasoning": "ok"}),
            json.dumps(
                {
                    "score": 0.4,
                    "action": "reretrieve",
                    "reasoning": "incomplete",
                    "refined_query": "kubernetes rollout",
                }
            ),
            json.dumps({"supported": True, "reasoning": "grounded"}),
            json.dumps(
                {
                    "score": 0.9,
                    "action": "accept",
                    "reasoning": "good",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=3)
        result = await pipeline.chat_full("kubernetes")
        assert result.actions == [AgentAction.RETRIEVE_MORE, AgentAction.ANSWER]
        assert chat.retrieval.retrieve.await_count == 2
        assert result.self_rag_decisions[1].need_retrieval is True

    @pytest.mark.asyncio
    async def test_empty_context_reretrieve_skips_retrieval_gate(self):
        _, result = await _run_empty_context_reretrieve_accept()
        assert result.self_rag_decisions[1].need_retrieval is True

    @pytest.mark.asyncio
    async def test_no_retrieval_reretrieve_labels_retrieve_more(self):
        chat = _chat_mock(direct_answer="partial")
        side_effect = [
            json.dumps({"need_retrieval": False, "reasoning": "maybe docs"}),
            json.dumps(
                {
                    "score": 0.4,
                    "action": "reretrieve",
                    "reasoning": "needs docs",
                    "refined_query": "kubernetes docs",
                }
            ),
            json.dumps({"supported": True, "reasoning": "grounded"}),
            json.dumps(
                {
                    "score": 0.9,
                    "action": "accept",
                    "reasoning": "good",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=3)
        result = await pipeline.chat_full("tell me about k8s")
        assert result.actions == [AgentAction.RETRIEVE_MORE, AgentAction.ANSWER]
        assert chat.retrieval.retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_unsupported_retry_uses_refined_query(self):
        chat, result = await _run_unsupported_reretrieve_accept()
        second_query = chat.retrieval.retrieve.await_args_list[1].args[0].text
        assert second_query == "kubernetes deployment steps"
        assert result.answer.text == "supported draft answer"

    @pytest.mark.asyncio
    async def test_chat_streams_self_rag_answer(self):
        chat = _chat_mock()
        chat.generation.call_llm.side_effect = _self_rag_responses()
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True)
        stream = await pipeline.chat("kubernetes question")
        tokens = [token async for token in stream]
        assert "".join(tokens) == "supported draft answer"

    @pytest.mark.asyncio
    async def test_chat_stream_preserves_whitespace(self):
        formatted = "Line one\n\n  - bullet\n  - item"
        chat = _chat_mock(draft_answer=formatted)
        chat.generation.call_llm.side_effect = _self_rag_responses()
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True)
        stream = await pipeline.chat("kubernetes question")
        tokens = [token async for token in stream]
        assert "".join(tokens) == formatted

    @pytest.mark.asyncio
    async def test_generate_uses_original_question_after_reretrieve(self):
        original = "tell me about k8s"
        chat, _ = await _run_self_rag(
            _unsupported_reretrieve_accept_responses(),
            question=original,
        )

        assert chat.generation.generate.call_count == 2  # type: ignore[attr-defined]
        for call in chat.generation.generate.call_args_list:  # type: ignore[attr-defined]
            assert call.args[0] == original

        second_query = chat.retrieval.retrieve.await_args_list[1].args[0].text
        assert second_query == "kubernetes deployment steps"
        assert second_query != original

    @pytest.mark.asyncio
    async def test_empty_context_retries_with_refined_query(self):
        chat, _ = await _run_empty_context_reretrieve_accept()
        second_query = chat.retrieval.retrieve.await_args_list[1].args[0].text
        assert second_query == "kubernetes deployment"

    @pytest.mark.asyncio
    async def test_empty_context_on_last_iteration_returns_no_info(self):
        chat = _chat_mock()
        chat.retrieval.retrieve = AsyncMock(return_value=_retrieval_result([], context=""))
        side_effect = [
            json.dumps({"need_retrieval": True, "reasoning": "needs docs"}),
            json.dumps(
                {
                    "score": 0.1,
                    "action": "reretrieve",
                    "reasoning": "nothing found",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=1)
        result = await pipeline.chat_full("k8s")
        assert result.answer.text == "I don't have information about this."
        assert result.actions == [AgentAction.RETRIEVE_MORE]

    @pytest.mark.asyncio
    async def test_empty_context_with_utility_refuse_returns_no_info(self):
        chat = _chat_mock()
        chat.retrieval.retrieve = AsyncMock(return_value=_retrieval_result([], context=""))
        side_effect = [
            json.dumps({"need_retrieval": True, "reasoning": "needs docs"}),
            json.dumps(
                {
                    "score": 0.0,
                    "action": "refuse",
                    "reasoning": "no relevant documents",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=3)
        result = await pipeline.chat_full("kubernetes")
        assert result.answer.text == "I don't have information about this."
        assert result.actions == [AgentAction.CLARIFY]
        assert chat.retrieval.retrieve.await_count == 1

    @pytest.mark.asyncio
    async def test_empty_context_refuse_on_last_iteration_labels_clarify(self):
        """Utility refuse must win over last-iteration reretrieve exhaustion."""
        chat = _chat_mock()
        chat.retrieval.retrieve = AsyncMock(return_value=_retrieval_result([], context=""))
        side_effect = [
            json.dumps({"need_retrieval": True, "reasoning": "needs docs"}),
            json.dumps(
                {
                    "score": 0.0,
                    "action": "refuse",
                    "reasoning": "no relevant documents",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=1)
        result = await pipeline.chat_full("kubernetes")
        assert result.answer.text == "I don't have information about this."
        assert result.actions == [AgentAction.CLARIFY]
        assert result.self_rag_decisions[-1].utility_action == "refuse"

    @pytest.mark.asyncio
    async def test_unsupported_with_utility_refuse_returns_no_info(self):
        chat = _chat_mock()
        side_effect = [
            json.dumps({"need_retrieval": True, "reasoning": "needs docs"}),
            json.dumps({"supported": False, "reasoning": "hallucinated"}),
            json.dumps(
                {
                    "score": 0.0,
                    "action": "refuse",
                    "reasoning": "unsupported claims",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=3)
        result = await pipeline.chat_full("kubernetes")
        assert result.answer.text == "I don't have information about this."
        assert result.actions == [AgentAction.CLARIFY]

    @pytest.mark.asyncio
    async def test_unsupported_refuse_on_last_iteration_labels_clarify(self):
        """Utility refuse on the final retry iteration must not be labeled RETRIEVE_MORE."""
        chat = _chat_mock()
        side_effect = [
            json.dumps({"need_retrieval": True, "reasoning": "needs docs"}),
            json.dumps({"supported": False, "reasoning": "not grounded"}),
            json.dumps(
                {
                    "score": 0.2,
                    "action": "reretrieve",
                    "reasoning": "retry",
                    "refined_query": "kubernetes details",
                }
            ),
            json.dumps({"supported": False, "reasoning": "still unsupported"}),
            json.dumps(
                {
                    "score": 0.0,
                    "action": "refuse",
                    "reasoning": "unsupported claims",
                    "refined_query": "",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=2)
        result = await pipeline.chat_full("kubernetes")
        assert result.answer.text == "I don't have information about this."
        assert result.actions == [AgentAction.RETRIEVE_MORE, AgentAction.CLARIFY]
        assert result.self_rag_decisions[-1].utility_action == "refuse"

    @pytest.mark.asyncio
    async def test_supported_reretrieve_exhausts_iterations_returns_last_grounded_draft(self):
        chat = _chat_mock()
        side_effect = [
            json.dumps({"need_retrieval": True, "reasoning": "needs docs"}),
            json.dumps({"supported": True, "reasoning": "ok"}),
            json.dumps(
                {
                    "score": 0.4,
                    "action": "reretrieve",
                    "reasoning": "incomplete",
                    "refined_query": "kubernetes rollout",
                }
            ),
            json.dumps({"supported": True, "reasoning": "ok"}),
            json.dumps(
                {
                    "score": 0.3,
                    "action": "reretrieve",
                    "reasoning": "still incomplete",
                    "refined_query": "kubernetes rollout details",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=2)
        result = await pipeline.chat_full("kubernetes")
        assert result.answer.text == "supported draft answer"
        assert result.actions == [AgentAction.RETRIEVE_MORE, AgentAction.ANSWER]
        assert result.self_rag_decisions[-1].utility_action == "reretrieve"
        assert chat.retrieval.retrieve.await_count == 2

    @pytest.mark.asyncio
    async def test_no_retrieval_reretrieve_exhausts_iterations_returns_no_info(self):
        """Direct drafts skip support check — no grounded draft to fall back to."""
        chat = _chat_mock(direct_answer="partial guess")
        side_effect = [
            json.dumps({"need_retrieval": False, "reasoning": "maybe greeting"}),
            json.dumps(
                {
                    "score": 0.4,
                    "action": "reretrieve",
                    "reasoning": "needs docs",
                    "refined_query": "kubernetes docs",
                }
            ),
        ]
        chat.generation.call_llm.side_effect = side_effect
        pipeline = AgentPipeline(pipeline=chat, self_rag_enabled=True, max_iterations=1)
        result = await pipeline.chat_full("hello")
        assert result.answer.text == "I don't have information about this."
        assert result.actions == [AgentAction.RETRIEVE_MORE]
        chat.retrieval.retrieve.assert_not_called()


class TestAgentChatResponseSelfRAG:
    def test_from_run_includes_self_rag_decisions(self):
        from src.api.schemas.agent import AgentChatResponse
        from src.rag.pipelines.agent_pipeline import SelfRAGStepDecision

        run = AgentRunResult(
            answer=Answer(query_id="q1", text="answer", sources=["c0"]),
            iterations=1,
            actions=[AgentAction.ANSWER],
            self_rag_decisions=[
                SelfRAGStepDecision(
                    iteration=1,
                    need_retrieval=True,
                    retrieval_reasoning="needs docs",
                    supported=True,
                    support_reasoning="grounded",
                    utility_score=0.9,
                    utility_action="accept",
                    utility_reasoning="useful",
                )
            ],
        )
        response = AgentChatResponse.from_run(run)
        assert len(response.self_rag_decisions) == 1
        assert response.self_rag_decisions[0].utility_action == "accept"
