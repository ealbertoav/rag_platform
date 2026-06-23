"""T-031 — GenerationService and ChatPipeline tests."""
from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.services.generation_service import GenerationService
from src.rag.pipelines.chat_pipeline import ChatPipeline

# ── helpers ────────────────────────────────────────────────────────────────────


_NO_INFO = "I don't have information about this."


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=f"relevant text {i}")


async def _async_tokens(*tokens: str) -> AsyncIterator[str]:
    for t in tokens:
        yield t


def _llm_mock(response: str = "LLM answer") -> MagicMock:
    m = MagicMock()
    m.generate.return_value = response
    m.generate_stream.return_value = _async_tokens("LLM", " answer")
    return m


def _service(llm: MagicMock | None = None) -> GenerationService:
    return GenerationService(llm=llm or _llm_mock())


def _retrieval_mock(
    context: str = "relevant context",
    chunks: list[Chunk] | None = None,
) -> MagicMock:
    from src.domain.services.retrieval_service import RetrievalResult

    chunks = chunks or [_chunk(0), _chunk(1)]
    result = RetrievalResult(
        query=Query(text="q"),
        chunks=chunks,
        context=context,
    )
    m = MagicMock()
    m.retrieve = AsyncMock(return_value=result)
    return m


def _pipeline(
    context: str = "relevant context",
    chunks: list[Chunk] | None = None,
    llm: MagicMock | None = None,
) -> ChatPipeline:
    llm = llm or _llm_mock()
    return ChatPipeline(
        retrieval=_retrieval_mock(context, chunks),
        generation=_service(llm),
    )


# ── GenerationService ──────────────────────────────────────────────────────────


class TestGenerationService:
    def test_generate_returns_answer(self):
        result = _service().generate("q", "ctx", ["c0"])
        assert isinstance(result, Answer)

    def test_generate_text_from_llm(self):
        llm = _llm_mock("my answer")
        result = _service(llm).generate("q", "ctx", ["c0"])
        assert result.text == "my answer"

    def test_generate_sources_set(self):
        result = _service().generate("q", "ctx", ["c0", "c1"])
        assert result.sources == ["c0", "c1"]

    def test_generate_empty_context_no_info(self):
        result = _service().generate("q", "", [])
        assert result.text == _NO_INFO
        assert result.sources == []

    def test_generate_blank_context_no_info(self):
        result = _service().generate("q", "   ", [])
        assert result.text == _NO_INFO

    def test_generate_empty_context_no_llm_call(self):
        llm = _llm_mock()
        _service(llm).generate("q", "", [])
        llm.generate.assert_not_called()

    def test_generate_passes_question_as_context_arg(self):
        llm = _llm_mock()
        _service(llm).generate("my question", "ctx", [])
        _, kwargs = llm.generate.call_args
        assert kwargs["context"] == "my question"

    def test_stream_returns_async_iterator(self):
        stream = _service().stream("q", "ctx")
        assert hasattr(stream, "__aiter__")

    @pytest.mark.asyncio
    async def test_stream_empty_context_yields_no_info(self):
        tokens = [t async for t in _service().stream("q", "")]
        assert "".join(tokens) == _NO_INFO

    @pytest.mark.asyncio
    async def test_stream_calls_generate_stream(self):
        llm = _llm_mock()
        stream = _service(llm).stream("q", "ctx")
        _ = [t async for t in stream]
        llm.generate_stream.assert_called_once()

    def test_from_settings_returns_service(self):
        svc = GenerationService.from_settings(_llm_mock())
        assert isinstance(svc, GenerationService)


# ── ChatPipeline ───────────────────────────────────────────────────────────────


class TestChatPipelineStream:
    @pytest.mark.asyncio
    async def test_chat_returns_async_iterator(self):
        stream = await _pipeline().chat("question")
        assert hasattr(stream, "__aiter__")

    @pytest.mark.asyncio
    async def test_chat_yields_tokens(self):
        llm = _llm_mock()
        llm.generate_stream.return_value = _async_tokens("tok1", "tok2")
        tokens = [t async for t in await _pipeline(llm=llm).chat("q")]
        assert tokens == ["tok1", "tok2"]

    @pytest.mark.asyncio
    async def test_chat_empty_context_yields_no_info(self):
        tokens = [t async for t in await _pipeline(context="").chat("q")]
        assert "".join(tokens) == _NO_INFO

    @pytest.mark.asyncio
    async def test_chat_calls_retrieval(self):
        p = _pipeline()
        await p.chat("question")
        p._retrieval.retrieve.assert_called_once()  # type: ignore[attr-defined]


class TestChatPipelineFull:
    @pytest.mark.asyncio
    async def test_chat_full_returns_answer(self):
        result = await _pipeline().chat_full("question")
        assert isinstance(result, Answer)

    @pytest.mark.asyncio
    async def test_chat_full_text_from_llm(self):
        llm = _llm_mock("complete response")
        result = await _pipeline(llm=llm).chat_full("q")
        assert result.text == "complete response"

    @pytest.mark.asyncio
    async def test_chat_full_sources_from_chunks(self):
        chunks = [_chunk(0), _chunk(1)]
        result = await _pipeline(chunks=chunks).chat_full("q")
        assert set(result.sources) == {"c0", "c1"}

    @pytest.mark.asyncio
    async def test_chat_full_latency_positive(self):
        result = await _pipeline().chat_full("q")
        assert result.latency_ms > 0

    @pytest.mark.asyncio
    async def test_chat_full_token_count_positive(self):
        llm = _llm_mock("one two three")
        result = await _pipeline(llm=llm).chat_full("q")
        assert result.token_count > 0

    @pytest.mark.asyncio
    async def test_chat_full_empty_context_no_info(self):
        result = await _pipeline(context="").chat_full("q")
        assert result.text == _NO_INFO
        assert result.sources == []

    @pytest.mark.asyncio
    async def test_chat_full_query_id_set(self):
        result = await _pipeline().chat_full("q")
        assert result.query_id != ""
