"""T-031 — GenerationService and ChatPipeline tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.constants import CHUNK_INDEX_KEY, MERGED_CHUNK_IDS_KEY
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
        llm=llm,
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

    def test_generate_direct_calls_llm(self):
        llm = _llm_mock("Hello there!")
        result = _service(llm).generate_direct("hello")
        assert result.text == "Hello there!"
        assert result.sources == []
        llm.generate.assert_called_once_with(prompt="hello", context="")

    def test_generate_direct_does_not_use_rag_prompt(self):
        llm = _llm_mock("Hi!")
        _service(llm).generate_direct("hello")
        prompt_arg = llm.generate.call_args.kwargs["prompt"]
        assert "Context:" not in prompt_arg

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
    async def test_chat_full_sources_expand_rse_merged_chunk(self):
        merged = Chunk(
            id="c0",
            document_id="doc",
            text="merged segment",
            metadata={
                CHUNK_INDEX_KEY: 0,
                MERGED_CHUNK_IDS_KEY: ["c0", "c1", "c2"],
            },
        )
        result = await _pipeline(chunks=[merged]).chat_full("q")
        assert set(result.sources) == {"c0", "c1", "c2"}

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

    @pytest.mark.asyncio
    async def test_chat_full_explain_false_skips_explanations(self):
        llm = _llm_mock("answer text")
        result = await _pipeline(llm=llm).chat_full("q", explain=False)
        assert result.explanations is None
        assert llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_chat_full_explain_true_attaches_explanations(self):
        import json

        llm = _llm_mock("answer text")
        llm.generate.side_effect = [
            "answer text",
            json.dumps(
                {
                    "explanations": [
                        {"chunk_id": "c0", "reason": "Mentions the topic."},
                        {"chunk_id": "c1", "reason": "Adds supporting detail."},
                    ]
                }
            ),
        ]
        result = await _pipeline(llm=llm).chat_full("q", explain=True)
        assert result.explanations is not None
        assert len(result.explanations) == 2
        assert result.explanations[0].chunk_id == "c0"
        assert llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_chat_full_explain_failure_omits_explanations(self):
        llm = _llm_mock("answer text")
        llm.generate.side_effect = ["answer text", "not json"]
        result = await _pipeline(llm=llm).chat_full("q", explain=True)
        assert result.explanations is None

    @pytest.mark.asyncio
    @patch("src.rag.pipelines.chat_pipeline.time.monotonic")
    async def test_chat_full_latency_includes_explain(self, mock_monotonic):
        import json

        mock_monotonic.side_effect = [0.0, 0.5]
        llm = _llm_mock("answer text")
        llm.generate.side_effect = [
            "answer text",
            json.dumps(
                {
                    "explanations": [
                        {"chunk_id": "c0", "reason": "Mentions the topic."},
                        {"chunk_id": "c1", "reason": "Adds supporting detail."},
                    ]
                }
            ),
        ]
        result = await _pipeline(llm=llm).chat_full("q", explain=True)
        assert result.latency_ms == pytest.approx(500.0)


class TestChatPipelineProperties:
    def test_retrieval_property(self):
        p = _pipeline()
        assert p.retrieval is p._retrieval

    def test_generation_property(self):
        p = _pipeline()
        assert p.generation is p._generation


class TestChatPipelineBenchmark:
    @pytest.mark.asyncio
    async def test_benchmark_returns_answer_and_context_texts(self):
        chunks = [_chunk(0), _chunk(1)]
        p = _pipeline(chunks=chunks)
        answer, context_texts = await p.benchmark("question")
        assert answer.text == "LLM answer"
        assert context_texts == [c.text for c in chunks]
        assert set(answer.sources) == {"c0", "c1"}


class TestChatPipelineFromSettings:
    def test_from_settings_builds_pipeline(self):
        mock_retrieval = MagicMock()
        mock_generation = MagicMock()
        with (
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as mock_llm,
            patch(
                "src.rag.pipelines.chat_pipeline.RetrievalPipeline.from_settings",
                return_value=mock_retrieval,
            ),
            patch(
                "src.rag.pipelines.chat_pipeline.GenerationService.from_settings",
                return_value=mock_generation,
            ) as mock_gen,
        ):
            mock_llm.return_value = MagicMock()
            pipeline = ChatPipeline.from_settings()

        assert isinstance(pipeline, ChatPipeline)
        assert pipeline.retrieval is mock_retrieval
        assert pipeline.generation is mock_generation
        mock_gen.assert_called_once()
