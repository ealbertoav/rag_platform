"""T-031 — GenerationService and ChatPipeline tests."""

from __future__ import annotations

import json
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
_ANSWER_TEXT = "answer text"
_SAMPLE_EXPLANATIONS = [
    {"chunk_id": "c0", "reason": "Mentions the topic."},
    {"chunk_id": "c1", "reason": "Adds supporting detail."},
]
_SAMPLE_HIGHLIGHT_ITEMS: list[dict[str, object]] = [
    {"chunk_id": "c0", "spans": ["relevant text 0"]},
    {"chunk_id": "c1", "spans": ["relevant text 1"]},
]
_SAMPLE_HIGHLIGHTS = {
    "c0": ["relevant text 0"],
    "c1": ["relevant text 1"],
}


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
    *,
    source_highlighting_enabled: bool = False,
) -> ChatPipeline:
    llm = llm or _llm_mock()
    return ChatPipeline(
        retrieval=_retrieval_mock(context, chunks),
        generation=_service(llm),
        llm=llm,
        source_highlighting_enabled=source_highlighting_enabled,
    )


def _explain_json(
    explanations: list[dict[str, str]] | None = None,
    *,
    highlights: list[dict[str, object]] | None = None,
) -> str:
    payload: dict[str, object] = {"explanations": explanations or _SAMPLE_EXPLANATIONS}
    if highlights is not None:
        payload["highlights"] = highlights
    return json.dumps(payload)


def _highlights_json(
    highlights: list[dict[str, object]] | None = None,
) -> str:
    return json.dumps({"highlights": highlights or _SAMPLE_HIGHLIGHT_ITEMS})


def _assert_both_post_gen(result: Answer, llm: MagicMock, *, call_count: int) -> None:
    assert result.explanations is not None
    assert len(result.explanations) == 2
    assert result.highlights == _SAMPLE_HIGHLIGHTS
    assert llm.generate.call_count == call_count


def _llm_with_post_gen_responses(*responses: str) -> MagicMock:
    llm = _llm_mock(_ANSWER_TEXT)
    llm.generate.side_effect = [_ANSWER_TEXT, *responses]
    return llm


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
        retrieval = _retrieval_mock()
        p = ChatPipeline(
            retrieval=retrieval,
            generation=_service(),
            llm=_llm_mock(),
        )
        await p.chat("question")
        retrieval.retrieve.assert_called_once()


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
        llm = _llm_with_post_gen_responses(_explain_json())
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
        mock_monotonic.side_effect = [0.0, 0.5, 0.5, 0.5]
        llm = _llm_with_post_gen_responses(_explain_json())
        result = await _pipeline(llm=llm).chat_full("q", explain=True)
        assert result.latency_ms == pytest.approx(500.0)

    @pytest.mark.asyncio
    async def test_chat_full_highlighting_disabled_skips_highlights(self):
        llm = _llm_mock("answer text")
        result = await _pipeline(llm=llm).chat_full("q")
        assert result.highlights is None
        assert llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_chat_full_highlighting_enabled_attaches_highlights(self):
        llm = _llm_with_post_gen_responses(_highlights_json())
        result = await _pipeline(llm=llm, source_highlighting_enabled=True).chat_full("q")
        assert result.highlights == _SAMPLE_HIGHLIGHTS
        assert llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_chat_full_highlighting_failure_omits_highlights(self):
        llm = _llm_mock("answer text")
        llm.generate.side_effect = ["answer text", "not json"]
        result = await _pipeline(llm=llm, source_highlighting_enabled=True).chat_full("q")
        assert result.highlights is None

    @pytest.mark.asyncio
    async def test_chat_full_highlights_param_without_config(self):
        llm = _llm_with_post_gen_responses(
            json.dumps({"highlights": [{"chunk_id": "c0", "spans": ["relevant text 0"]}]})
        )
        result = await _pipeline(llm=llm).chat_full("q", highlights=True)
        assert result.highlights == {"c0": ["relevant text 0"]}
        assert llm.generate.call_count == 2

    @pytest.mark.asyncio
    async def test_chat_full_explain_and_highlights_use_single_combined_llm_call(self):
        llm = _llm_with_post_gen_responses(_explain_json(highlights=_SAMPLE_HIGHLIGHT_ITEMS))
        result = await _pipeline(llm=llm).chat_full("q", explain=True, highlights=True)
        _assert_both_post_gen(result, llm, call_count=2)

    @pytest.mark.asyncio
    async def test_chat_full_combined_explanations_only_falls_back_to_highlights(self):
        llm = _llm_with_post_gen_responses(
            json.dumps({"explanations": _SAMPLE_EXPLANATIONS}),
            _highlights_json(),
        )
        result = await _pipeline(llm=llm).chat_full("q", explain=True, highlights=True)
        _assert_both_post_gen(result, llm, call_count=3)

    @pytest.mark.asyncio
    async def test_chat_full_combined_highlights_only_falls_back_to_explain(self):
        llm = _llm_with_post_gen_responses(_highlights_json(), _explain_json())
        result = await _pipeline(llm=llm).chat_full("q", explain=True, highlights=True)
        _assert_both_post_gen(result, llm, call_count=3)

    @pytest.mark.asyncio
    async def test_chat_full_explain_with_global_highlighting_falls_back_on_combined_failure(
        self,
    ):
        llm = _llm_mock("answer text")
        llm.generate.side_effect = ["answer text", "not json", _explain_json(), "not json"]
        result = await _pipeline(llm=llm, source_highlighting_enabled=True).chat_full(
            "q",
            explain=True,
        )
        assert result.explanations is not None
        assert len(result.explanations) == 2
        assert result.highlights is None
        assert llm.generate.call_count == 4

    @pytest.mark.asyncio
    async def test_chat_full_combined_failure_falls_back_to_dedicated_paths(self):
        llm = _llm_mock("answer text")
        llm.generate.side_effect = [
            "answer text",
            "not json",
            _explain_json(),
            _highlights_json(),
        ]
        result = await _pipeline(llm=llm).chat_full("q", explain=True, highlights=True)
        _assert_both_post_gen(result, llm, call_count=4)


class TestChatPipelineAttributes:
    def test_retrieval_attribute(self):
        retrieval = _retrieval_mock()
        p = ChatPipeline(retrieval=retrieval, generation=_service())
        assert p.retrieval is retrieval

    def test_generation_attribute(self):
        generation = _service()
        p = ChatPipeline(retrieval=_retrieval_mock(), generation=generation)
        assert p.generation is generation


class TestChatPipelineBenchmark:
    @pytest.mark.asyncio
    async def test_benchmark_returns_answer_and_context_texts(self):
        chunks = [_chunk(0), _chunk(1)]
        p = _pipeline(chunks=chunks)
        run = await p.benchmark("question")
        assert run.answer.text == "LLM answer"
        assert run.context_texts == [c.text for c in chunks]
        assert run.parametric_answer is False
        assert set(run.answer.sources) == {"c0", "c1"}


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
