"""T-142 — Corrective RAG (CRAG) web search fallback tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.repositories.web_search_repository import WebSearchResult
from src.domain.services.generation_service import GenerationService
from src.domain.services.retrieval_service import RetrievalResult
from src.infrastructure.search.web_search import (
    DuckDuckGoWebSearchProvider,
    NullWebSearchProvider,
    TavilyWebSearchProvider,
    format_web_results,
    get_web_search_provider,
    parse_duckduckgo_lite,
)
from src.rag.pipelines.chat_pipeline import ChatPipeline
from src.rag.quality.crag import (
    CRAGAction,
    determine_crag_action,
    eval_contexts_for_resolution,
    explainable_chunks_for_resolution,
    refine_knowledge,
    score_retrieval_quality,
)
from tests.unit.web_search_helpers import patch_web_search_httpx_post


def _chunk(chunk_id: str, *, relevance_score: float | None = None) -> Chunk:
    metadata: dict[str, object] = {}
    if relevance_score is not None:
        metadata["relevance_score"] = relevance_score
    return Chunk(id=chunk_id, document_id="doc-1", text=f"text for {chunk_id}", metadata=metadata)


class TestScoreRetrievalQuality:
    def test_empty_chunks_graded_as_zero(self):
        result = score_retrieval_quality([])
        assert result.score == pytest.approx(0.0)
        assert result.graded is True

    def test_mean_of_relevance_scores(self):
        chunks = [
            _chunk("c0", relevance_score=0.9),
            _chunk("c1", relevance_score=0.3),
        ]
        result = score_retrieval_quality(chunks)
        assert result.score == pytest.approx(0.6)
        assert result.graded is True

    def test_ungraded_chunks_not_graded(self):
        result = score_retrieval_quality([_chunk("c0")])
        assert result.graded is False

    def test_pre_graded_scores_include_filtered_out(self):
        """CRAG uses all Reliable RAG grades, not only surviving chunks."""
        kept = [_chunk("c2", relevance_score=0.55)]
        result = score_retrieval_quality(kept, relevance_scores=[0.1, 0.2, 0.55])
        assert result.score == pytest.approx((0.1 + 0.2 + 0.55) / 3)
        assert result.graded is True
        assert (
            determine_crag_action(result.score, lower_threshold=0.3, upper_threshold=0.7)
            == CRAGAction.WEB_ONLY
        )

    def test_empty_pre_graded_scores(self):
        result = score_retrieval_quality([], relevance_scores=[])
        assert result.score == pytest.approx(0.0)
        assert result.graded is True


class TestDetermineCRAGAction:
    def test_high_score_uses_retrieval(self):
        action = determine_crag_action(0.85, lower_threshold=0.3, upper_threshold=0.7)
        assert action == CRAGAction.USE_RETRIEVAL

    def test_low_score_web_only(self):
        action = determine_crag_action(0.1, lower_threshold=0.3, upper_threshold=0.7)
        assert action == CRAGAction.WEB_ONLY

    def test_mid_score_combine_and_refine(self):
        action = determine_crag_action(0.5, lower_threshold=0.3, upper_threshold=0.7)
        assert action == CRAGAction.COMBINE_AND_REFINE


class TestEvalContextsForResolution:
    def test_empty_resolved_context_returns_empty(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        assert (
            eval_contexts_for_resolution(
                chunks=chunks,
                resolved_context="",
                refined=False,
            )
            == []
        )

    def test_refined_context_returns_single_passage(self):
        assert eval_contexts_for_resolution(
            chunks=[_chunk("c0", relevance_score=0.1)],
            resolved_context="refined web context",
            refined=True,
        ) == ["refined web context"]

    def test_unrefined_resolved_returns_chunk_texts(self):
        chunks = [_chunk("c0", relevance_score=0.9), _chunk("c1", relevance_score=0.85)]
        assert eval_contexts_for_resolution(
            chunks=chunks,
            resolved_context="good context",
            refined=False,
        ) == [chunk.text for chunk in chunks]

    def test_crag_failure_does_not_fall_back_to_chunks(self):
        """When CRAG clears generation context, evals must not use discarded chunks."""
        chunks = [_chunk("c0", relevance_score=0.1)]
        assert (
            eval_contexts_for_resolution(
                chunks=chunks,
                resolved_context="",
                refined=True,
            )
            == []
        )


class TestExplainableChunksForResolution:
    def test_unrefined_returns_chunks(self):
        chunks = [_chunk("c0", relevance_score=0.9)]
        assert (
            explainable_chunks_for_resolution(
                chunks=chunks,
                refined=False,
                fallback_to_retrieval=False,
            )
            == chunks
        )

    def test_refined_without_fallback_returns_none(self):
        chunks = [_chunk("c0", relevance_score=0.5)]
        assert (
            explainable_chunks_for_resolution(
                chunks=chunks,
                refined=True,
                fallback_to_retrieval=False,
            )
            is None
        )

    def test_refined_with_fallback_returns_chunks(self):
        chunks = [_chunk("c0", relevance_score=0.5)]
        assert (
            explainable_chunks_for_resolution(
                chunks=chunks,
                refined=True,
                fallback_to_retrieval=True,
            )
            == chunks
        )


class TestRefineKnowledge:
    def test_returns_refined_text(self):
        llm = MagicMock()
        llm.generate.return_value = "Refined factual context."
        result = refine_knowledge(
            "What is CRAG?",
            "retrieved passage",
            [WebSearchResult(title="Web", url="https://example.com", snippet="web snippet")],
            llm,
        )
        assert result == "Refined factual context."
        llm.generate.assert_called_once()

    def test_insufficient_marker_returns_empty(self):
        llm = MagicMock()
        llm.generate.return_value = "INSUFFICIENT_INFORMATION"
        result = refine_knowledge("q", "ctx", [], llm)
        assert result == ""

    def test_llm_failure_returns_empty(self):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        result = refine_knowledge("q", "ctx", [], llm)
        assert result == ""


class TestWebSearchProviders:
    @pytest.mark.asyncio
    async def test_null_provider_returns_empty(self):
        provider = NullWebSearchProvider()
        assert await provider.search("test query") == []

    def test_format_web_results(self):
        text = format_web_results(
            [
                WebSearchResult(title="Title", url="https://a.com", snippet="Snippet"),
            ]
        )
        assert "Title" in text
        assert "https://a.com" in text
        assert "Snippet" in text

    def test_parse_duckduckgo_lite_html_same_row(self):
        html = """
        <html><body><table>
        <tr>
          <td><a class="result-link" href="https://example.com">Example</a></td>
          <td class="result-snippet">Example snippet text.</td>
        </tr>
        </table></body></html>
        """
        results = parse_duckduckgo_lite(html, max_results=3)
        assert len(results) == 1
        assert results[0].title == "Example"
        assert results[0].url == "https://example.com"
        assert results[0].snippet == "Example snippet text."

    def test_parse_duckduckgo_lite_html_split_rows(self):
        html = """
        <html><body><table>
        <tr class="result-sponsored">
          <td>1.&nbsp;</td>
          <td><a class="result-link" href="https://ads.example">Sponsored Ad</a></td>
        </tr>
        <tr class="result-sponsored">
          <td>&nbsp;&nbsp;&nbsp;</td>
          <td class="result-snippet">Paid snippet.</td>
        </tr>
        <tr>
          <td valign="top">2.&nbsp;</td>
          <td>
            <a class="result-link" href="https://www.python.org/">Welcome to Python.org</a>
          </td>
        </tr>
        <tr>
          <td>&nbsp;&nbsp;&nbsp;</td>
          <td class="result-snippet">The official home of the Python Programming Language.</td>
        </tr>
        <tr>
          <td>&nbsp;&nbsp;&nbsp;</td>
          <td><span class="link-text">www.python.org</span></td>
        </tr>
        <tr>
          <td valign="top">3.&nbsp;</td>
          <td>
            <a class="result-link" href="https://www.w3schools.com/python/">
              Python Tutorial - W3Schools
            </a>
          </td>
        </tr>
        <tr>
          <td>&nbsp;&nbsp;&nbsp;</td>
          <td class="result-snippet">
            Well organized and easy to understand Web building tutorials.
          </td>
        </tr>
        </table></body></html>
        """
        results = parse_duckduckgo_lite(html, max_results=5)
        assert len(results) == 2
        assert results[0].title == "Welcome to Python.org"
        assert results[0].url == "https://www.python.org/"
        assert "official home" in results[0].snippet
        assert results[1].title == "Python Tutorial - W3Schools"
        assert "Web building tutorials" in results[1].snippet

    def test_parse_duckduckgo_lite_link_without_snippet(self):
        html = """
        <html><body><table>
        <tr>
          <td><a class="result-link" href="https://example.com">Example</a></td>
        </tr>
        <tr>
          <td><a class="result-link" href="https://other.com">Other</a></td>
        </tr>
        </table></body></html>
        """
        results = parse_duckduckgo_lite(html, max_results=3)
        assert len(results) == 2
        assert results[0].title == "Example"
        assert results[0].snippet == ""
        assert results[1].title == "Other"

    @pytest.mark.asyncio
    async def test_duckduckgo_provider_parses_split_row_html(self):
        html = """
        <html><body><table>
        <tr>
          <td><a class="result-link" href="https://example.com">Example</a></td>
        </tr>
        <tr>
          <td class="result-snippet">Example snippet text.</td>
        </tr>
        </table></body></html>
        """
        provider = DuckDuckGoWebSearchProvider()
        with patch_web_search_httpx_post(text=html):
            results = await provider.search("example query", max_results=3)

        assert len(results) == 1
        assert results[0].title == "Example"
        assert results[0].snippet == "Example snippet text."

    def test_get_web_search_provider_none(self):
        from src.core.settings import Settings, WebSearchSettings

        settings = Settings(web_search=WebSearchSettings(provider="none"))
        assert isinstance(get_web_search_provider(settings), NullWebSearchProvider)

    def test_get_web_search_provider_duckduckgo(self):
        from src.core.settings import Settings, WebSearchSettings

        settings = Settings(web_search=WebSearchSettings(provider="duckduckgo"))
        assert isinstance(get_web_search_provider(settings), DuckDuckGoWebSearchProvider)

    def test_get_web_search_provider_tavily_requires_key(self):
        from src.core.exceptions import ConfigurationError
        from src.core.settings import Settings, WebSearchSettings

        settings = Settings(web_search=WebSearchSettings(provider="tavily"))
        with pytest.raises(ConfigurationError, match="API key"):
            get_web_search_provider(settings)

    def test_get_web_search_provider_tavily_with_key(self):
        from pydantic import SecretStr

        from src.core.settings import Settings, TavilySearchConfig, WebSearchSettings

        settings = Settings(
            web_search=WebSearchSettings(
                provider="tavily",
                tavily=TavilySearchConfig(api_key=SecretStr("test-key")),
            )
        )
        provider = get_web_search_provider(settings)
        assert isinstance(provider, TavilyWebSearchProvider)


def _retrieval_result(
    *,
    context: str = "retrieved context",
    chunks: list[Chunk] | None = None,
) -> RetrievalResult:
    if chunks is None:
        chunks = [_chunk("c0", relevance_score=0.9)]
    return RetrievalResult(
        query=Query(text="question"),
        chunks=chunks,
        context=context,
    )


def _llm_mock(response: str = "answer") -> MagicMock:
    llm = MagicMock()
    llm.generate.return_value = response
    llm.generate_stream.return_value = _async_tokens("answer")
    return llm


async def _async_tokens(*tokens: str):
    for token in tokens:
        yield token


def _web_search_mock(results: list[WebSearchResult] | None = None) -> MagicMock:
    mock = MagicMock()
    mock.search = AsyncMock(return_value=results or [])
    return mock


def _crag_pipeline(
    *,
    retrieval_result: RetrievalResult | None = None,
    crag_enabled: bool = True,
    lower: float = 0.3,
    upper: float = 0.7,
    web_results: list[WebSearchResult] | None = None,
    refined_context: str = "refined web context",
    generation_response: str = "answer",
    crag_llm: MagicMock | None = None,
    web_search: MagicMock | None = None,
    web_search_available: bool | None = None,
) -> ChatPipeline:
    retrieval_result = retrieval_result or _retrieval_result()
    retrieval = MagicMock()
    retrieval.retrieve = AsyncMock(return_value=retrieval_result)

    generation_llm = _llm_mock(generation_response)
    generation = GenerationService(generation_llm)

    if crag_llm is None:
        crag_llm = MagicMock()
        crag_llm.generate.return_value = refined_context

    if web_search is None and web_search_available is not False:
        web_search = _web_search_mock(web_results)
    if web_search_available is None:
        web_search_available = web_search is not None and crag_llm is not None

    return ChatPipeline(
        retrieval=retrieval,
        generation=generation,
        crag_enabled=crag_enabled,
        crag_lower_threshold=lower,
        crag_upper_threshold=upper,
        web_search=web_search,
        llm=crag_llm,
        web_search_available=web_search_available,
    )


_MID_QUALITY_WEB_RESULTS = [WebSearchResult(title="Hit", url="https://a.com", snippet="extra")]


def _refined_crag_pipeline(
    *,
    crag_llm: MagicMock | None = None,
) -> tuple[ChatPipeline, MagicMock]:
    """CRAG pipeline that refines mid-quality retrieval into a web-only passage."""
    chunks = [_chunk("c0", relevance_score=0.5)]
    if crag_llm is None:
        crag_llm = MagicMock()
        crag_llm.generate.return_value = "refined web context"
    pipeline = _crag_pipeline(
        retrieval_result=_retrieval_result(chunks=chunks, context="partial context"),
        web_results=_MID_QUALITY_WEB_RESULTS,
        crag_llm=crag_llm,
    )
    return pipeline, crag_llm


class TestChatPipelineCRAG:
    @pytest.mark.asyncio
    async def test_crag_disabled_uses_retrieval_context(self):
        pipeline = _crag_pipeline(crag_enabled=False)
        answer = await pipeline.chat_full("question")
        assert answer.text == "answer"
        pipeline._web_search.search.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_high_quality_skips_web_search(self):
        chunks = [_chunk("c0", relevance_score=0.95)]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="good context"),
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "answer"
        pipeline._web_search.search.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_low_quality_uses_web_only(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        web_results = [
            WebSearchResult(title="Hit", url="https://example.com", snippet="web fact"),
        ]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="weak context"),
            web_results=web_results,
            refined_context="web refined answer context",
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "answer"
        assert answer.sources == []
        pipeline._web_search.search.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_mid_quality_combines_and_refines(self):
        chunks = [_chunk("c0", relevance_score=0.5)]
        web_results = [WebSearchResult(title="Hit", url="https://a.com", snippet="extra")]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="partial context"),
            web_results=web_results,
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "answer"
        assert set(answer.sources) == {"c0"}
        pipeline._web_search.search.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_combine_and_refine_skips_explain_when_context_refined(self):
        pipeline, crag_llm = _refined_crag_pipeline()
        answer = await pipeline.chat_full("question", explain=True)
        assert answer.explanations is None
        assert crag_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_combine_and_refine_skips_highlights_when_context_refined(self):
        pipeline, crag_llm = _refined_crag_pipeline()
        answer = await pipeline.chat_full("question", highlights=True)
        assert answer.highlights is None
        assert crag_llm.generate.call_count == 1

    @pytest.mark.asyncio
    async def test_crag_fallback_to_retrieval_still_explains(self):
        import json

        chunks = [_chunk("c0", relevance_score=0.5)]
        web_search = MagicMock()
        web_search.search = AsyncMock(side_effect=RuntimeError("network down"))
        crag_llm = MagicMock()
        crag_llm.generate.side_effect = [
            json.dumps({"explanations": [{"chunk_id": "c0", "reason": "Relevant passage."}]}),
        ]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="partial context"),
            web_search=web_search,
            crag_llm=crag_llm,
        )
        answer = await pipeline.chat_full("question", explain=True)
        assert answer.explanations is not None
        assert answer.explanations[0].chunk_id == "c0"

    @pytest.mark.asyncio
    async def test_crag_fallback_to_retrieval_still_highlights(self):
        import json

        chunks = [_chunk("c0", relevance_score=0.5)]
        web_search = MagicMock()
        web_search.search = AsyncMock(side_effect=RuntimeError("network down"))
        crag_llm = MagicMock()
        crag_llm.generate.side_effect = [
            json.dumps({"highlights": [{"chunk_id": "c0", "spans": ["text for c0"]}]}),
        ]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="partial context"),
            web_search=web_search,
            crag_llm=crag_llm,
        )
        answer = await pipeline.chat_full("question", highlights=True)
        assert answer.highlights == {"c0": ["text for c0"]}

    @pytest.mark.asyncio
    async def test_web_search_failure_returns_no_info(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        web_search = MagicMock()
        web_search.search = AsyncMock(side_effect=RuntimeError("network down"))
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks),
            web_search=web_search,
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "I don't have information about this."

    @pytest.mark.asyncio
    async def test_empty_web_results_returns_no_info(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks),
            web_results=[],
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "I don't have information about this."

    @pytest.mark.asyncio
    async def test_refinement_insufficient_returns_no_info(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        crag_llm = MagicMock()
        crag_llm.generate.return_value = "INSUFFICIENT_INFORMATION"
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks),
            web_results=[WebSearchResult(title="Hit", url="https://a.com", snippet="x")],
            crag_llm=crag_llm,
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "I don't have information about this."

    @pytest.mark.asyncio
    async def test_crag_span_recorded(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks),
            web_results=[WebSearchResult(title="Hit", url="https://a.com", snippet="x")],
        )
        with patch("src.rag.pipelines.chat_pipeline.record_crag_span") as record_span:
            await pipeline.chat_full("question")
        record_span.assert_called_once()
        decision = record_span.call_args.args[1]
        assert decision.action == CRAGAction.WEB_ONLY
        assert decision.web_search_used is True

    @pytest.mark.asyncio
    async def test_ungraded_chunks_skip_crag_correction(self):
        chunks = [_chunk("c0")]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="plain context"),
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "answer"
        pipeline._web_search.search.assert_not_called()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_empty_retrieval_triggers_web_only(self):
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=[], context=""),
            web_results=[
                WebSearchResult(title="Hit", url="https://example.com", snippet="web fact"),
            ],
            refined_context="web refined answer context",
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "answer"
        assert answer.sources == []
        pipeline._web_search.search.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_empty_retrieval_crag_span_web_only(self):
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=[], context=""),
            web_results=[WebSearchResult(title="Hit", url="https://a.com", snippet="x")],
        )
        with patch("src.rag.pipelines.chat_pipeline.record_crag_span") as record_span:
            await pipeline.chat_full("question")
        record_span.assert_called_once()
        decision = record_span.call_args.args[1]
        assert decision.skipped is False
        assert decision.quality_graded is True
        assert decision.web_search_used is True
        assert decision.action == CRAGAction.WEB_ONLY

    @pytest.mark.asyncio
    async def test_combine_without_web_falls_back_to_retrieval(self):
        chunks = [_chunk("c0", relevance_score=0.5)]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="partial context"),
            web_search=None,
            crag_llm=None,
            web_search_available=False,
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "answer"
        assert set(answer.sources) == {"c0"}

    @pytest.mark.asyncio
    async def test_benchmark_uses_refined_eval_context(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        refined = "refined benchmark context"
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="weak context"),
            web_results=[WebSearchResult(title="Hit", url="https://a.com", snippet="x")],
            refined_context=refined,
        )
        answer, context_texts = await pipeline.benchmark("question")
        assert answer.text == "answer"
        assert context_texts == [refined]

    @pytest.mark.asyncio
    async def test_benchmark_web_search_failure_returns_empty_eval_context(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        web_search = MagicMock()
        web_search.search = AsyncMock(side_effect=RuntimeError("network down"))
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="weak context"),
            web_search=web_search,
        )
        answer, context_texts = await pipeline.benchmark("question")
        assert answer.text == "I don't have information about this."
        assert context_texts == []

    @pytest.mark.asyncio
    async def test_benchmark_empty_web_results_returns_empty_eval_context(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="weak context"),
            web_results=[],
        )
        answer, context_texts = await pipeline.benchmark("question")
        assert answer.text == "I don't have information about this."
        assert context_texts == []

    @pytest.mark.asyncio
    async def test_benchmark_refinement_failure_returns_empty_eval_context(self):
        chunks = [_chunk("c0", relevance_score=0.1)]
        crag_llm = MagicMock()
        crag_llm.generate.return_value = "INSUFFICIENT_INFORMATION"
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="weak context"),
            web_results=[WebSearchResult(title="Hit", url="https://a.com", snippet="x")],
            crag_llm=crag_llm,
        )
        answer, context_texts = await pipeline.benchmark("question")
        assert answer.text == "I don't have information about this."
        assert context_texts == []

    @pytest.mark.asyncio
    async def test_benchmark_combine_fallback_uses_retrieval_eval_context(self):
        chunks = [_chunk("c0", relevance_score=0.5)]
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=chunks, context="partial context"),
            web_search=None,
            crag_llm=None,
            web_search_available=False,
        )
        answer, context_texts = await pipeline.benchmark("question")
        assert answer.text == "answer"
        assert context_texts == [chunk.text for chunk in chunks]

    @pytest.mark.asyncio
    async def test_filtered_low_scores_trigger_web_only(self):
        """CRAG uses all Reliable RAG grades, not only chunks above min_score."""
        chunks = [_chunk("c2", relevance_score=0.55)]
        pipeline = _crag_pipeline(
            retrieval_result=RetrievalResult(
                query=Query(text="question"),
                chunks=chunks,
                context="partial context",
                relevance_scores=[0.1, 0.2, 0.55],
            ),
            web_results=[WebSearchResult(title="Hit", url="https://a.com", snippet="extra")],
            refined_context="web refined answer context",
        )
        answer = await pipeline.chat_full("question")
        assert answer.text == "answer"
        assert answer.sources == []
        pipeline._web_search.search.assert_awaited_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_benchmark_empty_retrieval_uses_web_refined_context(self):
        pipeline = _crag_pipeline(
            retrieval_result=_retrieval_result(chunks=[], context=""),
            web_results=[
                WebSearchResult(title="Hit", url="https://example.com", snippet="web fact"),
            ],
            refined_context="web refined answer context",
        )
        answer, context_texts = await pipeline.benchmark("question")
        assert answer.text == "answer"
        assert context_texts == ["web refined answer context"]
        pipeline._web_search.search.assert_awaited_once()  # type: ignore[attr-defined]


class TestChatPipelineCRAGFromSettings:
    def test_from_settings_respects_crag_disabled(self):
        with (
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as llm,
            patch("src.rag.pipelines.chat_pipeline.RetrievalPipeline.from_settings") as retrieval,
            patch("src.rag.pipelines.chat_pipeline.GenerationService.from_settings") as generation,
            patch("src.core.settings.settings") as mock_settings,
        ):
            llm.return_value = MagicMock()
            retrieval.return_value = MagicMock()
            generation.return_value = MagicMock()
            mock_settings.quality.crag.enabled = False
            mock_settings.web_search.provider = "duckduckgo"
            mock_settings.web_search.max_results = 5

            pipeline = ChatPipeline.from_settings()

        assert pipeline._crag_enabled is False
        assert pipeline._web_search is None
        assert pipeline._web_search_available is False

    def test_from_settings_builds_web_search_when_crag_enabled(self):
        with (
            patch(
                "src.infrastructure.llm.llama_cpp_provider.LlamaCppProvider.from_settings"
            ) as llm,
            patch("src.rag.pipelines.chat_pipeline.RetrievalPipeline.from_settings") as retrieval,
            patch("src.rag.pipelines.chat_pipeline.GenerationService.from_settings") as generation,
            patch(
                "src.infrastructure.search.web_search.get_web_search_provider",
                return_value=NullWebSearchProvider(),
            ) as get_provider,
            patch("src.core.settings.settings") as mock_settings,
        ):
            llm.return_value = MagicMock()
            retrieval.return_value = MagicMock()
            generation.return_value = MagicMock()
            mock_settings.quality.crag.enabled = True
            mock_settings.quality.crag.lower_threshold = 0.3
            mock_settings.quality.crag.upper_threshold = 0.7
            mock_settings.web_search.provider = "duckduckgo"
            mock_settings.web_search.max_results = 5

            pipeline = ChatPipeline.from_settings()

        get_provider.assert_called_once()
        assert pipeline._crag_enabled is True
        assert pipeline._web_search is not None
        assert pipeline._web_search_available is True


class TestCRAGSettings:
    def test_crag_defaults_from_yaml(self):
        from src.core.settings import settings

        assert settings.quality.crag.enabled is False
        assert settings.quality.crag.lower_threshold == pytest.approx(0.3)
        assert settings.quality.crag.upper_threshold == pytest.approx(0.7)

    def test_web_search_defaults_from_yaml(self):
        from src.core.settings import settings

        assert settings.web_search.provider == "none"
        assert settings.web_search.max_results == 5

    def test_invalid_threshold_order_raises(self):
        from pydantic import ValidationError

        from src.core.settings import CRAGSettings

        with pytest.raises(ValidationError, match="lower_threshold"):
            CRAGSettings(enabled=True, lower_threshold=0.8, upper_threshold=0.2)
