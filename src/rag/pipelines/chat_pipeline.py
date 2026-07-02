from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from opentelemetry import trace

from src.domain.entities.answer import Answer
from src.domain.entities.query import Query
from src.domain.services.generation_service import GenerationService
from src.observability.metrics import record_generation, record_request
from src.rag.enrichment.relevant_segment_extraction import chunk_source_ids
from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline
from src.rag.quality.crag import (
    ContextResolution,
    CRAGAction,
    CRAGDecision,
    crag_fallback_without_web,
    determine_crag_action,
    eval_contexts_for_resolution,
    record_crag_span,
    refine_knowledge,
    score_retrieval_quality,
)
from src.rag.quality.explainable_retrieval import explain_chunks, resolve_chunks_for_sources

if TYPE_CHECKING:
    from src.domain.repositories.llm_repository import LLMRepository
    from src.domain.repositories.web_search_repository import WebSearchRepository
    from src.domain.services.retrieval_service import RetrievalResult

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.chat")


class ChatPipeline:
    """End-to-end: question → retrieval → LLM → streamed answer.

    "chat()" — streams tokens, suitable for SSE responses.
    "chat_full()" — accumulates the stream and returns an "Answer"
                      with "sources" and "latency_ms" populated.
    """

    def __init__(
        self,
        retrieval: RetrievalPipeline,
        generation: GenerationService,
        *,
        crag_enabled: bool = False,
        crag_lower_threshold: float = 0.3,
        crag_upper_threshold: float = 0.7,
        web_search: WebSearchRepository | None = None,
        llm: LLMRepository | None = None,
        web_search_max_results: int = 5,
        web_search_available: bool = True,
    ) -> None:
        self._retrieval = retrieval
        self._generation = generation
        self._crag_enabled = crag_enabled
        self._crag_lower_threshold = crag_lower_threshold
        self._crag_upper_threshold = crag_upper_threshold
        self._web_search = web_search
        self._llm = llm
        self._web_search_max_results = web_search_max_results
        self._web_search_available = (
            web_search_available and web_search is not None and llm is not None
        )

    @property
    def retrieval(self) -> RetrievalPipeline:
        return self._retrieval

    @property
    def generation(self) -> GenerationService:
        return self._generation

    # ── Public ─────────────────────────────────────────────────────────────────

    async def chat(self, question: str | Query) -> AsyncIterator[str]:
        """Return a token stream for *question*.

        Callers iterate with:
            async for token in await pipeline.chat(question):
                yield token
        """
        query = question if isinstance(question, Query) else Query(text=question)
        result = await self._retrieval.retrieve(query)
        resolution = await self._resolve_context(query.text, result)
        return self._generation.stream(query.text, resolution.context)

    async def chat_full(self, question: str | Query, *, explain: bool = False) -> Answer:
        """Run the full pipeline and return a complete "Answer".

        Useful for non-streaming contexts (tests, scripts).
        When *explain* is True, attaches per-source retrieval explanations.
        """
        query = question if isinstance(question, Query) else Query(text=question)
        t0 = time.monotonic()
        result = await self._retrieval.retrieve(query)
        resolution = await self._resolve_context(query.text, result)

        answer = self._generation.generate(query.text, resolution.context, resolution.sources)

        elapsed = (time.monotonic() - t0) * 1000
        token_count = len(answer.text.split())
        record_generation(token_count, elapsed / 1000)
        record_request("chat", elapsed / 1000, success=True)

        explanations = None
        if explain and answer.sources and self._llm is not None:
            source_chunks = resolve_chunks_for_sources(answer.sources, result.chunks)
            if source_chunks:
                explanations = explain_chunks(query.text, source_chunks, self._llm)
                if not explanations:
                    explanations = None

        return answer.model_copy(
            update={
                "query_id": query.id,
                "latency_ms": elapsed,
                "token_count": token_count,
                "explanations": explanations,
            }
        )

    async def benchmark(self, question: str) -> tuple[Answer, list[str]]:
        """Return (Answer, context_text_list) for benchmarking.

        The context list mirrors what generation used: raw chunk texts for
        standard retrieval, a single refined passage after CRAG refinement, or
        empty when CRAG could not supply usable context.
        """
        t0 = time.monotonic()
        query = Query(text=question)
        retrieval_result = await self._retrieval.retrieve(query)
        resolution = await self._resolve_context(question, retrieval_result)

        answer = self._generation.generate(question, resolution.context, resolution.sources)

        elapsed = (time.monotonic() - t0) * 1000
        answer = answer.model_copy(
            update={
                "query_id": query.id,
                "latency_ms": elapsed,
                "token_count": len(answer.text.split()),
            }
        )
        return answer, resolution.eval_contexts

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, bm25_index: object | None = None) -> ChatPipeline:
        """Build the full pipeline from settings (lazy model loading)."""
        from src.core.settings import settings
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider
        from src.infrastructure.search.web_search import get_web_search_provider

        llm = LlamaCppProvider.from_settings()
        retrieval = RetrievalPipeline.from_settings(llm=llm, bm25_index=bm25_index)  # type: ignore[arg-type]
        generation = GenerationService.from_settings(llm=llm)

        crag_cfg = settings.quality.crag
        web_search = None
        web_search_available = False
        if crag_cfg.enabled:
            if settings.web_search.provider == "none":
                logger.warning(
                    "CRAG enabled but web_search.provider=none — "
                    "corrective web fallback disabled; enable Reliable RAG (T-140) "
                    "for graded retrieval or set web_search.provider"
                )
            else:
                try:
                    web_search = get_web_search_provider(settings)
                    web_search_available = True
                except Exception as exc:
                    logger.warning("CRAG enabled but web search unavailable: %s", exc)

        return cls(
            retrieval=retrieval,
            generation=generation,
            crag_enabled=crag_cfg.enabled,
            crag_lower_threshold=crag_cfg.lower_threshold,
            crag_upper_threshold=crag_cfg.upper_threshold,
            web_search=web_search,
            llm=llm,
            web_search_max_results=settings.web_search.max_results,
            web_search_available=web_search_available,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _resolve_context(
        self,
        query_text: str,
        retrieval_result: RetrievalResult,
    ) -> ContextResolution:
        """Return LLM context, sources, and eval passages, optionally applying CRAG."""
        sources = [
            chunk_id for chunk in retrieval_result.chunks for chunk_id in chunk_source_ids(chunk)
        ]
        if not self._crag_enabled:
            return ContextResolution(
                context=retrieval_result.context,
                sources=sources,
                eval_contexts=[chunk.text for chunk in retrieval_result.chunks],
            )

        with _tracer.start_as_current_span("chat.crag") as span:
            context, decision = await self._apply_crag(query_text, retrieval_result)
            record_crag_span(span, decision)
            resolved_sources = [] if decision.action == CRAGAction.WEB_ONLY else sources
            eval_contexts = eval_contexts_for_resolution(
                chunks=retrieval_result.chunks,
                resolved_context=context,
                refined=decision.refined,
            )
            return ContextResolution(
                context=context,
                sources=resolved_sources,
                eval_contexts=eval_contexts,
            )

    async def _apply_crag(
        self,
        query_text: str,
        retrieval_result: RetrievalResult,
    ) -> tuple[str, CRAGDecision]:
        quality = score_retrieval_quality(
            retrieval_result.chunks,
            relevance_scores=retrieval_result.relevance_scores or None,
        )

        if not quality.graded:
            logger.debug("CRAG skipped — no relevance_score metadata; enable Reliable RAG (T-140)")
            return retrieval_result.context, CRAGDecision(
                quality_score=quality.score,
                action=CRAGAction.USE_RETRIEVAL,
                web_search_used=False,
                quality_graded=False,
                skipped=True,
            )

        action = determine_crag_action(
            quality.score,
            lower_threshold=self._crag_lower_threshold,
            upper_threshold=self._crag_upper_threshold,
        )

        if action == CRAGAction.USE_RETRIEVAL:
            return retrieval_result.context, CRAGDecision(
                quality_score=quality.score,
                action=action,
                web_search_used=False,
            )

        if not self._web_search_available:
            return crag_fallback_without_web(
                query_text,
                retrieval_result.context,
                quality.score,
                action,
            )

        assert self._web_search is not None
        assert self._llm is not None

        try:
            web_results = await self._web_search.search(
                query_text,
                max_results=self._web_search_max_results,
            )
        except Exception as exc:
            logger.warning("CRAG web search failed for %r: %s", query_text[:60], exc)
            return crag_fallback_without_web(
                query_text,
                retrieval_result.context,
                quality.score,
                action,
                web_search_attempted=True,
            )

        if not web_results:
            logger.info("CRAG web search returned no results for %r", query_text[:60])
            return crag_fallback_without_web(
                query_text,
                retrieval_result.context,
                quality.score,
                action,
                web_search_attempted=True,
            )

        retrieval_context = "" if action == CRAGAction.WEB_ONLY else retrieval_result.context
        refined = refine_knowledge(
            query_text,
            retrieval_context,
            web_results,
            self._llm,
        )
        if not refined.strip():
            return crag_fallback_without_web(
                query_text,
                retrieval_result.context,
                quality.score,
                action,
                web_search_attempted=True,
                web_result_count=len(web_results),
                refinement_attempted=True,
            )

        return refined, CRAGDecision(
            quality_score=quality.score,
            action=action,
            web_search_used=True,
            web_result_count=len(web_results),
            refined=True,
        )
