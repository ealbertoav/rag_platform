from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING

from src.domain.entities.answer import Answer
from src.domain.entities.query import Query
from src.domain.services.generation_service import GenerationService
from src.observability.metrics import record_generation, record_request
from src.rag.enrichment.relevant_segment_extraction import chunk_source_ids
from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline

if TYPE_CHECKING:
    from src.infrastructure.vectordb.bm25 import BM25Index

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._retrieval = retrieval
        self._generation = generation

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
        return self._generation.stream(query.text, result.context)

    async def chat_full(self, question: str | Query) -> Answer:
        """Run the full pipeline and return a complete "Answer".

        Useful for non-streaming contexts (tests, scripts).
        """
        query = question if isinstance(question, Query) else Query(text=question)
        t0 = time.monotonic()
        result = await self._retrieval.retrieve(query)

        sources = [chunk_id for c in result.chunks for chunk_id in chunk_source_ids(c)]
        answer = self._generation.generate(query.text, result.context, sources)

        elapsed = (time.monotonic() - t0) * 1000
        token_count = len(answer.text.split())
        record_generation(token_count, elapsed / 1000)
        record_request("chat", elapsed / 1000, success=True)
        return answer.model_copy(
            update={
                "query_id": query.id,
                "latency_ms": elapsed,
                "token_count": token_count,
            }
        )

    async def benchmark(self, question: str) -> tuple[Answer, list[str]]:
        """Return (Answer, context_text_list) for benchmarking.

        The context list contains the raw text of each retrieved chunk, needed
        for faithfulness and relevance scoring against the generated answer.
        """
        from src.domain.services.retrieval_service import RetrievalResult

        t0 = time.monotonic()
        query = Query(text=question)
        retrieval_result: RetrievalResult = await self._retrieval.retrieve(query)

        context_texts = [c.text for c in retrieval_result.chunks]
        sources = [chunk_id for c in retrieval_result.chunks for chunk_id in chunk_source_ids(c)]
        answer = self._generation.generate(question, retrieval_result.context, sources)

        elapsed = (time.monotonic() - t0) * 1000
        answer = answer.model_copy(
            update={
                "query_id": query.id,
                "latency_ms": elapsed,
                "token_count": len(answer.text.split()),
            }
        )
        return answer, context_texts

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, bm25_index: BM25Index | None = None) -> ChatPipeline:
        """Build the full pipeline from settings (lazy model loading)."""
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider

        llm = LlamaCppProvider.from_settings()
        retrieval = RetrievalPipeline.from_settings(llm=llm, bm25_index=bm25_index)
        generation = GenerationService.from_settings(llm=llm)
        return cls(retrieval=retrieval, generation=generation)
