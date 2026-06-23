from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator

from src.domain.entities.answer import Answer
from src.domain.entities.query import Query
from src.domain.services.generation_service import GenerationService
from src.observability.metrics import record_generation, record_request
from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline

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

    # ── Public ─────────────────────────────────────────────────────────────────

    async def chat(self, question: str) -> AsyncIterator[str]:
        """Return a token stream for *question*.

        Callers iterate with:
            async for token in await pipeline.chat(question):
                yield token
        """
        query = Query(text=question)
        result = await self._retrieval.retrieve(query)
        return self._generation.stream(question, result.context)

    async def chat_full(self, question: str) -> Answer:
        """Run the full pipeline and return a complete "Answer".

        Useful for non-streaming contexts (tests, scripts).
        """
        t0 = time.monotonic()
        query = Query(text=question)
        result = await self._retrieval.retrieve(query)

        sources = [c.id for c in result.chunks]
        answer = self._generation.generate(question, result.context, sources)

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
        sources = [c.id for c in retrieval_result.chunks]
        answer = self._generation.generate(question, retrieval_result.context, sources)

        elapsed = (time.monotonic() - t0) * 1000
        answer = answer.model_copy(
            update={"query_id": query.id, "latency_ms": elapsed,
                    "token_count": len(answer.text.split())}
        )
        return answer, context_texts

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls) -> ChatPipeline:
        """Build the full pipeline from settings (lazy model loading)."""
        from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider

        llm = LlamaCppProvider.from_settings()
        retrieval = RetrievalPipeline.from_settings(llm=llm)
        generation = GenerationService.from_settings(llm=llm)
        return cls(retrieval=retrieval, generation=generation)
