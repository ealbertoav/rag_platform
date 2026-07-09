from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from string import Template
from uuid import uuid4

from opentelemetry import trace

from src.domain.entities.answer import Answer
from src.domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.generation")

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "system" / "rag_assistant.txt"
_NO_CONTEXT_REPLY = "I don't have information about this."


class GenerationService:
    """Builds prompts and calls the LLM for RAG-style generation.

    Keeps the domain service free of retrieval concerns — it only knows about
    the LLM interface and the Answer entity.
    """

    def __init__(self, llm: LLMRepository) -> None:
        self._llm: LLMRepository = llm
        self._template: Template | None = None

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate(self, question: str, context: str, sources: list[str]) -> Answer:
        """Return a fully formed Answer (blocking)."""
        query_id = str(uuid4())
        if not context.strip():
            logger.debug("Empty context — returning no-info reply")
            return Answer(query_id=query_id, text=_NO_CONTEXT_REPLY, sources=[])

        with _tracer.start_as_current_span("generation.llm") as span:
            span.set_attribute("chunk_count", len(sources))
            prompt = self._build_prompt(context)
            text = self._llm.generate(prompt=prompt, context=question)
            span.set_attribute("token_count", len(text.split()))
        return Answer(query_id=query_id, text=text, sources=sources)

    def generate_direct(self, question: str) -> Answer:
        """Generate an answer without retrieved context (e.g. greetings, chit-chat)."""
        query_id = str(uuid4())
        with _tracer.start_as_current_span("generation.llm") as span:
            span.set_attribute("direct", True)
            text = self._llm.generate(prompt=question, context="")
            span.set_attribute("token_count", len(text.split()))
        return Answer(query_id=query_id, text=text, sources=[])

    def stream(self, question: str, context: str) -> AsyncIterator[str]:
        """Return an async iterator that yields tokens as they are produced.

        If the context is empty, it yields the no-info reply as a single token.
        """
        if not context.strip():
            return self._no_context_stream()
        prompt = self._build_prompt(context)
        return self._llm.generate_stream(prompt=prompt, context=question)

    def call_llm(self, prompt: str) -> str:
        """Call the LLM directly with a raw prompt (no RAG context wrapping).

        Used by callers that need bare LLM inference (e.g. agent decision loop).
        """
        return self._llm.generate(prompt=prompt, context="")

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, llm: LLMRepository) -> GenerationService:
        return cls(llm=llm)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_prompt(self, context: str) -> str:
        if self._template is not None:
            return self._template.substitute(context=context)
        template = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
        self._template = template
        return template.substitute(context=context)

    @staticmethod
    async def _no_context_stream() -> AsyncGenerator[str, None]:
        yield _NO_CONTEXT_REPLY
