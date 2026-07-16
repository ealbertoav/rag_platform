from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from string import Template
from uuid import uuid4

from opentelemetry import trace

from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import has_mixed_modality, join_chunk_context_multimodal

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.generation")

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "system" / "rag_assistant.txt"
_MULTIMODAL_PROMPT_PATH = (
    Path(__file__).parents[2] / "prompts" / "system" / "rag_assistant_multimodal.txt"
)
_NO_CONTEXT_REPLY = "I don't have information about this."


class GenerationService:
    """Builds prompts and calls the LLM for RAG-style generation.

    Keeps the domain service free of retrieval concerns — it only knows about
    the LLM interface and the Answer entity.
    """

    def __init__(self, llm: LLMRepository, *, multimodal_prompt_enabled: bool = False) -> None:
        self._llm: LLMRepository = llm
        self._template: Template | None = None
        self._multimodal_template: Template | None = None
        self._multimodal_prompt_enabled: bool = multimodal_prompt_enabled

    # ── Public ─────────────────────────────────────────────────────────────────

    def generate(
        self,
        question: str,
        context: str,
        sources: list[str],
        chunks: list[Chunk] | None = None,
    ) -> Answer:
        """Return a fully formed Answer (blocking)."""
        query_id = str(uuid4())
        if not context.strip():
            logger.debug("Empty context — returning no-info reply")
            return Answer(query_id=query_id, text=_NO_CONTEXT_REPLY, sources=[])

        with _tracer.start_as_current_span("generation.llm") as span:
            span.set_attribute("chunk_count", len(sources))
            prompt = self._build_prompt(context, chunks)
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

    def stream(
        self,
        question: str,
        context: str,
        chunks: list[Chunk] | None = None,
    ) -> AsyncIterator[str]:
        """Return an async iterator that yields tokens as they are produced.

        If the context is empty, it yields the no-info reply as a single token.
        """
        if not context.strip():
            return self._no_context_stream()
        prompt = self._build_prompt(context, chunks)
        return self._llm.generate_stream(prompt=prompt, context=question)

    def call_llm(self, prompt: str) -> str:
        """Call the LLM directly with a raw prompt (no RAG context wrapping).

        Used by callers that need bare LLM inference (e.g. agent decision loop).
        """
        return self._llm.generate(prompt=prompt, context="")

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, llm: LLMRepository) -> GenerationService:
        from src.core.settings import settings

        return cls(
            llm=llm,
            multimodal_prompt_enabled=settings.generation.multimodal_prompt.enabled,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_prompt(self, context: str, chunks: list[Chunk] | None) -> str:
        """Build the system prompt, swapping in the multimodal template (T-270)
        when enabled and *chunks* span more than one modality. Falls back to the
        base template and the given *context* otherwise — byte-identical to
        pre-T-270 behavior when disabled or chunks are unavailable/single-modality.
        """
        if self._multimodal_prompt_enabled and chunks and has_mixed_modality(chunks):
            return self._multimodal_template_obj().substitute(
                context=join_chunk_context_multimodal(chunks)
            )
        return self._template_obj().substitute(context=context)

    def _template_obj(self) -> Template:
        if self._template is None:
            self._template = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
        return self._template

    def _multimodal_template_obj(self) -> Template:
        if self._multimodal_template is None:
            self._multimodal_template = Template(
                _MULTIMODAL_PROMPT_PATH.read_text(encoding="utf-8")
            )
        return self._multimodal_template

    @staticmethod
    async def _no_context_stream() -> AsyncGenerator[str, None]:
        yield _NO_CONTEXT_REPLY
