from __future__ import annotations

import logging
from collections.abc import AsyncGenerator, AsyncIterator
from pathlib import Path
from string import Template
from uuid import uuid4

from opentelemetry import trace

from src.core.constants import MODALITY_FIGURE, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.domain.repositories.vision_repository import VisionRepository
from src.rag.chunking.contextual_headers import (
    has_mixed_modality,
    join_chunk_context,
    join_chunk_context_multimodal,
)

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.generation")

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "system" / "rag_assistant.txt"
_MULTIMODAL_PROMPT_PATH = (
    Path(__file__).parents[2] / "prompts" / "system" / "rag_assistant_multimodal.txt"
)
_VISION_PROMPT_PATH = (
    Path(__file__).parents[2] / "prompts" / "system" / "vision_figure_description.txt"
)
_NO_CONTEXT_REPLY = "I don't have information about this."


class GenerationService:
    """Builds prompts and calls the LLM for RAG-style generation.

    Keeps the domain service free of retrieval concerns — it only knows about
    the LLM interface and the Answer entity.
    """

    def __init__(
        self,
        llm: LLMRepository,
        *,
        multimodal_prompt_enabled: bool = False,
        vision: VisionRepository | None = None,
        vision_generation_enabled: bool = False,
    ) -> None:
        self._llm: LLMRepository = llm
        self._template: Template | None = None
        self._multimodal_template: Template | None = None
        self._multimodal_prompt_enabled: bool = multimodal_prompt_enabled
        self._vision: VisionRepository | None = vision
        self._vision_generation_enabled: bool = vision_generation_enabled
        self._vision_prompt_template: Template | None = None

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
            chunks, context = self._apply_vision_descriptions(chunks, context, question)
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
        chunks, context = self._apply_vision_descriptions(chunks, context, question)
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
        from src.infrastructure.vision import get_generation_vision_provider

        vision_generation_enabled = settings.generation.vision_generation.enabled
        vision = get_generation_vision_provider(settings) if vision_generation_enabled else None
        return cls(
            llm=llm,
            multimodal_prompt_enabled=settings.generation.multimodal_prompt.enabled,
            vision=vision,
            vision_generation_enabled=vision_generation_enabled,
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

    def _apply_vision_descriptions(
        self,
        chunks: list[Chunk] | None,
        context: str,
        question: str,
    ) -> tuple[list[Chunk] | None, str]:
        """Swap figure chunks' text for a live vision-LLM description (T-271).

        For each chunk with ``modality=figure`` and a readable ``asset_path``,
        calls the configured vision provider with the current *question* and
        replaces the chunk's text with the returned description, then rebuilds
        *context* from the updated chunks so the description reaches the LLM.
        No-ops (returning *chunks*/*context* unchanged) when vision generation
        is disabled, no vision provider is configured, no chunks are given, or
        every figure description fails/comes back empty — byte-identical to
        pre-T-271 behavior in those cases.
        """
        if not self._vision_generation_enabled or self._vision is None or not chunks:
            return chunks, context

        updated: list[Chunk] = []
        changed = False
        for chunk in chunks:
            description = self._describe_figure(chunk, question)
            if description is not None:
                chunk = chunk.model_copy(
                    update={
                        "text": description,
                        "metadata": {
                            k: v for k, v in chunk.metadata.items() if k != PARENT_CONTEXT_TEXT_KEY
                        },
                    }
                )
                changed = True
            updated.append(chunk)

        if not changed:
            return chunks, context
        return updated, join_chunk_context(updated)

    def _describe_figure(self, chunk: Chunk, question: str) -> str | None:
        """Return a vision-LLM description of *chunk*'s image, or None to soft-fail."""
        if self._vision is None or chunk.modality != MODALITY_FIGURE or not chunk.asset_path:
            return None

        path = Path(chunk.asset_path)
        if not path.is_file():
            logger.warning("Vision generation skipped for %s: asset missing at %s", chunk.id, path)
            return None

        try:
            description = self._vision.caption_image(path, prompt=self._vision_prompt(question))
        except Exception as exc:
            logger.warning("Vision generation failed for figure chunk %s: %s", chunk.id, exc)
            return None

        description = description.strip()
        return description or None

    def _vision_prompt(self, question: str) -> str:
        return self._vision_prompt_template_obj().substitute(question=question)

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

    def _vision_prompt_template_obj(self) -> Template:
        if self._vision_prompt_template is None:
            self._vision_prompt_template = Template(_VISION_PROMPT_PATH.read_text(encoding="utf-8"))
        return self._vision_prompt_template

    @staticmethod
    async def _no_context_stream() -> AsyncGenerator[str, None]:
        yield _NO_CONTEXT_REPLY
