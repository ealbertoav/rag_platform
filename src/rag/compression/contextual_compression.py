from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from src.core.constants import CHUNK_RAW_TEXT_KEY, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import chunk_context_text, passage_context_key
from src.rag.compression.token_reducer import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "compression" / "extract_relevant.txt"


class ContextualCompressor:
    """Reduces context token count by extracting only query-relevant sentences.

    For each chunk, the LLM is asked to extract the passages that directly
    address the query.  On failure the original chunk text is preserved, so
    the pipeline always has something to send to the generator.

    The total compressed context is capped at "max_tokens"; chunks beyond
    the budget are dropped rather than truncated mid-thought.
    """

    def __init__(
        self,
        llm: LLMRepository,
        max_tokens: int = 1500,
        enabled: bool = True,
    ) -> None:
        self._llm: LLMRepository = llm
        self._max_tokens: int = max_tokens
        self._enabled: bool = enabled
        self._prompt_template: Template | None = None

    # ── Public ─────────────────────────────────────────────────────────────────

    def compress(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return *chunks* with text replaced by query-relevant extractions.

        If disabled, returns the input list unchanged.
        Chunks are processed in order; once the token budget is exhausted,
        remaining chunks are omitted.
        """
        if not self._enabled or not chunks:
            return chunks

        result: list[Chunk] = []
        remaining_tokens = self._max_tokens
        compressed_passages: dict[str, str] = {}

        for chunk in chunks:
            passage_key = passage_context_key(chunk)
            cached_text = compressed_passages.get(passage_key)
            if cached_text is not None:
                result.append(self._with_compressed_text(chunk, cached_text))
                continue

            if remaining_tokens <= 0:
                break

            text = self._extract(query, chunk)
            text = truncate_to_tokens(text, remaining_tokens)
            if not text:
                continue
            compressed_passages[passage_key] = text
            result.append(self._with_compressed_text(chunk, text))
            remaining_tokens -= count_tokens(text)

        logger.debug(
            "Compression: %d → %d chunks, budget used %d/%d tokens",
            len(chunks),
            len(result),
            self._max_tokens - remaining_tokens,
            self._max_tokens,
        )
        return result

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, llm: LLMRepository) -> ContextualCompressor:
        from src.core.settings import settings

        cfg = settings.compression
        return cls(llm=llm, max_tokens=cfg.max_tokens, enabled=cfg.enabled)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _load_template(self) -> Template:
        template = self._prompt_template
        if template is None:
            template = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
            self._prompt_template = template
        return template

    def _extract(self, query: str, chunk: Chunk) -> str:
        """Ask the LLM to extract relevant sentences; fall back to the full text."""
        source_text = chunk_context_text(chunk)
        try:
            prompt = self._load_template().substitute(query=query, passage=source_text)
            response = self._llm.generate(prompt=prompt, context="").strip()
            return response if response else source_text
        except Exception as exc:
            logger.warning("Compression failed for chunk %r, using original: %s", chunk.id, exc)
            return source_text

    @staticmethod
    def _with_compressed_text(chunk: Chunk, text: str) -> Chunk:
        metadata = dict(chunk.metadata)
        if CHUNK_RAW_TEXT_KEY in metadata:
            metadata[CHUNK_RAW_TEXT_KEY] = text
        if PARENT_CONTEXT_TEXT_KEY in metadata:
            metadata[PARENT_CONTEXT_TEXT_KEY] = text
        update: dict[str, object] = {"text": text}
        if metadata != chunk.metadata:
            update["metadata"] = metadata
        return chunk.model_copy(update=update)
