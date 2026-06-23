from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.compression.token_reducer import count_tokens, truncate_to_tokens

logger = logging.getLogger(__name__)

_PROMPT_PATH = (
    Path(__file__).parents[2] / "prompts" / "compression" / "extract_relevant.txt"
)


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
        self._llm = llm
        self._max_tokens = max_tokens
        self._enabled = enabled
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

        for chunk in chunks:
            if remaining_tokens <= 0:
                break
            text = self._extract(query, chunk)
            text = truncate_to_tokens(text, remaining_tokens)
            if not text:
                continue
            result.append(chunk.model_copy(update={"text": text}))
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
        if self._prompt_template is None:
            self._prompt_template = Template(_PROMPT_PATH.read_text(encoding="utf-8"))
        return self._prompt_template

    def _extract(self, query: str, chunk: Chunk) -> str:
        """Ask the LLM to extract relevant sentences; fall back to the full text."""
        try:
            prompt = self._load_template().substitute(
                query=query, passage=chunk.text
            )
            response = self._llm.generate(prompt=prompt, context="").strip()
            return response if response else chunk.text
        except Exception as exc:
            logger.warning(
                "Compression failed for chunk %r, using original: %s", chunk.id, exc
            )
            return chunk.text
