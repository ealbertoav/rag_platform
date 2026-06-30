from __future__ import annotations

import logging
import re
from pathlib import Path
from string import Template

from src.domain.entities.query import Query
from src.domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "retrieval" / "query_expansion.txt"

# Strip leading list markers such as "1.", "-", "•" from LLM output lines.
_LIST_PREFIX = re.compile(r"^\d+[.)]\s*|^[-•*]\s*")


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def _parse_variants(text: str, n: int) -> list[str]:
    """Extract up to *n* non-empty lines from the LLM response."""
    variants: list[str] = []
    for line in text.splitlines():
        line = _LIST_PREFIX.sub("", line).strip()
        if line:
            variants.append(line)
        if len(variants) >= n:
            break
    return variants


class QueryExpander:
    """Uses an LLM to rewrite a query into N semantically diverse variants.

    When disabled ("enabled=False") or when the LLM fails, the original
    Query is returned unchanged — retrieval falls back to the single query.

    Results are cached in-memory per query text for the lifetime of this
    instance. A cached entry is reused when the requested variant count is
    less than or equal to what was already generated; a higher limit triggers
    a fresh LLM call so adaptive per-category overrides are honored.
    """

    def __init__(
        self,
        llm: LLMRepository,
        n_variants: int = 3,
        enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._n_variants = n_variants
        self._enabled = enabled
        self._cache: dict[str, list[str]] = {}
        self._prompt_template: Template | None = None

    # ── Public ─────────────────────────────────────────────────────────────────

    def expand(self, query: Query, n_variants: int | None = None) -> Query:
        """Return *query* with "expanded_texts" populated.

        If disabled or the LLM call fails, it returns the original query unchanged.
        *n_variants* overrides the instance default when provided.
        """
        limit = self._n_variants if n_variants is None else n_variants
        if not self._enabled or limit < 1:
            return query

        cached = self._cache.get(query.text)
        if cached is None or len(cached) < limit:
            self._cache[query.text] = self._generate(query.text, limit)

        variants = self._cache[query.text][:limit]
        if not variants:
            return query

        return query.model_copy(update={"expanded_texts": variants})

    def clear_cache(self) -> None:
        self._cache.clear()

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, llm: LLMRepository) -> QueryExpander:
        from src.core.settings import settings

        cfg = settings.query_expansion
        return cls(llm=llm, n_variants=cfg.n_variants, enabled=cfg.enabled)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _build_prompt(self, query_text: str, n_variants: int) -> str:
        template = self._prompt_template or _load_prompt()
        self._prompt_template = template
        return template.substitute(
            n_variants=n_variants,
            query=query_text,
        )

    def _generate(self, query_text: str, n_variants: int) -> list[str]:
        try:
            prompt = self._build_prompt(query_text, n_variants)
            response = self._llm.generate(prompt=prompt, context="")
            variants = _parse_variants(response, n_variants)
            logger.debug("Query expanded: %d variants for %r", len(variants), query_text[:60])
            return variants
        except Exception as exc:
            logger.warning("Query expansion failed for %r: %s", query_text[:60], exc)
            return []
