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
_LIST_PREFIX = re.compile(r"^[\d]+[.)]\s*|^[-•*]\s*")


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
    instance, so the same question never triggers more than one LLM call.
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

    def expand(self, query: Query) -> Query:
        """Return *query* with "expanded_texts" populated.

        If disabled or the LLM call fails, it returns the original query unchanged.
        """
        if not self._enabled or self._n_variants < 1:
            return query

        if query.text not in self._cache:
            self._cache[query.text] = self._generate(query.text)

        variants = self._cache[query.text]
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

    def _build_prompt(self, query_text: str) -> str:
        if self._prompt_template is None:
            self._prompt_template = _load_prompt()
        return self._prompt_template.substitute(
            n_variants=self._n_variants,
            query=query_text,
        )

    def _generate(self, query_text: str) -> list[str]:
        try:
            prompt = self._build_prompt(query_text)
            response = self._llm.generate(prompt=prompt, context="")
            variants = _parse_variants(response, self._n_variants)
            logger.debug("Query expanded: %d variants for %r", len(variants), query_text[:60])
            return variants
        except Exception as exc:
            logger.warning("Query expansion failed for %r: %s", query_text[:60], exc)
            return []
