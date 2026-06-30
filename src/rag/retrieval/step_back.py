from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from src.domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "retrieval" / "step_back.txt"


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def generate_step_back(query: str, llm: LLMRepository) -> str:
    """Return a broader step-back question for background context retrieval."""
    generator = StepBackGenerator(llm=llm, enabled=True)
    return generator.generate(query)


class StepBackGenerator:
    """Uses an LLM to produce a broader step-back query for multi-query fusion.

    When disabled or when the LLM fails, it returns an empty string, so standard
    retrieval proceeds unchanged. Only successful non-empty results are cached;
    transient failures are retried on the next call.
    """

    def __init__(self, llm: LLMRepository, enabled: bool = False) -> None:
        self._llm = llm
        self._enabled = enabled
        self._cache: dict[str, str] = {}
        self._prompt_template: Template | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    def generate(self, query_text: str) -> str:
        """Return a step-back variant for *query_text*, or "" when disabled/failed."""
        if not self._enabled or not query_text.strip():
            return ""

        cached = self._cache.get(query_text)
        if cached is not None:
            return cached

        step_back = self._generate(query_text)
        if step_back:
            self._cache[query_text] = step_back
        return step_back

    def clear_cache(self) -> None:
        self._cache.clear()

    @classmethod
    def from_settings(cls, llm: LLMRepository) -> StepBackGenerator:
        from src.core.settings import settings

        cfg = settings.query_expansion.step_back
        return cls(llm=llm, enabled=cfg.enabled)

    def _build_prompt(self, query_text: str) -> str:
        template = self._prompt_template or _load_prompt()
        self._prompt_template = template
        return template.substitute(query=query_text)

    def _generate(self, query_text: str) -> str:
        try:
            prompt = self._build_prompt(query_text)
            response = self._llm.generate(prompt=prompt, context="").strip()
            if response:
                logger.debug("Step-back generated for %r: %r", query_text[:60], response[:80])
            return response
        except Exception as exc:
            logger.warning("Step-back generation failed for %r: %s", query_text[:60], exc)
            return ""
