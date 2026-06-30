from __future__ import annotations

import json
import logging
import re
from enum import StrEnum
from pathlib import Path
from string import Template

from opentelemetry import trace
from pydantic import BaseModel, ValidationError

from src.domain.entities.query import Query
from src.domain.repositories.llm_repository import LLMRepository

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.retrieval")

_PROMPT_PATH = Path(__file__).parents[3] / "prompts" / "retrieval" / "query_classification.txt"
_JSON_OBJECT = re.compile(r"\{.*}", re.DOTALL)


class QueryCategory(StrEnum):
    FACTUAL = "factual"
    ANALYTICAL = "analytical"
    OPINION = "opinion"
    CONTEXTUAL = "contextual"


class ClassificationOutput(BaseModel):
    """Structured LLM output for query classification."""

    category: QueryCategory
    reasoning: str = ""


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def parse_classification(text: str) -> ClassificationOutput:
    """Parse and validate structured classification JSON from an LLM response."""
    candidates = [text.strip()]
    if match := _JSON_OBJECT.search(text):
        candidates.append(match.group())

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return ClassificationOutput.model_validate_json(candidate)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            try:
                return ClassificationOutput.model_validate(data)
            except ValidationError as nested:
                last_error = nested

    msg = "Could not parse classification from LLM response"
    raise ValueError(msg) from last_error


class QueryClassifier:
    """Classifies queries by intent using structured LLM output.

    When disabled, returns the query unchanged with no LLM call. On parse or
    LLM failure, defaults to "QueryCategory.FACTUAL".
    """

    def __init__(
        self,
        llm: LLMRepository,
        enabled: bool = True,
    ) -> None:
        self._llm = llm
        self._enabled = enabled
        self._prompt_template: Template | None = None
        self._cache: dict[str, QueryCategory] = {}

    def classify(self, query: Query) -> Query:
        """Return *query* with "metadata["category"]" populated."""
        if not self._enabled:
            return query

        with _tracer.start_as_current_span("retrieval.adaptive.classification") as span:
            category = self._classify_text(query.text)
            span.set_attribute("query.category", category.value)
            metadata = {**query.metadata, "category": category.value}
            return query.model_copy(update={"metadata": metadata})

    def clear_cache(self) -> None:
        self._cache.clear()

    @classmethod
    def from_settings(cls, llm: LLMRepository) -> QueryClassifier:
        from src.core.settings import settings

        cfg = settings.retrieval.adaptive
        return cls(llm=llm, enabled=cfg.enabled)

    def _build_prompt(self, query_text: str) -> str:
        template = self._prompt_template or _load_prompt()
        self._prompt_template = template
        return template.substitute(query=query_text)

    def _classify_text(self, query_text: str) -> QueryCategory:
        if query_text in self._cache:
            return self._cache[query_text]

        category = QueryCategory.FACTUAL
        try:
            prompt = self._build_prompt(query_text)
            response = self._llm.generate(prompt=prompt, context="")
            output = parse_classification(response)
            category = output.category
            logger.debug(
                "Query classified as %s for %r (%s)",
                category.value,
                query_text[:60],
                output.reasoning[:80],
            )
        except Exception as exc:
            logger.warning("Query classification failed for %r: %s", query_text[:60], exc)

        self._cache[query_text] = category
        return category
