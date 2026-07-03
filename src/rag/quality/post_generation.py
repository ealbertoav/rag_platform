from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from opentelemetry import trace
from pydantic import BaseModel, Field, field_validator

from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import format_passages_for_llm
from src.rag.quality.explainable_retrieval import (
    ChunkExplanation,
    map_explanations_to_chunks,
    parse_explain_retrieval,
)
from src.rag.quality.source_highlighting import (
    ChunkHighlights,
    map_highlights_to_chunks,
    parse_source_highlighting,
)
from src.rag.structured_output import extract_json_object, parse_structured_output

if TYPE_CHECKING:
    from src.domain.entities.answer import Answer

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.quality")

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "quality" / "explain_and_highlight.txt"


class ExplainAndHighlightOutput(BaseModel):
    """Structured LLM response with per-chunk explanations and highlight spans."""

    explanations: list[ChunkExplanation] = Field(default_factory=list)
    highlights: list[ChunkHighlights] = Field(default_factory=list)

    @field_validator("explanations", "highlights", mode="before")
    @classmethod
    def _coerce_none_to_empty(cls, value: object) -> object:
        return [] if value is None else value


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def _parse_explanations_field(raw: object) -> list[ChunkExplanation]:
    if raw is None:
        return []
    try:
        return parse_explain_retrieval(json.dumps({"explanations": raw})).explanations
    except ValueError:
        logger.debug("Combined response: failed to parse explanations field")
        return []


def _parse_highlights_field(raw: object) -> list[ChunkHighlights]:
    if raw is None:
        return []
    try:
        return parse_source_highlighting(json.dumps({"highlights": raw})).highlights
    except ValueError:
        logger.debug("Combined response: failed to parse highlights field")
        return []


def parse_explain_and_highlight(text: str) -> ExplainAndHighlightOutput:
    """Parse combined to explain and highlight JSON from an LLM response.

    Accepts responses that include only one top-level field, null/omitted arrays,
    or a mix where one field is malformed but the other is valid.
    """
    try:
        return parse_structured_output(
            text, ExplainAndHighlightOutput, label="explain and highlight"
        )
    except ValueError:
        pass

    data = extract_json_object(text)
    if data is None:
        msg = "Could not parse explain and highlight from LLM response"
        raise ValueError(msg)

    explanations = _parse_explanations_field(data.get("explanations"))
    highlights = _parse_highlights_field(data.get("highlights"))
    if not explanations and not highlights:
        msg = "Could not parse explain and highlight from LLM response"
        raise ValueError(msg)

    return ExplainAndHighlightOutput(explanations=explanations, highlights=highlights)


def explain_and_highlight(
    query: str,
    answer: Answer,
    chunks: list[Chunk],
    llm: LLMRepository,
) -> tuple[list[ChunkExplanation], dict[str, list[str]]]:
    """Return explanations and highlight spans in a single LLM call.

    Each result is mapped independently; one side may succeed while the other is
    empty. Returns empty results only when parsing fails entirely or the LLM call
    raises.
    """
    if not chunks or not answer.text.strip():
        return [], {}

    template = _load_prompt()
    prompt = template.substitute(
        query=query.strip(),
        answer=answer.text.strip(),
        passages=format_passages_for_llm(chunks, normalize_newlines=False),
    )

    with _tracer.start_as_current_span("quality.explain_and_highlight") as span:
        t0 = time.monotonic()
        try:
            response = llm.generate(prompt=prompt, context="")
            output = parse_explain_and_highlight(response)
        except Exception as exc:
            logger.warning(
                "Combined explain/highlight failed for answer %r: %s",
                answer.query_id,
                exc,
            )
            span.set_attribute("quality.success", False)
            span.set_attribute("quality.explanations_success", False)
            span.set_attribute("quality.highlights_success", False)
            span.set_attribute("latency_ms", round((time.monotonic() - t0) * 1000, 1))
            return [], {}

        explanation_by_id = {item.chunk_id: item for item in output.explanations}
        highlights_by_id = {item.chunk_id: item.spans for item in output.highlights}
        explanations = map_explanations_to_chunks(explanation_by_id, chunks)
        highlights = map_highlights_to_chunks(highlights_by_id, chunks)
        explanations_ok = bool(explanations)
        highlights_ok = bool(highlights)
        span.set_attribute("quality.success", explanations_ok or highlights_ok)
        span.set_attribute("quality.explanations_success", explanations_ok)
        span.set_attribute("quality.highlights_success", highlights_ok)
        span.set_attribute("quality.explanation_count", len(explanations))
        span.set_attribute("quality.highlight_chunk_count", len(highlights))
        span.set_attribute("latency_ms", round((time.monotonic() - t0) * 1000, 1))
        return explanations, highlights
