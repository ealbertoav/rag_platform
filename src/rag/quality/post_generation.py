from __future__ import annotations

import logging
import time
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from opentelemetry import trace
from pydantic import BaseModel

from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import format_passages_for_llm
from src.rag.quality.explainable_retrieval import ChunkExplanation, map_explanations_to_chunks
from src.rag.quality.source_highlighting import ChunkHighlights, map_highlights_to_chunks
from src.rag.structured_output import parse_structured_output

if TYPE_CHECKING:
    from src.domain.entities.answer import Answer

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.quality")

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "quality" / "explain_and_highlight.txt"


class ExplainAndHighlightOutput(BaseModel):
    """Structured LLM response with per-chunk explanations and highlight spans."""

    explanations: list[ChunkExplanation]
    highlights: list[ChunkHighlights]


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def parse_explain_and_highlight(text: str) -> ExplainAndHighlightOutput:
    """Parse and validate combined explain and highlight JSON from an LLM response."""
    return parse_structured_output(text, ExplainAndHighlightOutput, label="explain and highlight")


def explain_and_highlight(
    query: str,
    answer: Answer,
    chunks: list[Chunk],
    llm: LLMRepository,
) -> tuple[list[ChunkExplanation], dict[str, list[str]]]:
    """Return explanations and highlight spans in a single LLM call.

    On LLM or parse failure, returns empty results (caller omits both fields).
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
            span.set_attribute("latency_ms", round((time.monotonic() - t0) * 1000, 1))
            return [], {}

        explanation_by_id = {item.chunk_id: item for item in output.explanations}
        highlights_by_id = {item.chunk_id: item.spans for item in output.highlights}
        explanations = map_explanations_to_chunks(explanation_by_id, chunks)
        highlights = map_highlights_to_chunks(highlights_by_id, chunks)
        span.set_attribute("quality.success", True)
        span.set_attribute("quality.explanation_count", len(explanations))
        span.set_attribute("quality.highlight_chunk_count", len(highlights))
        span.set_attribute("latency_ms", round((time.monotonic() - t0) * 1000, 1))
        return explanations, highlights
