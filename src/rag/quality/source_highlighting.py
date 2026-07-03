from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from opentelemetry import trace
from pydantic import BaseModel

from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import (
    chunk_context_text,
    format_passages_for_llm,
    group_chunks_by_passage,
)
from src.rag.structured_output import parse_structured_output

if TYPE_CHECKING:
    from src.domain.entities.answer import Answer

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.quality")

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "quality" / "source_highlighting.txt"
_WHITESPACE = re.compile(r"\s+")


class ChunkHighlights(BaseModel):
    """Supporting sentence spans within one source passage."""

    chunk_id: str
    spans: list[str]


class SourceHighlightingOutput(BaseModel):
    """Structured LLM response with per-chunk highlight spans."""

    highlights: list[ChunkHighlights]


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def _normalize_whitespace(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def _validate_span(span: str, passage_text: str) -> str | None:
    """Return a verbatim substring of *passage_text* that matches *span*, else None."""
    candidate = span.strip()
    if not candidate:
        return None
    if candidate in passage_text:
        return candidate

    tokens = _normalize_whitespace(candidate).split()
    if not tokens:
        return None

    pattern = r"\s+".join(re.escape(token) for token in tokens)
    match = re.search(pattern, passage_text)
    if match is None:
        return None

    verbatim = match.group(0)
    if verbatim not in passage_text:
        return None
    return verbatim


def _lookup_spans(
    highlights_by_id: dict[str, list[str]],
    representative: Chunk,
    group: list[Chunk],
) -> list[str] | None:
    """Resolve highlight spans by representative or any sibling chunk ID in a *group*."""
    if representative.id in highlights_by_id:
        return highlights_by_id[representative.id]
    for chunk in group:
        if chunk.id in highlights_by_id:
            return highlights_by_id[chunk.id]
    return None


def map_highlights_to_chunks(
    highlights_by_id: dict[str, list[str]],
    chunks: list[Chunk],
) -> dict[str, list[str]]:
    """Validate LLM highlight spans and map them to every chunk ID in each passage group."""
    result: dict[str, list[str]] = {}

    for representative, group in group_chunks_by_passage(chunks):
        raw_spans = _lookup_spans(highlights_by_id, representative, group)
        if not raw_spans:
            logger.debug(
                "No highlights for passage group led by %s — omitting %d chunk(s)",
                representative.id,
                len(group),
            )
            continue

        passage_text = chunk_context_text(representative)
        validated: list[str] = []
        seen: set[str] = set()
        for span in raw_spans:
            verbatim = _validate_span(span, passage_text)
            if verbatim is None:
                logger.debug(
                    "Dropped non-verbatim highlight for %s: %r",
                    representative.id,
                    span[:80],
                )
                continue
            if verbatim not in seen:
                seen.add(verbatim)
                validated.append(verbatim)

        if not validated:
            continue

        for chunk in group:
            result[chunk.id] = list(validated)

    return result


def parse_source_highlighting(text: str) -> SourceHighlightingOutput:
    """Parse and validate structured highlight JSON from an LLM response."""
    return parse_structured_output(text, SourceHighlightingOutput, label="source highlighting")


def extract_highlights(
    answer: Answer,
    chunks: list[Chunk],
    llm: LLMRepository,
) -> dict[str, list[str]]:
    """Return chunk ID → verbatim supporting sentence spans for *answer*.

    Spans are validated as substrings of "chunk_context_text" for each passage
    group (the same text shown to the answer generator), then copied to every
    chunk ID in that group. On LLM or parse failure, returns an empty dict
    (caller omits highlights).
    """
    if not chunks or not answer.text.strip():
        return {}

    template = _load_prompt()
    prompt = template.substitute(
        answer=answer.text.strip(),
        passages=format_passages_for_llm(chunks, normalize_newlines=False),
    )

    with _tracer.start_as_current_span("quality.source_highlighting") as span:
        t0 = time.monotonic()
        try:
            response = llm.generate(prompt=prompt, context="")
            output = parse_source_highlighting(response)
        except Exception as exc:
            logger.warning("Source highlighting failed for answer %r: %s", answer.query_id, exc)
            span.set_attribute("quality.success", False)
            span.set_attribute("latency_ms", round((time.monotonic() - t0) * 1000, 1))
            return {}

        highlights_by_id = {item.chunk_id: item.spans for item in output.highlights}
        result = map_highlights_to_chunks(highlights_by_id, chunks)
        span.set_attribute("quality.success", True)
        span.set_attribute("quality.highlight_chunk_count", len(result))
        span.set_attribute("latency_ms", round((time.monotonic() - t0) * 1000, 1))
        return result
