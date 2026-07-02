from __future__ import annotations

import logging
import re
from pathlib import Path
from string import Template

from pydantic import BaseModel

from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import chunk_context_text, group_chunks_by_passage
from src.rag.structured_output import parse_structured_output

logger = logging.getLogger(__name__)

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


def _format_passages(chunks: list[Chunk]) -> str:
    lines: list[str] = []
    for index, (representative, _) in enumerate(group_chunks_by_passage(chunks), start=1):
        text = chunk_context_text(representative).strip()
        lines.append(f"[{index}] chunk_id={representative.id}\n{text}")
    return "\n\n".join(lines)


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
        passages=_format_passages(chunks),
    )

    try:
        response = llm.generate(prompt=prompt, context="")
        output = parse_source_highlighting(response)
    except Exception as exc:
        logger.warning("Source highlighting failed for answer %r: %s", answer.query_id, exc)
        return {}

    highlights_by_id = {item.chunk_id: item.spans for item in output.highlights}
    result: dict[str, list[str]] = {}

    for representative, group in group_chunks_by_passage(chunks):
        raw_spans = highlights_by_id.get(representative.id)
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
