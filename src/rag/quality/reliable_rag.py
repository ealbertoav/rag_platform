from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template

from pydantic import BaseModel, Field, ValidationError

from src.core.constants import CHUNK_PARENT_ID_KEY, PARENT_CONTEXT_TEXT_KEY
from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import chunk_context_text

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "quality" / "relevance_grading.txt"
_JSON_OBJECT = re.compile(r"\{.*}", re.DOTALL)


class ChunkRelevance(BaseModel):
    """Structured LLM output for one retrieved chunk."""

    chunk_id: str
    relevance_score: float = Field(ge=0.0, le=1.0)
    supporting: bool


class RelevanceGradingOutput(BaseModel):
    """Structured LLM response grading all chunks in one call."""

    grades: list[ChunkRelevance]


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def _grading_passage_key(chunk: Chunk) -> str:
    """Group chunks that share the same LLM-facing passage (e.g. parent context siblings)."""
    parent_id = chunk.metadata.get(CHUNK_PARENT_ID_KEY)
    parent_context = chunk.metadata.get(PARENT_CONTEXT_TEXT_KEY)
    if (
        isinstance(parent_id, str)
        and parent_id
        and isinstance(parent_context, str)
        and parent_context
    ):
        return parent_id
    return chunk.id


def _group_chunks_for_grading(chunks: list[Chunk]) -> list[tuple[Chunk, list[Chunk]]]:
    """Return representative chunks and the chunk groups they stand for."""
    groups: dict[str, list[Chunk]] = {}
    order: list[str] = []
    for chunk in chunks:
        key = _grading_passage_key(chunk)
        if key not in groups:
            order.append(key)
            groups[key] = []
        groups[key].append(chunk)
    return [(groups[key][0], groups[key]) for key in order]


def _format_passages(chunks: list[Chunk]) -> str:
    lines: list[str] = []
    for index, (representative, _) in enumerate(_group_chunks_for_grading(chunks), start=1):
        text = chunk_context_text(representative).strip().replace("\n", " ")
        lines.append(f"[{index}] chunk_id={representative.id}\n{text}")
    return "\n\n".join(lines)


def parse_relevance_grading(text: str) -> RelevanceGradingOutput:
    """Parse and validate structured relevance JSON from an LLM response."""
    candidates = [text.strip()]
    if match := _JSON_OBJECT.search(text):
        candidates.append(match.group())

    last_error: Exception | None = None
    for candidate in candidates:
        if not candidate:
            continue
        try:
            return RelevanceGradingOutput.model_validate_json(candidate)
        except (ValidationError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
            try:
                data = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            try:
                return RelevanceGradingOutput.model_validate(data)
            except ValidationError as nested:
                last_error = nested

    msg = "Could not parse relevance grading from LLM response"
    raise ValueError(msg) from last_error


def grade_relevance(
    query: str,
    chunks: list[Chunk],
    llm: LLMRepository,
    *,
    min_score: float = 0.5,
) -> tuple[list[Chunk], int, int]:
    """Grade chunks and return those meeting *min_score*.

    Returns ``(filtered_chunks, pass_count, fail_count)``. On LLM or parse
    failure, returns all input chunks with ``pass_count=len(chunks)`` and
    ``fail_count=0`` so retrieval degrades gracefully.
    """
    if not chunks:
        return [], 0, 0

    template = _load_prompt()
    prompt = template.substitute(
        query=query.strip(),
        passages=_format_passages(chunks),
    )

    try:
        response = llm.generate(prompt=prompt, context="")
        output = parse_relevance_grading(response)
    except Exception as exc:
        logger.warning("Relevance grading failed for %r: %s", query[:60], exc)
        return chunks, len(chunks), 0

    grade_by_id = {grade.chunk_id: grade for grade in output.grades}
    kept: list[Chunk] = []
    pass_count = 0
    fail_count = 0

    for representative, group in _group_chunks_for_grading(chunks):
        grade = grade_by_id.get(representative.id)
        if grade is None:
            logger.debug(
                "No grade for passage group led by %s — excluding %d chunk(s)",
                representative.id,
                len(group),
            )
            fail_count += len(group)
            continue
        if grade.relevance_score < min_score:
            logger.debug(
                "Passage group led by %s below min_score (%.2f < %.2f)",
                representative.id,
                grade.relevance_score,
                min_score,
            )
            fail_count += len(group)
            continue
        pass_count += len(group)
        for chunk in group:
            metadata = {
                **chunk.metadata,
                "relevance_score": grade.relevance_score,
                "relevance_supporting": grade.supporting,
            }
            kept.append(chunk.model_copy(update={"metadata": metadata}))

    return kept, pass_count, fail_count
