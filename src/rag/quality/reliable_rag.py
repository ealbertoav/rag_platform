from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from pydantic import BaseModel, Field

from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import (
    format_passages_for_llm,
)
from src.rag.chunking.contextual_headers import (
    group_chunks_by_passage as _group_chunks_for_grading,
)
from src.rag.structured_output import parse_structured_output

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "quality" / "relevance_grading.txt"


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


def _format_passages(chunks: list[Chunk]) -> str:
    return format_passages_for_llm(chunks, normalize_newlines=True)


def parse_relevance_grading(text: str) -> RelevanceGradingOutput:
    """Parse and validate structured relevance JSON from an LLM response."""
    return parse_structured_output(text, RelevanceGradingOutput, label="relevance grading")


def grade_relevance(
    query: str,
    chunks: list[Chunk],
    llm: LLMRepository,
    *,
    min_score: float = 0.5,
) -> tuple[list[Chunk], int, int, list[float]]:
    """Grade chunks and return those meeting *min_score*.

    Returns "(filtered_chunks, pass_count, fail_count, all_graded_scores)".
    *all_graded_scores* includes every LLM-assigned score (pass and fail) for
    downstream CRAG thresholding. On LLM or parse failure, returns all input
    chunks with "pass_count=len(chunks)", "fail_count=0", and an empty score list.
    """
    if not chunks:
        return [], 0, 0, []

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
        return chunks, len(chunks), 0, []

    grade_by_id = {grade.chunk_id: grade for grade in output.grades}
    kept: list[Chunk] = []
    pass_count = 0
    fail_count = 0
    all_graded_scores: list[float] = []

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
        all_graded_scores.append(grade.relevance_score)
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

    return kept, pass_count, fail_count, all_graded_scores
