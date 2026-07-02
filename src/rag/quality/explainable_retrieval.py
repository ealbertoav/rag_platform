from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from pydantic import BaseModel

from src.core.constants import MERGED_CHUNK_IDS_KEY
from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.chunking.contextual_headers import (
    chunk_context_text,
    group_chunks_by_passage,
)
from src.rag.structured_output import parse_structured_output

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "quality" / "explain_retrieval.txt"


class ChunkExplanation(BaseModel):
    """Human-readable reason a retrieved chunk was selected for the query."""

    chunk_id: str
    reason: str


class ExplainRetrievalOutput(BaseModel):
    """Structured LLM response with per-chunk retrieval explanations."""

    explanations: list[ChunkExplanation]


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def _format_passages(chunks: list[Chunk]) -> str:
    lines: list[str] = []
    for index, (representative, _) in enumerate(group_chunks_by_passage(chunks), start=1):
        text = chunk_context_text(representative).strip().replace("\n", " ")
        lines.append(f"[{index}] chunk_id={representative.id}\n{text}")
    return "\n\n".join(lines)


def resolve_chunks_for_sources(sources: list[str], chunks: list[Chunk]) -> list[Chunk]:
    """Map citation source IDs to representative chunks for explanation."""
    by_id = {chunk.id: chunk for chunk in chunks}
    merged_lookup: dict[str, Chunk] = {}
    for chunk in chunks:
        merged_ids = chunk.metadata.get(MERGED_CHUNK_IDS_KEY)
        if isinstance(merged_ids, list):
            for merged_id in merged_ids:
                merged_lookup[str(merged_id)] = chunk

    resolved: list[Chunk] = []
    seen: set[str] = set()
    for source_id in sources:
        if source_id in seen:
            continue
        seen.add(source_id)
        if source_id in by_id:
            resolved.append(by_id[source_id])
        elif source_id in merged_lookup:
            resolved.append(merged_lookup[source_id].model_copy(update={"id": source_id}))
    return resolved


def parse_explain_retrieval(text: str) -> ExplainRetrievalOutput:
    """Parse and validate structured explanation JSON from an LLM response."""
    return parse_structured_output(text, ExplainRetrievalOutput, label="explain retrieval")


def explain_chunks(
    query: str,
    chunks: list[Chunk],
    llm: LLMRepository,
) -> list[ChunkExplanation]:
    """Return human-readable explanations for why each *chunk* was retrieved.

    On LLM or parse failure, returns an empty list (caller omits explanations).
    """
    if not chunks:
        return []

    template = _load_prompt()
    prompt = template.substitute(
        query=query.strip(),
        passages=_format_passages(chunks),
    )

    try:
        response = llm.generate(prompt=prompt, context="")
        output = parse_explain_retrieval(response)
    except Exception as exc:
        logger.warning("Explainable retrieval failed for %r: %s", query[:60], exc)
        return []

    explanation_by_id = {item.chunk_id: item for item in output.explanations}
    explanations: list[ChunkExplanation] = []
    for representative, group in group_chunks_by_passage(chunks):
        explanation = explanation_by_id.get(representative.id)
        if explanation is None:
            logger.debug(
                "No explanation for passage group led by %s — omitting %d chunk(s)",
                representative.id,
                len(group),
            )
            continue
        for chunk in group:
            explanations.append(ChunkExplanation(chunk_id=chunk.id, reason=explanation.reason))
    return explanations
