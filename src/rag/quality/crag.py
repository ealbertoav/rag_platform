from __future__ import annotations

import dataclasses
import logging
from enum import StrEnum
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from src.domain.entities.chunk import Chunk
from src.domain.repositories.llm_repository import LLMRepository
from src.domain.repositories.web_search_repository import WebSearchResult
from src.infrastructure.search.web_search import format_web_results

if TYPE_CHECKING:
    from opentelemetry.trace import Span

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "quality" / "crag_knowledge_refinement.txt"
_INSUFFICIENT = "INSUFFICIENT_INFORMATION"


class CRAGAction(StrEnum):
    """Corrective action chosen from the retrieval quality score."""

    USE_RETRIEVAL = "use_retrieval"
    COMBINE_AND_REFINE = "combine_and_refine"
    WEB_ONLY = "web_only"


@dataclasses.dataclass(frozen=True)
class RetrievalQualityScore:
    """Aggregate retrieval quality used to choose a CRAG branch."""

    score: float
    graded: bool
    """True when scores come from chunk "relevance_score" metadata (T-140)."""


@dataclasses.dataclass(frozen=True)
class CRAGDecision:
    """Outcome of one CRAG evaluation for observability and tests."""

    quality_score: float
    action: CRAGAction
    web_search_used: bool
    quality_graded: bool = True
    web_result_count: int = 0
    refined: bool = False
    fallback_to_retrieval: bool = False
    skipped: bool = False
    """Corrective branch skipped (e.g. no relevance grades from Reliable RAG)."""


@dataclasses.dataclass(frozen=True)
class ContextResolution:
    """LLM context plus eval passages after optional CRAG correction."""

    context: str
    sources: list[str]
    eval_contexts: list[str]


def score_retrieval_quality(chunks: list[Chunk]) -> RetrievalQualityScore:
    """Compute aggregate retrieval quality for CRAG thresholding.

    Uses mean "relevance_score" from Reliable RAG (T-140) metadata when present.
    Empty retrieval returns "score=0.0, graded=True". Chunks without grades return
    "graded=False" so callers skip the corrective branch until Reliable RAG runs.
    """
    if not chunks:
        return RetrievalQualityScore(score=0.0, graded=True)

    scores: list[float] = []
    for chunk in chunks:
        raw = chunk.metadata.get("relevance_score")
        if isinstance(raw, (int, float)):
            scores.append(float(raw))

    if not scores:
        return RetrievalQualityScore(score=0.0, graded=False)

    return RetrievalQualityScore(score=sum(scores) / len(scores), graded=True)


def determine_crag_action(
    quality_score: float,
    *,
    lower_threshold: float,
    upper_threshold: float,
) -> CRAGAction:
    """Map a quality score to the CRAG branch from RAG_Techniques thresholds."""
    if quality_score > upper_threshold:
        return CRAGAction.USE_RETRIEVAL
    if quality_score < lower_threshold:
        return CRAGAction.WEB_ONLY
    return CRAGAction.COMBINE_AND_REFINE


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


def refine_knowledge(
    query: str,
    retrieval_context: str,
    web_results: list[WebSearchResult],
    llm: LLMRepository,
) -> str:
    """Merge retrieval and web evidence into a single LLM-ready context string."""
    template = _load_prompt()
    prompt = template.substitute(
        query=query.strip(),
        retrieval_context=retrieval_context.strip() or "(none)",
        web_context=format_web_results(web_results) or "(none)",
    )

    try:
        refined = llm.generate(prompt=prompt, context="").strip()
    except Exception as exc:
        logger.warning("CRAG knowledge refinement failed for %r: %s", query[:60], exc)
        return ""

    if not refined or refined.upper() == _INSUFFICIENT:
        return ""
    return refined


def crag_fallback_without_web(
    query_text: str,
    retrieval_context: str,
    quality_score: float,
    action: CRAGAction,
    *,
    web_search_attempted: bool = False,
    web_result_count: int = 0,
    refinement_attempted: bool = False,
) -> tuple[str, CRAGDecision]:
    """Gracefully degrade when corrective web search cannot run."""
    if action == CRAGAction.COMBINE_AND_REFINE and retrieval_context.strip():
        logger.warning(
            "CRAG %s for %r unavailable — falling back to retrieved context",
            action.value,
            query_text[:60],
        )
        return retrieval_context, CRAGDecision(
            quality_score=quality_score,
            action=action,
            web_search_used=web_search_attempted,
            web_result_count=web_result_count,
            refined=refinement_attempted,
            fallback_to_retrieval=True,
        )

    logger.warning(
        "CRAG %s for %r could not be satisfied — insufficient information",
        action.value,
        query_text[:60],
    )
    return "", CRAGDecision(
        quality_score=quality_score,
        action=action,
        web_search_used=web_search_attempted,
        web_result_count=web_result_count,
        refined=refinement_attempted,
    )


def eval_contexts_for_resolution(
    *,
    chunks: list[Chunk],
    resolved_context: str,
    refined: bool,
) -> list[str]:
    """Return passage texts aligned with what generation/evals should treat as context."""
    if not resolved_context.strip():
        return []
    if refined:
        return [resolved_context]
    return [chunk.text for chunk in chunks]


def record_crag_span(span: Span, decision: CRAGDecision) -> None:
    """Attach CRAG decision attributes to an OTel span."""
    span.set_attribute("crag.quality_score", round(decision.quality_score, 4))
    span.set_attribute("crag.action", decision.action.value)
    span.set_attribute("crag.quality_graded", decision.quality_graded)
    span.set_attribute("crag.web_search_used", decision.web_search_used)
    span.set_attribute("crag.web_result_count", decision.web_result_count)
    span.set_attribute("crag.refined", decision.refined)
    span.set_attribute("crag.fallback_to_retrieval", decision.fallback_to_retrieval)
    span.set_attribute("crag.skipped", decision.skipped)
