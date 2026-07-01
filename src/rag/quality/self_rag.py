from __future__ import annotations

import logging
from enum import StrEnum
from pathlib import Path
from string import Template

from pydantic import BaseModel, Field

from src.domain.repositories.llm_repository import LLMRepository
from src.rag.structured_output import parse_structured_output

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parents[2] / "prompts" / "quality"


class UtilityAction(StrEnum):
    ACCEPT = "accept"
    RERETRIEVE = "reretrieve"
    REFUSE = "refuse"


class RetrievalDecision(BaseModel):
    """Structured LLM output: does this query need document retrieval?"""

    need_retrieval: bool
    reasoning: str


class SupportCheck(BaseModel):
    """Structured LLM output: is the draft answer supported by context?"""

    supported: bool
    reasoning: str


class UtilityScore(BaseModel):
    """Structured LLM output: utility of the draft answer."""

    score: float = Field(ge=0.0, le=1.0)
    action: UtilityAction
    reasoning: str
    refined_query: str = ""


def _load_prompt(name: str) -> Template:
    return Template((_PROMPTS_DIR / name).read_text(encoding="utf-8"))


def parse_retrieval_decision(text: str) -> RetrievalDecision:
    return parse_structured_output(text, RetrievalDecision, label="retrieval decision")


def parse_support_check(text: str) -> SupportCheck:
    return parse_structured_output(text, SupportCheck, label="support check")


def parse_utility_score(text: str) -> UtilityScore:
    return parse_structured_output(text, UtilityScore, label="utility score")


def decide_retrieval(query: str, llm: LLMRepository) -> RetrievalDecision:
    """Ask the LLM whether document retrieval is needed for a *query*."""
    template = _load_prompt("self_rag_decision.txt")
    prompt = template.substitute(query=query.strip())
    try:
        response = llm.generate(prompt=prompt, context="")
        return parse_retrieval_decision(response)
    except Exception as exc:
        logger.warning("Self-RAG retrieval decision failed for %r: %s", query[:60], exc)
        return RetrievalDecision(need_retrieval=True, reasoning="fallback: assume retrieval needed")


def check_support(
    query: str,
    draft_answer: str,
    context: str,
    llm: LLMRepository,
) -> SupportCheck:
    """Check whether *context* fully supports *draft_answer*."""
    template = _load_prompt("self_rag_support.txt")
    prompt = template.substitute(
        query=query.strip(),
        context=context.strip() or "(no context)",
        draft_answer=draft_answer.strip(),
    )
    try:
        response = llm.generate(prompt=prompt, context="")
        return parse_support_check(response)
    except Exception as exc:
        logger.warning("Self-RAG support check failed for %r: %s", query[:60], exc)
        return SupportCheck(supported=True, reasoning="fallback: assume supported")


def score_utility(
    query: str,
    draft_answer: str,
    context: str,
    llm: LLMRepository,
) -> UtilityScore:
    """Score draft answer utility and recommend accept / re-retrieve / refuse."""
    template = _load_prompt("self_rag_utility.txt")
    prompt = template.substitute(
        query=query.strip(),
        context=context.strip() or "(no context)",
        draft_answer=draft_answer.strip(),
    )
    try:
        response = llm.generate(prompt=prompt, context="")
        return parse_utility_score(response)
    except Exception as exc:
        logger.warning("Self-RAG utility scoring failed for %r: %s", query[:60], exc)
        return UtilityScore(
            score=0.5,
            action=UtilityAction.ACCEPT,
            reasoning="fallback: accept draft",
        )
