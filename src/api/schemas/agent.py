from __future__ import annotations

import dataclasses
import logging

from src.rag.pipelines.agent_pipeline import AgentRunResult, SelfRAGStepDecision

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class SelfRAGDecisionResponse:
    iteration: int
    need_retrieval: bool | None
    retrieval_reasoning: str
    supported: bool | None
    support_reasoning: str
    utility_score: float | None
    utility_action: str | None
    utility_reasoning: str
    refined_query: str

    @classmethod
    def from_step(cls, step: SelfRAGStepDecision) -> SelfRAGDecisionResponse:
        return cls(
            iteration=step.iteration,
            need_retrieval=step.need_retrieval,
            retrieval_reasoning=step.retrieval_reasoning,
            supported=step.supported,
            support_reasoning=step.support_reasoning,
            utility_score=step.utility_score,
            utility_action=step.utility_action,
            utility_reasoning=step.utility_reasoning,
            refined_query=step.refined_query,
        )


@dataclasses.dataclass(frozen=True)
class AgentChatResponse:
    """API response for agentic chat."""

    answer: str
    sources: list[str]
    latency_ms: float
    token_count: int
    iterations: int
    actions: list[str]
    self_rag_decisions: list[SelfRAGDecisionResponse]

    @classmethod
    def from_run(cls, result: AgentRunResult) -> AgentChatResponse:
        return cls(
            answer=result.answer.text,
            sources=result.answer.sources,
            latency_ms=result.answer.latency_ms,
            token_count=result.answer.token_count,
            iterations=result.iterations,
            actions=[a.value for a in result.actions],
            self_rag_decisions=[
                SelfRAGDecisionResponse.from_step(step) for step in result.self_rag_decisions
            ],
        )
