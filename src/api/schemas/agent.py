from __future__ import annotations

import dataclasses
import logging

from src.rag.pipelines.agent_pipeline import AgentRunResult

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class AgentChatResponse:
    """API response for agentic chat."""

    answer: str
    sources: list[str]
    latency_ms: float
    token_count: int
    iterations: int
    actions: list[str]

    @classmethod
    def from_run(cls, result: AgentRunResult) -> AgentChatResponse:
        return cls(
            answer=result.answer.text,
            sources=result.answer.sources,
            latency_ms=result.answer.latency_ms,
            token_count=result.answer.token_count,
            iterations=result.iterations,
            actions=[a.value for a in result.actions],
        )
