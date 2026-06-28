from __future__ import annotations

import dataclasses
import json
import logging
import re
from collections.abc import AsyncIterator
from enum import StrEnum
from pathlib import Path
from string import Template
from typing import TYPE_CHECKING

from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.chunking.contextual_headers import join_chunk_context
from src.rag.enrichment.relevant_segment_extraction import chunk_source_ids
from src.rag.pipelines.chat_pipeline import ChatPipeline
from src.rag.ranking.score_fusion import rrf_fuse

if TYPE_CHECKING:
    from src.infrastructure.vectordb.bm25 import BM25Index

logger = logging.getLogger(__name__)

_DECISION_PROMPT = Path(__file__).parents[2] / "prompts" / "system" / "agent_decision.txt"
_DEFAULT_MAX_ITERATIONS = 3


class AgentAction(StrEnum):
    ANSWER = "ANSWER"
    RETRIEVE_MORE = "RETRIEVE_MORE"
    GRAPH_LOOKUP = "GRAPH_LOOKUP"
    CLARIFY = "CLARIFY"


@dataclasses.dataclass
class AgentDecision:
    action: AgentAction
    reasoning: str
    refined_query: str = ""
    entities: list[str] = dataclasses.field(default_factory=list)
    clarification: str = ""


@dataclasses.dataclass
class AgentRunResult:
    """Output of an agentic retrieval and generation run."""

    answer: Answer
    iterations: int
    actions: list[AgentAction]


@dataclasses.dataclass
class AgentRetrieveResult:
    chunks: list[Chunk]
    iterations: int
    actions: list[AgentAction]


class AgentPipeline:
    """Tool-calling agent layer over the RAG stack.

    For each question the agent iterates:
      1. Retrieve context (dense + BM25 + optional graph)
      2. Ask the LLM: is this enough? (ANSWER / RETRIEVE_MORE / GRAPH_LOOKUP / CLARIFY)
      3. Act on the decision — re-retrieve with a refined query, look up entities in
         Neo4j, ask the user a clarifying question, or generate the final answer.

    Iteration is capped at *max_iterations* to prevent runaway loops.  On any
    LLM decision parse failure, the agent falls back to answering with whatever
    context is available.
    """

    def __init__(
        self,
        pipeline: ChatPipeline,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self._pipeline = pipeline
        self._max_iterations = max_iterations
        self._decision_template: Template | None = None

    # ── Public ─────────────────────────────────────────────────────────────────

    async def chat(
        self,
        question: str,
        *,
        max_iterations: int | None = None,
    ) -> AsyncIterator[str]:
        """Run the agentic loop and stream the final answer."""
        max_iter = max_iterations if max_iterations is not None else self._max_iterations
        result = await self._agentic_retrieve(question, max_iterations=max_iter)
        context = self._build_context(result.chunks)
        return self._pipeline.generation.stream(question, context)

    async def chat_full(
        self,
        question: str,
        *,
        max_iterations: int | None = None,
    ) -> AgentRunResult:
        """Run the agentic loop and return a complete result."""
        import time

        max_iter = max_iterations if max_iterations is not None else self._max_iterations
        t0 = time.monotonic()
        run = await self._agentic_retrieve(question, max_iterations=max_iter)
        context = self._build_context(run.chunks)
        sources = [chunk_id for c in run.chunks for chunk_id in chunk_source_ids(c)]
        answer = self._pipeline.generation.generate(question, context, sources)
        elapsed = (time.monotonic() - t0) * 1000
        final_answer = answer.model_copy(
            update={
                "query_id": Query(text=question).id,
                "latency_ms": elapsed,
                "token_count": len(answer.text.split()),
            }
        )
        return AgentRunResult(
            answer=final_answer,
            iterations=run.iterations,
            actions=run.actions,
        )

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(
        cls,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
        bm25_index: BM25Index | None = None,
    ) -> AgentPipeline:
        return cls(
            pipeline=ChatPipeline.from_settings(bm25_index=bm25_index),
            max_iterations=max_iterations,
        )

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _agentic_retrieve(
        self,
        question: str,
        *,
        max_iterations: int | None = None,
    ) -> AgentRetrieveResult:
        """Iterative retrieval loop — returns the final merged chunk list."""
        max_iter = max_iterations if max_iterations is not None else self._max_iterations
        query = Query(text=question)
        retrieval = await self._pipeline.retrieval.retrieve(query)
        chunks: list[Chunk] = list(retrieval.chunks)
        actions: list[AgentAction] = []

        for iteration in range(max_iter):
            if not chunks:
                break

            decision = self._decide(question, chunks)
            actions.append(decision.action)
            logger.debug(
                "Agent iteration %d: %s — %s",
                iteration + 1,
                decision.action,
                decision.reasoning,
            )

            if decision.action == AgentAction.ANSWER:
                return AgentRetrieveResult(
                    chunks=chunks,
                    iterations=iteration + 1,
                    actions=actions,
                )

            if decision.action == AgentAction.CLARIFY:
                logger.info("Agent: clarification needed — %s", decision.clarification)
                return AgentRetrieveResult(chunks=[], iterations=iteration + 1, actions=actions)

            if decision.action == AgentAction.RETRIEVE_MORE and decision.refined_query:
                refined_query = Query(text=decision.refined_query)
                refined = await self._pipeline.retrieval.retrieve(refined_query)
                from src.domain.repositories.vector_store_repository import SearchResult

                existing: list[SearchResult] = [(c, 1.0) for c in chunks]
                new_results: list[SearchResult] = [(c, 1.0) for c in refined.chunks]
                merged = rrf_fuse(existing, new_results, top_k=len(chunks))
                chunks = [c for c, _ in merged]

            elif decision.action == AgentAction.GRAPH_LOOKUP and decision.entities:
                graph_chunks = self._graph_lookup(decision.entities, chunks)
                if graph_chunks:
                    from src.domain.repositories.vector_store_repository import SearchResult

                    existing = [(c, 1.0) for c in chunks]
                    graph_sr: list[SearchResult] = [(c, 1.0) for c in graph_chunks]
                    merged = rrf_fuse(existing, graph_sr, top_k=len(chunks) + len(graph_chunks))
                    chunks = [c for c, _ in merged]
                return AgentRetrieveResult(
                    chunks=chunks,
                    iterations=iteration + 1,
                    actions=actions,
                )

        return AgentRetrieveResult(
            chunks=chunks,
            iterations=min(max_iter, max(1, len(actions))),
            actions=actions,
        )

    @staticmethod
    def _build_context(chunks: list[Chunk]) -> str:
        """Join chunk passages for LLM prompts (respects CCH raw_text and RSE merges)."""
        return join_chunk_context(chunks)

    def _decide(self, question: str, chunks: list[Chunk]) -> AgentDecision:
        """Ask the LLM whether to answer or refine retrieval."""
        context = self._build_context(chunks[:5])  # cap to 5 chunks
        template = self._get_decision_template()
        prompt = template.substitute(question=question, context=context)
        try:
            raw = self._pipeline.generation.call_llm(prompt)
            return parse_decision(raw)
        except Exception as exc:
            logger.warning("Agent decision parsing failed: %s — defaulting to ANSWER", exc)
            return AgentDecision(action=AgentAction.ANSWER, reasoning="fallback")

    def _graph_lookup(self, entities: list[str], existing: list[Chunk]) -> list[Chunk]:
        """Try graph retrieval if a GraphRetriever is wired into the hybrid retriever."""
        try:
            graph_retriever = self._pipeline.retrieval.service.hybrid.graph
            if graph_retriever is None:
                return []
            existing_ids = {c.id for c in existing}
            results = graph_retriever.search(" ".join(entities), top_k=5)
            return [c for c, _ in results if c.id not in existing_ids]
        except Exception as exc:
            logger.debug("Graph lookup skipped: %s", exc)
            return []

    def _get_decision_template(self) -> Template:
        if self._decision_template is not None:
            return self._decision_template
        tpl = Template(_DECISION_PROMPT.read_text(encoding="utf-8"))
        self._decision_template = tpl
        return tpl


# ── helpers ────────────────────────────────────────────────────────────────────


def parse_decision(text: str) -> AgentDecision:
    """Parse the LLM's JSON decision output."""

    def _try(src: str) -> AgentDecision | None:
        try:
            data: object = json.loads(src.strip())
            if not isinstance(data, dict):
                return None
            action_str = data.get("action", "ANSWER")
            action = AgentAction(str(action_str).upper())
            return AgentDecision(
                action=action,
                reasoning=str(data.get("reasoning", "")),
                refined_query=str(data.get("refined_query", "")),
                entities=[str(e) for e in (data.get("entities") or [])],
                clarification=str(data.get("clarification", "")),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return None

    if (d := _try(text)) is not None:
        return d
    match = re.search(r"\{.*}", text, re.DOTALL)
    if match and (d := _try(match.group())) is not None:
        return d
    return AgentDecision(action=AgentAction.ANSWER, reasoning="parse-fallback")
