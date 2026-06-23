from __future__ import annotations

import dataclasses
import enum
import json
import logging
import re
from collections.abc import AsyncIterator
from pathlib import Path
from string import Template

from src.domain.entities.answer import Answer
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.repositories.llm_repository import LLMRepository
from src.rag.pipelines.chat_pipeline import ChatPipeline
from src.rag.pipelines.retrieval_pipeline import RetrievalPipeline
from src.rag.ranking.score_fusion import rrf_fuse

logger = logging.getLogger(__name__)

_DECISION_PROMPT = Path(__file__).parents[2] / "prompts" / "system" / "agent_decision.txt"
_DEFAULT_MAX_ITERATIONS = 3


class AgentAction(str, enum.Enum):
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


class AgentPipeline:
    """Tool-calling agent layer over the RAG stack.

    For each question the agent iterates:
      1. Retrieve context (dense + BM25 + optional graph)
      2. Ask the LLM: is this enough? (ANSWER / RETRIEVE_MORE / GRAPH_LOOKUP / CLARIFY)
      3. Act on the decision — re-retrieve with a refined query, look up entities in
         Neo4j, ask the user a clarifying question, or generate the final answer.

    Iteration is capped at *max_iterations* to prevent runaway loops.  On any
    LLM decision parse failure the agent falls back to answering with whatever
    context is available.
    """

    def __init__(
        self,
        chat: ChatPipeline,
        max_iterations: int = _DEFAULT_MAX_ITERATIONS,
    ) -> None:
        self._chat = chat
        self._max_iterations = max_iterations
        self._decision_template: Template | None = None

    # ── Public ─────────────────────────────────────────────────────────────────

    async def chat(self, question: str) -> AsyncIterator[str]:
        """Run the agentic loop and stream the final answer."""
        chunks = await self._agentic_retrieve(question)
        context = "\n\n".join(c.text for c in chunks)
        return self._chat._generation.stream(question, context)

    async def chat_full(self, question: str) -> Answer:
        """Run the agentic loop and return a complete Answer."""
        import time

        t0 = time.monotonic()
        chunks = await self._agentic_retrieve(question)
        context = "\n\n".join(c.text for c in chunks)
        sources = [c.id for c in chunks]
        answer = self._chat._generation.generate(question, context, sources)
        elapsed = (time.monotonic() - t0) * 1000
        return answer.model_copy(
            update={
                "query_id": Query(text=question).id,
                "latency_ms": elapsed,
                "token_count": len(answer.text.split()),
            }
        )

    # ── Factory ────────────────────────────────────────────────────────────────

    @classmethod
    def from_settings(cls, max_iterations: int = _DEFAULT_MAX_ITERATIONS) -> AgentPipeline:
        return cls(chat=ChatPipeline.from_settings(), max_iterations=max_iterations)

    # ── Internals ──────────────────────────────────────────────────────────────

    async def _agentic_retrieve(self, question: str) -> list[Chunk]:
        """Iterative retrieval loop — returns the final merged chunk list."""
        query = Query(text=question)
        retrieval = await self._chat._retrieval.retrieve(query)
        chunks: list[Chunk] = list(retrieval.chunks)

        for iteration in range(self._max_iterations):
            if not chunks:
                break

            decision = self._decide(question, chunks)
            logger.debug(
                "Agent iteration %d: %s — %s",
                iteration + 1, decision.action, decision.reasoning,
            )

            if decision.action == AgentAction.ANSWER:
                break

            if decision.action == AgentAction.CLARIFY:
                # Surface the clarifying question as part of the answer text by
                # returning an empty chunk list; the generation layer will respond
                # with the no-context fallback message.
                logger.info("Agent: clarification needed — %s", decision.clarification)
                chunks = []
                break

            if decision.action == AgentAction.RETRIEVE_MORE and decision.refined_query:
                refined_query = Query(text=decision.refined_query)
                refined = await self._chat._retrieval.retrieve(refined_query)
                # Merge via RRF: existing chunks + new results
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
                break  # one graph lookup per session

        return chunks

    def _decide(self, question: str, chunks: list[Chunk]) -> AgentDecision:
        """Ask the LLM whether to answer or refine retrieval."""
        context = "\n\n".join(c.text for c in chunks[:5])  # cap to 5 chunks
        template = self._get_decision_template()
        prompt = template.substitute(question=question, context=context)
        try:
            raw = self._chat._generation._llm.generate(prompt=prompt, context="")
            return _parse_decision(raw)
        except Exception as exc:
            logger.warning("Agent decision parsing failed: %s — defaulting to ANSWER", exc)
            return AgentDecision(action=AgentAction.ANSWER, reasoning="fallback")

    def _graph_lookup(self, entities: list[str], existing: list[Chunk]) -> list[Chunk]:
        """Try graph retrieval if a GraphRetriever is wired into the hybrid retriever."""
        try:
            graph_retriever = self._chat._retrieval._service._hybrid._graph
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


def _parse_decision(text: str) -> AgentDecision:
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
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match and (d := _try(match.group())) is not None:
        return d
    return AgentDecision(action=AgentAction.ANSWER, reasoning="parse-fallback")
