from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from string import Template

from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.domain.repositories.llm_repository import LLMRepository
from src.domain.repositories.vector_store_repository import SearchResult
from src.infrastructure.vectordb.neo4j_graph import GraphRelation, Neo4jGraphRepository
from src.rag.retrieval.bm25_retriever import BM25Retriever
from src.rag.retrieval.filters import apply_chunk_filters

logger = logging.getLogger(__name__)

# Metadata lives only in BM25/Qdrant payloads, so graph search must over-fetch
# before post-filtering (same pattern as hybrid candidate expansion).
_METADATA_OVERFETCH = 10
_MAX_GRAPH_FETCH = 500

_EXTRACT_PROMPT = Path(__file__).parents[2] / "prompts" / "retrieval" / "entity_extraction.txt"
_ENTITY_SPLIT = re.compile(r"[,;.!?\s]+")


class EntityExtractor:
    """Extracts (subject, relation, object) triple from text using the LLM.

    Used at **ingestion** time to populate the knowledge graph and at
    **retrieval** time to pull entity names from the user query.
    """

    def __init__(self, llm: LLMRepository) -> None:
        self._llm: LLMRepository = llm
        self._template: Template | None = None

    def extract_relations(self, text: str) -> list[GraphRelation]:
        """Return triples extracted from *text*."""
        prompt = self._get_template().substitute(text=text)
        try:
            response = self._llm.generate(prompt=prompt, context="").strip()
            return _parse_relations(response)
        except Exception as exc:
            logger.warning("Entity extraction failed: %s", exc)
            return []

    @staticmethod
    def extract_entity_names(text: str) -> list[str]:
        """Cheap extraction: return capitalized tokens as candidate entity names.

        Used at query time to avoid an LLM call on the hot path.  The LLM-based
        "extract_relations" is used only during ingestion.
        """
        tokens = _ENTITY_SPLIT.split(text)
        return list({t for t in tokens if t and t[0].isupper()})

    # ── Internals ──────────────────────────────────────────────────────────────

    def _get_template(self) -> Template:
        if self._template is not None:
            return self._template
        tpl = Template(_EXTRACT_PROMPT.read_text(encoding="utf-8"))
        self._template = tpl
        return tpl


class GraphRetriever:
    """Retrieves chunks via entity-based graph traversal in Neo4j.

    Flow:
      1. Extract entity names from the query (lightweight heuristic)
      2. Search Neo4j for chunks that mention those entities
      3. Look up full Chunk objects from BM25 (source-of-truth)
      4. Return (Chunk, score) pairs ready for RRF fusion

    When Neo4j is unavailable, it returns an empty list so that the hybrid
    retriever degrades gracefully to dense + BM25 only.
    """

    def __init__(
        self,
        extractor: EntityExtractor,
        graph: Neo4jGraphRepository,
        bm25: BM25Retriever,
    ) -> None:
        self._extractor: EntityExtractor = extractor
        self._graph: Neo4jGraphRepository = graph
        self._bm25: BM25Retriever = bm25

    async def search(
        self,
        query: str,
        top_k: int,
        *,
        filters: RetrievalFilter | None = None,
    ) -> list[SearchResult]:
        """Return up to *top_k* (Chunk, score) pairs from the knowledge graph."""
        entity_names = self._extractor.extract_entity_names(query)
        if not entity_names:
            return []

        fetch_k = _graph_fetch_limit(top_k, filters)
        try:
            id_score_pairs = await self._graph.search_by_entities(
                entity_names,
                fetch_k,
                filters=filters,
            )
        except Exception as exc:
            logger.warning("Graph retrieval failed (continuing without it): %s", exc)
            return []

        results: list[SearchResult] = []
        for chunk_id, score in id_score_pairs:
            chunk = self._bm25.get_by_id(chunk_id)
            if isinstance(chunk, Chunk):
                results.append((chunk, score))

        filtered = apply_chunk_filters(results, filters)
        logger.debug(
            "Graph retrieval: %d entities → %d chunks (%d after filters, fetch_k=%d)",
            len(entity_names),
            len(results),
            len(filtered),
            fetch_k,
        )
        return filtered[:top_k]

    @classmethod
    def from_settings(
        cls,
        llm: LLMRepository,
        bm25: BM25Retriever,
    ) -> GraphRetriever:
        return cls(
            extractor=EntityExtractor(llm=llm),
            graph=Neo4jGraphRepository.from_settings(),
            bm25=bm25,
        )


# ── helpers ────────────────────────────────────────────────────────────────────


def _graph_fetch_limit(top_k: int, filters: RetrievalFilter | None) -> int:
    """Size the Neo4j candidate pool so post-filters can still fill *top_k* slots."""
    if filters and filters.metadata:
        return min(top_k * _METADATA_OVERFETCH, _MAX_GRAPH_FETCH)
    return top_k


def _parse_relations(text: str) -> list[GraphRelation]:
    """Parse LLM JSON output into GraphRelation objects."""

    def _try(src: str) -> list[GraphRelation] | None:
        try:
            parsed: object = json.loads(src.strip())
            if not isinstance(parsed, list):
                return None
            result: list[GraphRelation] = []
            for item in parsed:
                if not isinstance(item, dict):
                    continue
                subj = item.get("subject")
                rel = item.get("relation")
                obj = item.get("object")
                if isinstance(subj, str) and isinstance(rel, str) and isinstance(obj, str):
                    result.append(
                        GraphRelation(
                            subject=subj.strip(),
                            relation=rel.strip(),
                            object=obj.strip(),
                            subject_type=str(item.get("subject_type", "Entity")),
                            object_type=str(item.get("object_type", "Entity")),
                        )
                    )
            return result
        except (json.JSONDecodeError, TypeError):
            return None

    if (r := _try(text)) is not None:
        return r
    match = re.search(r"\[.*?]", text, re.DOTALL)
    if match and (r := _try(match.group())) is not None:
        return r
    return []
