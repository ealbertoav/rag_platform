from __future__ import annotations

import logging
from pathlib import Path
from string import Template

from opentelemetry import trace

from src.core.constants import CHUNK_TYPE_HYPE, CHUNK_TYPE_SUMMARY
from src.domain.entities.query import Query
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.llm_repository import LLMRepository
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository

logger = logging.getLogger(__name__)
_tracer = trace.get_tracer("rag-platform.retrieval")

_PROMPT_PATH = Path(__file__).parents[2] / "prompts" / "retrieval" / "hyde_generate.txt"
_HIERARCHICAL_EXCLUDE = frozenset({CHUNK_TYPE_HYPE, CHUNK_TYPE_SUMMARY})


def _load_prompt() -> Template:
    return Template(_PROMPT_PATH.read_text(encoding="utf-8"))


class HyDERetriever:
    """Hypothetical Document Embedding (HyDE) retriever.

    Generates a hypothetical answer passage via LLM, embeds it, and runs dense
    search — useful for vague or underspecified questions where the query
    embedding alone is a poor match.
    """

    def __init__(
        self,
        llm: LLMRepository,
        embedder: EmbeddingRepository,
        vector_store: VectorStoreRepository,
    ) -> None:
        self._llm: LLMRepository = llm
        self._embedder: EmbeddingRepository = embedder
        self._vector_store: VectorStoreRepository = vector_store
        self._prompt_template: Template | None = None

    def generate_hypothetical_doc(self, query_text: str) -> str:
        """Return an LLM-generated passage that hypothetically answers *query_text*."""
        template = self._prompt_template or _load_prompt()
        self._prompt_template = template
        prompt = template.substitute(query=query_text)
        return self._llm.generate(prompt=prompt, context="").strip()

    def retrieve(self, query: Query, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* chunks ranked by dense similarity to the hypothetical doc."""
        with _tracer.start_as_current_span("retrieval.hyde") as span:
            try:
                hypo_doc = self.generate_hypothetical_doc(query.text)
                span.set_attribute("hypothetical_doc_length", len(hypo_doc))
                if not hypo_doc:
                    logger.debug("HyDE: empty hypothetical doc for %r", query.text[:60])
                    return []

                embedding = self._embedder.embed_query([hypo_doc])[0]
                results = self._vector_store.search_dense(
                    embedding,
                    top_k=top_k,
                    exclude_types=_HIERARCHICAL_EXCLUDE,
                    filters=query.filters,
                )
                logger.debug(
                    "HyDE retrieval: %d results for %r (hypo_doc=%d chars)",
                    len(results),
                    query.text[:60],
                    len(hypo_doc),
                )
                return results
            except Exception as exc:
                logger.warning("HyDE retrieval failed for %r: %s", query.text[:60], exc)
                span.set_attribute("hypothetical_doc_length", 0)
                return []
