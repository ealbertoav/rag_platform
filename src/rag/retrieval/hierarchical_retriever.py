from __future__ import annotations

import logging

from src.core.constants import CHUNK_TYPE_DETAIL, CHUNK_TYPE_SUMMARY
from src.domain.entities.query import Query
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.domain.repositories.vector_store_repository import SearchResult, VectorStoreRepository
from src.rag.enrichment.hierarchical_indexer import is_summary_chunk
from src.rag.retrieval.filters import effective_document_ids

logger = logging.getLogger(__name__)


class HierarchicalRetriever:
    """Two-stage retrieval: match document summaries, then search detail chunks.

    Stage 1 finds the most relevant documents via summary vectors.  Stage 2
    searches detail chunks scoped to those documents.  Only detail chunks are
    returned — summary text is never passed to downstream generation.
    """

    def __init__(
        self,
        embedder: EmbeddingRepository,
        vector_store: VectorStoreRepository,
        summary_top_k: int = 3,
    ) -> None:
        self._embedder: EmbeddingRepository = embedder
        self._vector_store: VectorStoreRepository = vector_store
        self._summary_top_k: int = summary_top_k

    def retrieve(self, query: Query, top_k: int) -> list[SearchResult]:
        """Return up to *top_k* detail chunks ranked by hierarchical search."""
        embedding = query.embedding or self._embedder.embed_query([query.text])[0]

        summary_hits = self._vector_store.search_dense(
            embedding,
            top_k=self._summary_top_k,
            type_equals=CHUNK_TYPE_SUMMARY,
            filters=query.filters,
        )
        summary_doc_ids = _document_ids_from_summaries(summary_hits)
        if not summary_doc_ids:
            logger.debug("Hierarchical retrieval: no summary matches")
            return []

        document_ids = effective_document_ids(summary_doc_ids, query.filters)
        if not document_ids:
            logger.debug("Hierarchical retrieval: no documents after scope intersection")
            return []

        detail_hits = self._vector_store.search_dense(
            embedding,
            top_k=top_k,
            type_equals=CHUNK_TYPE_DETAIL,
            document_ids=document_ids,
            filters=query.filters,
        )
        detail_results = [
            (chunk, score) for chunk, score in detail_hits if not is_summary_chunk(chunk)
        ]
        logger.debug(
            "Hierarchical retrieval: %d summaries → %d docs → %d detail hits",
            len(summary_hits),
            len(document_ids),
            len(detail_results),
        )
        return detail_results[:top_k]


def _document_ids_from_summaries(summary_hits: list[SearchResult]) -> frozenset[str]:
    """Collect unique document IDs from summary search hits, preserving rank order."""
    seen: set[str] = set()
    ordered: list[str] = []
    for chunk, _score in summary_hits:
        doc_id = chunk.document_id
        if doc_id not in seen:
            seen.add(doc_id)
            ordered.append(doc_id)
    return frozenset(ordered)
