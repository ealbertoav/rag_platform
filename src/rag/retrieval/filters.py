from __future__ import annotations

from qdrant_client.models import Condition, FieldCondition, Filter, MatchAny, MatchValue

from src.core.constants import CHUNK_TYPE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.domain.repositories.vector_store_repository import SearchResult


def effective_document_ids(
    explicit: frozenset[str] | None,
    filters: RetrievalFilter | None,
) -> frozenset[str] | None:
    """Merge an explicit document scope with filter document IDs (intersection when both set).

    "None" explicitly means no caller-provided scope (use filter IDs when present).
    An empty "frozenset()" means the caller explicitly found no documents — it must
    not fall back to filter IDs (e.g. hierarchical stage 1 with zero summary hits).
    """
    from_filter = frozenset(filters.document_ids) if filters and filters.document_ids else None
    if explicit is not None and from_filter is not None:
        return explicit & from_filter
    if explicit is not None:
        return explicit
    return from_filter


def build_qdrant_filter(
    *,
    type_equals: str | None = None,
    exclude_types: frozenset[str] | None = None,
    document_ids: frozenset[str] | None = None,
    filters: RetrievalFilter | None = None,
) -> Filter | None:
    """Build a Qdrant payload filter from retrieval constraints."""
    must: list[Condition] = []

    if type_equals is not None:
        must.append(
            FieldCondition(
                key=CHUNK_TYPE_KEY,
                match=MatchValue(value=type_equals),
            )
        )

    scoped_ids = effective_document_ids(document_ids, filters)
    if scoped_ids:
        must.append(
            FieldCondition(
                key="document_id",
                match=MatchAny(any=sorted(scoped_ids)),
            )
        )

    if filters:
        for key, value in filters.metadata.items():
            must.append(
                FieldCondition(
                    key=f"metadata.{key}",
                    match=MatchValue(value=value),
                )
            )

    must_not: list[Condition] = []
    if exclude_types:
        must_not.extend(
            FieldCondition(key=CHUNK_TYPE_KEY, match=MatchValue(value=chunk_type))
            for chunk_type in exclude_types
        )

    if not must and not must_not:
        return None
    return Filter(
        must=must if must else None,
        must_not=must_not if must_not else None,
    )


def chunk_matches_filter(chunk: Chunk, filters: RetrievalFilter | None) -> bool:
    """Return True when *chunk* satisfies document scope and metadata constraints."""
    if filters is None or not (filters.document_ids or filters.metadata):
        return True
    if filters.document_ids and chunk.document_id not in filters.document_ids:
        return False
    return all(chunk.metadata.get(key) == value for key, value in filters.metadata.items())


def apply_chunk_filters(
    results: list[SearchResult],
    filters: RetrievalFilter | None,
) -> list[SearchResult]:
    """Drop results whose chunks fall outside the document scope or metadata constraints."""
    if filters is None or not (filters.document_ids or filters.metadata):
        return results
    return [(chunk, score) for chunk, score in results if chunk_matches_filter(chunk, filters)]


def apply_min_score(
    results: list[SearchResult],
    filters: RetrievalFilter | None,
) -> list[SearchResult]:
    """Drop results below the configured cosine-similarity threshold."""
    if filters is None or filters.min_score is None:
        return results
    return [(chunk, score) for chunk, score in results if score >= filters.min_score]


def filters_from_request(
    *,
    document_ids: list[str] | None = None,
    metadata_filters: dict[str, str] | None = None,
    min_score: float | None = None,
) -> RetrievalFilter | None:
    """Build a RetrievalFilter from API request fields; None when no constraints."""
    doc_ids = document_ids or []
    metadata = metadata_filters or {}
    if not doc_ids and not metadata and min_score is None:
        return None
    return RetrievalFilter(
        document_ids=doc_ids,
        metadata=metadata,
        min_score=min_score,
    )
