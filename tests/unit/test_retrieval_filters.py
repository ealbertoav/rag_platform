"""T-134 — Multi-faceted Qdrant filtering tests."""

from __future__ import annotations

from typing import cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from src.core.constants import CHUNK_TYPE_HYPE, CHUNK_TYPE_KEY
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.domain.entities.retrieval_filter import RetrievalFilter
from src.rag.retrieval.dense_retriever import DenseRetriever
from src.rag.retrieval.filters import (
    apply_chunk_filters,
    apply_min_score,
    build_qdrant_filter,
    chunk_matches_filter,
    effective_document_ids,
    filters_from_request,
)
from src.rag.retrieval.hybrid_retriever import HybridRetriever

# ── helpers ────────────────────────────────────────────────────────────────────

_VEC = [0.1, 0.2, 0.3, 0.4]
_CHUNK_A = Chunk(document_id="doc-a", text="chunk a", metadata={"source": "a.pdf"})
_CHUNK_B = Chunk(document_id="doc-b", text="chunk b", metadata={"source": "b.pdf"})


def _must_field(qdrant_filter: Filter, index: int = 0) -> FieldCondition:
    must = qdrant_filter.must
    assert isinstance(must, list)
    return cast(FieldCondition, must[index])


def _must_not_field(qdrant_filter: Filter, index: int = 0) -> FieldCondition:
    must_not = qdrant_filter.must_not
    assert isinstance(must_not, list)
    return cast(FieldCondition, must_not[index])


def _match_any(condition: FieldCondition) -> MatchAny:
    match = condition.match
    assert isinstance(match, MatchAny)
    return match


def _match_value(condition: FieldCondition) -> MatchValue:
    match = condition.match
    assert isinstance(match, MatchValue)
    return match


_MIN_SCORE_FILTER = RetrievalFilter(min_score=0.5)


def _hybrid_retriever(
    *,
    dense_results: list | None = None,
    bm25_results: list | None = None,
    graph: MagicMock | None = None,
) -> tuple[HybridRetriever, MagicMock, MagicMock, MagicMock | None]:
    dense = MagicMock()
    dense.retrieve.return_value = dense_results if dense_results is not None else []
    bm25 = MagicMock()
    bm25.search.return_value = bm25_results if bm25_results is not None else []
    bm25.get_by_id.return_value = None
    kwargs: dict[str, object] = {"dense": dense, "bm25": bm25}
    if graph is not None:
        kwargs["graph_retriever"] = graph
    hybrid = HybridRetriever(**kwargs)  # type: ignore[arg-type]
    return hybrid, dense, bm25, graph


async def _retrieve_hybrid(
    hybrid: HybridRetriever,
    *,
    filters: RetrievalFilter | None = None,
    top_k: int = 5,
    text: str = "q",
):
    return await hybrid.retrieve(Query(text=text, filters=filters), top_k=top_k)


# ── RetrievalFilter ────────────────────────────────────────────────────────────


class TestRetrievalFilter:
    def test_is_active_when_document_ids_set(self):
        assert RetrievalFilter(document_ids=["doc-1"]).is_active()

    def test_is_active_when_metadata_set(self):
        assert RetrievalFilter(metadata={"section": "intro"}).is_active()

    def test_is_active_when_min_score_set(self):
        assert RetrievalFilter(min_score=0.5).is_active()

    def test_inactive_when_empty(self):
        assert not RetrievalFilter().is_active()


class TestFiltersFromRequest:
    def test_returns_none_when_no_constraints(self):
        assert filters_from_request() is None

    def test_builds_from_document_ids(self):
        filt = filters_from_request(document_ids=["doc-1", "doc-2"])
        assert filt is not None
        assert filt.document_ids == ["doc-1", "doc-2"]

    def test_builds_from_metadata_filters(self):
        filt = filters_from_request(metadata_filters={"section": "revenue"})
        assert filt is not None
        assert filt.metadata == {"section": "revenue"}

    def test_builds_from_min_score(self):
        filt = filters_from_request(min_score=0.75)
        assert filt is not None
        assert filt.min_score == 0.75


# ── build_qdrant_filter ────────────────────────────────────────────────────────


class TestBuildQdrantFilter:
    def test_returns_none_without_constraints(self):
        assert build_qdrant_filter() is None

    def test_document_ids_from_filter(self):
        filt = build_qdrant_filter(filters=RetrievalFilter(document_ids=["doc-a", "doc-b"]))
        assert filt is not None
        condition = _must_field(filt)
        assert condition.key == "document_id"
        assert set(cast(list[str], _match_any(condition).any)) == {"doc-a", "doc-b"}

    def test_metadata_exact_match_uses_nested_key(self):
        filt = build_qdrant_filter(filters=RetrievalFilter(metadata={"source": "report.pdf"}))
        assert filt is not None
        condition = _must_field(filt)
        assert condition.key == "metadata.source"
        assert _match_value(condition).value == "report.pdf"

    def test_type_equals_and_exclude_types(self):
        filt = build_qdrant_filter(
            type_equals=CHUNK_TYPE_HYPE,
            exclude_types=frozenset({"summary"}),
        )
        assert filt is not None
        condition = _must_field(filt)
        assert condition.key == CHUNK_TYPE_KEY
        assert isinstance(condition.match, MatchValue)
        assert _match_value(_must_not_field(filt)).value == "summary"

    def test_intersects_explicit_and_filter_document_ids(self):
        filt = build_qdrant_filter(
            document_ids=frozenset({"doc-a", "doc-b"}),
            filters=RetrievalFilter(document_ids=["doc-b", "doc-c"]),
        )
        assert filt is not None
        assert set(cast(list[str], _match_any(_must_field(filt)).any)) == {"doc-b"}


class TestEffectiveDocumentIds:
    def test_returns_explicit_when_no_filter(self):
        assert effective_document_ids(frozenset({"a"}), None) == frozenset({"a"})

    def test_returns_filter_ids_when_no_explicit(self):
        filt = RetrievalFilter(document_ids=["a", "b"])
        assert effective_document_ids(None, filt) == frozenset({"a", "b"})

    def test_intersection_when_both_set(self):
        filt = RetrievalFilter(document_ids=["b", "c"])
        assert effective_document_ids(frozenset({"a", "b"}), filt) == frozenset({"b"})

    def test_empty_explicit_does_not_fallback_to_filter(self):
        filt = RetrievalFilter(document_ids=["a", "b"])
        assert effective_document_ids(frozenset(), filt) == frozenset()

    def test_empty_explicit_without_filter(self):
        assert effective_document_ids(frozenset(), None) == frozenset()


# ── chunk_matches_filter / apply_chunk_filters ────────────────────────────────


class TestChunkMatchesFilter:
    def test_no_filter_matches_all(self):
        assert chunk_matches_filter(_CHUNK_A, None)
        assert chunk_matches_filter(_CHUNK_A, RetrievalFilter())

    def test_document_id_scope(self):
        filt = RetrievalFilter(document_ids=["doc-a"])
        assert chunk_matches_filter(_CHUNK_A, filt)
        assert not chunk_matches_filter(_CHUNK_B, filt)

    def test_metadata_exact_match(self):
        filt = RetrievalFilter(metadata={"source": "a.pdf"})
        assert chunk_matches_filter(_CHUNK_A, filt)
        assert not chunk_matches_filter(_CHUNK_B, filt)


class TestApplyChunkFilters:
    def test_noop_without_scope(self):
        results = [(_CHUNK_A, 0.9), (_CHUNK_B, 0.4)]
        assert apply_chunk_filters(results, None) == results

    def test_drops_out_of_scope(self):
        results = [(_CHUNK_A, 0.9), (_CHUNK_B, 0.4)]
        filt = RetrievalFilter(document_ids=["doc-a"])
        assert apply_chunk_filters(results, filt) == [(_CHUNK_A, 0.9)]


# ── apply_min_score ────────────────────────────────────────────────────────────


class TestApplyMinScore:
    def test_noop_without_filter(self):
        results = [(_CHUNK_A, 0.9), (_CHUNK_B, 0.4)]
        assert apply_min_score(results, None) == results

    def test_noop_without_min_score(self):
        results = [(_CHUNK_A, 0.9), (_CHUNK_B, 0.4)]
        assert apply_min_score(results, RetrievalFilter()) == results

    def test_drops_below_threshold(self):
        results = [(_CHUNK_A, 0.9), (_CHUNK_B, 0.4)]
        filt = RetrievalFilter(min_score=0.5)
        filtered = apply_min_score(results, filt)
        assert filtered == [(_CHUNK_A, 0.9)]

    def test_keeps_equal_threshold(self):
        results = [(_CHUNK_A, 0.5)]
        filt = RetrievalFilter(min_score=0.5)
        assert apply_min_score(results, filt) == results


# ── DenseRetriever integration ─────────────────────────────────────────────────


class TestDenseRetrieverFilters:
    def test_passes_query_filters_to_vector_store(self):
        embedder = MagicMock()
        embedder.embed_query.return_value = [_VEC]
        vector_store = MagicMock()
        vector_store.search_dense.return_value = []
        filt = RetrievalFilter(document_ids=["doc-1"])
        query = Query(text="What is IAM?", filters=filt)

        DenseRetriever(embedder=embedder, vector_store=vector_store).retrieve(query, top_k=5)

        _, kwargs = vector_store.search_dense.call_args
        assert kwargs["filters"] == filt


# ── HybridRetriever min_score ──────────────────────────────────────────────────


class TestHybridRetrieverMinScore:
    @pytest.mark.asyncio
    async def test_min_score_applied_before_fusion(self):
        hybrid, _, _, _ = _hybrid_retriever(
            dense_results=[(_CHUNK_A, 0.9), (_CHUNK_B, 0.3)],
        )
        results = await _retrieve_hybrid(hybrid, filters=_MIN_SCORE_FILTER)

        assert len(results) == 1
        assert results[0][0].id == _CHUNK_A.id

    @pytest.mark.asyncio
    async def test_min_score_not_applied_to_bm25_leg(self):
        """BM25 scores are not cosine similarity; min_score must not drop them."""
        hybrid, _, _, _ = _hybrid_retriever(bm25_results=[(_CHUNK_B, 0.2)])
        results = await _retrieve_hybrid(hybrid, filters=_MIN_SCORE_FILTER)

        assert len(results) == 1
        assert results[0][0].id == _CHUNK_B.id

    @pytest.mark.asyncio
    async def test_passes_filters_to_bm25_and_graph(self):
        graph = MagicMock()
        graph.search = AsyncMock(return_value=[])
        hybrid, _, bm25, _ = _hybrid_retriever(graph=graph)
        filt = RetrievalFilter(document_ids=["doc-a"])

        await _retrieve_hybrid(hybrid, filters=filt)

        bm25.search.assert_called_once_with("q", 15, filters=filt)
        graph.search.assert_awaited_once_with("q", 15, filters=filt)

    @pytest.mark.asyncio
    async def test_document_scope_excludes_out_of_scope_bm25_hits(self):
        hybrid, _, _, _ = _hybrid_retriever(bm25_results=[(_CHUNK_A, 1.0)])
        results = await _retrieve_hybrid(hybrid, filters=RetrievalFilter(document_ids=["doc-b"]))

        assert results == []


# ── Query entity ───────────────────────────────────────────────────────────────


class TestQueryFiltersField:
    def test_filters_default_none(self):
        assert Query(text="q").filters is None

    def test_round_trip_with_filters(self):
        filt = RetrievalFilter(document_ids=["doc-1"], min_score=0.6)
        q = Query(text="q", filters=filt)
        restored = Query.model_validate(q.model_dump())
        assert restored.filters == filt
