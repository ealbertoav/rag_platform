"""T-022 — HybridRetriever tests."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.core.constants import CHUNK_TYPE_KEY, CHUNK_TYPE_SYNTHETIC, SOURCE_CHUNK_ID_KEY
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.domain.entities.query import Query
from src.rag.retrieval.hybrid_retriever import HybridRetriever
from tests.unit.hybrid_retriever_helpers import feedback_boost_retriever

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int) -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=f"chunk {i}")


def _query(text: str = "What is EKS?") -> Query:
    return Query(text=text)


def _retriever(
    dense_results: list[tuple[Chunk, float]] | None = None,
    bm25_results: list[tuple[Chunk, float]] | None = None,
    alpha: float = 0.7,
) -> HybridRetriever:
    dense_mock = MagicMock()
    dense_mock.retrieve.return_value = (
        dense_results if dense_results is not None else [(_chunk(0), 0.9), (_chunk(1), 0.7)]
    )
    bm25_mock = MagicMock()
    bm25_mock.search.return_value = (
        bm25_results if bm25_results is not None else [(_chunk(1), 1.2), (_chunk(2), 0.8)]
    )
    return HybridRetriever(dense=dense_mock, bm25=bm25_mock, alpha=alpha)


# ── async retrieve ────────────────────────────────────────────────────────────


class TestHybridRetrieverAsync:
    @pytest.mark.asyncio
    async def test_returns_list(self):
        result = await _retriever().retrieve(_query(), top_k=3)
        assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_chunk_and_score_types(self):
        results = await _retriever().retrieve(_query(), top_k=3)
        for chunk, score in results:
            assert isinstance(chunk, Chunk)
            assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_top_k_respected(self):
        result = await _retriever().retrieve(_query(), top_k=2)
        assert len(result) <= 2

    @pytest.mark.asyncio
    async def test_calls_dense_retrieve(self):
        hr = _retriever()
        await hr.retrieve(_query(), top_k=3)
        hr._dense.retrieve.assert_called_once()  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_calls_bm25_search(self):
        hr = _retriever()
        await hr.retrieve(_query("my query"), top_k=3)
        hr._bm25.search.assert_called_once_with("my query", 9, filters=None)  # type: ignore[attr-defined]

    @pytest.mark.asyncio
    async def test_deduplicates_shared_chunk(self):
        shared = _chunk(0)
        hr = _retriever(
            dense_results=[(shared, 0.9)],
            bm25_results=[(shared, 1.2)],
        )
        results = await hr.retrieve(_query(), top_k=5)
        assert sum(1 for c, _ in results if c.id == shared.id) == 1

    @pytest.mark.asyncio
    async def test_shared_chunk_gets_boost(self):
        c0 = _chunk(0)  # in both lists
        c1 = _chunk(1)  # dense only, higher raw score
        hr = _retriever(
            dense_results=[(c1, 0.99), (c0, 0.5)],
            bm25_results=[(c0, 1.5)],
        )
        results = await hr.retrieve(_query(), top_k=2)
        assert results[0][0].id == c0.id

    @pytest.mark.asyncio
    async def test_empty_results_returns_empty(self):
        hr = _retriever(dense_results=[], bm25_results=[])
        assert await hr.retrieve(_query(), top_k=5) == []

    @pytest.mark.asyncio
    async def test_feedback_lookup_failure_still_returns_fused_results(self):
        hr, _, _ = feedback_boost_retriever(
            feedback_scores_side_effect=VectorStoreError("Qdrant retrieve failed"),
        )
        results = await hr.retrieve(_query(), top_k=2)
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_resolves_synthetic_question_to_source(self):
        source = _chunk(0)
        question = Chunk(
            id="q0",
            document_id="doc",
            text="What is chunk 0?",
            metadata={
                CHUNK_TYPE_KEY: CHUNK_TYPE_SYNTHETIC,
                SOURCE_CHUNK_ID_KEY: source.id,
            },
        )
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = []
        bm25_mock.get_by_id.return_value = source
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(question, 0.95)]
        hr = HybridRetriever(dense=dense_mock, bm25=bm25_mock, alpha=0.7)
        results = await hr.retrieve(_query(), top_k=3)
        assert len(results) == 1
        assert results[0][0].id == source.id


# ── retrieve_sync ─────────────────────────────────────────────────────────────


class TestRetrieveSync:
    def test_returns_same_as_async(self):
        hr = _retriever()
        results = hr.retrieve_sync(_query(), top_k=3)
        assert isinstance(results, list)
        assert all(isinstance(c, Chunk) for c, _ in results)


# ── alpha stored ──────────────────────────────────────────────────────────────


class TestAlpha:
    def test_alpha_stored(self):
        hr = _retriever(alpha=0.3)
        assert hr.alpha == pytest.approx(0.3)


# ── fusion mode ───────────────────────────────────────────────────────────────


class TestFusionMode:
    @pytest.mark.asyncio
    async def test_weighted_linear_when_hyde_present_but_disabled(self):
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(_chunk(0), 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [(_chunk(1), 1.2)]
        hyde_mock = MagicMock()
        hr = HybridRetriever(
            dense=dense_mock,
            bm25=bm25_mock,
            hyde_retriever=hyde_mock,
            fusion_mode="weighted_linear",
        )
        with (
            patch("src.rag.retrieval.hybrid_retriever.weighted_linear_fuse") as wl,
            patch("src.rag.retrieval.hybrid_retriever.rrf_fuse") as rrf,
        ):
            wl.return_value = [(_chunk(0), 0.5)]
            await hr.retrieve(_query(), top_k=3, use_hyde=False)
            wl.assert_called_once()
            rrf.assert_not_called()
        hyde_mock.retrieve.assert_not_called()

    @pytest.mark.asyncio
    async def test_rrf_when_hyde_active(self):
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(_chunk(0), 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [(_chunk(1), 1.2)]
        hyde_mock = MagicMock()
        hyde_mock.retrieve.return_value = [(_chunk(2), 0.6)]
        hr = HybridRetriever(
            dense=dense_mock,
            bm25=bm25_mock,
            hyde_retriever=hyde_mock,
            fusion_mode="weighted_linear",
        )
        with (
            patch("src.rag.retrieval.hybrid_retriever.weighted_linear_fuse") as wl,
            patch("src.rag.retrieval.hybrid_retriever.rrf_fuse") as rrf,
        ):
            rrf.return_value = [(_chunk(0), 0.5)]
            await hr.retrieve(_query(), top_k=3, use_hyde=True)
            rrf.assert_called_once()
            wl.assert_not_called()
        hyde_mock.retrieve.assert_called_once()
