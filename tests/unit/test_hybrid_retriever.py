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


# ── image leg (T-261) ─────────────────────────────────────────────────────────


class TestImageLeg:
    @pytest.mark.asyncio
    async def test_calls_image_retrieve(self):
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(_chunk(0), 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = []
        image_mock = MagicMock()
        image_mock.retrieve.return_value = [(_chunk(2), 0.8)]
        hr = HybridRetriever(dense=dense_mock, bm25=bm25_mock, image_retriever=image_mock)
        await hr.retrieve(_query(), top_k=3)
        image_mock.retrieve.assert_called_once()

    @pytest.mark.asyncio
    async def test_image_results_included_in_fusion(self):
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = []
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = []
        image_mock = MagicMock()
        figure = _chunk(9)
        image_mock.retrieve.return_value = [(figure, 0.8)]
        hr = HybridRetriever(dense=dense_mock, bm25=bm25_mock, image_retriever=image_mock)
        results = await hr.retrieve(_query(), top_k=3)
        assert [c.id for c, _ in results] == [figure.id]

    @pytest.mark.asyncio
    async def test_shared_chunk_across_dense_and_image_gets_boost(self):
        shared = _chunk(0)
        other = _chunk(1)
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(other, 0.99), (shared, 0.5)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = []
        image_mock = MagicMock()
        image_mock.retrieve.return_value = [(shared, 0.9)]
        hr = HybridRetriever(dense=dense_mock, bm25=bm25_mock, image_retriever=image_mock)
        results = await hr.retrieve(_query(), top_k=2)
        assert results[0][0].id == shared.id

    @pytest.mark.asyncio
    async def test_no_image_retriever_by_default(self):
        hr = _retriever()
        results = await hr.retrieve(_query(), top_k=3)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_image_disabled_leg_returns_empty_is_a_noop(self):
        """An ImageDenseRetriever wired in but disabled (non-multimodal provider)
        returns [] from retrieve() — the fused result must match not wiring it at all."""
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(_chunk(0), 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [(_chunk(1), 1.2)]
        image_mock = MagicMock()
        image_mock.retrieve.return_value = []

        without_image = HybridRetriever(dense=dense_mock, bm25=bm25_mock)
        with_disabled_image = HybridRetriever(
            dense=dense_mock, bm25=bm25_mock, image_retriever=image_mock
        )
        expected = await without_image.retrieve(_query(), top_k=3)
        actual = await with_disabled_image.retrieve(_query(), top_k=3)
        assert [c.id for c, _ in actual] == [c.id for c, _ in expected]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("image_enabled", [True, False])
    async def test_weighted_linear_guard_follows_image_enabled(self, image_enabled):
        """`_build_image_retriever` wires the image leg in unconditionally (no
        feature flag, T-260 convention), so the weighted_linear guard must key
        off `image.enabled`, not just presence — else a non-multimodal provider
        would silently lose weighted_linear fusion for everyone (regression)."""
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(_chunk(0), 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [(_chunk(1), 1.2)]
        image_mock = MagicMock()
        image_mock.enabled = image_enabled
        image_mock.retrieve.return_value = [(_chunk(2), 0.6)] if image_enabled else []
        hr = HybridRetriever(
            dense=dense_mock,
            bm25=bm25_mock,
            image_retriever=image_mock,
            fusion_mode="weighted_linear",
        )
        with (
            patch("src.rag.retrieval.hybrid_retriever.weighted_linear_fuse") as wl,
            patch("src.rag.retrieval.hybrid_retriever.rrf_fuse") as rrf,
        ):
            wl.return_value = [(_chunk(0), 0.5)]
            rrf.return_value = [(_chunk(0), 0.5)]
            await hr.retrieve(_query(), top_k=3)
            if image_enabled:
                rrf.assert_called_once()
                wl.assert_not_called()
            else:
                wl.assert_called_once()
                rrf.assert_not_called()


# ── per-leg RRF weights (T-263) ───────────────────────────────────────────────


class TestRrfLegWeights:
    @pytest.mark.asyncio
    async def test_default_rrf_weights_none_passed_to_rrf_fuse(self):
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(_chunk(0), 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [(_chunk(1), 1.2)]
        hr = HybridRetriever(dense=dense_mock, bm25=bm25_mock)
        with patch("src.rag.retrieval.hybrid_retriever.rrf_fuse") as rrf:
            rrf.return_value = [(_chunk(0), 0.5)]
            await hr.retrieve(_query(), top_k=3)
            _, kwargs = rrf.call_args
            assert kwargs["weights"] is None

    @pytest.mark.asyncio
    async def test_configured_weights_forwarded_in_leg_order(self):
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(_chunk(0), 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [(_chunk(1), 1.2)]
        image_mock = MagicMock()
        image_mock.enabled = True
        image_mock.retrieve.return_value = [(_chunk(2), 0.6)]
        hr = HybridRetriever(
            dense=dense_mock,
            bm25=bm25_mock,
            image_retriever=image_mock,
            rrf_weights={"dense": 2.0, "image": 0.5},
        )
        with patch("src.rag.retrieval.hybrid_retriever.rrf_fuse") as rrf:
            rrf.return_value = [(_chunk(0), 0.5)]
            await hr.retrieve(_query(), top_k=3)
            _, kwargs = rrf.call_args
            # order: dense, bm25, graph, hype, hyde, hierarchical, image
            assert kwargs["weights"] == [2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5]

    @pytest.mark.asyncio
    async def test_configured_weights_change_fusion_outcome(self):
        c0 = _chunk(0)  # ranked #1 in dense
        c1 = _chunk(1)  # sole hit in bm25, would otherwise lose to c0
        dense_mock = MagicMock()
        dense_mock.retrieve.return_value = [(c0, 0.9)]
        bm25_mock = MagicMock()
        bm25_mock.search.return_value = [(c1, 0.8)]

        default_hr = HybridRetriever(dense=dense_mock, bm25=bm25_mock)
        default_results = await default_hr.retrieve(_query(), top_k=2)
        assert default_results[0][0].id == c0.id

        boosted_hr = HybridRetriever(
            dense=dense_mock, bm25=bm25_mock, rrf_weights={"bm25": 5.0}
        )
        boosted_results = await boosted_hr.retrieve(_query(), top_k=2)
        assert boosted_results[0][0].id == c1.id

    def test_rrf_weights_stored(self):
        hr = HybridRetriever(
            dense=MagicMock(), bm25=MagicMock(), rrf_weights={"dense": 1.5}
        )
        assert hr.rrf_weights == {"dense": 1.5}
