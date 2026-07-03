"""T-145 — Retrieval feedback loop tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.constants import FEEDBACK_SCORE_KEY
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.main import create_app
from src.rag.quality.feedback_loop import (
    apply_feedback_boost,
    feedback_score_from_metadata,
    merge_chunk_views,
    record_feedback,
    resolve_feedback_score,
    score_from_relevant,
)


def _chunk(chunk_id: str, *, feedback_score: float | None = None) -> Chunk:
    metadata: dict[str, object] = {}
    if feedback_score is not None:
        metadata[FEEDBACK_SCORE_KEY] = feedback_score
    return Chunk(id=chunk_id, document_id="doc-1", text="sample", metadata=metadata)


def _result(
    chunk_id: str, score: float, *, feedback_score: float | None = None
) -> tuple[Chunk, float]:
    return _chunk(chunk_id, feedback_score=feedback_score), score


class TestScoreFromRelevant:
    def test_positive_vote(self):
        assert score_from_relevant(True) == 1.0

    def test_negative_vote(self):
        assert score_from_relevant(False) == -1.0


class TestFeedbackScoreFromMetadata:
    def test_bool_metadata_returns_zero(self):
        assert feedback_score_from_metadata({FEEDBACK_SCORE_KEY: True}) == 0.0


class TestMergeChunkViews:
    def test_ignores_stale_feedback_score_from_right(self):
        left = _chunk("c0", feedback_score=1.0)
        right = _chunk("c0", feedback_score=5.0)
        merged = merge_chunk_views(left, right)
        assert merged.metadata[FEEDBACK_SCORE_KEY] == 1.0

    def test_does_not_copy_feedback_score_from_right_only(self):
        left = _chunk("c0")
        right = _chunk("c0", feedback_score=-1.0)
        merged = merge_chunk_views(left, right)
        assert FEEDBACK_SCORE_KEY not in merged.metadata

    def test_copies_missing_metadata_keys_from_right(self):
        left = Chunk(id="c0", document_id="doc-1", text="sample", metadata={"source": "left"})
        right = Chunk(id="c0", document_id="doc-1", text="sample", metadata={"section": "intro"})
        merged = merge_chunk_views(left, right)
        assert merged.metadata["source"] == "left"
        assert merged.metadata["section"] == "intro"


class TestResolveFeedbackScore:
    def test_prefers_vector_store_over_stale_metadata(self):
        chunk = _chunk("c0", feedback_score=5.0)
        assert resolve_feedback_score(chunk, vector_scores={"c0": -1.0}) == -1.0

    def test_falls_back_to_chunk_metadata_without_vector_store(self):
        assert resolve_feedback_score(_chunk("c0", feedback_score=2.0)) == 2.0

    def test_ignores_stale_bm25_when_vector_store_supplied(self):
        from src.infrastructure.vectordb.bm25 import BM25Index

        index = BM25Index()
        index.index([_chunk("c0", feedback_score=5.0)])
        chunk = _chunk("c0", feedback_score=0.0)
        assert resolve_feedback_score(chunk, vector_scores={"c0": -1.0}) == -1.0


class TestRecordFeedback:
    def test_accumulates_and_persists_to_vector_store(self):
        store = MagicMock()
        store.accumulate_feedback_score.return_value = 2.0
        record_feedback(store, "query-1", "chunk-a", 1.0)
        store.accumulate_feedback_score.assert_called_once_with("chunk-a", 1.0)


class TestApplyFeedbackBoost:
    def test_no_multiplier_returns_unchanged(self):
        results = [_result("c0", 0.5), _result("c1", 0.4, feedback_score=2.0)]
        assert apply_feedback_boost(results, boost_multiplier=0.0) == results

    def test_positive_feedback_reorders_results(self):
        results = [
            _result("low-feedback", 0.9),
            _result("high-feedback", 0.8, feedback_score=3.0),
        ]
        boosted = apply_feedback_boost(results, boost_multiplier=0.1)
        assert boosted[0][0].id == "high-feedback"
        assert boosted[0][1] > boosted[1][1]

    def test_non_positive_feedback_unchanged(self):
        results = [_result("c0", 0.5, feedback_score=-1.0)]
        boosted = apply_feedback_boost(results, boost_multiplier=0.1)
        assert boosted[0][1] == 0.5

    def test_vector_store_lookup_applies_when_metadata_stale(self):
        store = MagicMock()
        store.get_feedback_scores.return_value = {"c0": 4.0}
        results = [_result("c0", 0.5)]
        boosted = apply_feedback_boost(
            results,
            boost_multiplier=0.1,
            vector_store=store,
        )
        assert boosted[0][1] == pytest.approx(0.9)
        store.get_feedback_scores.assert_called_once_with(["c0"])

    def test_vector_store_lookup_failure_skips_boost(self):
        store = MagicMock()
        store.get_feedback_scores.side_effect = VectorStoreError("Qdrant retrieve failed")
        results = [_result("c0", 0.5, feedback_score=3.0)]
        boosted = apply_feedback_boost(
            results,
            boost_multiplier=0.1,
            vector_store=store,
        )
        assert boosted[0][1] == 0.5


class TestFeedbackApi:
    @pytest.fixture
    def app_client(self):
        app = create_app()
        app.state.models_loaded = True
        return app

    @staticmethod
    def _client(app):
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_submit_feedback_returns_204(self, app_client):
        store = MagicMock()
        store.accumulate_feedback_score.return_value = 1.0
        app_client.state.vector_store = store
        async with self._client(app_client) as client:
            resp = await client.post(
                "/feedback",
                json={"query_id": "q-1", "chunk_id": "chunk-a", "relevant": True},
            )
        assert resp.status_code == 204
        store.accumulate_feedback_score.assert_called_once_with("chunk-a", 1.0)

    @pytest.mark.asyncio
    async def test_missing_chunk_returns_404(self, app_client):
        store = MagicMock()
        store.accumulate_feedback_score.side_effect = VectorStoreError("Chunk 'missing' not found")
        app_client.state.vector_store = store
        async with self._client(app_client) as client:
            resp = await client.post(
                "/feedback",
                json={"query_id": "q-1", "chunk_id": "missing", "relevant": False},
            )
        assert resp.status_code == 404
