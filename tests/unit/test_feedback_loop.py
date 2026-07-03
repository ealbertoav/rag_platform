"""T-145 — Retrieval feedback loop tests."""

from __future__ import annotations

import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.core.constants import FEEDBACK_SCORE_KEY
from src.core.exceptions import VectorStoreError
from src.domain.entities.chunk import Chunk
from src.infrastructure.vectordb.bm25 import BM25Index
from src.main import create_app
from src.rag.quality.feedback_loop import (
    apply_feedback_boost,
    record_feedback,
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


class TestRecordFeedback:
    def test_accumulates_and_persists_to_vector_store(self):
        store = MagicMock()
        store.get_feedback_score.return_value = 1.0
        record_feedback(store, "query-1", "chunk-a", 1.0)
        store.set_feedback_score.assert_called_once_with("chunk-a", 2.0)

    def test_updates_bm25_metadata_when_index_provided(self, tmp_path: Path):
        store = MagicMock()
        store.get_feedback_score.return_value = 0.0
        index = BM25Index(index_path=tmp_path / "bm25.json")
        index.index([_chunk("chunk-a")])
        record_feedback(store, "query-1", "chunk-a", 1.0, bm25_index=index)
        updated = index.get_by_id("chunk-a")
        assert updated is not None
        assert updated.metadata[FEEDBACK_SCORE_KEY] == 1.0
        reloaded = BM25Index(index_path=tmp_path / "bm25.json")
        reloaded.load()
        persisted = reloaded.get_by_id("chunk-a")
        assert persisted is not None
        assert persisted.metadata[FEEDBACK_SCORE_KEY] == 1.0

    def test_missing_bm25_chunk_logs_warning(self, caplog):
        store = MagicMock()
        store.get_feedback_score.return_value = 0.0
        index = BM25Index()
        with caplog.at_level(logging.WARNING):
            record_feedback(store, "query-1", "missing", 1.0, bm25_index=index)
        assert "not found in BM25 index" in caplog.text


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


class TestFeedbackApi:
    @pytest.fixture
    def app_client(self):
        app = create_app()
        app.state.models_loaded = True
        app.state.bm25_index = BM25Index()
        return app

    @staticmethod
    def _client(app):
        return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    @pytest.mark.asyncio
    async def test_submit_feedback_returns_204(self, app_client):
        with patch("src.api.routers.feedback.QdrantVectorStore.from_settings") as factory:
            store = MagicMock()
            store.get_feedback_score.return_value = 0.0
            factory.return_value = store
            async with self._client(app_client) as client:
                resp = await client.post(
                    "/feedback",
                    json={"query_id": "q-1", "chunk_id": "chunk-a", "relevant": True},
                )
        assert resp.status_code == 204
        store.set_feedback_score.assert_called_once_with("chunk-a", 1.0)

    @pytest.mark.asyncio
    async def test_missing_chunk_returns_404(self, app_client):
        with patch("src.api.routers.feedback.QdrantVectorStore.from_settings") as factory:
            store = MagicMock()
            store.get_feedback_score.return_value = 0.0
            store.set_feedback_score.side_effect = VectorStoreError("Chunk 'missing' not found")
            factory.return_value = store
            async with self._client(app_client) as client:
                resp = await client.post(
                    "/feedback",
                    json={"query_id": "q-1", "chunk_id": "missing", "relevant": False},
                )
        assert resp.status_code == 404
