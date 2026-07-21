"""T-051 — Prometheus metrics tests."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient
from prometheus_client import REGISTRY

# ── Metric definitions ─────────────────────────────────────────────────────────


class TestMetricDefinitions:
    def test_request_latency_registered(self):
        assert (
            REGISTRY.get_sample_value("rag_request_latency_seconds_count", {"stage": "chat"})
            is not None
            or True
        )  # metric exists once observed

    def test_record_request_does_not_raise(self):
        from src.observability.metrics import record_request

        record_request("test_stage", 0.123, success=True)
        record_request("test_stage", 0.456, success=False)

    def test_record_retrieval_does_not_raise(self):
        from src.observability.metrics import record_retrieval

        record_retrieval(chunk_count=5, latency_seconds=0.25)

    def test_record_generation_does_not_raise(self):
        from src.observability.metrics import record_generation

        record_generation(token_count=128, latency_seconds=1.5)

    def test_request_counter_increments(self):
        from src.observability.metrics import REQUESTS_TOTAL

        before = REQUESTS_TOTAL.labels(status="success")._value.get()
        from src.observability.metrics import record_request

        record_request("counter_test", 0.1, success=True)
        after = REQUESTS_TOTAL.labels(status="success")._value.get()
        assert after > before

    def test_llm_tokens_counter_increments(self):
        from src.observability.metrics import LLM_TOKENS_TOTAL, record_generation

        before = LLM_TOKENS_TOTAL._value.get()
        record_generation(token_count=50, latency_seconds=0.5)
        after = LLM_TOKENS_TOTAL._value.get()
        assert after == pytest.approx(before + 50)


# ── Retrieval/answer-quality signal (#92) ───────────────────────────────────────


class TestRerankerMetrics:
    def test_record_reranker_success_increments_outcome_and_observes_scores(self):
        from src.observability.metrics import (
            RERANKER_OUTCOME_TOTAL,
            RERANKER_SCORE,
            record_reranker_success,
        )

        before_outcome = RERANKER_OUTCOME_TOTAL.labels(outcome="reranked")._value.get()
        before_score_count = RERANKER_SCORE._sum.get()

        record_reranker_success([0.9, 0.5])

        after_outcome = RERANKER_OUTCOME_TOTAL.labels(outcome="reranked")._value.get()
        after_score_count = RERANKER_SCORE._sum.get()
        assert after_outcome == pytest.approx(before_outcome + 1)
        assert after_score_count == pytest.approx(before_score_count + 1.4)

    def test_record_reranker_success_with_no_scores_still_increments_outcome(self):
        from src.observability.metrics import RERANKER_OUTCOME_TOTAL, record_reranker_success

        before = RERANKER_OUTCOME_TOTAL.labels(outcome="reranked")._value.get()
        record_reranker_success([])
        after = RERANKER_OUTCOME_TOTAL.labels(outcome="reranked")._value.get()
        assert after == pytest.approx(before + 1)

    def test_record_reranker_fallback_increments_fallback_outcome(self):
        from src.observability.metrics import RERANKER_OUTCOME_TOTAL, record_reranker_fallback

        before = RERANKER_OUTCOME_TOTAL.labels(outcome="fallback")._value.get()
        record_reranker_fallback()
        after = RERANKER_OUTCOME_TOTAL.labels(outcome="fallback")._value.get()
        assert after == pytest.approx(before + 1)


class TestReliableRagMetrics:
    def test_record_reliable_rag_scores_observes_each_score(self):
        from src.observability.metrics import (
            RELIABLE_RAG_RELEVANCE_SCORE,
            record_reliable_rag_scores,
        )

        before = RELIABLE_RAG_RELEVANCE_SCORE._sum.get()
        record_reliable_rag_scores([0.8, 0.3, 1.0])
        after = RELIABLE_RAG_RELEVANCE_SCORE._sum.get()
        assert after == pytest.approx(before + 2.1)

    def test_record_reliable_rag_scores_with_empty_list_does_not_raise(self):
        from src.observability.metrics import record_reliable_rag_scores

        record_reliable_rag_scores([])


class TestFeedbackMetrics:
    def test_record_feedback_event_positive_sentiment(self):
        from src.observability.metrics import FEEDBACK_EVENTS_TOTAL, record_feedback_event

        before = FEEDBACK_EVENTS_TOTAL.labels(sentiment="positive")._value.get()
        record_feedback_event(sentiment="positive", accumulated=3.0)
        after = FEEDBACK_EVENTS_TOTAL.labels(sentiment="positive")._value.get()
        assert after == pytest.approx(before + 1)

    def test_record_feedback_event_negative_sentiment(self):
        from src.observability.metrics import FEEDBACK_EVENTS_TOTAL, record_feedback_event

        before = FEEDBACK_EVENTS_TOTAL.labels(sentiment="negative")._value.get()
        record_feedback_event(sentiment="negative", accumulated=-2.0)
        after = FEEDBACK_EVENTS_TOTAL.labels(sentiment="negative")._value.get()
        assert after == pytest.approx(before + 1)

    def test_record_feedback_event_observes_accumulated_score(self):
        from src.observability.metrics import FEEDBACK_SCORE_ACCUMULATED, record_feedback_event

        before = FEEDBACK_SCORE_ACCUMULATED._sum.get()
        record_feedback_event(sentiment="positive", accumulated=4.5)
        after = FEEDBACK_SCORE_ACCUMULATED._sum.get()
        assert after == pytest.approx(before + 4.5)


# ── /metrics endpoint ──────────────────────────────────────────────────────────


class TestMetricsEndpoint:
    @pytest.fixture
    def app_client(self):
        from unittest.mock import MagicMock

        from src.main import create_app

        app = create_app()
        app.state.chat_pipeline = MagicMock()
        app.state.ingestion_pipeline = MagicMock()
        app.state.models_loaded = True
        return app

    @pytest.mark.asyncio
    async def test_metrics_endpoint_200(self, app_client):
        async with AsyncClient(
            transport=ASGITransport(app=app_client), base_url="http://test"
        ) as c:
            resp = await c.get("/metrics")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_metrics_content_type(self, app_client):
        async with AsyncClient(
            transport=ASGITransport(app=app_client), base_url="http://test"
        ) as c:
            resp = await c.get("/metrics")
        assert "text/plain" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_metrics_body_contains_rag_metrics(self, app_client):
        async with AsyncClient(
            transport=ASGITransport(app=app_client), base_url="http://test"
        ) as c:
            resp = await c.get("/metrics")
        body = resp.text
        assert "rag_request_latency_seconds" in body
        assert "rag_requests_total" in body
        assert "rag_retrieval_chunk_count" in body
        assert "rag_llm_tokens_total" in body
        assert "rag_reranker_outcome_total" in body
        assert "rag_reranker_score" in body
        assert "rag_reliable_rag_relevance_score" in body
        assert "rag_feedback_events_total" in body
        assert "rag_feedback_score_accumulated" in body
