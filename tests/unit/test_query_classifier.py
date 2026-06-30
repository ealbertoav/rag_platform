"""T-131 — Adaptive query classification tests."""

from __future__ import annotations

import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.domain.entities.query import Query
from src.rag.retrieval.adaptive.query_classifier import (
    ClassificationOutput,
    QueryCategory,
    QueryClassifier,
    parse_classification,
)


class TestParseClassification:
    def test_parses_clean_json(self):
        output = parse_classification(
            '{"category": "analytical", "reasoning": "Compares two metrics."}',
        )
        assert output == ClassificationOutput(
            category=QueryCategory.ANALYTICAL,
            reasoning="Compares two metrics.",
        )

    def test_extracts_json_from_prose(self):
        output = parse_classification(
            'Here is the result:\n{"category": "opinion", "reasoning": "Asks for advice."}',
        )
        assert output.category is QueryCategory.OPINION

    def test_rejects_invalid_category(self):
        with pytest.raises(ValueError, match="Could not parse classification"):
            parse_classification('{"category": "unknown", "reasoning": "n/a"}')


class TestQueryClassifier:
    def test_classifies_factual_query(self):
        llm = MagicMock()
        llm.generate.return_value = (
            '{"category": "factual", "reasoning": "Asks for a specific date."}'
        )
        classifier = QueryClassifier(llm=llm, enabled=True)

        result = classifier.classify(Query(text="When was the company founded?"))

        assert result.metadata["category"] == "factual"
        llm.generate.assert_called_once()
        prompt = llm.generate.call_args.kwargs["prompt"]
        assert "When was the company founded?" in prompt

    @pytest.mark.parametrize(
        ("category", "query_text"),
        [
            (QueryCategory.ANALYTICAL, "Why did revenue decline compared to last year?"),
            (QueryCategory.OPINION, "Which cloud provider would you recommend?"),
            (QueryCategory.CONTEXTUAL, "What about the other option we discussed?"),
        ],
    )
    def test_classifies_all_categories(self, category: QueryCategory, query_text: str):
        llm = MagicMock()
        llm.generate.return_value = f'{{"category": "{category.value}", "reasoning": "test"}}'
        classifier = QueryClassifier(llm=llm, enabled=True)

        result = classifier.classify(Query(text=query_text))

        assert result.metadata["category"] == category.value

    def test_disabled_skips_llm_call(self):
        llm = MagicMock()
        classifier = QueryClassifier(llm=llm, enabled=False)

        result = classifier.classify(Query(text="Any question?"))

        assert result.metadata == {}
        llm.generate.assert_not_called()

    def test_llm_failure_defaults_to_factual(self, caplog):
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("LLM down")
        classifier = QueryClassifier(llm=llm, enabled=True)

        with caplog.at_level(logging.WARNING):
            result = classifier.classify(Query(text="What is EBITDA?"))

        assert result.metadata["category"] == "factual"
        assert "Query classification failed" in caplog.text

    def test_invalid_response_defaults_to_factual(self):
        llm = MagicMock()
        llm.generate.return_value = "not json at all"
        classifier = QueryClassifier(llm=llm, enabled=True)

        result = classifier.classify(Query(text="What is EBITDA?"))

        assert result.metadata["category"] == "factual"

    def test_preserves_existing_metadata(self):
        llm = MagicMock()
        llm.generate.return_value = '{"category": "factual", "reasoning": "ok"}'
        classifier = QueryClassifier(llm=llm, enabled=True)

        result = classifier.classify(Query(text="q", metadata={"session_id": "abc"}))

        assert result.metadata == {"session_id": "abc", "category": "factual"}

    def test_caches_classification_per_query_text(self):
        llm = MagicMock()
        llm.generate.return_value = '{"category": "factual", "reasoning": "ok"}'
        classifier = QueryClassifier(llm=llm, enabled=True)

        classifier.classify(Query(text="same question"))
        classifier.classify(Query(text="same question"))

        llm.generate.assert_called_once()

    def test_otel_span_records_category(self):
        import src.rag.retrieval.adaptive.query_classifier as qc_module

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        test_tracer = provider.get_tracer("rag-platform.retrieval")

        llm = MagicMock()
        llm.generate.return_value = '{"category": "analytical", "reasoning": "trend"}'
        classifier = QueryClassifier(llm=llm, enabled=True)

        with patch.object(qc_module, "_tracer", test_tracer):
            classifier.classify(Query(text="How do margins trend over time?"))

        spans = exporter.get_finished_spans()
        assert any(s.name == "retrieval.adaptive.classification" for s in spans)
        span = next(s for s in spans if s.name == "retrieval.adaptive.classification")
        attrs = span.attributes or {}
        assert attrs["query.category"] == "analytical"


class TestRetrievalServiceClassification:
    @pytest.mark.asyncio
    async def test_classifier_runs_before_retrieval(self):
        from src.domain.services.retrieval_service import RetrievalService

        dense = MagicMock()
        dense.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[])
        classifier = MagicMock()
        classifier.classify.side_effect = lambda q: q.model_copy(
            update={"metadata": {**q.metadata, "category": "analytical"}},
        )

        svc = RetrievalService(
            dense_retriever=dense,
            hybrid_retriever=hybrid,
            query_classifier=classifier,
        )
        result = await svc.retrieve(Query(text="Compare Q1 and Q2 revenue"))

        classifier.classify.assert_called_once()
        assert result.query.metadata["category"] == "analytical"
