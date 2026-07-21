"""T-132 — Adaptive retrieval strategy tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.core.settings import CategoryStrategySettings, Settings
from src.domain.entities.query import Query
from src.domain.services.retrieval_service import RetrievalService
from src.rag.retrieval.adaptive.query_classifier import QueryCategory
from src.rag.retrieval.adaptive.strategies import (
    AdaptiveStrategyRegistry,
    AnalyticalRetrievalStrategy,
    ContextualRetrievalStrategy,
    FactualRetrievalStrategy,
    OpinionRetrievalStrategy,
    RetrievalStrategyParams,
)


class TestRetrievalStrategyParams:
    @pytest.mark.parametrize(
        ("strategy_cls", "expected"),
        [
            (FactualRetrievalStrategy, (30, 1, False, True)),
            (AnalyticalRetrievalStrategy, (50, 3, True, True)),
            (OpinionRetrievalStrategy, (20, 2, False, False)),
            (ContextualRetrievalStrategy, (40, 2, False, True)),
        ],
    )
    def test_default_category_params(self, strategy_cls, expected):
        params = strategy_cls().params
        assert (params.top_k, params.n_variants, params.hyde, params.compression) == expected


class TestAdaptiveStrategyRegistry:
    def test_resolves_each_category(self):
        registry = AdaptiveStrategyRegistry()
        for category in QueryCategory:
            params = registry.resolve_params(category.value)
            assert params.top_k > 0
            assert params.n_variants >= 1

    def test_unknown_category_falls_back_to_factual(self):
        registry = AdaptiveStrategyRegistry()
        factual = registry.resolve_params(QueryCategory.FACTUAL.value)
        unknown = registry.resolve_params("not-a-category")
        assert unknown == factual

    def test_missing_category_key_falls_back_to_factual(self):
        registry = AdaptiveStrategyRegistry()
        assert registry.resolve_params(None) == registry.resolve_params("factual")

    def test_yaml_overrides_applied(self):
        registry = AdaptiveStrategyRegistry(
            strategies={
                "analytical": RetrievalStrategyParams(
                    top_k=99,
                    n_variants=5,
                    hyde=False,
                    compression=False,
                ),
            },
        )
        params = registry.resolve_params("analytical")
        assert params.top_k == 99
        assert params.n_variants == 5
        assert params.hyde is False
        assert params.compression is False

    def test_from_settings_loads_yaml_defaults(self):
        registry = AdaptiveStrategyRegistry.from_settings()
        analytical = registry.resolve_params("analytical")
        assert analytical.top_k == 50
        assert analytical.hyde is True

    def test_from_settings_respects_env_override(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("RETRIEVAL__ADAPTIVE__STRATEGIES__FACTUAL__TOP_K", "12")
        with patch("src.core.settings.settings", Settings()):
            registry = AdaptiveStrategyRegistry.from_settings()
        factual = registry.resolve_params("factual")
        assert factual.top_k == 12


class TestRetrievalServiceAdaptiveStrategies:
    @pytest.mark.asyncio
    async def test_analytical_strategy_tunes_retrieval(self):
        from src.domain.entities.chunk import Chunk

        dense = MagicMock()
        dense.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
        chunk = Chunk(id="c1", document_id="doc", text="revenue data")
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[(chunk, 0.9)])
        expander = MagicMock()
        expander.expand.side_effect = lambda q, n_variants=None: q
        compressor = MagicMock()
        compressor.compress = AsyncMock(return_value=[chunk])

        registry = AdaptiveStrategyRegistry()
        svc = RetrievalService(
            dense_retriever=dense,
            hybrid_retriever=hybrid,
            query_expander=expander,
            compressor=compressor,
            strategy_registry=registry,
            top_k_retrieval=10,
        )

        await svc.retrieve(
            Query(text="Compare revenue trends", metadata={"category": "analytical"}),
        )

        expander.expand.assert_called_once()
        assert expander.expand.call_args.kwargs["n_variants"] == 3
        hybrid.retrieve.assert_awaited()
        assert hybrid.retrieve.await_args.kwargs["top_k"] == 50
        assert hybrid.retrieve.await_args.kwargs["use_hyde"] is True
        compressor.compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_opinion_strategy_skips_compression_and_hyde(self):
        dense = MagicMock()
        dense.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[])
        compressor = MagicMock()

        svc = RetrievalService(
            dense_retriever=dense,
            hybrid_retriever=hybrid,
            compressor=compressor,
            strategy_registry=AdaptiveStrategyRegistry(),
        )

        await svc.retrieve(Query(text="Which option is better?", metadata={"category": "opinion"}))

        assert hybrid.retrieve.await_args.kwargs["top_k"] == 20
        assert hybrid.retrieve.await_args.kwargs["use_hyde"] is False
        compressor.compress.assert_not_called()

    @pytest.mark.asyncio
    async def test_unknown_category_uses_factual_strategy(self):
        dense = MagicMock()
        dense.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[])

        svc = RetrievalService(
            dense_retriever=dense,
            hybrid_retriever=hybrid,
            strategy_registry=AdaptiveStrategyRegistry(),
        )

        await svc.retrieve(Query(text="What year?", metadata={"category": "unknown"}))

        assert hybrid.retrieve.await_args.kwargs["top_k"] == 30

    @pytest.mark.asyncio
    async def test_no_registry_uses_service_defaults(self):
        dense = MagicMock()
        dense.embed_query.side_effect = lambda q: q.model_copy(update={"embedding": [0.1]})
        hybrid = MagicMock()
        hybrid.retrieve = AsyncMock(return_value=[])

        svc = RetrievalService(
            dense_retriever=dense,
            hybrid_retriever=hybrid,
            top_k_retrieval=77,
        )

        await svc.retrieve(Query(text="plain query"))

        assert hybrid.retrieve.await_args.kwargs["top_k"] == 77
        assert hybrid.retrieve.await_args.kwargs["use_hyde"] is True

    def test_otel_span_records_strategy_params(self):
        import src.domain.services.retrieval_service as rs_module

        exporter = InMemorySpanExporter()
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(exporter))
        test_tracer = provider.get_tracer("rag-platform.retrieval")

        registry = AdaptiveStrategyRegistry()
        svc = RetrievalService(
            dense_retriever=MagicMock(),
            hybrid_retriever=MagicMock(),
            strategy_registry=registry,
        )

        with patch.object(rs_module, "_tracer", test_tracer):
            svc._resolve_strategy(Query(text="q", metadata={"category": "contextual"}))

        spans = exporter.get_finished_spans()
        assert any(s.name == "retrieval.adaptive.strategy" for s in spans)
        span = next(s for s in spans if s.name == "retrieval.adaptive.strategy")
        attrs = span.attributes or {}
        assert attrs["query.category"] == "contextual"
        assert attrs["retrieval.strategy.top_k"] == 40
        assert attrs["retrieval.strategy.n_variants"] == 2


class TestCategoryStrategySettings:
    def test_validates_positive_top_k(self):
        with pytest.raises(ValueError):
            CategoryStrategySettings(top_k=0, n_variants=1)
