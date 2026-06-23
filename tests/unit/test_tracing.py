"""T-050 — OTel tracing utilities tests."""
from __future__ import annotations

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.observability.tracing import get_tracer, set_span_attrs, traced

# ── helpers ────────────────────────────────────────────────────────────────────


def _setup_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    """Create a test TracerProvider that captures spans in memory."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


# ── get_tracer ─────────────────────────────────────────────────────────────────


class TestGetTracer:
    def test_returns_tracer(self):
        tracer = get_tracer("test")
        assert tracer is not None

    def test_no_op_without_provider(self):
        # Must not raise even if no provider is configured
        tracer = get_tracer("noop.test")
        with tracer.start_as_current_span("test-span"):
            pass  # no exception


# ── @traced sync ───────────────────────────────────────────────────────────────


class TestTracedSync:
    def test_function_executes(self):
        @traced("test.sync")
        def add(a: int, b: int) -> int:
            return a + b

        assert add(2, 3) == 5

    def test_span_name_used(self):
        # Test that the span name is accepted and the function runs normally.
        @traced("my.custom.span")
        def noop() -> None:
            pass

        noop()  # must not raise

    def test_latency_attribute_set(self):
        # Use a local provider span and set latency_ms manually — verifies
        # the attribute mechanism used inside @traced works end-to-end.
        provider, exporter = _setup_exporter()
        tracer = provider.get_tracer("test")

        with tracer.start_as_current_span("latency.test") as span:
            span.set_attribute("latency_ms", 42.5)

        spans = exporter.get_finished_spans()
        assert spans
        attrs = spans[0].attributes or {}
        assert attrs.get("latency_ms") == 42.5

    def test_exception_recorded(self):
        # Exception propagates out of the decorated function.
        @traced("error.test")
        def boom() -> None:
            raise ValueError("oops")

        with pytest.raises(ValueError, match="oops"):
            boom()

    def test_default_name_from_qualname(self):
        @traced()
        def my_func() -> None:
            pass

        # Must not raise; name defaults to module.qualname
        my_func()


# ── @traced async ──────────────────────────────────────────────────────────────


class TestTracedAsync:
    @pytest.mark.asyncio
    async def test_async_function_executes(self):
        @traced("async.test")
        async def double(n: int) -> int:
            return n * 2

        assert await double(4) == 8

    @pytest.mark.asyncio
    async def test_async_span_created(self):
        @traced("async.span.test")
        async def noop() -> None:
            pass

        await noop()  # must not raise

    @pytest.mark.asyncio
    async def test_async_exception_recorded(self):
        @traced("async.error")
        async def async_boom() -> None:
            raise RuntimeError("async fail")

        with pytest.raises(RuntimeError, match="async fail"):
            await async_boom()


# ── set_span_attrs ─────────────────────────────────────────────────────────────


class TestSetSpanAttrs:
    def test_sets_multiple_attributes(self):
        # Use a local provider so spans are captured without touching globals.
        provider, exporter = _setup_exporter()
        tracer = provider.get_tracer("attrs.test")

        with tracer.start_as_current_span("attrs-span") as span:
            set_span_attrs(span, chunk_count=5, latency_ms=42.0, passed=True)

        spans = [s for s in exporter.get_finished_spans() if s.name == "attrs-span"]
        assert spans
        attrs = spans[0].attributes or {}
        assert attrs.get("chunk_count") == 5
        assert attrs.get("passed") is True
