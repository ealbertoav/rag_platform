"""T-163 unit tests — async llama.cpp streaming (concurrent load + OTel)."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.core.exceptions import GenerationError
from src.infrastructure.llm.llama_cpp_provider import LlamaCppProvider


def _provider(model: MagicMock | None = None) -> LlamaCppProvider:
    p = LlamaCppProvider(
        model_path="fake/model.gguf",
        context_size=512,
        n_gpu_layers=0,
        temperature=0.0,
        max_tokens=64,
    )
    if model is not None:
        p._model = model  # type: ignore[assignment]
    return p


def _stream_mock(tokens: list[str], *, delay_s: float = 0.0) -> MagicMock:
    m = MagicMock()

    def _iter_chunks(**_: object):
        for token in tokens:
            if delay_s:
                import time

                time.sleep(delay_s)
            yield {"choices": [{"delta": {"content": token}}]}

    m.create_chat_completion.side_effect = _iter_chunks
    return m


def _setup_span_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


class TestConcurrentStreaming:
    @pytest.mark.asyncio
    async def test_ten_concurrent_streams_complete(self):
        """Ten overlapping streams must all finish with expected tokens."""
        mock = _stream_mock(["a", "b"], delay_s=0.005)
        provider = _provider(mock)

        async def _collect(idx: int) -> list[str]:
            return [t async for t in provider.generate_stream(f"q-{idx}", "")]

        results = await asyncio.gather(*(_collect(i) for i in range(10)))
        assert len(results) == 10
        assert all(tokens == ["a", "b"] for tokens in results)
        assert mock.create_chat_completion.call_count == 10

    @pytest.mark.asyncio
    async def test_concurrent_streams_do_not_starve_event_loop(self):
        """Event-loop heartbeat must keep ticking while streams are active."""
        mock = _stream_mock(["t1", "t2", "t3"], delay_s=0.02)
        provider = _provider(mock)
        heartbeat_ticks = 0
        stop = asyncio.Event()

        async def _heartbeat() -> None:
            nonlocal heartbeat_ticks
            while not stop.is_set():
                await asyncio.sleep(0.005)
                heartbeat_ticks += 1

        async def _collect(idx: int) -> list[str]:
            return [t async for t in provider.generate_stream(f"q-{idx}", "")]

        heartbeat_task = asyncio.create_task(_heartbeat())
        try:
            await asyncio.wait_for(asyncio.gather(*(_collect(i) for i in range(10))), timeout=30.0)
        finally:
            stop.set()
            await heartbeat_task

        assert heartbeat_ticks >= 5, "event loop was starved during concurrent streaming"

    @pytest.mark.asyncio
    async def test_single_stream_latency_unchanged(self):
        """Single-request path should not add measurable overhead vs. direct mock."""
        mock = _stream_mock(["x", "y", "z"])
        provider = _provider(mock)

        async def _once() -> list[str]:
            return [t async for t in provider.generate_stream("q", "")]

        import time

        t0 = time.monotonic()
        first = await _once()
        first_ms = (time.monotonic() - t0) * 1000

        t0 = time.monotonic()
        second = await _once()
        second_ms = (time.monotonic() - t0) * 1000

        assert first == second == ["x", "y", "z"]
        assert second_ms <= first_ms * 1.05 + 1.0


class TestStreamTracing:
    @pytest.mark.asyncio
    async def test_llm_stream_span_records_queue_wait_ms(self):
        provider, exporter = _setup_span_exporter()
        mock = _stream_mock(["hello", "world"])
        llm = _provider(mock)

        with patch(
            "src.infrastructure.llm.llama_cpp_provider._tracer",
            provider.get_tracer("test"),
        ):
            tokens = [t async for t in llm.generate_stream("q", "")]

        assert tokens == ["hello", "world"]
        spans = exporter.get_finished_spans()
        stream_spans = [s for s in spans if s.name == "llm.stream"]
        assert len(stream_spans) == 1
        attrs = stream_spans[0].attributes or {}
        queue_wait_ms = attrs.get("queue_wait_ms")
        assert isinstance(queue_wait_ms, (int, float))
        assert float(queue_wait_ms) >= 0.0

    @pytest.mark.asyncio
    async def test_stream_worker_exception_raises_generation_error(self):
        provider = LlamaCppProvider(model_path="fake.gguf")
        mock_llama = MagicMock()
        mock_llama.create_chat_completion.side_effect = RuntimeError("boom")
        provider._model = mock_llama

        with pytest.raises(GenerationError, match="stream failed"):
            async for _ in provider.generate_stream("prompt", "context"):
                pass
