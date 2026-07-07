"""T-163 unit tests — async llama.cpp streaming (concurrent load + OTel)."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from typing import cast
from unittest.mock import MagicMock, patch

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from src.core.exceptions import GenerationError
from src.infrastructure.llm import llama_cpp_provider
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

    def _iter_chunks(**kwargs: object):
        if not kwargs.get("stream"):
            return {"choices": [{"message": {"content": "test response"}}]}

        def _token_stream():
            for token in tokens:
                if delay_s:
                    import time

                    time.sleep(delay_s)
                yield {"choices": [{"delta": {"content": token}}]}

        return _token_stream()

    m.create_chat_completion.side_effect = _iter_chunks
    return m


def _setup_span_exporter() -> tuple[TracerProvider, InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return provider, exporter


def _async_gen(
    provider: LlamaCppProvider, prompt: str, context: str = ""
) -> AsyncGenerator[str, None]:
    return cast(AsyncGenerator[str, None], provider.generate_stream(prompt, context))


async def _collect_stream(provider: LlamaCppProvider, question: str) -> list[str]:
    return [t async for t in provider.generate_stream(question, "")]


def _token_stream_mock(count: int, *, delay_s: float = 0.005) -> MagicMock:
    return _stream_mock([f"t{i}" for i in range(count)], delay_s=delay_s)


async def _cancel_stream_after_first_token(
    provider: LlamaCppProvider,
    *,
    prompt: str = "q",
    expected_first: str = "t0",
    queue_maxsize: int = 1,
) -> None:
    with patch.object(llama_cpp_provider, "_STREAM_QUEUE_MAXSIZE", queue_maxsize):
        stream = _async_gen(provider, prompt)
        assert await stream.__anext__() == expected_first
        await asyncio.wait_for(stream.aclose(), timeout=2.0)


async def _assert_generate_after_cancel(
    provider: LlamaCppProvider, prompt: str = "after-cancel"
) -> None:
    result = await asyncio.wait_for(
        asyncio.to_thread(provider.generate, prompt, ""),
        timeout=2.0,
    )
    assert result == "test response"


def _inference_lock_held(provider: LlamaCppProvider) -> bool:
    """Return whether the provider serialization lock is currently held."""
    lock = provider._lock  # noqa: SLF001
    return lock.locked()


async def _assert_lock_released(provider: LlamaCppProvider) -> None:
    assert not _inference_lock_held(provider)


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

    @pytest.mark.asyncio
    async def test_backpressure_does_not_block_inference_lock(self):
        """Slow SSE consumers must not hold self._lock once tokens are produced."""
        mock = _stream_mock([f"t{i}" for i in range(32)], delay_s=0.0)
        provider = _provider(mock)

        with patch.object(llama_cpp_provider, "_STREAM_QUEUE_MAXSIZE", 2):
            stream = _async_gen(provider, "slow-consumer")
            assert await stream.__anext__() == "t0"

            result = await asyncio.wait_for(
                asyncio.to_thread(provider.generate, "concurrent", ""),
                timeout=2.0,
            )
            assert result == "test response"
            await asyncio.wait_for(stream.aclose(), timeout=2.0)


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


class TestStreamBridge:
    @pytest.mark.asyncio
    async def test_bridge_put_failure_raises_generation_error(self):
        """Bridge delivery failures must surface as GenerationError, not empty output."""
        provider = _provider(_stream_mock(["only"]))
        real_put = asyncio.Queue.put
        bridge_calls = 0

        async def _failing_put(queue: asyncio.Queue[object], item: object) -> None:
            nonlocal bridge_calls
            bridge_calls += 1
            if bridge_calls == 1 and isinstance(item, str):
                raise RuntimeError("bridge put failed")
            await real_put(queue, item)

        with (
            patch.object(asyncio.Queue, "put", _failing_put),
            pytest.raises(GenerationError, match="stream failed") as exc_info,
        ):
            await _collect_stream(provider, "q")

        assert isinstance(exc_info.value.cause, RuntimeError)
        await _assert_lock_released(provider)

    @pytest.mark.asyncio
    async def test_bridge_drops_items_after_cancelled(self):
        """Cover bridge discard path when cancellation races with pending thread items."""
        provider = _provider(_token_stream_mock(8))
        await _cancel_stream_after_first_token(provider)
        await _assert_lock_released(provider)


class TestStreamCancellation:
    @pytest.mark.asyncio
    async def test_early_close_releases_inference_lock(self):
        """Client disconnect must not leave self._lock held (Bugbot T-163)."""
        provider = _provider(_token_stream_mock(32))
        await _cancel_stream_after_first_token(provider, queue_maxsize=2)
        await _assert_generate_after_cancel(provider)

    @pytest.mark.asyncio
    async def test_cancelled_stream_allows_subsequent_streams(self):
        """A canceled stream must not block later streaming requests."""
        provider = _provider(_stream_mock(["a", "b", "c"], delay_s=0.01))
        await _cancel_stream_after_first_token(provider, expected_first="a")

        tokens = await asyncio.wait_for(_collect_stream(provider, "q2"), timeout=2.0)
        assert tokens == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_cancelled_stream_allows_generate_while_queue_saturated(self):
        """Cover cancel during bridge backpressure without leaving the lock held."""
        provider = _provider(_token_stream_mock(16, delay_s=0.01))
        await _cancel_stream_after_first_token(provider)
        await _assert_lock_released(provider)
        await _assert_generate_after_cancel(provider)

    @pytest.mark.asyncio
    async def test_bridge_thread_get_failure_raises_generation_error(self):
        """Cover the bridge exception handler when thread_queue.get fails."""
        provider = _provider(_stream_mock(["x"]))
        real_get = llama_cpp_provider.queue.Queue.get
        calls = 0

        def _failing_get(
            thread_queue: llama_cpp_provider.queue.Queue,
            block: bool = True,
            timeout: float | None = None,
        ) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise OSError("thread bridge failed")
            return real_get(thread_queue, block, timeout)

        with (
            patch.object(llama_cpp_provider.queue.Queue, "get", _failing_get),
            pytest.raises(GenerationError, match="stream failed") as exc_info,
        ):
            await _collect_stream(provider, "q")

        assert isinstance(exc_info.value.cause, OSError)

    @pytest.mark.asyncio
    async def test_bridge_exits_on_sentinel_after_cancel(self):
        """Cover bridge sentinel exit path after client disconnect."""
        provider = _provider(_token_stream_mock(12))
        await _cancel_stream_after_first_token(provider)
        await _assert_lock_released(provider)
