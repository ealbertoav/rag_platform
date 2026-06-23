from __future__ import annotations

import functools
import inspect
import time
from collections.abc import Callable
from typing import TypeVar

from opentelemetry import trace
from opentelemetry.trace import Span, StatusCode

_F = TypeVar("_F", bound=Callable[..., object])

# Module-level tracer — the global TracerProvider is set up in core.logging.
_tracer = trace.get_tracer("rag-platform")


def get_tracer(name: str = "rag-platform") -> trace.Tracer:
    """Return a named tracer backed by the configured global TracerProvider.

    When no provider is configured (e.g., in tests without OTel), the OTel
    SDK returns a no-op tracer, so calls are safe to make regardless.
    """
    return trace.get_tracer(name)


def traced(
    span_name: str | None = None,
    *,
    record_exception: bool = True,
) -> Callable[[_F], _F]:
    """Decorator that wraps a sync or async function in an OTel span.

    Usage:

        @traced("retrieval.hybrid")
        async def retrieve (self, query: Query, top_k: int) -> list[SearchResult]: ...

        @traced() # defaults to module.ClassName.method
        def rerank(self, ...): ...

    The span automatically records "latency_ms" and, on error, the exception
    and sets the span status to ERROR.
    """
    def decorator(fn: _F) -> _F:
        module = str(getattr(fn, "__module__", None) or __name__)
        qualname = str(getattr(fn, "__qualname__", None) or repr(fn))
        name = span_name or f"{module}.{qualname}"
        tracer = get_tracer(module)

        if inspect.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def async_wrapper(*args: object, **kwargs: object) -> object:
                with tracer.start_as_current_span(name) as span:
                    t0 = time.monotonic()
                    try:
                        result = await fn(*args, **kwargs)  # type: ignore[misc]
                        _set_latency(span, t0)
                        return result
                    except Exception as exc:
                        if record_exception:
                            span.record_exception(exc)
                            span.set_status(StatusCode.ERROR, str(exc))
                        raise
            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: object, **kwargs: object) -> object:
            with tracer.start_as_current_span(name) as span:
                t0 = time.monotonic()
                try:
                    result = fn(*args, **kwargs)  # type: ignore[misc]
                    _set_latency(span, t0)
                    return result
                except Exception as exc:
                    if record_exception:
                        span.record_exception(exc)
                        span.set_status(StatusCode.ERROR, str(exc))
                    raise
        return sync_wrapper  # type: ignore[return-value]

    return decorator


# ── helpers ────────────────────────────────────────────────────────────────────


def _set_latency(span: Span, t0: float) -> None:
    span.set_attribute("latency_ms", round((time.monotonic() - t0) * 1000, 1))


def set_span_attrs(span: Span, **attrs: str | int | float | bool) -> None:
    """Convenience wrapper for setting multiple span attributes at once."""
    for key, value in attrs.items():
        span.set_attribute(key, value)
