from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any, ClassVar, override

from opentelemetry import trace

# Standard LogRecord attributes that should NOT be forwarded as extra fields.
_BUILTIN_ATTRS: frozenset[str] = frozenset(
    {
        "args",
        "created",
        "exc_info",
        "exc_text",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "message",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "taskName",
        "thread",
        "threadName",
    }
)

# Marker attribute placed on the handler we install so configure_logging()
# can detect and replace it instead of stacking duplicates.
_MARKER = "_rag_platform_handler"


def _otel_context() -> tuple[str, str]:
    """Return (trace_id, span_id) hex strings from the active OTel span.

    Returns ("", "") when there is no active span — no collector required.
    """
    ctx = trace.get_current_span().get_span_context()
    if ctx and ctx.is_valid:
        return format(ctx.trace_id, "032x"), format(ctx.span_id, "016x")
    return "", ""


# ── Formatters ─────────────────────────────────────────────────────────────────


class JsonFormatter(logging.Formatter):
    """Emits one JSON object per log line, always including OTel trace context."""

    @override
    def format(self, record: logging.LogRecord) -> str:
        record.message = record.getMessage()
        trace_id, span_id = _otel_context()

        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.message,
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
            "trace_id": trace_id,
            "span_id": span_id,
        }

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        if record.stack_info:
            payload["stack_info"] = self.formatStack(record.stack_info)

        # Forward any caller-supplied extra={} fields.
        for key, val in record.__dict__.items():
            if key not in _BUILTIN_ATTRS and key not in payload:
                payload[key] = val

        return json.dumps(payload, default=str)


class TextFormatter(logging.Formatter):
    """Human-readable formatter for local development, appends trace context when present."""

    _FMT: ClassVar[str] = "%(asctime)s %(levelname)-8s %(name)s — %(message)s"
    _DATEFMT: ClassVar[str] = "%Y-%m-%dT%H:%M:%S"

    def __init__(self) -> None:
        super().__init__(fmt=self._FMT, datefmt=self._DATEFMT)

    @override
    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        trace_id, span_id = _otel_context()
        if trace_id:
            return f"{base}  trace={trace_id[:8]} span={span_id[:8]}"
        return base


# ── OTel setup ─────────────────────────────────────────────────────────────────


def _setup_otel(endpoint: str, sampling_rate: float) -> None:
    """Configure a global OTel TracerProvider with OTLP gRPC export.

    Silently skips if the SDK is unavailable or the endpoint is unreachable —
    the rest of the app continues without distributed tracing.
    """
    try:
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import SERVICE_NAME, Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.sdk.trace.sampling import ALWAYS_ON, TraceIdRatioBased

        resource = Resource.create({SERVICE_NAME: "rag-platform"})
        sampler = ALWAYS_ON if sampling_rate >= 1.0 else TraceIdRatioBased(sampling_rate)
        provider = TracerProvider(resource=resource, sampler=sampler)
        exporter = OTLPSpanExporter(endpoint=endpoint, insecure=True)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
    except (ImportError, ValueError, OSError) as exc:
        logging.getLogger(__name__).debug("OTel setup skipped: %s", exc)


# ── Public API ─────────────────────────────────────────────────────────────────


def configure_logging() -> None:
    """Set up the root logger from settings. Idempotent — safe to call multiple times.

    Priority: existing call → env var LOG_LEVEL override → settings.logging values.
    """
    # Deferred import to avoid circular dependency at module load time.
    from src.core.settings import settings

    cfg = settings.logging
    level = getattr(logging, cfg.level, logging.INFO)
    formatter: logging.Formatter = JsonFormatter() if cfg.format == "json" else TextFormatter()

    root = logging.getLogger()

    # Remove any previously installed rag-platform handler (idempotency).
    root.handlers = [h for h in root.handlers if not getattr(h, _MARKER, False)]

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    setattr(handler, _MARKER, True)
    root.addHandler(handler)
    root.setLevel(level)

    # Silence chatty third-party loggers.
    for name in ("httpx", "httpcore", "urllib3", "hpack", "grpc"):
        logging.getLogger(name).setLevel(logging.WARNING)

    _setup_otel(cfg.otel_endpoint, cfg.trace_sampling_rate)


_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger for *name*, initializing the logging system on the first call."""
    global _configured
    if not _configured:
        configure_logging()
        _configured = True
    return logging.getLogger(name)
