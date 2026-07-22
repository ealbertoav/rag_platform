import json
import logging
import sys
from unittest.mock import patch

import pytest
from opentelemetry.sdk.trace import TracerProvider

import src.core.logging as logging_module
from src.core.logging import JsonFormatter, TextFormatter, get_logger


def _make_record(
    msg: str = "hello",
    level: int = logging.INFO,
    name: str = "test.logger",
    extra: dict | None = None,
) -> logging.LogRecord:
    record = logging.LogRecord(
        name=name,
        level=level,
        pathname=__file__,
        lineno=1,
        msg=msg,
        args=(),
        exc_info=None,
    )
    if extra:
        for k, v in extra.items():
            setattr(record, k, v)
    return record


# ── JsonFormatter ──────────────────────────────────────────────────────────────


class TestJsonFormatter:
    def test_output_is_valid_json(self):
        raw = JsonFormatter().format(_make_record())
        parsed = json.loads(raw)
        assert isinstance(parsed, dict)

    def test_required_keys_present(self):
        parsed = json.loads(JsonFormatter().format(_make_record()))
        required = (
            "timestamp",
            "level",
            "logger",
            "message",
            "module",
            "function",
            "line",
            "trace_id",
            "span_id",
        )
        for key in required:
            assert key in parsed, f"missing key: {key}"

    def test_message_content(self):
        parsed = json.loads(JsonFormatter().format(_make_record("test message")))
        assert parsed["message"] == "test message"

    def test_level_name(self):
        parsed = json.loads(JsonFormatter().format(_make_record(level=logging.WARNING)))
        assert parsed["level"] == "WARNING"

    def test_no_active_span_gives_empty_trace_context(self):
        parsed = json.loads(JsonFormatter().format(_make_record()))
        assert parsed["trace_id"] == ""
        assert parsed["span_id"] == ""

    def test_extra_fields_forwarded(self):
        parsed = json.loads(JsonFormatter().format(_make_record(extra={"request_id": "abc123"})))
        assert parsed.get("request_id") == "abc123"

    def test_exception_info_included(self):
        try:
            raise ValueError("boom")
        except ValueError:
            record = _make_record()
            record.exc_info = sys.exc_info()
            parsed = json.loads(JsonFormatter().format(record))
            assert "exception" in parsed
            assert "ValueError" in parsed["exception"]

    def test_timestamp_is_iso_format(self):
        parsed = json.loads(JsonFormatter().format(_make_record()))
        ts = parsed["timestamp"]
        # ISO 8601 with UTC offset contains "T" and "+"
        assert "T" in ts


# ── TextFormatter ──────────────────────────────────────────────────────────────


class TestTextFormatter:
    def test_output_is_string(self):
        out = TextFormatter().format(_make_record())
        assert isinstance(out, str)

    def test_contains_message(self):
        out = TextFormatter().format(_make_record("hello text"))
        assert "hello text" in out

    def test_no_trace_suffix_without_span(self):
        out = TextFormatter().format(_make_record())
        assert "trace=" not in out

    def test_level_in_output(self):
        out = TextFormatter().format(_make_record(level=logging.ERROR))
        assert "ERROR" in out


# ── OTel trace context ─────────────────────────────────────────────────────────


def _format_in_span(formatter: logging.Formatter) -> str:
    """Format a log record inside an active OTel span and return the result."""
    provider = TracerProvider()
    with provider.get_tracer("test").start_as_current_span("test-span"):
        return formatter.format(_make_record())


class TestOtelContext:
    def test_no_span_returns_empty_strings(self):
        parsed = json.loads(JsonFormatter().format(_make_record()))
        assert parsed["trace_id"] == ""
        assert parsed["span_id"] == ""

    def test_active_span_injects_valid_ids(self):
        parsed = json.loads(_format_in_span(JsonFormatter()))
        assert len(parsed["trace_id"]) == 32
        assert len(parsed["span_id"]) == 16
        assert all(c in "0123456789abcdef" for c in parsed["trace_id"])

    def test_text_formatter_appends_trace_suffix_inside_span(self):
        out = _format_in_span(TextFormatter())
        assert "trace=" in out
        assert "span=" in out


# ── get_logger ─────────────────────────────────────────────────────────────────


class TestGetLogger:
    @pytest.fixture(autouse=True)
    def _stub_configure_logging(self, monkeypatch: pytest.MonkeyPatch):
        """get_logger() lazily calls the real configure_logging() on its first-ever
        invocation, which sets up a real OTel TracerProvider against whatever
        settings.logging.otel_endpoint happens to be configured — exporting every
        subsequently-traced span in the rest of the test session to a real
        collector if one happens to be reachable (e.g. a local dev Docker stack).
        Stub it out, matching test_main.py's lifespan tests' existing pattern."""
        monkeypatch.setattr(logging_module, "_configured", False)
        with patch("src.core.logging.configure_logging"):
            yield

    def test_returns_logger_instance(self):
        logger = get_logger("src.core.test")
        assert isinstance(logger, logging.Logger)

    def test_name_preserved(self):
        logger = get_logger("my.module")
        assert logger.name == "my.module"

    def test_idempotent_calls(self):
        logger1 = get_logger("app.a")
        logger2 = get_logger("app.a")
        assert logger1 is logger2
