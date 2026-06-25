"""Shared fixtures and hooks for unit tests."""

from __future__ import annotations

import os

# CI sets OTEL_SDK_DISABLED=true to avoid OTLP export during app startup, but several
# unit tests assert real span/trace behavior. Force the SDK on before test modules
# import opentelemetry (conftest loads before a collection imports test files).
if os.environ.get("OTEL_SDK_DISABLED", "").lower() in {"1", "true", "yes"}:
    os.environ["OTEL_SDK_DISABLED"] = "false"
