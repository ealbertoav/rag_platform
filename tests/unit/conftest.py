"""Shared fixtures and hooks for unit tests."""

from __future__ import annotations

import os

import pytest

# CI sets OTEL_SDK_DISABLED=true to avoid OTLP export during app startup, but several
# unit tests assert real span/trace behavior. Force the SDK on before test modules
# import opentelemetry (conftest loads before a collection imports test files).
if os.environ.get("OTEL_SDK_DISABLED", "").lower() in {"1", "true", "yes"}:
    os.environ["OTEL_SDK_DISABLED"] = "false"


@pytest.fixture(scope="session", autouse=True)
def _preload_datasets_once() -> None:
    """Import datasets once per xdist worker to avoid pyarrow extension re-registration."""
    try:
        import datasets  # noqa: F401
    except ImportError:
        pass
