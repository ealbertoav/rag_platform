"""#109 integration test — Jaeger UI (requires running Docker stack).

Run with:
    docker compose up -d jaeger
    uv run pytest tests/integration/test_jaeger.py -v

Only confirms Jaeger's HTTP API is reachable — not which traces appear (see
tests/unit/test_jaeger_config.py for the static wiring checks, and #109's
Testing Decisions for why trace-content assertions are out of scope here).
"""

from __future__ import annotations

import httpx
import pytest

_JAEGER_API_URL = "http://localhost:16686/api/services"


def _reachable() -> bool:
    try:
        httpx.get(_JAEGER_API_URL, timeout=2).raise_for_status()
        return True
    except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason="Jaeger not reachable at localhost:16686")


class TestJaegerReachable:
    def test_services_api_returns_200(self):
        response = httpx.get(_JAEGER_API_URL, timeout=2)
        assert response.status_code == 200

    def test_services_api_returns_json_data(self):
        response = httpx.get(_JAEGER_API_URL, timeout=2)
        body = response.json()
        assert "data" in body
