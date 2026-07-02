"""Shared helpers for web search unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, patch


@contextmanager
def patch_web_search_httpx_post(
    *,
    text: str | None = None,
    json_payload: dict | None = None,
) -> Iterator[MagicMock]:
    """Patch httpx.AsyncClient used by web search providers and yield the mock client."""
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()
    if text is not None:
        mock_response.text = text
    if json_payload is not None:
        mock_response.json.return_value = json_payload

    mock_client = MagicMock()
    mock_client.post = AsyncMock(return_value=mock_response)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    with patch(
        "src.infrastructure.search.web_search.httpx.AsyncClient",
        return_value=mock_client,
    ):
        yield mock_client
