"""T-273 — Chunk Lookup API tests."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from src.core.exceptions import VectorStoreError
from src.core.settings import settings
from src.domain.entities.chunk import Chunk
from src.main import create_app


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


@pytest.fixture
def app_client(monkeypatch):
    monkeypatch.setattr(settings.quality.chunk_lookup, "enabled", True)
    app = create_app()
    app.state.models_loaded = True
    return app


class TestGetChunkEndpoint:
    @pytest.mark.asyncio
    async def test_returns_200_with_chunk(self, app_client):
        chunk = Chunk(
            id="chunk-a",
            document_id="doc-1",
            text="hello world",
            metadata={"source": "a.pdf", "page": 3},
        )
        store = MagicMock()
        store.get_chunk.return_value = chunk
        app_client.state.vector_store = store
        async with _client(app_client) as c:
            resp = await c.get("/chunks/chunk-a")
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_id"] == "chunk-a"
        assert data["document_id"] == "doc-1"
        assert data["text"] == "hello world"
        assert data["page"] == 3
        store.get_chunk.assert_called_once_with("chunk-a")

    @pytest.mark.asyncio
    async def test_returns_404_when_missing(self, app_client):
        store = MagicMock()
        store.get_chunk.return_value = None
        app_client.state.vector_store = store
        async with _client(app_client) as c:
            resp = await c.get("/chunks/missing")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_404_when_disabled(self, monkeypatch):
        monkeypatch.setattr(settings.quality.chunk_lookup, "enabled", False)
        app = create_app()
        app.state.models_loaded = True
        store = MagicMock()
        store.get_chunk.return_value = Chunk(id="chunk-a", document_id="doc-1", text="hi")
        app.state.vector_store = store
        async with _client(app) as c:
            resp = await c.get("/chunks/chunk-a")
        assert resp.status_code == 404
        store.get_chunk.assert_not_called()

    @pytest.mark.asyncio
    async def test_returns_502_on_vector_store_error(self, app_client):
        store = MagicMock()
        store.get_chunk.side_effect = VectorStoreError("Qdrant retrieve failed")
        app_client.state.vector_store = store
        async with _client(app_client) as c:
            resp = await c.get("/chunks/chunk-a")
        assert resp.status_code == 502

    @pytest.mark.asyncio
    async def test_requires_api_key_when_configured(self, app_client, monkeypatch):
        monkeypatch.setattr(settings.api, "api_key", SecretStr("secret-key"))
        app_client.state.vector_store = MagicMock()
        async with _client(app_client) as c:
            resp = await c.get("/chunks/chunk-a")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_accepts_valid_api_key(self, app_client, monkeypatch):
        monkeypatch.setattr(settings.api, "api_key", SecretStr("secret-key"))
        store = MagicMock()
        store.get_chunk.return_value = Chunk(id="chunk-a", document_id="doc-1", text="hi")
        app_client.state.vector_store = store
        async with _client(app_client) as c:
            resp = await c.get("/chunks/chunk-a", headers={"X-API-Key": "secret-key"})
        assert resp.status_code == 200
