"""Unit tests for FastAPI app lifecycle (main.py)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import create_app, lifespan


@pytest.mark.asyncio
async def test_lifespan_startup_sets_pipeline_state():
    app = create_app()
    mock_chat = MagicMock()
    mock_ingest = MagicMock()

    with (
        patch("src.core.logging.configure_logging"),
        patch(
            "src.rag.pipelines.chat_pipeline.ChatPipeline.from_settings",
            return_value=mock_chat,
        ),
        patch(
            "src.rag.pipelines.ingestion_pipeline.IngestionPipeline.from_settings",
            return_value=mock_ingest,
        ),
    ):
        async with lifespan(app):
            assert app.state.chat_pipeline is mock_chat
            assert app.state.ingestion_pipeline is mock_ingest
            assert app.state.models_loaded is True


@pytest.mark.asyncio
async def test_lifespan_shutdown_saves_indexes():
    app = create_app()
    mock_ingest = MagicMock()

    with (
        patch("src.core.logging.configure_logging"),
        patch("src.rag.pipelines.chat_pipeline.ChatPipeline.from_settings"),
        patch(
            "src.rag.pipelines.ingestion_pipeline.IngestionPipeline.from_settings",
            return_value=mock_ingest,
        ),
    ):
        async with lifespan(app):
            pass
        mock_ingest.save_indexes.assert_called_once()


@pytest.mark.asyncio
async def test_lifespan_shutdown_save_failure_logged(caplog):
    import logging

    app = create_app()
    mock_ingest = MagicMock()
    mock_ingest.save_indexes.side_effect = RuntimeError("disk full")

    with (
        patch("src.core.logging.configure_logging"),
        patch("src.rag.pipelines.chat_pipeline.ChatPipeline.from_settings"),
        patch(
            "src.rag.pipelines.ingestion_pipeline.IngestionPipeline.from_settings",
            return_value=mock_ingest,
        ),
        caplog.at_level(logging.WARNING),
    ):
        async with lifespan(app):
            pass
        assert "Failed to save BM25 index" in caplog.text


@pytest.mark.asyncio
async def test_create_app_serves_health():
    app = create_app()
    app.state.chat_pipeline = MagicMock()
    app.state.ingestion_pipeline = MagicMock()
    app.state.models_loaded = True

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
