"""T-032 — FastAPI application tests."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr

from src.core.settings import settings
from src.domain.entities.answer import Answer
from src.main import create_app
from src.rag.pipelines.ingestion_pipeline import IngestionResult

# ── app fixture ────────────────────────────────────────────────────────────────


async def _token_stream(*tokens: str) -> AsyncIterator[str]:
    for t in tokens:
        yield t


@pytest.fixture
def chat_pipeline_mock() -> MagicMock:
    m = MagicMock()
    m.chat = AsyncMock(return_value=_token_stream("Hello", " world"))
    m.chat_full = AsyncMock(
        return_value=Answer(
            query_id="q-1",
            text="Hello world",
            sources=["c0"],
            latency_ms=42.0,
            token_count=2,
        )
    )
    return m


@pytest.fixture
def ingestion_pipeline_mock() -> MagicMock:
    m = MagicMock()
    m.ingest_file.return_value = IngestionResult(
        source="/tmp/doc.md", chunk_count=3, content_hash="abc123"
    )
    m.ingest_directory.return_value = [
        IngestionResult(source="/tmp/doc.md", chunk_count=3, content_hash="abc123")
    ]
    m.save_indexes = MagicMock()
    return m


@pytest.fixture
def agent_pipeline_mock() -> MagicMock:
    from src.domain.entities.answer import Answer
    from src.rag.pipelines.agent_pipeline import AgentAction, AgentRunResult

    m = MagicMock()
    m.chat = AsyncMock(return_value=_token_stream("Agent", " answer"))
    m.chat_full = AsyncMock(
        return_value=AgentRunResult(
            answer=Answer(
                query_id="q-1",
                text="Agent answer",
                sources=["c0"],
                latency_ms=55.0,
                token_count=2,
            ),
            iterations=2,
            actions=[AgentAction.ANSWER],
            self_rag_decisions=[],
        )
    )
    return m


@pytest.fixture
def app_client(chat_pipeline_mock, ingestion_pipeline_mock, agent_pipeline_mock):
    app = create_app()
    app.state.chat_pipeline = chat_pipeline_mock
    app.state.agent_pipeline = agent_pipeline_mock
    app.state.ingestion_pipeline = ingestion_pipeline_mock
    app.state.models_loaded = True
    return app


def _client(app):
    return AsyncClient(transport=ASGITransport(app=app), base_url="http://test")


def _allow_ingest_root(monkeypatch: pytest.MonkeyPatch, root: Path) -> None:
    monkeypatch.setattr(settings.api, "ingest_allowed_roots", [str(root)])


# ── /health ────────────────────────────────────────────────────────────────────


class TestHealth:
    @pytest.mark.asyncio
    async def test_returns_200(self, app_client):
        async with _client(app_client) as c:
            resp = await c.get("/health")
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_status_ok(self, app_client):
        async with _client(app_client) as c:
            data = (await c.get("/health")).json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_models_loaded_true(self, app_client):
        async with _client(app_client) as c:
            data = (await c.get("/health")).json()
        assert data["models_loaded"] is True


# ── /ingest/path ───────────────────────────────────────────────────────────────


class TestIngestPath:
    @pytest.mark.asyncio
    async def test_existing_file_returns_200(self, app_client, tmp_path, monkeypatch):
        _allow_ingest_root(monkeypatch, tmp_path)
        md = tmp_path / "doc.md"
        md.write_text("# Hello")
        app_client.state.ingestion_pipeline.ingest_file.return_value = IngestionResult(
            source=str(md), chunk_count=1, content_hash="aaa"
        )
        async with _client(app_client) as c:
            resp = await c.post("/ingest/path", json={"source": str(md)})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_returns_chunk_count(self, app_client, tmp_path, monkeypatch):
        _allow_ingest_root(monkeypatch, tmp_path)
        md = tmp_path / "doc.md"
        md.write_text("# Hello")
        app_client.state.ingestion_pipeline.ingest_file.return_value = IngestionResult(
            source=str(md), chunk_count=7, content_hash="aaa"
        )
        async with _client(app_client) as c:
            data = (await c.post("/ingest/path", json={"source": str(md)})).json()
        assert data["chunk_count"] == 7

    @pytest.mark.asyncio
    async def test_missing_file_returns_404(self, app_client, tmp_path, monkeypatch):
        _allow_ingest_root(monkeypatch, tmp_path)
        async with _client(app_client) as c:
            resp = await c.post("/ingest/path", json={"source": str(tmp_path / "missing.md")})
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_directory_ingest_sums_chunk_count(self, app_client, tmp_path, monkeypatch):
        _allow_ingest_root(monkeypatch, tmp_path)
        sub = tmp_path / "docs"
        sub.mkdir()
        app_client.state.ingestion_pipeline.ingest_directory.return_value = [
            IngestionResult(source=str(sub / "a.md"), chunk_count=2, content_hash="a"),
            IngestionResult(source=str(sub / "b.md"), chunk_count=3, content_hash="b"),
        ]
        async with _client(app_client) as c:
            data = (await c.post("/ingest/path", json={"source": str(sub)})).json()
        assert data["chunk_count"] == 5
        assert data["content_hash"] == ""

    @pytest.mark.asyncio
    async def test_ingestion_error_returns_422(self, app_client, tmp_path, monkeypatch):
        _allow_ingest_root(monkeypatch, tmp_path)
        from src.core.exceptions import IngestionError

        md = tmp_path / "doc.md"
        md.write_text("# Hello")
        app_client.state.ingestion_pipeline.ingest_file.side_effect = IngestionError("bad file")
        async with _client(app_client) as c:
            resp = await c.post("/ingest/path", json={"source": str(md)})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_path_outside_allowed_root_returns_403(self, app_client, monkeypatch):
        _allow_ingest_root(monkeypatch, Path("/allowed/only"))
        async with _client(app_client) as c:
            resp = await c.post("/ingest/path", json={"source": "/etc/passwd"})
        assert resp.status_code == 403


# ── /ingest/upload ─────────────────────────────────────────────────────────────


class TestIngestUpload:
    @pytest.mark.asyncio
    async def test_upload_returns_200(self, app_client):
        app_client.state.ingestion_pipeline.ingest_file.return_value = IngestionResult(
            source="upload.md", chunk_count=4, content_hash="xyz"
        )
        async with _client(app_client) as c:
            resp = await c.post(
                "/ingest/upload",
                files={"file": ("upload.md", b"# Title\n\nBody", "text/markdown")},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["chunk_count"] == 4
        assert data["content_hash"] == "xyz"

    @pytest.mark.asyncio
    async def test_upload_ingestion_error_returns_422(self, app_client):
        from src.core.exceptions import IngestionError

        app_client.state.ingestion_pipeline.ingest_file.side_effect = IngestionError("fail")
        async with _client(app_client) as c:
            resp = await c.post(
                "/ingest/upload",
                files={"file": ("bad.md", b"content", "text/markdown")},
            )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_upload_too_large_returns_413(self, app_client, monkeypatch):
        monkeypatch.setattr(settings.api, "max_upload_bytes", 4)
        async with _client(app_client) as c:
            resp = await c.post(
                "/ingest/upload",
                files={"file": ("big.md", b"12345", "text/markdown")},
            )
        assert resp.status_code == 413


# ── API key auth ───────────────────────────────────────────────────────────────


class TestApiKeyAuth:
    @pytest.mark.asyncio
    async def test_chat_requires_api_key_when_configured(self, app_client, monkeypatch):
        monkeypatch.setattr(settings.api, "api_key", SecretStr("secret-key"))
        async with _client(app_client) as c:
            resp = await c.post("/chat", json={"question": "q"})
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_chat_accepts_valid_api_key(self, app_client, monkeypatch):
        monkeypatch.setattr(settings.api, "api_key", SecretStr("secret-key"))
        async with _client(app_client) as c:
            resp = await c.post(
                "/chat",
                json={"question": "q"},
                headers={"X-API-Key": "secret-key"},
            )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_health_unauthenticated_when_api_key_configured(self, app_client, monkeypatch):
        monkeypatch.setattr(settings.api, "api_key", SecretStr("secret-key"))
        async with _client(app_client) as c:
            resp = await c.get("/health")
        assert resp.status_code == 200


class TestChatStream:
    @pytest.mark.asyncio
    async def test_returns_200(self, app_client):
        async with _client(app_client) as c:
            resp = await c.post("/chat", json={"question": "What is EKS?"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_content_type_event_stream(self, app_client):
        async with _client(app_client) as c:
            resp = await c.post("/chat", json={"question": "q"})
        assert "text/event-stream" in resp.headers["content-type"]

    @pytest.mark.asyncio
    async def test_sse_contains_tokens(self, app_client):
        async with _client(app_client) as c:
            resp = await c.post("/chat", json={"question": "q"})
        lines = [ln for ln in resp.text.splitlines() if ln.startswith("data:")]
        payloads = [json.loads(ln[5:].strip()) for ln in lines if ln.strip() != "data: [DONE]"]
        tokens = [p["token"] for p in payloads]
        assert "Hello" in tokens
        assert " world" in tokens

    @pytest.mark.asyncio
    async def test_sse_ends_with_done(self, app_client):
        async with _client(app_client) as c:
            resp = await c.post("/chat", json={"question": "q"})
        assert "data: [DONE]" in resp.text


# ── /chat/full ─────────────────────────────────────────────────────────────────


class TestChatFull:
    @pytest.mark.asyncio
    async def test_returns_200(self, app_client):
        async with _client(app_client) as c:
            resp = await c.post("/chat/full", json={"question": "q"})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_answer_in_response(self, app_client):
        async with _client(app_client) as c:
            data = (await c.post("/chat/full", json={"question": "q"})).json()
        assert data["answer"] == "Hello world"
        assert "c0" in data["sources"]

    @pytest.mark.asyncio
    async def test_latency_present(self, app_client):
        async with _client(app_client) as c:
            data = (await c.post("/chat/full", json={"question": "q"})).json()
        assert "latency_ms" in data

    @pytest.mark.asyncio
    async def test_explain_false_omits_explanations(self, app_client):
        async with _client(app_client) as c:
            data = (await c.post("/chat/full", json={"question": "q"})).json()
        assert "explanations" not in data

    @pytest.mark.asyncio
    async def test_explain_true_passes_flag_to_pipeline(self, app_client, chat_pipeline_mock):
        async with _client(app_client) as c:
            await c.post("/chat/full?explain=true", json={"question": "q"})
        chat_pipeline_mock.chat_full.assert_awaited_once()
        assert chat_pipeline_mock.chat_full.await_args.kwargs["explain"] is True


# ── /chat/agent ────────────────────────────────────────────────────────────────


class TestChatAgent:
    @pytest.mark.asyncio
    async def test_agent_stream_returns_200(self, app_client):
        async with _client(app_client) as c:
            resp = await c.post("/chat/agent", json={"question": "q", "max_iterations": 2})
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_agent_full_returns_metadata(self, app_client):
        async with _client(app_client) as c:
            data = (await c.post("/chat/agent/full", json={"question": "q"})).json()
        assert data["answer"] == "Agent answer"
        assert data["iterations"] == 2
        assert data["actions"] == ["ANSWER"]
        assert data["self_rag_decisions"] == []


# ── /evals/run ─────────────────────────────────────────────────────────────────


class TestEvals:
    @pytest.mark.asyncio
    async def test_empty_dataset_returns_204(self, app_client):
        # QA dataset has only placeholder rows in the test environment.
        async with _client(app_client) as c:
            resp = await c.post("/evals/run")
        assert resp.status_code == 204

    @pytest.mark.asyncio
    async def test_real_dataset_returns_200(self, app_client):
        from unittest.mock import AsyncMock, patch

        from src.evals.e2e.rag_benchmark import BenchmarkReport

        report = BenchmarkReport(
            timestamp="20250101T000000",
            total_samples=5,
            mean_recall_at_5=0.8,
            mean_faithfulness=0.9,
            mean_relevance=0.85,
            mean_context_precision=0.8,
            mean_hallucination=0.05,
            recall_threshold=0.5,
            faithfulness_threshold=0.8,
            relevance_threshold=0.75,
            context_precision_threshold=0.7,
            hallucination_threshold=0.1,
            passed=True,
        )
        with (
            patch(
                "src.domain.services.evaluation_service.EvaluationService.run",
                new_callable=AsyncMock,
                return_value=report,
            ),
            patch(
                "src.evals.e2e.rag_benchmark.BenchmarkReport.save",
            ),
        ):
            async with _client(app_client) as c:
                resp = await c.post("/evals/run")
        assert resp.status_code == 200
        assert resp.json()["status"] == "passed"
        assert resp.json()["total_samples"] == 5


# ── OpenAPI docs ───────────────────────────────────────────────────────────────


class TestOpenAPI:
    @pytest.mark.asyncio
    async def test_docs_available(self, app_client):
        async with _client(app_client) as c:
            resp = await c.get("/docs")
        assert resp.status_code == 200  # noqa: E501

    @pytest.mark.asyncio
    async def test_openapi_json(self, app_client):
        async with _client(app_client) as c:
            data = (await c.get("/openapi.json")).json()
        assert data["info"]["title"] == "RAG Platform"
