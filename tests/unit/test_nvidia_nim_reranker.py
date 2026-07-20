"""Unit tests for NvidiaNimRerankerProvider (HTTP mocked, no real credentials)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from src.core.exceptions import ConfigurationError, RetrievalError
from src.domain.entities.chunk import Chunk
from src.domain.repositories.reranker_repository import RerankerRepository
from src.infrastructure.rerankers.nvidia_nim_reranker import NvidiaNimRerankerProvider

# ── helpers ────────────────────────────────────────────────────────────────────


def _chunk(i: int, text: str = "") -> Chunk:
    return Chunk(id=f"c{i}", document_id="doc", text=text or f"chunk text {i}")


def _chunks(n: int) -> list[Chunk]:
    return [_chunk(i) for i in range(n)]


_INVOKE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia/llama-nemotron-rerank-1b-v2/reranking"


def _ranking_response(rankings: list[dict[str, Any]]) -> httpx.Response:
    request = httpx.Request("POST", _INVOKE_URL)
    return httpx.Response(status_code=200, json={"rankings": rankings}, request=request)


def _provider_and_mock(
    client: MagicMock | None = None,
) -> tuple[NvidiaNimRerankerProvider, MagicMock]:
    mock_client = client or MagicMock()
    p = NvidiaNimRerankerProvider(
        api_key="nvapi-test",
        model="nvidia/llama-nemotron-rerank-1b-v2",
        client=mock_client,
    )
    return p, mock_client


# ── NvidiaNimRerankerProvider ───────────────────────────────────────────────────


class TestNvidiaNimRerankerProvider:
    def test_implements_reranker_repository(self) -> None:
        p, _ = _provider_and_mock()
        assert isinstance(p, RerankerRepository)

    def test_score_returns_chunk_score_pairs_in_input_order(self) -> None:
        p, mock_client = _provider_and_mock()
        mock_client.post.return_value = _ranking_response(
            [{"index": 1, "logit": 2.5}, {"index": 0, "logit": -0.3}]
        )
        chunks = _chunks(2)
        result = p.score("q", chunks)
        assert result == [(chunks[0], -0.3), (chunks[1], 2.5)]

    def test_score_empty_chunks_returns_empty(self) -> None:
        p, mock_client = _provider_and_mock()
        assert p.score("q", []) == []
        mock_client.post.assert_not_called()

    def test_request_shape(self) -> None:
        p, mock_client = _provider_and_mock()
        mock_client.post.return_value = _ranking_response([{"index": 0, "logit": 1.0}])
        p.score("which way should i go?", [_chunk(0, text="a passage")])

        _, kwargs = mock_client.post.call_args
        assert kwargs["json"] == {
            "model": "nvidia/llama-nemotron-rerank-1b-v2",
            "query": {"text": "which way should i go?"},
            "passages": [{"text": "a passage"}],
        }
        assert kwargs["headers"]["Authorization"] == "Bearer nvapi-test"

    def test_invoke_url_is_per_model_retrieval_path(self) -> None:
        """Reranking NIMs use "{base_url}/retrieval/{model}/reranking", NOT
        "{base_url}/ranking" — the latter 404s against the real API (found
        during #79's live validation)."""
        p, mock_client = _provider_and_mock()
        mock_client.post.return_value = _ranking_response([{"index": 0, "logit": 1.0}])
        p.score("q", [_chunk(0)])

        args, _ = mock_client.post.call_args
        assert args[0] == _INVOKE_URL

    def test_rerank_sorted_by_score_descending(self) -> None:
        p, mock_client = _provider_and_mock()
        mock_client.post.return_value = _ranking_response(
            [{"index": 0, "logit": 0.3}, {"index": 1, "logit": 0.9}, {"index": 2, "logit": 0.5}]
        )
        chunks = [_chunk(0), _chunk(1), _chunk(2)]
        result = p.rerank("q", chunks, top_k=3)
        assert [c.id for c in result] == ["c1", "c2", "c0"]

    def test_top_k_respected(self) -> None:
        p, mock_client = _provider_and_mock()
        mock_client.post.return_value = _ranking_response(
            [{"index": 0, "logit": 0.9}, {"index": 1, "logit": 0.5}, {"index": 2, "logit": 0.3}]
        )
        result = p.rerank("q", _chunks(3), top_k=2)
        assert len(result) == 2

    def test_empty_chunks_returns_empty(self) -> None:
        p, _ = _provider_and_mock()
        assert p.rerank("q", [], top_k=5) == []

    def test_http_error_raises_retrieval_error(self) -> None:
        p, mock_client = _provider_and_mock()
        request = httpx.Request("POST", _INVOKE_URL)
        mock_client.post.return_value = httpx.Response(status_code=500, request=request)
        with pytest.raises(RetrievalError) as exc_info:
            p.rerank("q", _chunks(2), top_k=2)
        assert exc_info.value.cause is not None

    def test_http_error_message_includes_status_code(self) -> None:
        p, mock_client = _provider_and_mock()
        request = httpx.Request("POST", _INVOKE_URL)
        mock_client.post.return_value = httpx.Response(status_code=503, request=request)
        with pytest.raises(RetrievalError, match="HTTP 503"):
            p.rerank("q", _chunks(2), top_k=2)

    def test_blank_api_key_raises_configuration_error(self) -> None:
        with pytest.raises(ConfigurationError, match="RERANKER__NVIDIA_NIM__API_KEY"):
            NvidiaNimRerankerProvider(api_key="   ")

    def test_network_error_raises_retrieval_error(self) -> None:
        p, mock_client = _provider_and_mock()
        mock_client.post.side_effect = httpx.ConnectError("connection refused")
        with pytest.raises(RetrievalError):
            p.rerank("q", _chunks(2), top_k=2)

    def test_from_settings(self) -> None:
        from pydantic import SecretStr

        mock_settings = MagicMock()
        mock_settings.reranker.nvidia_nim = MagicMock(
            api_key=SecretStr("nvapi-test"),
            model="nvidia/llama-nemotron-rerank-1b-v2",
            base_url="https://ai.api.nvidia.com/v1",
        )
        with patch("src.core.settings.settings", mock_settings):
            provider = NvidiaNimRerankerProvider.from_settings()
        assert provider.api_key == "nvapi-test"
        assert provider.model == "nvidia/llama-nemotron-rerank-1b-v2"
        assert provider.base_url == "https://ai.api.nvidia.com/v1"
