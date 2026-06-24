"""Unit tests for the four API-based embedding providers.

All HTTP calls are mocked — no real API keys or network access required.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

if TYPE_CHECKING:
    from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider
    from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider
    from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider
    from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

import pytest

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import EmbeddingRepository

_TEXTS = ["Hello world.", "Kubernetes IAM roles.", "Vector databases are cool."]
_DIM = 1536


# ── Helpers ────────────────────────────────────────────────────────────────────


def _fake_vecs(n: int, dim: int = _DIM) -> list[list[float]]:
    return [[float(i) / dim] * dim for i in range(n)]


# ── OpenAIEmbeddingProvider ────────────────────────────────────────────────────


class TestOpenAIEmbeddingProvider:
    @staticmethod
    def _provider() -> OpenAIEmbeddingProvider:
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        return OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-3-small")

    def test_implements_repository(self) -> None:
        assert isinstance(self._provider(), EmbeddingRepository)

    def test_embed_returns_dense_vectors(self) -> None:
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-3-small")
        vecs = _fake_vecs(len(_TEXTS))

        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=v, index=i) for i, v in enumerate(vecs)]
        mock_client = MagicMock()
        mock_client.embeddings.create.return_value = mock_response
        provider._client = mock_client

        result = provider.embed(_TEXTS)
        assert len(result) == len(_TEXTS)
        assert all(isinstance(r, list) for r in result)
        mock_client.embeddings.create.assert_called_once()

    def test_embed_sparse_returns_empty_dicts(self) -> None:
        result = self._provider().embed_sparse(_TEXTS)
        assert result == [{}, {}, {}]

    def test_embed_empty_list_returns_empty(self) -> None:
        assert self._provider().embed([]) == []

    def test_dimension_truncation_applied_for_v3_models(self) -> None:
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-3-large", dimensions=512
        )
        assert provider.dimensions == 512

    def test_dimension_truncation_ignored_for_ada(self) -> None:
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(
            api_key="sk-test", model="text-embedding-ada-002", dimensions=512
        )
        assert provider.dimensions is None

    def test_import_error_raises_embedding_error(self) -> None:
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="sk-test")
        with patch.dict("sys.modules", {"openai": None}):
            provider._client = None
            with pytest.raises(EmbeddingError, match="openai package is not installed"):
                provider._get_client()

    def test_retry_on_rate_limit(self) -> None:
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-3-small")
        vecs = _fake_vecs(1)
        mock_response = MagicMock()
        mock_response.data = [MagicMock(embedding=vecs[0], index=0)]
        mock_client = MagicMock()
        # Fail twice with 429, succeed on third call
        mock_client.embeddings.create.side_effect = [
            Exception("429 rate_limit"),
            Exception("429 rate_limit"),
            mock_response,
        ]
        provider._client = mock_client

        result = provider.embed(["test"])
        assert len(result) == 1
        assert mock_client.embeddings.create.call_count == 3

    def test_non_rate_limit_error_not_retried(self) -> None:
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        provider = OpenAIEmbeddingProvider(api_key="sk-test", model="text-embedding-3-small")
        mock_client = MagicMock()
        mock_client.embeddings.create.side_effect = Exception("Authentication failed")
        provider._client = mock_client

        with pytest.raises(EmbeddingError):
            provider.embed(["test"])
        assert mock_client.embeddings.create.call_count == 1


# ── VoyageEmbeddingProvider ───────────────────────────────────────────────────


class TestVoyageEmbeddingProvider:
    @staticmethod
    def _provider() -> VoyageEmbeddingProvider:
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        return VoyageEmbeddingProvider(api_key="voy-test")

    def test_implements_repository(self) -> None:
        assert isinstance(self._provider(), EmbeddingRepository)

    def test_embed_returns_dense_vectors(self) -> None:
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        provider = VoyageEmbeddingProvider(api_key="voy-test")
        vecs = _fake_vecs(len(_TEXTS))
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=vecs)
        provider._client = mock_client

        result = provider.embed(_TEXTS)
        assert len(result) == len(_TEXTS)
        mock_client.embed.assert_called_once_with(
            _TEXTS, model="voyage-large-2", input_type="document"
        )

    def test_embed_query_uses_query_input_type(self) -> None:
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        provider = VoyageEmbeddingProvider(api_key="voy-test")
        vecs = _fake_vecs(1)
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=vecs)
        provider._client = mock_client

        provider.embed_query(["what is EKS?"])
        mock_client.embed.assert_called_once_with(
            ["what is EKS?"], model="voyage-large-2", input_type="query"
        )

    def test_embed_sparse_returns_empty_dicts(self) -> None:
        result = self._provider().embed_sparse(_TEXTS)
        assert result == [{}, {}, {}]

    def test_embed_empty_list_returns_empty(self) -> None:
        assert self._provider().embed([]) == []

    def test_import_error_raises_embedding_error(self) -> None:
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        provider = VoyageEmbeddingProvider(api_key="voy-test")
        with patch.dict("sys.modules", {"voyageai": None}):
            provider._client = None
            with pytest.raises(EmbeddingError, match="voyageai package is not installed"):
                provider._get_client()

    def test_retry_on_rate_limit(self) -> None:
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        provider = VoyageEmbeddingProvider(api_key="voy-test")
        vecs = _fake_vecs(1)
        mock_client = MagicMock()
        mock_client.embed.side_effect = [
            Exception("429 too many requests"),
            MagicMock(embeddings=vecs),
        ]
        provider._client = mock_client

        result = provider.embed(["test"])
        assert len(result) == 1
        assert mock_client.embed.call_count == 2

    def test_non_rate_limit_error_not_retried(self) -> None:
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        provider = VoyageEmbeddingProvider(api_key="voy-test")
        mock_client = MagicMock()
        mock_client.embed.side_effect = Exception("Invalid model")
        provider._client = mock_client

        with pytest.raises(EmbeddingError):
            provider.embed(["test"])
        assert mock_client.embed.call_count == 1


# ── CohereEmbeddingProvider ───────────────────────────────────────────────────


class TestCohereEmbeddingProvider:
    @staticmethod
    def _provider() -> CohereEmbeddingProvider:
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        return CohereEmbeddingProvider(api_key="co-test")

    def test_implements_repository(self) -> None:
        assert isinstance(self._provider(), EmbeddingRepository)

    def test_embed_uses_search_document_input_type(self) -> None:
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="co-test")
        vecs = _fake_vecs(len(_TEXTS), dim=1024)
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=MagicMock(float_=vecs))
        provider._client = mock_client

        provider.embed(_TEXTS)
        mock_client.embed.assert_called_once_with(
            texts=_TEXTS,
            model="embed-english-v3.0",
            input_type="search_document",
            embedding_types=["float"],
        )

    def test_embed_query_uses_search_query_input_type(self) -> None:
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="co-test")
        vecs = _fake_vecs(1, dim=1024)
        mock_client = MagicMock()
        mock_client.embed.return_value = MagicMock(embeddings=MagicMock(float_=vecs))
        provider._client = mock_client

        provider.embed_query(["what is EKS?"])
        mock_client.embed.assert_called_once_with(
            texts=["what is EKS?"],
            model="embed-english-v3.0",
            input_type="search_query",
            embedding_types=["float"],
        )

    def test_embed_sparse_returns_empty_dicts(self) -> None:
        result = self._provider().embed_sparse(_TEXTS)
        assert result == [{}, {}, {}]

    def test_embed_empty_list_returns_empty(self) -> None:
        assert self._provider().embed([]) == []

    def test_import_error_raises_embedding_error(self) -> None:
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="co-test")
        with patch.dict("sys.modules", {"cohere": None}):
            provider._client = None
            with pytest.raises(EmbeddingError, match="cohere package is not installed"):
                provider._get_client()

    def test_retry_on_rate_limit(self) -> None:
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        provider = CohereEmbeddingProvider(api_key="co-test")
        vecs = _fake_vecs(1, dim=1024)
        mock_client = MagicMock()
        mock_client.embed.side_effect = [
            Exception("429 rate_limit exceeded"),
            MagicMock(embeddings=MagicMock(float_=vecs)),
        ]
        provider._client = mock_client

        result = provider.embed(["test"])
        assert len(result) == 1
        assert mock_client.embed.call_count == 2


# ── GeminiEmbeddingProvider ────────────────────────────────────────────────────


class TestGeminiEmbeddingProvider:
    @staticmethod
    def _provider() -> GeminiEmbeddingProvider:
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        return GeminiEmbeddingProvider(api_key="gem-test")

    def test_implements_repository(self) -> None:
        assert isinstance(self._provider(), EmbeddingRepository)

    def test_embed_uses_retrieval_document_task_type(self) -> None:
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="gem-test")
        vecs = _fake_vecs(len(_TEXTS), dim=768)

        with patch(
            "src.infrastructure.embeddings.gemini_provider.GeminiEmbeddingProvider._call_api",
            return_value=vecs,
        ) as mock_call:
            provider.embed(_TEXTS)
            mock_call.assert_called_once_with(_TEXTS, "RETRIEVAL_DOCUMENT")

    def test_embed_query_uses_retrieval_query_task_type(self) -> None:
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="gem-test")
        vecs = _fake_vecs(1, dim=768)

        with patch(
            "src.infrastructure.embeddings.gemini_provider.GeminiEmbeddingProvider._call_api",
            return_value=vecs,
        ) as mock_call:
            provider.embed_query(["what is EKS?"])
            mock_call.assert_called_once_with(["what is EKS?"], "RETRIEVAL_QUERY")

    def test_embed_sparse_returns_empty_dicts(self) -> None:
        result = self._provider().embed_sparse(_TEXTS)
        assert result == [{}, {}, {}]

    def test_embed_empty_list_returns_empty(self) -> None:
        assert self._provider().embed([]) == []

    def test_import_error_raises_embedding_error(self) -> None:
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="gem-test")
        with (
            patch.dict("sys.modules", {"google": None, "google.generativeai": None}),
            pytest.raises(EmbeddingError, match="google-generativeai package is not installed"),
        ):
            provider._call_api(["test"], "RETRIEVAL_DOCUMENT")

    def test_configure_called_per_api_call(self) -> None:
        """genai.configure() must be called before every API request with this instance's key."""
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="gem-test-key")
        vecs = _fake_vecs(1, dim=768)
        mock_genai = MagicMock()
        mock_genai.embed_content.return_value = {"embedding": vecs}

        # Patch only google.generativeai — do NOT also patch "google" (the namespace
        # package may already be installed via other deps and overriding it breaks import
        # resolution for sys.modules["google.generativeai"]).
        with patch.dict("sys.modules", {"google.generativeai": mock_genai}):
            provider._call_api(["hello"], "RETRIEVAL_DOCUMENT")
            mock_genai.configure.assert_called_once_with(api_key="gem-test-key")

    def test_retry_on_rate_limit(self) -> None:
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        provider = GeminiEmbeddingProvider(api_key="gem-test")
        vecs = _fake_vecs(1, dim=768)

        call_count = 0

        def _side_effect(_texts: list[str], _task_type: str) -> list[list[float]]:
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("429 quota exceeded")
            return vecs

        with patch.object(provider, "_call_api", side_effect=_side_effect):
            result = provider.embed(["test"])
        assert len(result) == 1
        assert call_count == 3
