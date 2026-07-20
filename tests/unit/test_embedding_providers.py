"""Tests for Nomic and Qwen3 embedding providers (models mocked)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.infrastructure.embeddings import get_embedding_provider
from src.infrastructure.embeddings.nomic import NomicEmbeddingProvider
from src.infrastructure.embeddings.qwen_embedding import QwenEmbeddingProvider

_TEXTS = ["Hello world.", "Kubernetes EKS IAM roles.", "Vector databases."]
_DIM = 768


def _st_mock(n: int, dim: int = _DIM) -> MagicMock:
    """Fake SentenceTransformer that returns random unit vectors."""
    rng = np.random.default_rng(0)
    vecs = rng.random((n, dim)).astype("float32")
    m = MagicMock()
    m.encode.return_value = vecs
    return m


def _nomic(model: MagicMock | None = None) -> NomicEmbeddingProvider:
    p = NomicEmbeddingProvider(model_path="nomic-ai/nomic-embed-text-v1.5", device="cpu")
    if model is not None:
        p._model = model
    return p


def _qwen(model: MagicMock | None = None) -> QwenEmbeddingProvider:
    p = QwenEmbeddingProvider(model_path="Qwen/Qwen3-Embedding-0.6B", device="cpu")
    if model is not None:
        p._model = model
    return p


def _sentence_transformer_import_error() -> patch.dict:
    """Block real sentence-transformers import (avoids heavy deps and SWIG noise)."""
    fake_module = MagicMock()
    fake_module.SentenceTransformer = MagicMock(side_effect=OSError("not found"))
    return patch.dict(sys.modules, {"sentence_transformers": fake_module})


# ── NomicEmbeddingProvider ─────────────────────────────────────────────────────


class TestNomicProvider:
    def test_implements_embedding_repository(self):
        assert isinstance(_nomic(), EmbeddingRepository)

    def test_embed_returns_vectors(self):
        p = _nomic(_st_mock(len(_TEXTS)))
        result = p.embed(_TEXTS)
        assert len(result) == len(_TEXTS)
        assert all(len(v) == _DIM for v in result)

    def test_embed_returns_floats(self):
        p = _nomic(_st_mock(len(_TEXTS)))
        result = p.embed(_TEXTS)
        assert all(isinstance(v[0], float) for v in result)

    def test_embed_empty_returns_empty(self):
        assert _nomic().embed([]) == []

    def test_embed_sparse_returns_empty_dicts(self):
        result = _nomic().embed_sparse(_TEXTS)
        assert result == [{}, {}, {}]

    def test_embed_both_uses_embed(self):
        mock = _st_mock(len(_TEXTS))
        p = _nomic(mock)
        dense, sparse = p.embed_both(_TEXTS)
        assert len(dense) == len(_TEXTS)
        assert sparse == [{}, {}, {}]
        mock.encode.assert_called_once()

    def test_model_load_error_raises_embedding_error(self):
        p = NomicEmbeddingProvider(model_path="bad/path")
        with _sentence_transformer_import_error(), pytest.raises(EmbeddingError):
            p._get_model()

    def test_encode_failure_raises_embedding_error(self):
        mock = MagicMock()
        mock.encode.side_effect = RuntimeError("OOM")
        p = _nomic(mock)
        with pytest.raises(EmbeddingError):
            p.embed(_TEXTS)

    def test_from_settings_returns_instance(self):
        assert isinstance(NomicEmbeddingProvider.from_settings(), NomicEmbeddingProvider)


# ── QwenEmbeddingProvider ──────────────────────────────────────────────────────


class TestQwenProvider:
    def test_implements_embedding_repository(self):
        assert isinstance(_qwen(), EmbeddingRepository)

    def test_embed_returns_vectors(self):
        p = _qwen(_st_mock(len(_TEXTS)))
        result = p.embed(_TEXTS)
        assert len(result) == len(_TEXTS)
        assert all(len(v) == _DIM for v in result)

    def test_embed_empty_returns_empty(self):
        assert _qwen().embed([]) == []

    def test_embed_sparse_returns_empty_dicts(self):
        result = _qwen().embed_sparse(_TEXTS)
        assert result == [{}, {}, {}]

    def test_embed_both_uses_embed(self):
        mock = _st_mock(len(_TEXTS))
        p = _qwen(mock)
        dense, sparse = p.embed_both(_TEXTS)
        assert len(dense) == len(_TEXTS)
        assert sparse == [{}, {}, {}]
        mock.encode.assert_called_once()

    def test_model_load_error_raises_embedding_error(self):
        p = QwenEmbeddingProvider(model_path="bad/path")
        with _sentence_transformer_import_error(), pytest.raises(EmbeddingError):
            p._get_model()

    def test_from_settings_returns_instance(self):
        assert isinstance(QwenEmbeddingProvider.from_settings(), QwenEmbeddingProvider)


# ── get_embedding_provider factory ────────────────────────────────────────────


def _apply_dotted_attrs(mock: MagicMock, attrs: dict[str, object]) -> None:
    """Set nested attributes on *mock* from dotted keys (e.g., embeddings.provider)."""
    for key, value in attrs.items():
        parts = key.split(".")
        target = mock
        for part in parts[:-1]:
            child = getattr(target, part, None)
            if not isinstance(child, MagicMock):
                child = MagicMock()
                setattr(target, part, child)
            target = child
        setattr(target, parts[-1], value)


def _mock_settings(**kwargs: object) -> MagicMock:
    """Build a settings mock with cache disabled and sensible embedding defaults."""
    defaults: dict[str, object] = {
        "embeddings.provider": "bge_m3",
        "embeddings.model_path": "models/embeddings/bge-m3",
        "embeddings.device": "cpu",
        "embeddings.batch_size": 32,
        "embeddings.normalize": True,
        "embeddings.cache.enabled": False,  # disable cache in unit tests
    }
    defaults.update(kwargs)
    settings = MagicMock()
    _apply_dotted_attrs(settings, defaults)
    return settings


class TestGetEmbeddingProvider:
    def test_bge_m3_default(self):
        from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
        from src.infrastructure.embeddings.cached_embedding_provider import CachedEmbeddingProvider

        provider = get_embedding_provider()
        # May be wrapped in CachedEmbeddingProvider when cache is enabled; unwrap.
        inner = provider._inner if isinstance(provider, CachedEmbeddingProvider) else provider
        assert isinstance(inner, BGEM3EmbeddingProvider)

    def test_nomic_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EMBEDDINGS__PROVIDER", "nomic")
        monkeypatch.setenv("EMBEDDINGS__MODEL_PATH", "nomic-ai/nomic-embed-text-v1.5")
        with patch(
            "src.core.settings.settings",
            _mock_settings(
                **{
                    "embeddings.provider": "nomic",
                    "embeddings.model_path": "nomic-ai/nomic-embed-text-v1.5",
                }
            ),
        ):
            provider = get_embedding_provider()
        assert isinstance(provider, NomicEmbeddingProvider)

    def test_qwen_provider(self):
        with patch(
            "src.core.settings.settings",
            _mock_settings(**{"embeddings.provider": "qwen_embedding"}),
        ):
            provider = get_embedding_provider()
        assert isinstance(provider, QwenEmbeddingProvider)

    def test_clip_provider(self):
        from src.infrastructure.embeddings.clip_provider import ClipEmbeddingProvider

        with patch(
            "src.core.settings.settings",
            _mock_settings(**{"embeddings.provider": "clip"}),
        ):
            provider = get_embedding_provider()
        assert isinstance(provider, ClipEmbeddingProvider)

    def test_unknown_provider_raises(self):
        bad_settings = _mock_settings(**{"embeddings.provider": "bad_provider"})
        with (
            patch("src.core.settings.settings", bad_settings),
            pytest.raises(ValueError, match="Unknown"),
        ):
            get_embedding_provider()


class TestCreateEmbeddingProvider:
    def test_uses_provider_default_model_path_when_not_active(self):
        from src.infrastructure.embeddings import create_embedding_provider

        settings = _mock_settings(
            **{
                "embeddings.provider": "bge_m3",
                "embeddings.model_path": "models/embeddings/bge-m3",
            }
        )
        provider = create_embedding_provider("nomic", settings)
        assert isinstance(provider, NomicEmbeddingProvider), (
            f"expected NomicEmbeddingProvider, got {provider.__class__.__name__}"
        )
        assert provider.model_path == "nomic-ai/nomic-embed-text-v1.5"

    def test_uses_configured_model_path_for_active_provider(self):
        from src.infrastructure.embeddings import create_embedding_provider

        settings = _mock_settings(
            **{
                "embeddings.provider": "nomic",
                "embeddings.model_path": "custom/nomic-path",
            }
        )
        provider = create_embedding_provider("nomic", settings)
        assert isinstance(provider, NomicEmbeddingProvider), (
            f"expected NomicEmbeddingProvider, got {provider.__class__.__name__}"
        )
        assert provider.model_path == "custom/nomic-path"


class TestEmbeddingModelIdentifier:
    def test_includes_api_model_name(self):
        from src.infrastructure.embeddings import embedding_model_identifier

        settings = _api_settings("openai")
        settings.embeddings.openai.model = "text-embedding-3-small"
        settings.embeddings.openai.dimensions = 1536
        assert (
            embedding_model_identifier("openai", settings) == "openai:text-embedding-3-small@1536"
        )

    def test_includes_self_hosted_model_path(self):
        from src.infrastructure.embeddings import embedding_model_identifier

        settings = _mock_settings(
            **{
                "embeddings.provider": "bge_m3",
                "embeddings.model_path": "models/embeddings/bge-m3",
            }
        )
        assert embedding_model_identifier("bge_m3", settings) == "bge_m3:models/embeddings/bge-m3"

    def test_openai_includes_effective_dimensions(self):
        from src.infrastructure.embeddings import embedding_model_identifier

        settings = _api_settings("openai")
        settings.embeddings.openai.model = "text-embedding-3-large"
        settings.embeddings.openai.dimensions = 512
        assert embedding_model_identifier("openai", settings) == "openai:text-embedding-3-large@512"


class TestProviderDenseDim:
    def test_openai_ada_uses_native_dim_not_configured_truncation(self):
        from src.infrastructure.embeddings import provider_dense_dim

        settings = _mock_settings(
            **{
                "embeddings.openai.model": "text-embedding-ada-002",
                "embeddings.openai.dimensions": 512,
            }
        )
        assert provider_dense_dim("openai", settings) == 1536

    def test_openai_v3_respects_configured_dimensions(self):
        from src.infrastructure.embeddings import provider_dense_dim

        settings = _mock_settings(
            **{
                "embeddings.openai.model": "text-embedding-3-large",
                "embeddings.openai.dimensions": 512,
            }
        )
        assert provider_dense_dim("openai", settings) == 512


class TestCreateEmbeddingProviderApiSettings:
    def test_openai_uses_passed_settings_not_global_singleton(self):
        from pydantic import SecretStr

        from src.infrastructure.embeddings import create_embedding_provider
        from src.infrastructure.embeddings.openai_provider import OpenAIEmbeddingProvider

        settings = _mock_settings(
            **{
                "embeddings.openai.api_key": SecretStr("passed-key"),
                "embeddings.openai.model": "text-embedding-3-small",
                "embeddings.openai.dimensions": 256,
            }
        )
        with patch(
            "src.core.settings.settings",
            _mock_settings(
                **{
                    "embeddings.openai.api_key": SecretStr("global-key"),
                    "embeddings.openai.model": "text-embedding-3-large",
                    "embeddings.openai.dimensions": 3072,
                }
            ),
        ):
            provider = create_embedding_provider("openai", settings)

        assert isinstance(provider, OpenAIEmbeddingProvider)
        assert provider.api_key == "passed-key"
        assert provider.model == "text-embedding-3-small"
        assert provider.dimensions == 256


def _api_settings(provider: str = "openai", *, api_key: str = "sk-test") -> MagicMock:
    """Build a nested settings mock for API embedding provider tests."""
    from pydantic import SecretStr

    settings = MagicMock()
    emb = MagicMock()
    emb.provider = provider
    emb.device = "cpu"
    emb.batch_size = 32
    emb.normalize = True
    emb.model_path = "models/embeddings/bge-m3"
    emb.cache = MagicMock(enabled=False, ttl_seconds=604800)
    emb.openai = MagicMock(
        api_key=SecretStr(api_key),
        model="text-embedding-3-small",
        dimensions=1536,
    )
    emb.voyage = MagicMock(
        api_key=SecretStr(api_key),
        model="voyage-large-2",
        dimensions=1024,
        multimodal_model="voyage-multimodal-3",
    )
    emb.cohere = MagicMock(api_key=SecretStr(api_key), model="embed-english-v3.0", dimensions=1024)
    emb.gemini = MagicMock(api_key=SecretStr(api_key), model="text-embedding-004", dimensions=768)
    emb.nvidia_nim = MagicMock(
        api_key=SecretStr(api_key),
        model="nvidia/llama-3.2-nv-embedqa-1b-v2",
        base_url="https://integrate.api.nvidia.com/v1",
        dimensions=2048,
    )
    settings.embeddings = emb
    settings.redis = MagicMock(url="redis://localhost:6379", password=SecretStr(""))
    return settings


class TestProviderDenseDimExtended:
    def test_voyage_cohere_gemini_dims(self):
        from src.infrastructure.embeddings import provider_dense_dim

        settings = _api_settings()
        assert provider_dense_dim("voyage", settings) == 1024
        assert provider_dense_dim("cohere", settings) == 1024
        assert provider_dense_dim("gemini", settings) == 768
        assert provider_dense_dim("nvidia_nim", settings) == 2048

    def test_self_hosted_provider_dims(self):
        from src.infrastructure.embeddings import provider_dense_dim

        settings = _mock_settings(**{"embeddings.provider": "nomic"})
        assert provider_dense_dim("nomic", settings) == 768

    def test_clip_dim(self):
        from src.infrastructure.embeddings import provider_dense_dim

        settings = _mock_settings(**{"embeddings.provider": "clip"})
        assert provider_dense_dim("clip", settings) == 512

    def test_unhandled_api_provider_raises(self, monkeypatch):
        from src.infrastructure.embeddings import provider_dense_dim

        monkeypatch.setattr(
            "src.infrastructure.embeddings.API_EMBEDDING_PROVIDERS",
            frozenset({"future_api"}),
        )
        settings = _api_settings()
        with pytest.raises(AssertionError, match="Unhandled API embedding provider"):
            provider_dense_dim("future_api", settings)


class TestProviderImageDim:
    """T-252: image_dense dimension lookup for the Qdrant schema."""

    def test_clip_returns_512(self):
        from src.infrastructure.embeddings import provider_image_dim

        settings = _mock_settings(**{"embeddings.provider": "clip"})
        assert provider_image_dim("clip", settings) == 512

    def test_voyage_returns_configured_multimodal_dimensions(self):
        from src.infrastructure.embeddings import provider_image_dim

        settings = _api_settings("voyage")
        settings.embeddings.voyage.multimodal_dimensions = 1024
        assert provider_image_dim("voyage", settings) == 1024

    def test_text_only_providers_return_none(self):
        from src.infrastructure.embeddings import provider_image_dim

        settings = _mock_settings(**{"embeddings.provider": "bge_m3"})
        assert provider_image_dim("bge_m3", settings) is None
        assert provider_image_dim("openai", _api_settings("openai")) is None
        assert provider_image_dim("cohere", _api_settings("cohere")) is None
        assert provider_image_dim("gemini", _api_settings("gemini")) is None
        assert provider_image_dim("nvidia_nim", _api_settings("nvidia_nim")) is None

    def test_unhandled_multimodal_provider_raises(self, monkeypatch):
        from src.infrastructure.embeddings import provider_image_dim

        monkeypatch.setattr(
            "src.infrastructure.embeddings.MULTIMODAL_EMBEDDING_PROVIDERS",
            frozenset({"future_multimodal"}),
        )
        settings = _api_settings()
        with pytest.raises(AssertionError, match="Unhandled multimodal embedding provider"):
            provider_image_dim("future_multimodal", settings)


class TestEmbeddingModelIdentifierExtended:
    def test_voyage_cohere_gemini(self):
        from src.infrastructure.embeddings import embedding_model_identifier

        settings = _api_settings()
        assert embedding_model_identifier("voyage", settings) == "voyage:voyage-large-2@1024"
        assert embedding_model_identifier("cohere", settings) == "cohere:embed-english-v3.0@1024"
        assert embedding_model_identifier("gemini", settings) == "gemini:text-embedding-004@768"
        assert (
            embedding_model_identifier("nvidia_nim", settings)
            == "nvidia_nim:nvidia/llama-3.2-nv-embedqa-1b-v2@2048"
        )


class TestCreateProviderApi:
    def test_voyage_uses_passed_settings(self) -> None:
        from src.infrastructure.embeddings import create_embedding_provider
        from src.infrastructure.embeddings.voyage_provider import VoyageEmbeddingProvider

        settings = _api_settings("voyage")
        provider = create_embedding_provider("voyage", settings)
        assert isinstance(provider, VoyageEmbeddingProvider)
        assert provider.api_key == "sk-test"

    def test_cohere_uses_passed_settings(self) -> None:
        from src.infrastructure.embeddings import create_embedding_provider
        from src.infrastructure.embeddings.cohere_provider import CohereEmbeddingProvider

        settings = _api_settings("cohere")
        provider = create_embedding_provider("cohere", settings)
        assert isinstance(provider, CohereEmbeddingProvider)
        assert provider.api_key == "sk-test"

    def test_gemini_uses_passed_settings(self) -> None:
        from src.infrastructure.embeddings import create_embedding_provider
        from src.infrastructure.embeddings.gemini_provider import GeminiEmbeddingProvider

        settings = _api_settings("gemini")
        provider = create_embedding_provider("gemini", settings)
        assert isinstance(provider, GeminiEmbeddingProvider)
        assert provider.api_key == "sk-test"

    def test_nvidia_nim_uses_passed_settings(self) -> None:
        from src.infrastructure.embeddings import create_embedding_provider
        from src.infrastructure.embeddings.nvidia_nim_provider import NvidiaNimEmbeddingProvider

        settings = _api_settings("nvidia_nim")
        provider = create_embedding_provider("nvidia_nim", settings)
        assert isinstance(provider, NvidiaNimEmbeddingProvider)
        assert provider.api_key == "sk-test"
        assert provider.base_url == "https://integrate.api.nvidia.com/v1"

    @pytest.mark.parametrize("provider", ["openai", "voyage", "cohere", "gemini", "nvidia_nim"])
    def test_missing_api_key_raises(self, provider: str) -> None:
        from src.core.exceptions import ConfigurationError
        from src.infrastructure.embeddings import create_embedding_provider

        settings = _api_settings(provider, api_key="")
        with pytest.raises(ConfigurationError, match="API key"):
            create_embedding_provider(provider, settings)


class TestCreateProviderUnknown:
    def test_unknown_provider_raises(self):
        from src.infrastructure.embeddings import create_embedding_provider

        with pytest.raises(ValueError, match="Unknown"):
            create_embedding_provider("not_a_provider", _mock_settings())


class TestGetEmbeddingProviderCache:
    def test_wraps_with_cache_when_enabled(self):
        from src.infrastructure.embeddings.cached_embedding_provider import CachedEmbeddingProvider

        settings = _mock_settings(**{"embeddings.cache.enabled": True})
        settings.redis = MagicMock(
            url="redis://localhost:6379", password=MagicMock(get_secret_value=lambda: "")
        )
        settings.embeddings.cache = MagicMock(enabled=True, ttl_seconds=3600)
        with patch("src.core.settings.settings", settings):
            provider = get_embedding_provider()
        assert isinstance(provider, CachedEmbeddingProvider)
