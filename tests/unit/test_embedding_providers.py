"""Tests for Nomic and Qwen3 embedding providers (models mocked)."""

from __future__ import annotations

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
        _ST = "src.infrastructure.embeddings.sentence_transformer_base.SentenceTransformer"
        with (
            patch(_ST, side_effect=OSError("not found"), create=True),
            pytest.raises(EmbeddingError),
        ):
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
        _ST = "src.infrastructure.embeddings.sentence_transformer_base.SentenceTransformer"
        with (
            patch(_ST, side_effect=OSError("not found"), create=True),
            pytest.raises(EmbeddingError),
        ):
            p._get_model()

    def test_from_settings_returns_instance(self):
        assert isinstance(QwenEmbeddingProvider.from_settings(), QwenEmbeddingProvider)


# ── get_embedding_provider factory ────────────────────────────────────────────


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
    return MagicMock(**defaults)


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

        settings = _mock_settings(
            **{
                "embeddings.provider": "openai",
                "embeddings.openai.model": "text-embedding-3-small",
            }
        )
        assert embedding_model_identifier("openai", settings) == "openai:text-embedding-3-small"

    def test_includes_self_hosted_model_path(self):
        from src.infrastructure.embeddings import embedding_model_identifier

        settings = _mock_settings(
            **{
                "embeddings.provider": "bge_m3",
                "embeddings.model_path": "models/embeddings/bge-m3",
            }
        )
        assert embedding_model_identifier("bge_m3", settings) == "bge_m3:models/embeddings/bge-m3"
