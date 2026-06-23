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
        with patch(_ST, side_effect=OSError("not found"), create=True), \
                pytest.raises(EmbeddingError):
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
        with patch(_ST, side_effect=OSError("not found"), create=True), \
                pytest.raises(EmbeddingError):
            p._get_model()

    def test_from_settings_returns_instance(self):
        assert isinstance(QwenEmbeddingProvider.from_settings(), QwenEmbeddingProvider)


# ── get_embedding_provider factory ────────────────────────────────────────────


class TestGetEmbeddingProvider:
    def test_bge_m3_default(self):
        from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider
        provider = get_embedding_provider()
        assert isinstance(provider, BGEM3EmbeddingProvider)

    def test_nomic_provider(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setenv("EMBEDDINGS__PROVIDER", "nomic")
        monkeypatch.setenv("EMBEDDINGS__MODEL_PATH", "nomic-ai/nomic-embed-text-v1.5")
        # Force re-evaluation of settings inside the factory
        with patch("src.core.settings.settings",
                   **{"embeddings.provider": "nomic",
                      "embeddings.model_path": "nomic-ai/nomic-embed-text-v1.5",
                      "embeddings.device": "cpu",
                      "embeddings.batch_size": 32,
                      "embeddings.normalize": True}):
            provider = get_embedding_provider()
        assert isinstance(provider, NomicEmbeddingProvider)

    def test_qwen_provider(self):
        with patch("src.core.settings.settings",
                   **{"embeddings.provider": "qwen_embedding",
                      "embeddings.model_path": "Qwen/Qwen3-Embedding-0.6B",
                      "embeddings.device": "cpu",
                      "embeddings.batch_size": 32,
                      "embeddings.normalize": True}):
            provider = get_embedding_provider()
        assert isinstance(provider, QwenEmbeddingProvider)

    def test_unknown_provider_raises(self):
        with patch("src.core.settings.settings", **{"embeddings.provider": "openai"}), \
                pytest.raises(ValueError, match="Unknown"):
            get_embedding_provider()
