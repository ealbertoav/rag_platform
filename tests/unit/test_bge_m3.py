"""T-012 unit tests — BGE-M3 provider (model mocked)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.core.exceptions import EmbeddingError
from src.domain.repositories.embedding_repository import EmbeddingRepository
from src.infrastructure.embeddings.bge_m3 import BGEM3EmbeddingProvider

_TEXTS = ["Hello world.", "The sky is blue.", "Fast vector search."]
_DIM = 1024


def _make_model_mock(n: int) -> MagicMock:
    """Return a mock BGEM3FlagModel whose encode() returns plausible output."""
    dense = np.random.default_rng(0).random((n, _DIM)).astype("float32")
    sparse = [{str(i * 10 + j): float(j) / 10 for j in range(1, 4)} for i in range(n)]

    mock = MagicMock()
    mock.encode.return_value = {
        "dense_vecs": dense,
        "lexical_weights": sparse,
    }
    return mock


@pytest.fixture
def provider() -> BGEM3EmbeddingProvider:
    return BGEM3EmbeddingProvider(model_path="fake/path", device="cpu", batch_size=8)


@pytest.fixture
def provider_with_model(provider: BGEM3EmbeddingProvider) -> BGEM3EmbeddingProvider:
    provider._model = _make_model_mock(len(_TEXTS))
    return provider


# ── Interface conformance ──────────────────────────────────────────────────────


class TestInterfaceConformance:
    def test_implements_embedding_repository(self, provider: BGEM3EmbeddingProvider):
        assert isinstance(provider, EmbeddingRepository)

    def test_from_settings_returns_instance(self):
        p = BGEM3EmbeddingProvider.from_settings()
        assert isinstance(p, BGEM3EmbeddingProvider)


# ── embed() ────────────────────────────────────────────────────────────────────


class TestEmbed:
    def test_returns_one_vector_per_text(self, provider_with_model: BGEM3EmbeddingProvider):
        result = provider_with_model.embed(_TEXTS)
        assert len(result) == len(_TEXTS)

    def test_vector_dimension(self, provider_with_model: BGEM3EmbeddingProvider):
        result = provider_with_model.embed(_TEXTS)
        assert all(len(v) == _DIM for v in result)

    def test_returns_list_of_floats(self, provider_with_model: BGEM3EmbeddingProvider):
        result = provider_with_model.embed(_TEXTS)
        assert all(isinstance(v[0], float) for v in result)

    def test_calls_model_with_return_dense_true(self, provider: BGEM3EmbeddingProvider):
        mock = _make_model_mock(len(_TEXTS))
        provider._model = mock
        provider.embed(_TEXTS)
        _, kwargs = mock.encode.call_args
        assert kwargs["return_dense"] is True

    def test_calls_model_with_return_sparse_false(self, provider: BGEM3EmbeddingProvider):
        mock = _make_model_mock(len(_TEXTS))
        provider._model = mock
        provider.embed(_TEXTS)
        _, kwargs = mock.encode.call_args
        assert kwargs["return_sparse"] is False

    def test_empty_input_returns_empty(self, provider: BGEM3EmbeddingProvider):
        assert provider.embed([]) == []

    def test_batch_size_forwarded(self, provider: BGEM3EmbeddingProvider):
        mock = _make_model_mock(1)
        provider._model = mock
        provider.embed(["text"])
        _, kwargs = mock.encode.call_args
        assert kwargs["batch_size"] == provider.batch_size


# ── embed_sparse() ─────────────────────────────────────────────────────────────


class TestEmbedSparse:
    def test_returns_one_dict_per_text(self, provider_with_model: BGEM3EmbeddingProvider):
        result = provider_with_model.embed_sparse(_TEXTS)
        assert len(result) == len(_TEXTS)

    def test_keys_are_ints(self, provider_with_model: BGEM3EmbeddingProvider):
        result = provider_with_model.embed_sparse(_TEXTS)
        for d in result:
            assert all(isinstance(k, int) for k in d)

    def test_values_are_floats(self, provider_with_model: BGEM3EmbeddingProvider):
        result = provider_with_model.embed_sparse(_TEXTS)
        for d in result:
            assert all(isinstance(v, float) for v in d.values())

    def test_zero_weights_filtered(self, provider: BGEM3EmbeddingProvider):
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": np.zeros((1, _DIM), dtype="float32"),
            "lexical_weights": [{"1": 0.0, "2": 0.5, "3": 0.0}],
        }
        provider._model = mock
        result = provider.embed_sparse(["text"])
        assert 0.0 not in result[0].values()
        assert result[0].get(2) == pytest.approx(0.5)

    def test_string_keys_converted_to_int(self, provider: BGEM3EmbeddingProvider):
        mock = MagicMock()
        mock.encode.return_value = {
            "dense_vecs": np.zeros((1, _DIM), dtype="float32"),
            "lexical_weights": [{"101": 0.9, "2023": 0.4}],
        }
        provider._model = mock
        result = provider.embed_sparse(["text"])
        assert 101 in result[0]
        assert 2023 in result[0]

    def test_empty_input_returns_empty(self, provider: BGEM3EmbeddingProvider):
        assert provider.embed_sparse([]) == []


# ── embed_both() ───────────────────────────────────────────────────────────────


class TestEmbedBoth:
    def test_returns_tuple_of_dense_and_sparse(self, provider: BGEM3EmbeddingProvider):
        mock = _make_model_mock(len(_TEXTS))
        provider._model = mock
        dense, sparse = provider.embed_both(_TEXTS)
        assert len(dense) == len(_TEXTS)
        assert len(sparse) == len(_TEXTS)

    def test_single_model_call(self, provider: BGEM3EmbeddingProvider):
        mock = _make_model_mock(len(_TEXTS))
        provider._model = mock
        provider.embed_both(_TEXTS)
        assert mock.encode.call_count == 1

    def test_both_flags_set(self, provider: BGEM3EmbeddingProvider):
        mock = _make_model_mock(len(_TEXTS))
        provider._model = mock
        provider.embed_both(_TEXTS)
        _, kwargs = mock.encode.call_args
        assert kwargs["return_dense"] is True
        assert kwargs["return_sparse"] is True


# ── Error handling ─────────────────────────────────────────────────────────────


class TestErrorHandling:
    def test_missing_model_raises_embedding_error(self, provider: BGEM3EmbeddingProvider):
        with (
            patch(
                "src.infrastructure.embeddings.bge_m3.BGEM3EmbeddingProvider._get_model",
                side_effect=EmbeddingError("model not found"),
            ),
            pytest.raises(EmbeddingError),
        ):
            provider.embed(["text"])

    def test_encode_failure_raises_embedding_error(self, provider: BGEM3EmbeddingProvider):
        mock = MagicMock()
        mock.encode.side_effect = RuntimeError("CUDA OOM")
        provider._model = mock
        with pytest.raises(EmbeddingError) as exc_info:
            provider.embed(["text"])
        assert exc_info.value.cause is not None

    def test_import_error_raises_embedding_error(self, provider: BGEM3EmbeddingProvider):
        with patch.dict("sys.modules", {"FlagEmbedding": None}), pytest.raises(EmbeddingError):
            provider._get_model()
